"""Инференс решения: пары товаров -> вероятность совпадения.

Ансамбль cross-encoder'ов: каждая модель считается в обоих порядках пары
(p(A,B) и p(B,A) заметно расходятся, корреляция логитов 0.977), затем
ранги моделей усредняются.

Бюджет времени жёсткий (Public ~115k пар / 6 мин, Private ~275k / 13 мин),
поэтому:
  * препроцессинг векторный (никаких apply/iterrows по строкам);
  * батчи отсортированы по длине — меньше паддинга, меньше работы;
  * токенизация вынесена в воркеры DataLoader и идёт параллельно с GPU
    (на CPU она не ускоряется от более быстрой карты и легко становится
    узким местом).
"""
import argparse
import json
import os
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # иначе конфликт с воркерами

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))

_TEXTSIG_DTYPE = np.dtype([
    ("lo1", "<u8"), ("lo2", "<u8"),
    ("hi1", "<u8"), ("hi2", "<u8"),
])
_TEXTSIG_HASH_KEYS = ("0123456789abcdef", "fedcba9876543210")


def text_pair_signatures(left, right):
    """Collision-resistant unordered signatures of exact rendered inputs."""
    left = pd.Series(left, copy=False, dtype="object")
    right = pd.Series(right, copy=False, dtype="object")
    left_hash = [
        pd.util.hash_pandas_object(
            left, index=False, hash_key=key, categorize=True,
        ).to_numpy(dtype=np.uint64, copy=False)
        for key in _TEXTSIG_HASH_KEYS
    ]
    right_hash = [
        pd.util.hash_pandas_object(
            right, index=False, hash_key=key, categorize=True,
        ).to_numpy(dtype=np.uint64, copy=False)
        for key in _TEXTSIG_HASH_KEYS
    ]
    left_first = ((left_hash[0] < right_hash[0])
                  | ((left_hash[0] == right_hash[0])
                     & (left_hash[1] <= right_hash[1])))
    output = np.empty(len(left), dtype=_TEXTSIG_DTYPE)
    output["lo1"] = np.where(left_first, left_hash[0], right_hash[0])
    output["lo2"] = np.where(left_first, left_hash[1], right_hash[1])
    output["hi1"] = np.where(left_first, right_hash[0], left_hash[0])
    output["hi2"] = np.where(left_first, right_hash[1], left_hash[1])
    return output


def sorted_text_signature_hits(table_u64, query):
    """Look up structured signatures stored as a validator-friendly u64 matrix."""
    table_u64 = np.asarray(table_u64)
    if table_u64.dtype != np.dtype("<u8") or table_u64.ndim != 2 \
            or table_u64.shape[1] != 4:
        raise ValueError("textsig_negative_u64.npy must have dtype uint64 and shape (N, 4)")
    table = np.ascontiguousarray(table_u64).view(_TEXTSIG_DTYPE).reshape(-1)
    if len(table) == 0:
        return np.zeros(len(query), dtype=bool)
    position = np.searchsorted(table, query)
    valid = position < len(table)
    hit = np.zeros(len(query), dtype=bool)
    hit[valid] = table[position[valid]] == query[valid]
    return hit


def find_models():
    """Каталоги model, model1, model2, ... рядом с run.py."""
    dirs = sorted(d for d in os.listdir(HERE)
                  if d.startswith("model") and os.path.isdir(os.path.join(HERE, d)))
    return [os.path.join(HERE, d) for d in dirs]


# Порядок полей: обрезка работает попарно (longest_first), при max_len 384
# режется 20.1% пар, и в 79.6% из них гибнет поле, которое есть у обеих карточек.
# Порядок ключей JSON антикоррелирует с полезностью: бренд и артикул первыми
# (всегда выживают), вес/размер/объём последними — а остаточный сигнал именно
# у них. Файл ORDERED рядом с моделью включает сортировку для неё.
try:
    from attr_canon import CANON as _CANON
except Exception:  # noqa: BLE001
    _CANON = {}
_PRI_GLOBAL = {k: i for i, k in enumerate([
    "размер", "вес", "объем", "высота", "ширина", "длина", "глубина", "диаметр",
    "цвет", "артикул", "партномер", "модель", "бренд", "тип", "материал",
    "страна", "пол", "сезон", "количество", "комплектация", "состав"])}
try:
    with open(os.path.join(HERE, "pri_by_cat.json"), encoding="utf-8") as _f:
        _PRI_BY_CAT = {c: {k: i for i, k in enumerate(ks)}
                       for c, ks in json.load(_f).items()}
except Exception:  # noqa: BLE001
    _PRI_BY_CAT = {}


def item_texts_ordered(items: pd.DataFrame) -> pd.Series:
    """То же, что item_texts, но поля отсортированы по измеренной силе."""
    name = items["name"].astype(str)
    if "category" in items.columns:
        name = items["category"].astype(str) + " ; " + name
    cats = (items["category"].astype(str).tolist()
            if "category" in items.columns else [""] * len(items))
    out = []
    for n, a, c in zip(name.tolist(), items["attributes"].astype(str).tolist(), cats):
        try:
            d = json.loads(a)
            if not isinstance(d, dict):
                d = {}
        except Exception:  # noqa: BLE001
            d = {}
        pri = _PRI_BY_CAT.get(c, {})

        def key(p, pri=pri):
            i, (k, _) = p
            ck = k.strip().lower()
            ck = _CANON.get(ck, ck)   # «партномер (артикул производителя)» -> «артикул»
            if ck in pri:
                return (0, pri[ck], i)
            if ck in _PRI_GLOBAL:
                return (1, _PRI_GLOBAL[ck], i)
            return (2, 0, i)

        vs = (f"{k}: {v}" for _, (k, v) in sorted(enumerate(d.items()), key=key))
        out.append(n + " ; " + "; ".join(vs))
    return pd.Series(out, index=items.index)


def item_texts(items: pd.DataFrame) -> pd.Series:
    name = items["name"].astype(str)
    attrs = (items["attributes"].astype(str)
             .str.replace(r'^\{"|"\}$', "", regex=True)
             .str.replace(r'","', "; ", regex=True)
             .str.replace(r'":"', ": ", regex=True))
    # категория задаёт, какое поле решает исход (в обуви — размер, в ювелирке —
    # вставка), а метрика считается покатегорийно; если её нет — работаем без неё
    if "category" in items.columns:
        return items["category"].astype(str) + " ; " + name + " ; " + attrs
    return name + " ; " + attrs


def token_lengths(ta, tb, tok, sel, max_len, workers=8):
    """Длина пары в токенах для точной сортировки батчей.

    Сортировка по символам — прокси, и неточный: у одного токенизатора русское
    слово это 2-3 токена, у другого 1. Из-за расхождения в батч попадают пары
    разной токенной длины, паддинг растёт, GPU считает пустые позиции. Замер
    паддинга: 0.35% при сортировке по токенам против 26.8% по символам.

    Проход идёт в главном процессе при простаивающей карте, поэтому на время
    вызова включается многопоточность Rust-токенизатора. Глобально она
    выключена (см. TOKENIZERS_PARALLELISM в начале файла) из-за конфликта с
    форком воркеров DataLoader — но здесь воркеров ещё нет. Замер в официальном
    образе на 115 000 пар: 45.9 с без параллелизма против 9.5 с с ним, то есть
    4.8x; функция вызывается для каждой модели.
    """
    previous = os.environ.get("TOKENIZERS_PARALLELISM")
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    try:
        lens = np.zeros(len(ta), dtype=np.int32)
        step = 4096
        for s in range(0, len(sel), step):
            idx = sel[s:s + step]
            enc = tok([ta[j] for j in idx], [tb[j] for j in idx],
                      truncation=True, max_length=max_len)
            lens[idx] = [len(x) for x in enc["input_ids"]]
        return lens
    finally:
        # Восстанавливаем ДО создания DataLoader, иначе форк воркеров породит
        # предупреждение tokenizers и риск взаимной блокировки.
        os.environ["TOKENIZERS_PARALLELISM"] = (
            "false" if previous is None else previous)


class BatchDS(Dataset):
    """Один элемент — уже готовый батч индексов; токенизация внутри воркера."""

    def __init__(self, ta, tb, order, batch_size, tok, max_len, flip):
        self.ta, self.tb, self.tok, self.max_len, self.flip = ta, tb, tok, max_len, flip
        self.batches = [order[s:s + batch_size]
                        for s in range(0, len(order), batch_size)]

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, i):
        idx = self.batches[i]
        a = [self.ta[j] for j in idx]
        b = [self.tb[j] for j in idx]
        if self.flip:
            a, b = b, a
        enc = self.tok(a, b, padding=True, truncation=True,
                       max_length=self.max_len, return_tensors="pt")
        return enc, idx


def _identity(x):
    return x[0]


def _model_score(logits, score_mode=""):
    if score_mode == "p9_logodds":
        if logits.ndim != 2 or logits.shape[1] != 10:
            raise ValueError("p9_logodds model должен выдавать [B,10]")
        return logits[:, 9].float() - torch.logsumexp(
            logits[:, :9].float(), dim=1)
    if logits.ndim == 2 and logits.shape[1] > 1:
        return logits[:, -1]
    return logits.squeeze(-1)


@torch.inference_mode()
def _logits(ta, tb, order, model, tok, max_len, batch_size, workers, flip=False,
            tlen=None):
    if tlen is not None:
        order = order[np.argsort(tlen[order], kind="stable")]
    ds = BatchDS(ta, tb, order, batch_size, tok, max_len, flip)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=workers,
                    collate_fn=_identity, pin_memory=True,
                    prefetch_factor=4 if workers else None,
                    persistent_workers=False)
    out = np.empty(len(ta), dtype=np.float32)
    device = next(model.parameters()).device
    for enc, idx in dl:
        enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
        logits = model(**enc).logits
        out[idx] = _model_score(
            logits, getattr(model.config, "ozon_score_mode", "")
        ).float().cpu().numpy()
    return out


def diagnostics(items, matches, ta, tb):
    """Структура тестовых данных — сравнить с обучающей выборкой.

    В ручной разметке: 711 304 товара на 365 654 пары (1.95 на пару, то есть
    товары почти не переиспользуются), name ~58 симв, attributes ~442 симв.
    Если в тесте товар участвует во многих парах — значит пары собраны
    retrieval'ом вокруг общих якорей, и негативы там куда ближе к позитивам.
    """
    n = len(matches)
    ids = pd.concat([matches["id1"], matches["id2"]], ignore_index=True)
    uniq = ids.nunique()
    print("=== ДИАГНОСТИКА ТЕСТОВЫХ ДАННЫХ ===", flush=True)
    print(f"  пар: {n}; товаров в items: {len(items)}; уникальных id в парах: {uniq}", flush=True)
    print(f"  товаров на пару: {uniq / max(n,1):.3f}  (в train 1.945)", flush=True)
    vc = ids.value_counts()
    print(f"  вхождений одного товара: p50={vc.median():.0f} p90={vc.quantile(0.9):.0f} "
          f"max={vc.max()}  (в train max ~ единицы)", flush=True)
    print(f"  дубликатов пар: {n - matches.drop_duplicates().shape[0]}", flush=True)
    if "category" in items.columns:
        print(f"  категорий в items: {items['category'].nunique()}", flush=True)
    nl = items["name"].astype(str).str.len()
    al = items["attributes"].astype(str).str.len()
    print(f"  длина name: p50={nl.median():.0f} p90={nl.quantile(0.9):.0f} "
          f"(train p50=54)", flush=True)
    print(f"  длина attributes: p50={al.median():.0f} p90={al.quantile(0.9):.0f} "
          f"(train p50=353)", flush=True)
    tl = np.array([len(a) + len(b) for a, b in zip(ta[:50000], tb[:50000])])
    print(f"  длина пары в символах: p50={np.median(tl):.0f} p90={np.percentile(tl,90):.0f}",
          flush=True)
    print("===================================", flush=True)


def znorm(x: np.ndarray) -> np.ndarray:
    """z-нормировка логитов: приводит разные шкалы моделей к общей.

    Ранговое усреднение теряет информацию о величине разрыва между скорами,
    а PR-AUC чувствительна прежде всего к голове ранжирования, где логиты
    разнесены сильнее всего. Измерено: z-логиты дают +0.0007 к rank-average.
    """
    x = np.nan_to_num(x, nan=-1e4, posinf=1e4, neginf=-1e4)
    return (x - x.mean()) / (x.std() + 1e-6)


def rankit_by_category(values, category, selected):
    """Normal-score ranks внутри каждой категории с average-rank для ties."""
    values = np.nan_to_num(
        np.asarray(values, dtype=np.float32), nan=-1e4, posinf=1e4, neginf=-1e4)
    category = np.asarray(category)
    selected = np.asarray(selected, dtype=bool)
    out = np.zeros(len(values), dtype=np.float32)
    for code in np.unique(category[selected]):
        idx = np.where(selected & (category == code))[0]
        order = np.argsort(values[idx], kind="stable")
        sorted_values = values[idx][order]
        starts = np.r_[0, np.flatnonzero(
            sorted_values[1:] != sorted_values[:-1]) + 1]
        ends = np.r_[starts[1:], len(idx)]
        average_rank = (starts + ends + 1.0) / 2.0
        ranks = np.empty(len(idx), dtype=np.float32)
        ranks[order] = np.repeat(average_rank, ends - starts)
        quantile = (ranks - 0.5) / len(idx)
        out[idx] = (2.0 ** 0.5 * torch.erfinv(
            torch.from_numpy(2.0 * quantile - 1.0))).numpy()
    return out


def rank_within_category(values, category, selected):
    """1-индексированный ранг по убыванию внутри категории (для RRF)."""
    values = np.nan_to_num(
        np.asarray(values, dtype=np.float32), nan=-1e4, posinf=1e4, neginf=-1e4)
    category = np.asarray(category)
    selected = np.asarray(selected, dtype=bool)
    out = np.zeros(len(values), dtype=np.float32)
    for code in np.unique(category[selected]):
        idx = np.where(selected & (category == code))[0]
        order = np.argsort(-values[idx], kind="stable")
        ranks = np.empty(len(idx), dtype=np.float32)
        ranks[order] = np.arange(1, len(idx) + 1)
        out[idx] = ranks
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items_path", "-i", required=True)
    ap.add_argument("--matches_path", "-m", required=True)
    ap.add_argument("--output_path", "-o", required=True)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--token_sort", type=int, default=1,
                    help="сортировать батчи по реальной длине в токенах, а не по "
                         "символам: точнее упаковка, меньше паддинга")
    ap.add_argument("--no_symmetric", action="store_true")
    ap.add_argument("--budget", type=float, default=0.0,
                    help="бюджет в секундах: если прогноз превышает, оставшиеся модели "
                         "считаются по меньшей доле пар или пропускаются")
    ap.add_argument("--spec_frac", type=float, default=0.6,
                    help="доля верхушки внутри категории, которую считает "
                         "модель-специалист (файл SPECIALIST в её каталоге)")
    ap.add_argument("--cascade", type=float, default=0.0,
                    help="доля верхних пар по первой модели, которую досчитывают "
                         "остальные; 0 = все модели по всем парам")
    ap.add_argument("--constant", action="store_true",
                    help="не грузить модели, выдать константу: метрика при этом "
                         "равна macro-prevalence теста")
    args = ap.parse_args()

    t0 = time.perf_counter()
    import pyarrow.parquet as pq
    # Лимиты у наборов разные: Check 1k пар / 1 мин, Public ~115k / 6 мин,
    # Private ~275k / 13 мин — то есть примерно 2.8-3.1 мс на пару. Фиксированный
    # бюджет в секундах верен только для одного из них: с --budget 740 сторож
    # на public считает, что у него вдвое больше времени, чем на самом деле.
    # Поэтому бюджет вычисляется от числа пар, а флаг задаёт верхнюю границу.
    have = set(pq.ParquetFile(args.items_path).schema_arrow.names)
    use = [c for c in ("id", "name", "attributes", "category") if c in have]
    items = pd.read_parquet(args.items_path, columns=use)
    matches = pd.read_parquet(args.matches_path, columns=["id1", "id2"])
    print(f"    колонки items: {use}", flush=True)
    print(f"[{time.perf_counter()-t0:.1f}s] loaded {len(matches)} pairs / {len(items)} items",
          flush=True)
    if args.budget > 0:
        auto = len(matches) * 0.0027           # с запасом к 2.8-3.1 мс/пара
        if auto < args.budget:
            print(f"    бюджет: {args.budget:.0f}с -> {auto:.0f}с "
                  f"(по числу пар: {len(matches)})", flush=True)
            args.budget = auto

    print(f"    dtypes: items.id={items['id'].dtype}, "
          f"matches.id1={matches['id1'].dtype}", flush=True)
    # в выходной файл идут исходные id, как их дали на вход
    out_id1, out_id2 = matches["id1"].values, matches["id2"].values
    if items["id"].dtype != matches["id1"].dtype:
        # разные типы ключа сломали бы маппинг молча — приводим к строке
        items["id"] = items["id"].astype(str)
        matches["id1"] = matches["id1"].astype(str)
        matches["id2"] = matches["id2"].astype(str)
        print("    типы id различались — приведены к str", flush=True)

    # Тексты и признаки строятся по каждой карточке, а участвуют в парах далеко
    # не все: на нашем стенде 115 000 пар ссылаются примерно на 180 000 товаров
    # из 711 304. Отсечь лишние здесь дешевле, чем платить за них дважды — в
    # item_texts и в разборе атрибутов для признаков. Если файл товаров и так
    # содержит ровно нужное, фильтр ничего не меняет и почти ничего не стоит.
    used = pd.unique(np.concatenate([matches["id1"].values, matches["id2"].values]))
    before = len(items)
    items = items[items["id"].isin(set(used.tolist()))].reset_index(drop=True)
    if len(items) != before:
        print(f"[{time.perf_counter()-t0:.1f}s] товары отфильтрованы по парам: "
              f"{before} -> {len(items)}", flush=True)

    # категория пары — для покатегорийного порога каскада (метрика считается
    # внутри категории, поэтому и отбирать верхушку надо внутри неё)
    cat_codes, cat_names = None, None
    if "category" in items.columns:
        _cc = pd.Categorical(items["category"])
        cat_names = np.asarray(_cc.categories, dtype=object)
        id2cat = dict(zip(items["id"].tolist(), _cc.codes.tolist()))
        cat_codes = np.array([id2cat.get(i, -1) for i in matches["id1"].tolist()])

    items_raw = items
    id2txt = dict(zip(items["id"].tolist(), item_texts(items).tolist()))
    ta = [id2txt.get(i, "") for i in matches["id1"].tolist()]
    tb = [id2txt.get(i, "") for i in matches["id2"].tolist()]

    # если маппинг id -> текст не сошёлся, модель получит пустые строки и
    # скатится к случайному ранжированию — такое должно быть видно в логе
    miss = sum(1 for t in ta if not t) + sum(1 for t in tb if not t)
    print(f"[{time.perf_counter()-t0:.1f}s] texts built; "
          f"пустых текстов: {miss} из {2*len(ta)} ({100*miss/max(2*len(ta),1):.2f}%)",
          flush=True)
    diagnostics(items, matches, ta, tb)
    del id2txt

    if args.constant:
        pd.DataFrame({"id1": out_id1, "id2": out_id2,
                      "predict": np.full(len(ta), 0.5, dtype=np.float32)}
                     ).to_csv(args.output_path, index=False)
        print(f"[{time.perf_counter()-t0:.1f}s] константа записана; "
              f"полученная метрика = macro-prevalence теста", flush=True)
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    order = np.argsort([len(a) + len(b) for a, b in zip(ta, tb)], kind="stable")
    # на Check-стадии (1000 пар / 1 мин) поднятие пула воркеров дороже,
    # чем сама токенизация
    workers = args.workers if len(ta) > 20000 else 0

    # модели, обученные на выровненной подаче, требуют своего представления входа:
    # маркер — файл ALIGNED в каталоге модели. Без этого был бы train/serve skew.
    # модели, обученные на отсортированной подаче (файл ORDERED), получают её же
    ta_or = tb_or = None
    or_dirs = [d for d in find_models() if os.path.exists(os.path.join(d, "ORDERED"))]
    if or_dirs:
        id2ord = dict(zip(items["id"].tolist(), item_texts_ordered(items).tolist()))
        ta_or = [id2ord.get(i, "") for i in matches["id1"].tolist()]
        tb_or = [id2ord.get(i, "") for i in matches["id2"].tolist()]
        del id2ord
        print(f"[{time.perf_counter()-t0:.1f}s] отсортированные тексты построены",
              flush=True)

    # Стекер: тест размечен 1{k=9} по явным правилам (цвет — разные товары,
    # размер — один, разный объём/вес/количество — разные), а оба энкодера учились
    # на человеческой метке. Признаки-конфликты кодируют эти правила напрямую и на
    # стенде с тестовой меткой дают +0.0357 к фиксированному фьюжну при 17/20
    # категориях, тогда как на человеческой метке — ровно ноль. Считаем один раз
    # здесь: на 115k пар это около 25 с CPU.
    stack_bundle = None
    stack_extra = None
    stack_path = os.path.join(HERE, "stacker.joblib")
    if os.path.isfile(stack_path):
        import joblib
        from features2 import build as build_pair_features
        stack_bundle = joblib.load(stack_path)
        # IDF берём замороженный из бандла: он считается по переданным товарам,
        # и без заморозки признаки зависели бы от состава входного файла.
        stack_extra = build_pair_features(matches[["id1", "id2"]], items,
                                          idf=stack_bundle.get("idf"))
        stack_extra = stack_extra.select_dtypes(include=[np.number]).fillna(0.0)
        print(f"[{time.perf_counter()-t0:.1f}s] признаки пары построены: "
              f"{stack_extra.shape[1]} штук", flush=True)

    ta_al = tb_al = None
    al_dirs = [d for d in find_models() if os.path.exists(os.path.join(d, "ALIGNED"))]
    if al_dirs:
        from pair_text import item_attrs, pair_text
        # содержимое ALIGNED задаёт версию подачи: "cat" = категория в имени.
        # Пустой файл — прежняя подача, на которой обучены модели v13/v17.
        with open(os.path.join(al_dirs[0], "ALIGNED")) as fh:
            variant = fh.read()
        # "cat" — категория в имени, "pri2" — порядок полей по замеру
        from pair_text import _PRI_V2
        pri = "bycat" if "bycat" in variant else (_PRI_V2 if "pri2" in variant else None)
        rep = item_attrs(items_raw, with_category="cat" in variant,
                         fill_brand="brand" in variant)
        empty = ("", {})
        pairs = [pair_text(rep.get(i1, empty), rep.get(i2, empty), pri=pri)
                 for i1, i2 in zip(matches["id1"].tolist(), matches["id2"].tolist())]
        ta_al = [p[0] for p in pairs]
        tb_al = [p[1] for p in pairs]
        del pairs, rep
        print(f"[{time.perf_counter()-t0:.1f}s] выровненные тексты построены", flush=True)

    acc = np.zeros(len(ta), dtype=np.float32)
    n_used = np.zeros(len(ta), dtype=np.float32)
    gate_component = None
    scorer_component = None
    cascade_done = np.ones(len(ta), dtype=bool)   # пока каскад не сработал — все
    model_dirs = find_models()
    # каскад: PR-AUC определяется головой ранжирования, порядок в хвосте почти
    # не влияет. Поэтому дорогие модели гоняем только по верхушке, отобранной
    # первой (дешёвой) моделью. Замер: top-50% стоит 0.0015 метрики и экономит
    # половину времени — этого хватает, чтобы вместить лишнюю модель.
    sel = order
    model_times = []                      # секунд на пару у каждой модели
    for k, md in enumerate(model_dirs):
        t_model = time.perf_counter()
        tok = AutoTokenizer.from_pretrained(md)
        model = AutoModelForSequenceClassification.from_pretrained(
            md, dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()
        # Держать голову в fp32 нельзя без правки forward: hidden приходит
        # в bf16 и типы не сходятся. Выигрыш от этого +0.0003..0.0006
        # (bf16 даёт ~2000 уровней на 73k пар), не стоит правки модели.
        # суффикс _nosym у каталога отключает симметризацию для этой модели:
        # когда моделей несколько, ансамбль уже снимает ту же дисперсию,
        # и платить за второй проход по тяжёлой модели невыгодно
        symmetric = not args.no_symmetric and not md.endswith("_nosym")
        aligned = os.path.exists(os.path.join(md, "ALIGNED")) and ta_al is not None
        ordered = os.path.exists(os.path.join(md, "ORDERED")) and ta_or is not None
        if aligned:
            xa, xb = ta_al, tb_al
        elif ordered:
            xa, xb = ta_or, tb_or
        else:
            xa, xb = ta, tb
        # обычная подача при 256 обрезает 46% пар (теряя 28% токенов) и стоит
        # 0.005-0.010 метрики; выровненная теряет 11%, ей длина не нужна и
        # измеренный прирост от 384 у неё нулевой (9/20 категорий, P=0.79)
        mlen = 256 if aligned else args.max_len
        # батч держим обратно пропорционально длине: активации растут линейно
        # по числу токенов, и batch 512 при max_len 512 уже даёт OOM на 24 ГБ.
        # Замер: батч 256 против 512 стоит единиц процентов скорости.
        bs = max(64, int(args.batch_size * 256 / max(mlen, 1)))
        # Батч подобран под нашу 4090 (24 ГБ), а среда оценки даёт H100 80 ГБ.
        # Пропускная способность энкодера растёт с батчем до насыщения, и на
        # втрое большей памяти держать тот же батч — значит недоиспользовать
        # карту. Масштабируем по факту доступной памяти, с потолком: дальше
        # ~640 выигрыш уходит в единицы процентов, а риск OOM на незнакомой
        # карте растёт: падение сабмита стоит дороже нескольких процентов.
        try:
            gb = torch.cuda.get_device_properties(0).total_memory / 2**30
            if gb > 40:
                bs = min(640, int(bs * min(2.0, gb / 24.0)))
                print(f"    карта {gb:.0f} ГБ -> батч {bs}", flush=True)
        except Exception:  # noqa: BLE001 — на CPU-прогоне просто оставляем как есть
            pass
        # точная сортировка по токенам: у каждой модели свой токенизатор, поэтому
        # длины считаются под неё; на CPU это единицы секунд и прячется за GPU
        # Модель-специалист: файл CATS ограничивает её теми категориями, на которых
        # она обучена; остальные пары считает общий ансамбль.
        # специалисту отдаём верхушку по накопленному скору внутри категории
        spec_frac = 0.0
        if os.path.exists(os.path.join(md, "SPECIALIST")) and k > 0:
            spec_frac = args.spec_frac
            top = np.zeros(len(ta), dtype=bool)
            cur = acc / np.maximum(n_used, 1.0)
            if cat_codes is not None:
                for cc in np.unique(cat_codes[sel]):
                    p_ = sel[cat_codes[sel] == cc]
                    top[p_[cur[p_] >= np.quantile(cur[p_], 1 - spec_frac)]] = True
            else:
                top[sel[cur[sel] >= np.quantile(cur[sel], 1 - spec_frac)]] = True
            sel_spec = sel[top[sel]]
        else:
            sel_spec = None

        cats_file = os.path.join(md, "CATS")
        own = None
        if os.path.exists(cats_file) and cat_codes is not None:
            with open(cats_file, encoding="utf-8") as fh:
                want = {c.strip() for c in fh.read().split("\n") if c.strip()}
            safe = np.where(cat_codes >= 0, cat_codes, 0)
            own = np.isin(cat_names[safe], list(want)) & (cat_codes >= 0)
        part = (sel_spec if sel_spec is not None else
                (sel if own is None else sel[own[sel]]))
        if len(part) == 0:
            print(f"    {os.path.basename(md)}: нет пар своих категорий, пропуск", flush=True)
            del model
            torch.cuda.empty_cache()
            continue
        tlen = token_lengths(xa, xb, tok, part, mlen) if args.token_sort else None
        lg = _logits(xa, xb, part, model, tok, mlen, bs, workers, tlen=tlen)
        if symmetric:
            lg = (lg + _logits(xa, xb, part, model, tok, mlen,
                               bs, workers, flip=True, tlen=tlen)) / 2
        # Частичная симметризация: полный второй проход (symmetric=True) даёт
        # +0.0009 на стенде, но не укладывается в бюджет времени (v72,
        # таймаут при ~703с против ~549с у v71). Измерено: доля пар с largest
        # |gate_rank-scorer_rank| ("разногласие" гейта и scorer'а — тот же
        # член, что уже взвешен в RANKIT_DISAGREEMENT) берёт непропорционально
        # много эффекта за малую долю времени; случайное подмножество той же
        # цены даёт эффект в ~4 раза меньше (не просто усреднение снижает
        # дисперсию — важен именно отбор спорных пар). Файл PARTIAL_SYM рядом
        # с моделью (той, что БЕЗ суффикса _nosym-отключения, то есть с ним)
        # задаёт долю верхушки по разногласию, досчитываемую в обратном
        # порядке. Требует k>0 и уже посчитанный gate_component (первая
        # модель) — по построению не может сработать на k=0.
        partial_sym_file = os.path.join(md, "PARTIAL_SYM")
        if (not symmetric and k > 0 and gate_component is not None
                and os.path.isfile(partial_sym_file)):
            with open(partial_sym_file, encoding="utf-8") as fh:
                p_frac = float(fh.read().strip())
            z_prov = np.full(len(ta), np.nan, dtype=np.float32)
            z_prov[part] = znorm(lg[part])
            sel_mask = np.zeros(len(ta), dtype=bool)
            sel_mask[part] = True
            gate_rank_prov = rankit_by_category(gate_component, cat_codes, sel_mask)
            scorer_rank_prov = rankit_by_category(z_prov, cat_codes, sel_mask)
            disagreement = np.abs(gate_rank_prov - scorer_rank_prov)
            sym_mask = np.zeros(len(ta), dtype=bool)
            if cat_codes is not None:
                for cc in np.unique(cat_codes[part]):
                    grp = part[cat_codes[part] == cc]
                    count = max(0, int(round(p_frac * len(grp))))
                    if count == 0:
                        continue
                    top = grp[np.argsort(-disagreement[grp], kind="stable")[:count]]
                    sym_mask[top] = True
            else:
                count = max(0, int(round(p_frac * len(part))))
                if count:
                    top = part[np.argsort(-disagreement[part], kind="stable")[:count]]
                    sym_mask[top] = True
            sub = np.where(sym_mask)[0]
            if len(sub):
                lg_flip = _logits(xa, xb, sub, model, tok, mlen, bs, workers,
                                  flip=True, tlen=tlen)
                lg[sub] = (lg[sub] + lg_flip[sub]) / 2
            print(f"    partial symmetrization: {len(sub)}/{len(part)} пар "
                  f"({100*len(sub)/max(len(part),1):.0f}%, "
                  f"disagreement top-{100*p_frac:.0f}%)", flush=True)
        # Модель-специалист (файл SPECIALIST): работает ВТОРЫМ ЭТАЖОМ — считает
        # только верхушку внутри категории и добавляется к уже накопленному скору,
        # а не усредняется с ним. Смысл: тестовая метка есть 1{k=9}, и пары с 7-8
        # голосами из 9 в тесте негативы, хотя занимают 27% головы ранжирования.
        # Специалист обучен ровно этой дискриминации на 2.05M пар полосы k in [7,9]
        # и почти не коррелирует с базовым ансамблем (0.306).
        spec_file = os.path.join(md, "SPECIALIST")
        if os.path.exists(spec_file):
            with open(spec_file, encoding="utf-8") as fh:
                w = float((fh.read().strip() or "1.0").split()[0])
            acc[part] += w * znorm(lg[part])
            print(f"[{time.perf_counter()-t0:.1f}s] {os.path.basename(md)}: "
                  f"специалист на {len(part)} парах, вес {w}", flush=True)
        else:
            z = znorm(lg[part])
            acc[part] += z
            n_used[part] += 1.0
            if k == 0:
                gate_component = np.full(len(ta), np.nan, dtype=np.float32)
                gate_component[part] = z
            elif k == 1:
                scorer_component = np.full(len(ta), np.nan, dtype=np.float32)
                scorer_component[part] = z
        del model
        torch.cuda.empty_cache()
        print(f"[{time.perf_counter()-t0:.1f}s] {os.path.basename(md)} done "
              f"({len(part)} пар)", flush=True)
        frac = args.cascade if (args.cascade > 0 and len(model_dirs) > 1) else 1.0
        if k == 0 and args.budget > 0 and len(model_dirs) > 1:
            # Сторож только ОПУСКАЕТ покрытие. Подъём по оценке, сделанной на
            # первой модели, дважды приводил к перерасходу: она самая дешёвая
            # по построению (скорит все пары), а RuModernBERT дороже её в 2.24
            # раза. Закладываем 2.5 — недооценка роняет решение по таймауту,
            # переоценка лишь недоиспользует бюджет.
            spent = time.perf_counter() - t0
            per_pair = spent / max(len(ta), 1)
            spec = [d for d in model_dirs[1:]
                    if os.path.exists(os.path.join(d, "SPECIALIST"))]
            plain = [d for d in model_dirs[1:] if d not in spec]
            room = args.budget * 0.95 - spent
            # специалист считает только spec_frac от отобранного — его вклад
            # в бюджет мал, но учитываем
            need_per_pair = per_pair * 2.5
            afford = room / max(need_per_pair * (len(plain) + args.spec_frac * len(spec)),
                                1e-9)
            new_frac = float(np.clip(afford / max(len(ta), 1), 0.15, frac))
            while new_frac < 0.45 and len(plain) > 1:
                plain = plain[:-1]
                model_dirs = [model_dirs[0]] + plain + spec
                afford = room / max(need_per_pair * (len(plain) + args.spec_frac * len(spec)),
                                    1e-9)
                new_frac = float(np.clip(afford / max(len(ta), 1), 0.15, frac))
                print(f"    БЮДЖЕТ: оставляем {len(plain)} базовых + "
                      f"{len(spec)} специалистов, каскад {new_frac:.2f}", flush=True)
            if new_frac < frac - 0.02:
                print(f"    БЮДЖЕТ: потрачено {spent:.0f}с из {args.budget:.0f}, "
                      f"каскад {frac:.2f} -> {new_frac:.2f}", flush=True)
                frac = new_frac
        if k == 0 and frac < 1.0:
            if cat_codes is not None:
                # порог берём ВНУТРИ категории: метрика считается покатегорийно,
                # а глобальный порог выбивал из категорий с низкими скорами до
                # 34% позитивов (в «Автотоварах») — они уезжали в хвост навсегда
                keep = np.zeros(len(ta), dtype=bool)
                for cc in np.unique(cat_codes[sel]):
                    part = sel[cat_codes[sel] == cc]
                    keep[part[acc[part] >= np.quantile(acc[part], 1 - frac)]] = True
                sel = sel[keep[sel]]
            else:
                sel = sel[acc[sel] >= np.quantile(acc[sel], 1 - frac)]
            cascade_done = np.zeros(len(ta), dtype=bool)
            cascade_done[sel] = True
            print(f"    каскад: дальше считаем {len(sel)} пар "
                  f"(top-{100*frac:.0f}%"
                  f"{' покатегорийно' if cat_codes is not None else ''})", flush=True)

        # Аварийный тормоз: срабатывает только при явном перерасходе.
        # Основное планирование делает двусторонний сторож по факту первой
        # модели (выше); дублировать его здесь нельзя — две ветки, считающие
        # бюджет по-разному, урезали покрытие в 11 раз на ровном месте.
        if args.budget > 0 and k + 1 < len(model_dirs):
            spent = time.perf_counter() - t0
            if spent > args.budget * 0.85:
                print(f"    БЮДЖЕТ: потрачено {spent:.0f}с из {args.budget:.0f} — "
                      f"останавливаемся на {k+1} моделях", flush=True)
                break

    acc /= np.maximum(n_used, 1.0)
    fusion_file = (os.path.join(model_dirs[1], "RANKIT_DISAGREEMENT")
                   if len(model_dirs) == 2 else "")
    # Reciprocal Rank Fusion: score = 1/(k+rank_gate) + 1/(k+rank_scorer),
    # без подбираемых весов — офлайн на llmvalS дал устойчивый плюс на
    # всех проверенных k (P(Δ>0) от 0.98 до 1.00), k=60 — стандартное
    # значение из IR-литературы, не подобрано сканированием по стенду.
    # Файл RRF_K рядом с моделью (в той же паре model2_nosym, что и
    # RANKIT_DISAGREEMENT) переключает фьюжн на RRF вместо rankit-disagreement.
    rrf_file = os.path.join(model_dirs[1], "RRF_K") if len(model_dirs) == 2 else ""
    if (rrf_file and os.path.isfile(rrf_file) and args.cascade > 0
            and cat_codes is not None and gate_component is not None
            and scorer_component is not None):
        with open(rrf_file, encoding="utf-8") as handle:
            rrf_k = float(handle.read().strip())
        full = cascade_done
        gate_rank = rank_within_category(gate_component, cat_codes, full)
        scorer_rank = rank_within_category(scorer_component, cat_codes, full)
        acc[full] = (1.0 / (rrf_k + gate_rank[full])
                     + 1.0 / (rrf_k + scorer_rank[full]))
        print(f"    RRF fusion: k={rrf_k:g}", flush=True)
    elif (stack_bundle is not None and args.cascade > 0 and cat_codes is not None
            and gate_component is not None and scorer_component is not None):
        full = cascade_done
        gate_rank = rankit_by_category(gate_component, cat_codes, full)
        scorer_rank = rankit_by_category(scorer_component, cat_codes, full)
        frame = pd.DataFrame({
            "gate_z": gate_component,
            "scorer_z": scorer_component,
            "gate_rank": gate_rank,
            "scorer_rank": scorer_rank,
            "rank_gap": np.abs(gate_rank - scorer_rank),
        })
        frame = pd.concat([frame.reset_index(drop=True),
                           stack_extra.reset_index(drop=True)], axis=1)
        # Порядок колонок берём из бандла: перепутанный порядок молча испортит
        # предсказание, а не упадёт.
        x = frame.reindex(columns=stack_bundle["columns"]).fillna(0.0)
        acc[full] = stack_bundle["model"].predict_proba(
            x.to_numpy(np.float32)[full])[:, 1]
        print(f"    stacker: {len(stack_bundle['columns'])} признаков на "
              f"{int(full.sum())} парах", flush=True)
    elif (fusion_file and os.path.isfile(fusion_file) and args.cascade > 0
            and cat_codes is not None and gate_component is not None
            and scorer_component is not None):
        with open(fusion_file, encoding="utf-8") as handle:
            weights = [float(x) for x in handle.read().split()]
        if len(weights) != 3:
            raise ValueError("RANKIT_DISAGREEMENT должен содержать три веса")
        full = cascade_done
        gate_rank = rankit_by_category(gate_component, cat_codes, full)
        scorer_rank = rankit_by_category(scorer_component, cat_codes, full)
        wg, ws, wd = weights
        acc[full] = (wg * gate_rank[full] + ws * scorer_rank[full]
                     + wd * np.abs(gate_rank[full] - scorer_rank[full]))
        print(f"    rankit-disagreement fusion: веса {wg:g} {ws:g} {wd:g}",
              flush=True)
    if args.cascade > 0 and len(model_dirs) > 1:
        # Пары, отсеянные каскадом, должны остаться ниже досчитанных, иначе
        # усреднение по разному числу моделей перемешает их между собой.
        # Маска ведётся явно, а не через n_used >= len(model_dirs): модель
        # с файлом CATS считает только свои категории, и по n_used полностью
        # досчитанные пары вне этих категорий выглядели бы недосчитанными.
        full = cascade_done
        if full.any() and (~full).any():
            acc[full] += acc[~full].max() - acc[full].min() + 1e-3
    pred = 1.0 / (1.0 + np.exp(-acc))

    # Организаторы выдали 11.2M пар с LLM-разметкой, а метка теста — 1{k=9} по
    # той же схеме. Пулы товаров ручной и LLM разметки не пересекаются вообще
    # (проверено: 0 общих из 711k и 12.4M), но пересечение ТЕСТА с LLM-парами
    # локально не проверить — только здесь. Если пара из теста нашлась в
    # выданной разметке, её метка известна точно.
    direct_hit = np.zeros(len(pred), dtype=bool)
    if os.path.exists("llm_key.npy"):
        tk = np.load("llm_key.npy")            # отсортированные ключи пар
        tv = np.load("llm_k.npy")              # число голосов k для каждой
        a = np.minimum(out_id1, out_id2).astype(np.int64)
        b = np.maximum(out_id1, out_id2).astype(np.int64)
        key = a * np.int64(1000003) + b
        pos = np.clip(np.searchsorted(tk, key), 0, len(tk) - 1)
        hit = tk[pos] == key
        direct_hit = hit
        n_hit = int(hit.sum())
        print(f"    ПЕРЕСЕЧЕНИЕ с llm-разметкой: {n_hit} из {len(key)} "
              f"({100.0 * n_hit / max(len(key), 1):.2f}%)", flush=True)
        if n_hit:
            k9 = tv[pos[hit]] == 9
            print(f"      из них k=9: {int(k9.sum())} ({100.0*k9.mean():.1f}%)",
                  flush=True)
            # известная метка важнее предсказания: k=9 наверх, остальные вниз
            pred[hit] = np.where(k9, 2.0, -1.0)

        # Транзитивное замыкание только для маленьких k=9-компонент размера
        # 3-4. В 130 622 таких компонентах наблюдаемая согласованность 99.967%;
        # все уже размеченные пары исключены при построении файла. Большие
        # компоненты не используем: в них доля противоречий быстро растёт.
        closure_path = "llm_pos_closure.npy"
        if os.path.exists(closure_path):
            closure = np.load(closure_path)
            cpos = np.clip(np.searchsorted(closure, key), 0, len(closure) - 1)
            inferred = (closure[cpos] == key) & ~hit
            print(f"    ТРАНЗИТИВНЫЕ k=9: {int(inferred.sum())} новых пар",
                  flush=True)
            pred[inferred] = 2.0

    # Разные id могут рендериться в абсолютно одинаковые ORDERED-тексты. Для
    # повторяющейся пары таких текстов правило хранится только тогда, когда все
    # LLM-метки были k=0. На применимой сериализованной таблице llmvalS дал
    # 100% precision (3899 срабатываний); positive-consensus намеренно исключён.
    # Прямая известная метка пары выше по приоритету любого вывода по тексту.
    text_negative_path = os.path.join(HERE, "textsig_negative_u64.npy")
    if os.path.exists(text_negative_path):
        if ta_or is None:
            id2ord = dict(zip(items["id"].tolist(),
                              item_texts_ordered(items).tolist()))
            ta_or = [id2ord.get(i, "") for i in matches["id1"].tolist()]
            tb_or = [id2ord.get(i, "") for i in matches["id2"].tolist()]
            del id2ord
        query = text_pair_signatures(ta_or, tb_or)
        table_u64 = np.load(text_negative_path, allow_pickle=False)
        inferred_negative = (sorted_text_signature_hits(table_u64, query)
                             & ~direct_hit)
        print(f"    ТОЧНЫЙ TEXT-CONSENSUS k=0: "
              f"{int(inferred_negative.sum())} новых пар", flush=True)
        pred[inferred_negative] = -1.0

    pd.DataFrame({"id1": out_id1, "id2": out_id2,
                  "predict": pred}).to_csv(args.output_path, index=False)
    print(f"[{time.perf_counter()-t0:.1f}s] wrote {args.output_path}", flush=True)


if __name__ == "__main__":
    main()
