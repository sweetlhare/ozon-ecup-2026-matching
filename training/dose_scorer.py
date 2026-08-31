"""Доза новой супервизии в продовый скорер: головные пары с меткой k=9.

Единственный класс вмешательств, когда-либо давший крупный плюс на public, —
новая супервизия в ОБУЧЕНИИ (`closure408` +0.0136 инъекцией конвенции k=9
в модель, которая её не видела). Инференс-голоса провалились дважды подряд.

Здесь дообучается сам продовый скорер (не новая модель!) на 130 000 парах
головного пула с истинной меткой организаторов. Эти пары в обучении не
участвовали: пул строился из poolA_testshape с исключением эндпоинтов llmvalS.

Доза малая и по образцу closure408 (408 шагов): переучивание внутри уже
усвоенной конвенции измерено отрицательным дважды, поэтому цель — добавить
рабочую область, а не переписать модель.

Приёмка — тем же критерием, что и голоса: СОБСТВЕННЫЙ macro PR-AUC модели
против нынешнего скорера. Ни head-AUC, ни дельта фьюжна не используются.
"""
import os
import random as _random
import numpy as np, pandas as pd, torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification

SRC = os.environ.get("SRC", "artifacts/a10_scorer_parent")
OUT = os.environ.get("OUT", "artifacts/scorer_dose408")
POOL = os.environ.get("POOL", "data/head_pool_v2.parquet")
STEPS = int(os.environ.get("STEPS", "408"))
BS = int(os.environ.get("BS", "16"))
LR = float(os.environ.get("LR", "8e-6"))
MAXLEN = int(os.environ.get("MAXLEN", "640"))

d = pd.read_parquet(POOL)
# SOFT=1: мягкая метка k/9 вместо жёсткой 1{k=9}. Метрика ранговая, ей нужен
# ПОРЯДОК; жёсткий порог учит различие, определяемое одним голосом из девяти,
# то есть максимально неустойчивое к перезапуску разметки на тесте.
if os.environ.get("CALIB"):
    # Мишень = P(k'9=9 | k), откалиброванная leave-one-out по повторно
    # размеченным одинаковым парам (data/llm_k9_category_calibration.json,
    # 6.9M пар). Это вероятность ИМЕННО ТЕСТОВОЙ МЕТКИ, а не доля голосов.
    # k/9 приписывает уровню k=4 значение 0.444, тогда как честная
    # вероятность там 0.0139 — как у k=0. Отсюда пик и спад: ранговый сигнал
    # полезен, абсолютная шкала k/9 систематически смещена.
    import json as _json
    from calibrate_llm_targets import target_to_k as _t2k
    _c = _json.load(open("data/llm_k9_category_calibration.json"))
    _p = {int(r["k"]): float(r["probability"]) for r in _c["table"]}
    _iso = [_p[i] for i in range(10)]
    for i in range(1, 10):            # изотоническое сглаживание: убираем
        _iso[i] = max(_iso[i], _iso[i-1])   # немонотонность k=6 < k=5
    _kk = _t2k(d.target.to_numpy())
    d["y"] = np.asarray([_iso[int(x)] for x in _kk], dtype=np.float32)
    print("мишень: калиброванная P(k9|k) =",
          [round(v, 4) for v in _iso], flush=True)
elif os.environ.get("SOFT"):
    from calibrate_llm_targets import target_to_k
    d["y"] = (target_to_k(d.target.to_numpy()) / 9.0).astype(np.float32)
else:
    d["y"] = (d.target >= 0.999).astype(np.float32)
print(f"доза: {len(d)} пар, позитивов {int(d.y.sum())} ({d.y.mean():.4f})", flush=True)

tok = AutoTokenizer.from_pretrained(SRC)
# DROPOUT=0.1: в продовом скорере ВСЕ поля dropout ровно нулевые
# (attention/classifier/embedding/mlp) — проверено в config.json. Без
# регуляризации дообучение запоминает обучающее распределение, что даёт
# механическое объяснение инверсии: чем больше локальный прирост, тем хуже
# public. Инверсия измерена на моделях из ЭТОГО дефектного режима, поэтому
# режим с dropout прежними данными не покрыт.
_cfg = AutoConfig.from_pretrained(SRC)
_dp = float(os.environ.get("DROPOUT", "0") or 0)
if _dp > 0:
    _n = 0
    for _k in list(vars(_cfg).keys()):
        if "drop" in _k.lower() and isinstance(getattr(_cfg, _k), float):
            setattr(_cfg, _k, _dp); _n += 1
    print(f"dropout={_dp} выставлен в {_n} полях конфига", flush=True)
model = AutoModelForSequenceClassification.from_pretrained(
    SRC, config=_cfg, dtype=torch.float32).cuda()
model.gradient_checkpointing_enable()
model.train()


class DS(Dataset):
    def __init__(self, a, b, y, w=None):
        self.a, self.b, self.y = list(a), list(b), np.asarray(y, dtype=np.float32)
        self.w = np.ones(len(self.a), dtype=np.float32) if w is None else np.asarray(w, dtype=np.float32)

    def __len__(self):
        return len(self.a)

    def __getitem__(self, i):
        return self.a[i], self.b[i], self.y[i], self.w[i]


def collate(batch):
    a, b, y, w = zip(*batch)
    enc = tok(list(a), list(b), truncation=True, max_length=MAXLEN,
              padding=True, return_tensors="pt")
    enc["labels"] = torch.tensor(y, dtype=torch.float32)
    enc["sw"] = torch.tensor(w, dtype=torch.float32)
    return enc


# ANCHOR: подмешать пары ЧЕЛОВЕЧЕСКОЙ разметки к корпусу учителя.
# Измерено: доза монотонно улучшает согласие с учителем (llmvalS) и монотонно
# ухудшает согласие с правдой (human_holdout, 73k пар). Публичный пик — точка
# пересечения, а не свойство мягкой метки. Якорь удерживает вторую ось и
# должен позволить взять дозу 900-1600, где по учителю +0.017..+0.032.
SEED = int(os.environ.get("SEED", "7"))
torch.manual_seed(SEED); np.random.seed(SEED); _random.seed(SEED)
_anc = float(os.environ.get("ANCHOR", "0") or 0)
if _anc > 0:
    # Пул задаётся переменной: старый hum_train содержал ровно 80k строк
    # и пересекался с human_holdout на 21.8%, поэтому доля якоря выше 1.0
    # упиралась в размер пула и мерилась с утечкой.
    _h = pd.read_parquet(os.environ.get("ANCHOR_POOL", "data/hum_train.parquet"))
    _n = int(len(d) * _anc)
    _h = _h.sample(n=min(_n, len(_h)), random_state=42)
    _h = _h.rename(columns={"target": "y"})[["text1", "text2", "y"]]
    _h["y"] = _h.y.astype("float32")
    d = pd.concat([d[["text1", "text2", "y"]], _h], ignore_index=True)
    d = d.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    print(f"якорь: добавлено {len(_h)} человеческих пар, всего {len(d)}", flush=True)

if os.environ.get("HEADROOM") and "category" in d.columns:
    # ВЕС ПО РЕЗЕРВУ. Метрика — невзвешенное среднее AP по 20 категориям, но
    # пул уже стратифицирован поровну, поэтому вес по размеру категории даёт
    # ровно единицы. Реальная асимметрия не в размере, а в потолке: Автотовары
    # стоят на 0.856 и расти им некуда, Одежда на 0.197 — там весь резерв.
    # Вес пропорционален (1 - AP) категории.
    import json as _j
    _ap = _j.load(open("data/percat_ap.json"))
    _m = float(os.environ.get("HEADROOM"))          # 0 = равномерно, 1 = полный резерв
    _w = d.category.astype(str).map(lambda c: 1.0 - _ap.get(c, 0.544))
    _w = _w / _w.mean()
    d["w"] = ((1.0 - _m) + _m * _w).astype("float32")
    print(f"вес по резерву m={_m}: разброс {d.w.min():.3f}..{d.w.max():.3f}", flush=True)
elif os.environ.get("CATW") and "category" in d.columns:
    _cnt = d.category.value_counts()
    d["w"] = (1.0 / d.category.map(_cnt)).astype("float32")
    d["w"] = (d.w / d.w.mean()).astype("float32")
    print(f"вес по категориям: {d.category.nunique()} категорий, "
          f"разброс веса {d.w.min():.3f}..{d.w.max():.3f}", flush=True)
else:
    d["w"] = np.float32(1.0)

dl = DataLoader(DS(d.text1, d.text2, d.y, d.w), batch_size=BS, shuffle=True,
                collate_fn=collate, num_workers=2, drop_last=True)
# LP-FT. Полное дообучение с первого шага портит предобученные признаки,
# подстраивая их под СЛУЧАЙНО инициализированную голову; при сильном сдвиге
# это стоит больше, чем даёт сама задача (feature distortion). Сначала учим
# только голову на замороженном энкодере, затем отпускаем всё.
LP_STEPS = int(os.environ.get("LP_STEPS", "0"))
_head_names = [n for n, _ in model.named_parameters()
               if "classifier" in n or n.startswith("score") or "head" in n]
if LP_STEPS > 0:
    print(f"LP-фаза {LP_STEPS} шагов, обучаем только: {_head_names}", flush=True)
    for n, prm in model.named_parameters():
        prm.requires_grad = n in _head_names

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=STEPS,
                                            pct_start=0.1)
step = 0
done = False
import time
t0 = time.time()
while not done:
    for batch in dl:
        batch = {k: v.cuda() for k, v in batch.items()}
        if LP_STEPS > 0 and step == LP_STEPS:
            # LP закончена: отпускаем энкодер и продолжаем с тем же малым LR
            for prm in model.parameters():
                prm.requires_grad = True
            print(f"  шаг {step}: энкодер разморожен", flush=True)
        y = batch.pop("labels")
        batch_w = batch.pop("sw")
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            lg = model(**batch).logits.float()
        p = lg[:, 1] - lg[:, 0] if lg.shape[1] == 2 else lg[:, 0]
        _cw = os.environ.get("CATW")
        if _cw:
            # ВЫРАВНИВАНИЕ ЛОССА С МЕТРИКОЙ. Метрика — НЕВЗВЕШЕННОЕ среднее по
            # 20 категориям, а лосс равномерен по парам: крупные категории
            # доминируют в градиенте, слабые (Одежда 0.195, Обувь 0.221)
            # недополучают. Вес примера обратно пропорционален размеру его
            # категории, чтобы каждая категория весила одинаково.
            w = batch_w.to(p.device).to(p.dtype)
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(
                p, y, reduction="none") * w).sum() / w.sum()
        else:
            loss = torch.nn.functional.binary_cross_entropy_with_logits(p, y)
        # RANK_W: попарный ранговый лосс поверх BCE. Метрика ранговая
        # (average_precision), а подтверждённый механизм выигрыша — «учи
        # порядок, а не порог». В train_ce.py ранговые лоссы были закрыты при
        # дельте -0.000011, но там rank_warmup=1500 при прогонах на 408 шагов,
        # то есть они НЕ ВКЛЮЧАЛИСЬ ни разу. Здесь без warmup.
        _rw = float(os.environ.get("RANK_W", "0") or 0)
        if _rw > 0:
            di = y.unsqueeze(0) - y.unsqueeze(1)      # разница целей
            m = (di.abs() > 1e-6)                     # только различающиеся пары
            if m.any():
                ds = p.unsqueeze(0) - p.unsqueeze(1)  # разница предсказаний
                tgt = (di > 0).float()
                rank = torch.nn.functional.binary_cross_entropy_with_logits(
                    ds[m], tgt[m])
                loss = loss + _rw * rank
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        step += 1
        if step % 50 == 0:
            print(f"  шаг {step}/{STEPS} loss {loss.item():.4f} {time.time()-t0:.0f}s",
                  flush=True)
        if step >= STEPS:
            done = True
            break

model.eval()
os.makedirs(OUT, exist_ok=True)
model.half().save_pretrained(OUT, safe_serialization=True)
tok.save_pretrained(OUT)
for mk in ("ORDERED", "ALIGNED", "RANKIT_DISAGREEMENT", "RRF_K", "PARTIAL_SYM"):
    p_ = os.path.join(SRC, mk)
    if os.path.isfile(p_):
        import shutil
        shutil.copy2(p_, os.path.join(OUT, mk))
print(f"ГОТОВО -> {OUT}", flush=True)

