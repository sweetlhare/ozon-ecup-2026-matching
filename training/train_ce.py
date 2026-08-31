"""Fine-tune cross-encoder на парах товаров.

Две фазы: (1) необязательное предобучение на LLM-разметке (мягкие метки),
(2) дообучение на ручной разметке. Валидация всегда на ручном hold-out —
он ближе всего к тестовым данным. Метрика — macro PR-AUC по категориям.
"""
import argparse
import copy
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

from split import make_split
from text import item_texts


class PairDS(Dataset):
    """swap=True случайно меняет порядок товаров в паре.

    Модель становится почти симметричной сама, и на инференсе можно не платить
    второй проход за усреднение p(A,B)/p(B,A) — освободившийся бюджет выгоднее
    отдать другой модели (измерено: +0.004 против симметризации при той же цене).
    """

    def __init__(self, ta, tb, y, swap=False, seed=0, teacher=None,
                 sample_weight=None):
        if teacher is not None and sample_weight is not None:
            raise ValueError("PairDS не поддерживает teacher и sample_weight одновременно")
        self.ta, self.tb, self.y, self.swap = ta, tb, y, swap
        self.teacher = teacher            # логиты учителя для дистилляции
        self.sample_weight = sample_weight
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        a, b = self.ta[i], self.tb[i]
        if self.swap and self.rng.random() < 0.5:
            a, b = b, a
        if self.teacher is not None:
            return a, b, self.y[i], self.teacher[i]
        if self.sample_weight is not None:
            return a, b, self.y[i], self.sample_weight[i]
        return a, b, self.y[i]


class MixedDS(Dataset):
    """Weak (LLM) и gold (ручная разметка) одним потоком, у каждой пары свой вес.

    Последовательное pretrain → finetune заканчивается эпохами чистого gold,
    а gold — это ровно то распределение, которого в тесте нет: 1.5% ретривальных
    негативов против 41% в LLM, prevalence 0.26 против 0.11, товары почти не
    переиспользуются. Дообучение поэтому частично стирает то, ради чего делалось
    предобучение: ретривальные пары дали +0.010 на полной val, но лишь +0.006
    на hard-slice. Здесь gold есть в каждом батче с первого шага, а weak — до
    последнего.
    """

    def __init__(self, texts, i1, i2, y, w, plan, n_weak=None):
        self.texts, self.i1, self.i2 = texts, i1, i2
        self.y, self.w, self.plan = y, w, plan
        self.n_weak = n_weak

    def __len__(self):
        return len(self.plan)

    def __getitem__(self, k):
        i = self.plan[k]
        row = (self.texts[self.i1[i]], self.texts[self.i2[i]], self.y[i], self.w[i])
        if self.n_weak is None:
            return row
        return (*row, int(i >= self.n_weak))


def build_plan(n_weak, n_gold, gold_frac):
    """План обхода: gold-индексы повторяются до нужной доли в потоке.

    Наивная конкатенация дала бы долю gold 1/21 (отношение объёмов), а нужно
    ~1/5 — Noisy Student показывает, что batch-ratio надо держать ниже
    отношения объёмов, и при разнице <100x joint обгоняет последовательное.
    """
    reps = max(1, int(round(gold_frac / (1 - gold_frac) * n_weak / n_gold)))
    plan = np.concatenate([np.arange(n_weak),
                           np.tile(n_weak + np.arange(n_gold), reps)])
    return plan, reps


class LookupPairDS(Dataset):
    """Пары как индексы в общий список текстов.

    Для 6M LLM-пар материализация text1/text2 на пару даёт ~12M строк
    и десятки гигабайт после форка dataloader-воркеров; здесь тексты
    хранятся в одном экземпляре, а пары — два int-массива.
    """

    def __init__(self, texts, i1, i2, y, w=None, target_table=None,
                 swap=False, seed=0):
        self.texts, self.i1, self.i2, self.y = texts, i1, i2, y
        self.w = w                        # вес примера (покатегорийная балансировка)
        self.target_table = target_table  # mmap [M,10], y содержит target_code
        self.swap = swap
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        a, b = self.texts[self.i1[i]], self.texts[self.i2[i]]
        if self.swap and self.rng.random() < 0.5:
            a, b = b, a
        target = (self.y[i] if self.target_table is None
                  else self.target_table[self.y[i]])
        if self.w is None:
            return a, b, target
        return a, b, target, self.w[i]


def collate(batch, tok, max_len, teacher_col=False):
    cols = list(zip(*batch))
    enc = tok(list(cols[0]), list(cols[1]), padding=True, truncation=True,
              max_length=max_len, return_tensors="pt")
    enc["labels"] = torch.as_tensor(np.asarray(cols[2]), dtype=torch.float32)
    if len(cols) > 3:
        # MixedDS отдаёт вес примера, PairDS с дистилляцией — логит учителя
        key = "teacher" if teacher_col else "sample_weight"
        enc[key] = torch.tensor(cols[3], dtype=torch.float32)
    if len(cols) > 4:
        enc["human_task"] = torch.tensor(cols[4], dtype=torch.bool)
    return enc


def macro_pr_auc(y, p, cat):
    scores = {}
    for c in np.unique(cat):
        m = cat == c
        if y[m].min() == y[m].max():
            continue
        scores[c] = average_precision_score(y[m], p[m])
    return float(np.mean(list(scores.values()))), scores


def cascade_score(gate, partner, cat, fraction):
    """Скор двухэтажного runtime: gate на всех парах, partner на top-доле."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction должен быть в интервале (0, 1]")
    if len(gate) != len(partner) or len(gate) != len(cat):
        raise ValueError("gate, partner и cat должны иметь одинаковую длину")

    def znorm(values):
        values = np.nan_to_num(values, nan=-1e4, posinf=1e4, neginf=-1e4)
        return (values - values.mean()) / (values.std() + 1e-6)

    first = znorm(np.asarray(gate, dtype=np.float32))
    selected = np.zeros(len(first), dtype=bool)
    for category in np.unique(cat):
        idx = np.where(cat == category)[0]
        count = max(1, int(round(fraction * len(idx))))
        selected[idx[np.argsort(-first[idx])[:count]]] = True

    out = first.copy()
    out[selected] = (first[selected] + znorm(
        np.asarray(partner, dtype=np.float32)[selected])) / 2
    if selected.any() and (~selected).any():
        out[selected] += out[~selected].max() - out[selected].min() + 1e-3
    return out


def candidate_cascade_score(candidate, cat, fraction, partner=None, gate=None):
    """Оценить кандидата ровно в той роли, которую он займёт в каскаде.

    ``partner`` сохраняет прежний режим: кандидат работает gate на всех парах,
    а фиксированная модель пересчитывает top-долю. ``gate`` — обратная и нужная
    для v37 роль: фиксированный mcat отбирает top-долю, кандидат её пересчитывает.
    """
    if (partner is None) == (gate is None):
        raise ValueError("нужно задать ровно один из partner или gate")
    if gate is not None:
        return cascade_score(gate, candidate, cat, fraction)
    return cascade_score(candidate, partner, cat, fraction)


def p9_logodds(logits):
    """Логит события k=9 для нормализованной 10-class головы."""
    if logits.ndim != 2 or logits.shape[1] != 10:
        raise ValueError("p9_logodds ожидает logits shape [B,10]")
    return logits[:, 9] - torch.logsumexp(logits[:, :9], dim=1)


def model_score(logits, score_mode=""):
    if score_mode == "p9_logodds":
        return p9_logodds(logits.float())
    if logits.dim() == 2 and logits.shape[1] > 1:
        return logits[:, -1]
    return logits.squeeze(-1)


def dual_head_logits(model, batch, human_task):
    """Run one shared encoder and route rows to consensus or human head."""
    required = ("model", "head", "drop", "classifier", "human_classifier")
    missing = [name for name in required if not hasattr(model, name)]
    if missing:
        raise ValueError(f"dual-head model misses modules: {missing}")
    hidden = model.model(**batch).last_hidden_state
    pooling = getattr(model.config, "classifier_pooling", "cls")
    if pooling == "cls":
        pooled = hidden[:, 0]
    elif pooling == "mean":
        mask = batch.get("attention_mask")
        if mask is None:
            mask = torch.ones(hidden.shape[:2], device=hidden.device,
                              dtype=torch.bool)
        pooled = ((hidden * mask.unsqueeze(-1)).sum(1)
                  / mask.sum(1, keepdim=True).clamp_min(1))
    else:
        raise ValueError(f"unsupported classifier_pooling={pooling!r}")
    features = model.drop(model.head(pooled))
    weak = model.classifier(features).squeeze(-1)
    human = model.human_classifier(features).squeeze(-1)
    return torch.where(human_task, human, weak)


def supervised_per_example(logits, labels):
    """CE для full distribution либо BCE для gold/обычной binary головы."""
    if labels.ndim == 2:
        if labels.shape[1] != 10 or logits.shape != labels.shape:
            raise ValueError("soft targets и logits должны иметь shape [B,10]")
        return -(labels * torch.log_softmax(logits.float(), dim=1)).sum(dim=1)
    if labels.ndim != 1:
        raise ValueError("labels должны иметь shape [B] или [B,10]")
    score = (p9_logodds(logits.float())
             if logits.ndim == 2 and logits.shape[1] == 10
             else logits.squeeze(-1).float())
    return nn.functional.binary_cross_entropy_with_logits(
        score, labels.float(), reduction="none")


def bernoulli_symmetric_kl(logits_a, logits_b):
    """Симметричный KL между Bernoulli-распределениями двух forward-pass.

    У binary classifier один логит, поэтому softmax по последней размерности
    всегда равен единице и сделал бы R-Drop точным no-op. Здесь логит задаёт
    Bernoulli ``P(y=1)`` и KL считается для обоих исходов.
    """
    if logits_a.shape != logits_b.shape:
        raise ValueError("R-Drop logits должны иметь одинаковую форму")
    a, b = logits_a.float(), logits_b.float()
    pa, pb = a.sigmoid(), b.sigmoid()
    kl_ab = (pa * (nn.functional.logsigmoid(a) - nn.functional.logsigmoid(b))
             + (1 - pa) * (nn.functional.logsigmoid(-a)
                           - nn.functional.logsigmoid(-b)))
    kl_ba = (pb * (nn.functional.logsigmoid(b) - nn.functional.logsigmoid(a))
             + (1 - pb) * (nn.functional.logsigmoid(-b)
                           - nn.functional.logsigmoid(-a)))
    return 0.5 * (kl_ab + kl_ba)


def weighted_mean(values, weights=None):
    if weights is None:
        return values.mean()
    weights = weights.to(values.device, dtype=values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1e-6)


def override_dropout_fields(config, value):
    """Явно включить dropout в известных encoder/classifier config-полях."""
    if not 0 <= value < 1:
        raise ValueError("dropout_override должен быть в интервале [0, 1)")
    fields = (
        "attention_dropout", "embedding_dropout", "mlp_dropout",
        "classifier_dropout", "hidden_dropout_prob",
        "attention_probs_dropout_prob",
    )
    touched = []
    for name in fields:
        if hasattr(config, name):
            setattr(config, name, value)
            touched.append(name)
    if not touched:
        raise ValueError("config не содержит известных dropout-полей")
    return touched


def optimizer_steps_for_epochs(n_batches, epochs, accum=1):
    """Число optimizer steps для заданного числа проходов microbatch-loader.

    При gradient accumulation один optimizer step потребляет ``accum``
    microbatch-ей. Делить здесь обязательно: иначе ``accum=2`` незаметно
    удваивает и число проходов данных, и длину scheduler.
    """
    if n_batches <= 0 or epochs <= 0 or accum <= 0:
        raise ValueError("n_batches, epochs и accum должны быть положительными")
    return max(1, int(n_batches * epochs / accum))


def select_train_keep(tr_idx, mask, expected_length):
    """Оставить в train только строки, отмеченные внешней маской.

    Перевзвешивание малой доли пула через ``--train_weights`` оказалось no-op по
    построению: gold-refresh видит 48 000 примеров, то есть 16.4% train-пула,
    поэтому 2.5% перевзвешенных пар задевают около 1 200 примеров. Чтобы менять
    обучающий сигнал, нужен отбор строк, а не их вес.
    """
    mask = np.asarray(mask)
    if mask.shape != (expected_length,):
        raise ValueError(
            f"--train_keep: shape {mask.shape}, ожидалось ({expected_length},)")
    tr_idx = np.asarray(tr_idx)
    kept = tr_idx[mask.astype(bool)[tr_idx]]
    if len(kept) == 0:
        raise ValueError("--train_keep не оставил ни одной train-пары")
    return kept


def apply_train_labels(y, tr_idx, labels, expected_length):
    """Подменить мишень на train-строках, не трогая val.

    Тест размечен ``1{k=9}`` силами LLM по явным правилам, а gold-фаза учится
    против человеческой метки. Флаг позволяет обучать в конвенции теста, оставляя
    val с человеческой меткой — иначе метрика перестанет быть сравнимой с
    прежними прогонами.
    """
    labels = np.asarray(labels, dtype=np.float32)
    if labels.shape != (expected_length,):
        raise ValueError(
            f"--train_labels: shape {labels.shape}, ожидалось ({expected_length},)")
    if not np.isfinite(labels).all() or labels.min() < 0 or labels.max() > 1:
        raise ValueError("--train_labels должны лежать в [0, 1]")
    out = np.asarray(y, dtype=np.float32).copy()
    out[tr_idx] = labels[tr_idx]
    return out


def validate_train_weights(values, expected_length):
    """Проверить внешний train-only массив весов в порядке matches.parquet."""
    values = np.asarray(values)
    if values.shape != (expected_length,):
        raise ValueError(
            f"train weights: shape {values.shape}, ожидалось ({expected_length},)")
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("train weights должны быть конечными и положительными")
    return values.astype(np.float32, copy=False)


@torch.inference_mode()
def predict(model, loader, device):
    model.eval()
    out = []
    for batch in loader:
        batch.pop("labels", None)
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = model(**batch).logits
            # Ordinal: последний порог. Full distribution: logit P(k=9),
            # нормализованный относительно остальных девяти состояний.
            logits = model_score(
                lg, getattr(model.config, "ozon_score_mode", ""))
        out.append(logits.float().cpu().numpy())
    model.train()
    return np.concatenate(out)


def mask_tokens(input_ids, tok, prob=0.15):
    """Стандартное BERT-маскирование: 80% [MASK], 10% случайный, 10% без замены."""
    labels = input_ids.clone()
    special = torch.zeros_like(input_ids, dtype=torch.bool)
    for t in (tok.cls_token_id, tok.sep_token_id, tok.pad_token_id):
        if t is not None:
            special |= input_ids == t
    probs = torch.full(labels.shape, prob)
    probs.masked_fill_(special, 0.0)
    masked = torch.bernoulli(probs).bool()
    labels[~masked] = -100
    ids = input_ids.clone()
    repl = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked
    ids[repl] = tok.mask_token_id
    rnd = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked & ~repl
    ids[rnd] = torch.randint(len(tok), labels.shape, dtype=input_ids.dtype)[rnd]
    return ids, labels


class FGM:
    """Adversarial training: возмущение эмбеддингов на шаг epsilon вдоль градиента.

    Присутствует практически во всех призовых решениях парных текстовых задач
    (USPPPM 10-е место: public 0.8394 -> 0.8418; ESCI/Kakao +0.0028 — крупнейший
    их отдельный вклад; USPPPM 8-е ~+0.005), а у нас ни разу не проверялось.
    Стоит один дополнительный forward-backward, то есть примерно +70% времени шага.
    """

    def __init__(self, model, eps=1.0):
        self.embedding = model.get_input_embeddings().weight
        self.eps = eps
        self.backup = None

    def attack(self):
        p = self.embedding
        if p.requires_grad and p.grad is not None:
            self.backup = p.data.clone()
            norm = torch.norm(p.grad)
            if norm != 0 and not torch.isnan(norm):
                p.data.add_(self.eps * p.grad / norm)

    def restore(self):
        if self.backup is not None:
            self.embedding.data.copy_(self.backup)
        self.backup = None


def save_input_marker(path, args):
    """Сохранить рядом с весами признаки train-time представления входа."""
    if args.order:
        with open(os.path.join(path, "ORDERED"), "w", encoding="utf-8"):
            pass
    if args.cats:
        cats = [category.strip() for category in args.cats.split(",")
                if category.strip()]
        with open(os.path.join(path, "CATS"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(cats) + "\n")
    if args.train_weights:
        import hashlib
        digest = hashlib.sha256()
        with open(args.train_weights, "rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        with open(os.path.join(path, "TRAIN_WEIGHTS"), "w",
                  encoding="utf-8") as fh:
            fh.write(f"{os.path.basename(args.train_weights)}\n{digest.hexdigest()}\n")


def freeze_bottom_layers(model, count):
    """Заморозить embeddings и первые ``count`` transformer-блоков.

    Короткий refresh сильного checkpoint не обязан держать Adam-состояния и
    backward-граф всего backbone. Это позволяет безопасно доучивать верхние
    слои, когда общая GPU временно оставляет мало свободной памяти.
    """
    if count <= 0:
        return
    backbone = getattr(model, "model", None)
    layers = getattr(backbone, "layers", None)
    embeddings = getattr(backbone, "embeddings", None)
    if layers is None or embeddings is None:
        raise ValueError("--freeze_bottom поддержан только для model.layers architecture")
    if count > len(layers):
        raise ValueError(
            f"--freeze_bottom={count}, но у модели только {len(layers)} слоёв")
    for parameter in embeddings.parameters():
        parameter.requires_grad_(False)
    for layer in layers[:count]:
        for parameter in layer.parameters():
            parameter.requires_grad_(False)

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters()
                    if parameter.requires_grad)
    print(f"заморожены embeddings и нижние {count}/{len(layers)} слоёв; "
          f"обучаемых параметров {trainable:,}/{total:,} "
          f"({100 * trainable / total:.1f}%)", flush=True)


class CatBatchSampler(torch.utils.data.Sampler):
    """Батч целиком из одной категории — того списка, по которому считается AP.

    В смешанном батче 95.6% пар «позитив против негатива» берутся из РАЗНЫХ
    категорий, то есть сравнения, которых в метрике нет вовсе (и они же самые
    лёгкие: 0.928 против 0.906 уже упорядочено верно). Группировка полезна ровно
    настолько, насколько лосс сравнивает элементы внутри батча — при чистом BCE
    она бесполезна, поэтому включается вместе с ранжирующим членом.
    """

    def __init__(self, cats, batch_size, seed=0):
        self.groups = [np.where(cats == c)[0] for c in np.unique(cats)]
        self.bs, self.seed, self.epoch = batch_size, seed, 0

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        batches = []
        for g in self.groups:
            g = g[rng.permutation(len(g))]
            n = (len(g) // self.bs) * self.bs
            if n:
                batches += list(g[:n].reshape(-1, self.bs))
        for i in rng.permutation(len(batches)):
            yield batches[i].tolist()

    def __len__(self):
        return sum(len(g) // self.bs for g in self.groups)


class LengthBucketSampler(torch.utils.data.Sampler):
    """Батч из пар примерно одной длины: паддинг 37.5% -> 3.6%, обучение в 1.54x.

    На инференсе мы давно сортируем пары по длине (паддинг 8.2%), а при обучении
    шла обычная случайная перетасовка, то есть в каждом батче короткие пары
    дополнялись до самой длинной. Медиана пары 242 токена при max_len 384 —
    больше трети вычислений уходило в никуда.

    Случайность сохраняется: перемешиваем всё, режем на мега-батчи (bucket
    батчей), сортируем по длине только внутри мега-батча и перемешиваем
    получившиеся батчи. Длина оценивается символами — токенизировать 5.75M пар
    заранее дороже, чем выигрыш, а корреляция символов с токенами достаточная.
    """

    def __init__(self, lengths, batch_size, bucket=50, seed=0):
        self.lens = np.asarray(lengths)
        self.bs, self.bucket, self.seed, self.epoch = batch_size, bucket, seed, 0

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        order = rng.permutation(len(self.lens))
        mega = self.bs * self.bucket
        batches = []
        for s in range(0, len(order), mega):
            part = order[s:s + mega]
            part = part[np.argsort(self.lens[part], kind="stable")]
            n = (len(part) // self.bs) * self.bs
            if n:
                batches += list(part[:n].reshape(-1, self.bs))
        for i in rng.permutation(len(batches)):
            yield batches[i].tolist()

    def __len__(self):
        return len(self.lens) // self.bs


def ordinal_loss(logits, y, n_bins=10):
    """Порядковая голова по числу голосов k вместо регрессии на k/9.

    BCE на мягкой метке тянет выход к среднему числу голосов, а метрика
    определяется одной границей: k=9 против всего остального. Здесь модель
    предсказывает распределение по 10 градациям, а скор берётся как P(k=9) —
    это другой функционал, а не другая шкала той же величины.

    Порядок градаций учитывается кумулятивно: вместо 10 независимых классов
    учим 9 порогов «k >= 1», «k >= 2», ..., «k >= 9», что сохраняет
    упорядоченность и не даёт модели путать соседние градации с далёкими.
    """
    k = torch.clamp((y * (n_bins - 1)).round().long(), 0, n_bins - 1)
    tgt = (k.unsqueeze(1) >= torch.arange(1, n_bins, device=k.device)).float()
    return nn.functional.binary_cross_entropy_with_logits(logits, tgt)


def margin_mse_loss(s, t, y, correct_only=False):
    """Дистилляция по РАЗНОСТЯМ логитов внутри батча (Hofstatter et al.).

    Учим студента воспроизводить не сам логит учителя, а зазор между парами —
    это инвариантно к сдвигу и масштабу шкалы, что критично при разных
    архитектурах в ансамбле. Поточечный MSE по логитам в литературе рушил
    ранжирующую метрику (MAP 0.21 -> 0.09 на ранг-преобразованных таргетах),
    поэтому именно margin, а не MSE.
    """
    pos, neg = y >= 0.5, y < 0.5
    if not pos.any() or not neg.any():
        return s.new_zeros(())
    ds = s[pos][:, None] - s[neg][None, :]
    dt = t[pos][:, None] - t[neg][None, :]
    if correct_only:
        # Teacher ошибается и на части gold pos-neg пар. Копировать отрицательный
        # teacher-margin значит напрямую спорить с BCE на истинной метке.
        keep = dt > 0
        if not keep.any():
            return s.new_zeros(())
        ds, dt = ds[keep], dt[keep]
    return nn.functional.mse_loss(ds, dt)


def ranknet_loss(s, y, thr=0.5):
    """Попарный ранжирующий член внутри батча.

    Метрика ранжирующая, а BCE — поточечная: она штрафует за уровень скора,
    а не за порядок. Инвертированных пар «позитив ниже негатива» внутри
    категории 8.4%, из них 36% с зазором меньше логита — по ним и бьёт этот член.
    Чистый ranking-лосс разрушает калибровку (PCOC до 25), поэтому только добавкой.
    """
    pos, neg = y >= thr, y < thr
    if not pos.any() or not neg.any():
        return s.new_zeros(())
    return nn.functional.softplus(-(s[pos][:, None] - s[neg][None, :])).mean()


def ap_loss(s, y, thr=0.5, tau=0.05):
    """Гладкое приближение Average Precision внутри батча (SmoothAP).

    Метрика соревнования — PR-AUC по категориям, а учим BCE. Это не то же, что
    ranknet_loss: тот штрафует за каждую инвертированную пару одинаково, а AP
    взвешивает инверсии по позиции — ошибка в голове ранжирования дороже, что
    и соответствует PR-AUC. И это не focal/LDAM (отвергнуты как константный
    сдвиг логита, не меняющий порядок).

    Индикатор [s_j > s_i] заменён сигмоидой с температурой tau. Осмысленно
    только когда батч — одна категория (--cat_batch), иначе AP считается по
    смеси категорий, которой в метрике нет.
    """
    pos = y >= thr
    n_pos = int(pos.sum())
    if n_pos == 0 or n_pos == len(y):
        return s.new_zeros(())
    d = (s[None, :] - s[:, None]) / tau          # d[i,j] = (s_j - s_i)/tau
    ind = torch.sigmoid(d)
    eye = torch.eye(len(s), device=s.device, dtype=s.dtype)
    rank = 1.0 + (ind * (1 - eye)).sum(1)        # мягкий ранг: сколько выше
    rank_pos = 1.0 + (ind * pos[None, :].to(s.dtype) * (1 - eye)).sum(1)
    ap = (rank_pos[pos] / rank[pos]).mean()      # precision@позиция, среднее по позитивам
    return 1.0 - ap


def gate_recall_loss(s, y, fraction=0.75, thr=0.5, tau=0.1):
    """Поднять позитивы выше границы top-доли, которую runtime оставляет gate.

    Батч должен содержать одну категорию: его квантиль приближает фактический
    покатегорийный cutoff каскада. Порог отделён от графа, поэтому добавка
    двигает только потерянные позитивы и не пытается сделать верхние 75% парами
    одного класса (при prevalence около 0.26 это невозможно и не требуется).
    """
    pos = y >= thr
    if not pos.any() or len(s) < 2:
        return s.new_zeros(())
    cutoff = torch.quantile(s.detach(), 1 - fraction)
    return tau * nn.functional.softplus((cutoff - s[pos]) / tau).mean()


def run_phase(name, model, dl_tr, steps, lr, args, evaluate, t0, mlm_head=None):
    device = "cuda"
    # weight decay не применяем к нормировкам и смещениям
    decay, no_decay = [], []
    for n_, p_ in model.named_parameters():
        if not p_.requires_grad:
            continue
        (no_decay if p_.ndim <= 1 or n_.endswith(".bias") else decay).append(p_)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.01},
                             {"params": no_decay, "weight_decay": 0.0}], lr=lr)
    sched = get_cosine_schedule_with_warmup(opt, int(0.05 * steps), steps)
    fgm = FGM(model, eps=args.fgm) if args.fgm > 0 else None
    model.train()

    # берём лучший чекпоинт, а не последний: на 3 эпохах пик был на 10000 шагов
    # (0.7737), а финал просел до 0.7710 — это 0.003 метрики даром
    best, best_state, best_step = -1.0, None, 0
    # Текущий checkpoint недостаточен для честного resume: лучший eval может
    # остаться далеко позади текущего шага и раньше жил только в RAM процесса.
    # Для длинного pretrain сохраняем score+state_dict одним атомарным файлом.
    if name == "pretrain" and args.ckpt_every:
        best_path = os.path.join(args.ckpt_dir, "BEST_STATE.pt")
    elif args.phase_best_dir:
        # Gold fine-tune короче pretrain, но его лучший checkpoint раньше жил
        # только в RAM. На общих GPU это всё равно часы работы, которые нельзя
        # терять при OOM/перезапуске. Фазы разделены именами, чтобы joint и
        # finetune одного запуска не перезаписывали друг друга.
        os.makedirs(args.phase_best_dir, exist_ok=True)
        best_path = os.path.join(args.phase_best_dir, f"{name}_BEST_STATE.pt")
    else:
        best_path = None
    if best_path and os.path.isfile(best_path):
        saved = torch.load(best_path, map_location="cpu", weights_only=True)
        best = float(saved["macro"])
        best_step = int(saved["step"])
        best_state = saved["state_dict"]
        print(f"[{time.perf_counter()-t0:.0f}s] {name}: загружен устойчивый лучший "
              f"чекпоинт со шага {best_step} (macro {best:.4f})", flush=True)
    elif ((name == "pretrain" and args.eval_before_pretrain)
          or (name != "pretrain" and args.eval_before_finetune)) \
            and not args.use_all_data:
        # При коротком refresh исходная модель уже сильная. Без точки step 0
        # первый же (возможно худший) eval становился «лучшим» и необратимо
        # заменял базу. Снимаем baseline до первого градиентного шага и кладём
        # его в тот же durable-файл, что и последующие улучшения.
        best = evaluate()
        best_state = {k: v.detach().to("cpu", copy=True)
                      for k, v in model.state_dict().items()}
        if best_path:
            tmp = f"{best_path}.tmp-{os.getpid()}"
            torch.save({"macro": best, "step": 0, "state_dict": best_state}, tmp)
            os.replace(tmp, best_path)
        print(f"[{time.perf_counter()-t0:.0f}s] {name}: исходный checkpoint "
              f"зафиксирован как baseline (macro {best:.4f})", flush=True)
    step, done = 0, False
    # накопление градиента: батч в памяти делится на accum частей, а
    # оптимизационный шаг остаётся тем же. Нужно, когда карту делят с чужими
    # процессами и физический батч 32 в остаток памяти не влезает
    accum = max(1, args.accum)
    micro = 0
    opt.zero_grad(set_to_none=True)
    while not done:
        for batch in dl_tr:
            labels = batch.pop("labels").to(device, non_blocking=True)
            teacher = batch.pop("teacher", None)
            sw = batch.pop("sample_weight", None)  # иначе улетит в forward
            human_task = batch.pop("human_task", None)
            if human_task is not None:
                human_task = human_task.to(device, non_blocking=True)
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            use_rdrop = args.rdrop_w > 0 and name != "pretrain"
            if human_task is not None:
                if use_rdrop or args.ordinal:
                    raise ValueError("dual-head несовместим с R-Drop/ordinal")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = dual_head_logits(model, batch, human_task)
                per = nn.functional.binary_cross_entropy_with_logits(
                    logits.float(), labels.float(), reduction="none")
            elif use_rdrop:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out_a = model(**batch).logits
                    out_b = model(**batch).logits
                score_mode = getattr(model.config, "ozon_score_mode", "")
                score_a = model_score(out_a, score_mode).float()
                score_b = model_score(out_b, score_mode).float()
                logits = 0.5 * (score_a + score_b)
                per = 0.5 * (
                    supervised_per_example(out_a.float(), labels)
                    + supervised_per_example(out_b.float(), labels)
                )
                rdrop_per = bernoulli_symmetric_kl(score_a, score_b)
            else:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(**batch).logits
                if args.ordinal:
                    logits = out[:, -1].float()    # порог k>=9 = мишень теста
                    per = ordinal_loss(out.float(), labels).expand(labels.shape)
                else:
                    logits = model_score(
                        out, getattr(model.config, "ozon_score_mode", ""))
                    per = supervised_per_example(out.float(), labels)
            if sw is not None:
                sw = sw.to(device, non_blocking=True)
            loss = weighted_mean(per, sw)
            if use_rdrop:
                loss = loss + args.rdrop_w * weighted_mean(rdrop_per, sw)
            if args.rank_w > 0 and step >= args.rank_warmup:
                loss = loss + args.rank_w * ranknet_loss(logits.float(), labels)
            if args.ap_w > 0 and step >= args.rank_warmup:
                loss = loss + args.ap_w * ap_loss(logits.float(), labels)
            if (args.gate_w > 0 and name != "pretrain"
                    and step >= args.rank_warmup):
                loss = loss + args.gate_w * gate_recall_loss(
                    logits.float(), labels, fraction=args.gate_fraction)
            if args.distill_w > 0 and teacher is not None:
                # чистая дистилляция ни разу не была оптимальной в трёх
                # независимых работах — держим смесь с настоящими метками
                loss = loss + args.distill_w * margin_mse_loss(
                    logits.float(), teacher.to(device), labels,
                    correct_only=args.distill_correct_only)
            if mlm_head is not None:
                # предобучение только на парах при большой разметке даёт ~0
                # (+0.14 F1 у Peeters et al.), а то же самое с MLM-головой +2.90
                # и не затухает с ростом разметки — это и есть наш «потолок»
                ids, mlm_labels = mask_tokens(batch["input_ids"].cpu(), mlm_head[1])
                mb = dict(batch)
                mb["input_ids"] = ids.to(device, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = mlm_head[0](**mb, labels=mlm_labels.to(device))
                loss = loss + args.mlm_w * out.loss.float()
            (loss / accum).backward()
            # FGM только в дообучении: в предобучении он удваивает стоимость шага
            # (лишний forward+backward), то есть 290k шагов из 9 ч превращаются в 17,
            # а измеренный эффект приёма (+0.003..0.005) относится к фазе дообучения
            if fgm is not None and name != "pretrain" and step >= args.fgm_warmup:
                fgm.attack()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    adv = (dual_head_logits(model, batch, human_task)
                           if human_task is not None else model(**batch).logits)
                adv_loss = supervised_per_example(adv.float(), labels)
                adv_loss = (adv_loss.mean() if sw is None
                            else (adv_loss * sw).sum() / sw.sum().clamp_min(1e-6))
                (adv_loss / accum).backward()   # градиенты складываются с основными
                fgm.restore()
            micro += 1
            if micro % accum:
                continue                        # копим дальше, шага ещё не было
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 500 == 0:
                print(f"[{time.perf_counter()-t0:.0f}s] {name} {step}/{steps} "
                      f"loss {loss.item():.4f}", flush=True)
            if args.ckpt_every and step % args.ckpt_every == 0:
                # длинное предобучение переживает смерть процесса: сервер общий,
                # два прогона уже погибли от чужой нагрузки на 60-70% пути
                model.save_pretrained(args.ckpt_dir)
                save_input_marker(args.ckpt_dir, args)
                with open(f"{args.ckpt_dir}/STEP", "w") as f:
                    f.write(str(step))
                print(f"[{time.perf_counter()-t0:.0f}s] {name}: чекпоинт на шаге "
                      f"{step} -> {args.ckpt_dir}", flush=True)
            # Отбор чекпоинта работает, только если замеров больше одного.
            # В mbert_x4 при --eval_every 20000 и 18278 шагах дообучения замер
            # был ровно один, и «лучший чекпоинт» оказался слепым финалом —
            # а RuModernBERT теряет до 0.0078 на добеге лишней эпохи.
            ev = args.ft_eval_every if (name != "pretrain" and args.ft_eval_every) \
                else args.eval_every
            if step % ev == 0 or step == steps:
                macro = evaluate()
                # при обучении на всех данных val входит в train: метрика растёт
                # монотонно и «лучший» чекпоинт — всегда последний. Тогда выбор
                # отключаем и берём финальный, а число эпох задаём по пику,
                # найденному на честном прогоне.
                if macro > best and not args.use_all_data:
                    best, best_step = macro, step
                    best_state = {k: v.detach().to("cpu", copy=True)
                                  for k, v in model.state_dict().items()}
                    if best_path:
                        tmp = f"{best_path}.tmp-{os.getpid()}"
                        torch.save({"macro": best, "step": best_step,
                                    "state_dict": best_state}, tmp)
                        os.replace(tmp, best_path)
                        print(f"[{time.perf_counter()-t0:.0f}s] {name}: лучший "
                              f"чекпоинт атомарно сохранён -> {best_path}", flush=True)
                print(f"[{time.perf_counter()-t0:.0f}s] {name} {step}  "
                      f"macro PR-AUC = {macro:.4f}  (лучший {best:.4f} на {best_step})",
                      flush=True)
            if step >= steps:
                done = True
                break

    if args.use_all_data:
        print(f"[{time.perf_counter()-t0:.0f}s] {name}: берём финальный чекпоинт "
              f"(val в train, отбор по ней недостоверен)", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"[{time.perf_counter()-t0:.0f}s] {name}: восстановлен чекпоинт "
              f"со шага {best_step} (macro {best:.4f})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="intfloat/multilingual-e5-small")
    ap.add_argument("--no_trust_remote_code", action="store_true",
                    help="использовать native Transformers implementation; нужен "
                         "для EuroBERT, чей bundled remote config несовместим")
    ap.add_argument("--attn_implementation", choices=["", "sdpa", "eager"],
                    default="", help="зафиксировать train/serve attention kernel")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--pretrain", default=None, help="parquet с text1/text2/target")
    ap.add_argument("--pretrain_targets", default="",
                    help="mmap .npy [M,10]; pair parquet должен содержать "
                         "uint32 target_code вместо scalar weak target")
    ap.add_argument("--pretrain_steps", type=int, default=0, help="0 = одна эпоха")
    ap.add_argument("--pretrain_lr", type=float, default=5e-5)
    ap.add_argument("--eval_before_pretrain", action="store_true",
                    help="до первого шага измерить и сохранить исходную модель "
                         "как baseline; для refresh, который не должен ухудшить "
                         "готовый pretrain-checkpoint")
    ap.add_argument("--eval_before_finetune", action="store_true",
                    help="до первого шага finetune сохранить исходную модель как "
                         "baseline; короткий gate-refresh не сможет тихо ухудшить её")
    ap.add_argument("--value_max", type=int, default=0,
                    help="обрезка значений атрибутов, 0 = сырая JSON-строка")
    ap.add_argument("--train_prevalence", type=float, default=0.0,
                    help="проредить позитивы в обучении до этой доли; "
                         "в тесте 0.111 против 0.263 в ручной разметке")
    ap.add_argument("--train_weights", default="",
                    help=".npy положительных train-only весов в точном порядке "
                         "matches.parquet; val-веса загружаются, но не используются")
    ap.add_argument("--save_after_pretrain", action="store_true",
                    help="сохранить модель сразу после предобучения на LLM")
    ap.add_argument("--pretrain_only", action="store_true",
                    help="после лучшего pretrain-checkpoint сохранить модель и "
                         "val prediction, не начинать gold fine-tune")
    ap.add_argument("--order", action="store_true",
                    help="поля карточки в порядке измеренной силы, а не ключей "
                         "JSON: обрезка попарная (20.1%% пар при max_len 384), и "
                         "вес/размер/объём уезжали под неё, хотя остаточный "
                         "сигнал есть только у них")
    ap.add_argument("--cat_weight", type=float, default=0.0,
                    help="покатегорийные веса лосса в предобучении: вес ∝ "
                         "(n_max/n_cat)^степень. 1.0 = полная балансировка, "
                         "0.5 = мягкая (1/sqrt), 0 = выключено")
    ap.add_argument("--len_bucket", action="store_true",
                    help="батчи из пар близкой длины: паддинг 37.5%% -> 3.6%%, "
                         "обучение в 1.54 раза быстрее при той же случайности")
    ap.add_argument("--ap_w", type=float, default=0.0,
                    help="вес гладкого AP внутри батча; включать только с "
                         "--cat_batch, иначе AP считается по смеси категорий")
    ap.add_argument("--accum", type=int, default=1,
                    help="накопление градиента: физический батч = --batch, "
                         "эффективный = --batch * --accum")
    ap.add_argument("--dropout_override", type=float, default=-1.0,
                    help="явно заменить dropout config-поля; mmBERT по умолчанию "
                         "использует 0, поэтому без override R-Drop является no-op")
    ap.add_argument("--rdrop_w", type=float, default=0.0,
                    help="вес Bernoulli symmetric-KL между двумя train forward; "
                         "работает только в gold-фазе")
    ap.add_argument("--freeze_bottom", type=int, default=0,
                    help="заморозить embeddings и N нижних transformer-слоёв; "
                         "снижает VRAM короткого refresh сильного checkpoint")
    ap.add_argument("--ckpt_every", type=int, default=0,
                    help="сохранять модель на диск каждые N шагов (0 = не сохранять)")
    ap.add_argument("--ckpt_dir", default=None,
                    help="куда писать промежуточные чекпоинты; продолжение "
                         "прогона — это запуск с --model <ckpt_dir> и остатком шагов")
    ap.add_argument("--phase_best_dir", default="",
                    help="атомарно хранить лучший state_dict коротких фаз "
                         "(например finetune_BEST_STATE.pt); pretrain по-прежнему "
                         "использует BEST_STATE.pt внутри --ckpt_dir")
    ap.add_argument("--joint", action="store_true",
                    help="weak и gold одним потоком вместо pretrain->finetune")
    ap.add_argument("--dual_head", action="store_true",
                    help="отдельная human-head в joint-режиме; production "
                         "classifier остаётся consensus-head")
    ap.add_argument("--gold_frac", type=float, default=0.2,
                    help="доля gold в батче: 0.2 = 4:1, 0.125 = 7:1")
    ap.add_argument("--weak_w", type=float, default=1.0, help="вес BCE на weak-примерах")
    ap.add_argument("--llm_binarize", type=float, default=0.0,
                    help="порог бинаризации weak-метки; 1.0 = как в тесте (9/9)")
    ap.add_argument("--joint_steps", type=int, default=0)
    ap.add_argument("--novelty", default="data/val_maxsim.npy",
                    help="сходство val-товаров с каталогом: отбор чекпоинта по "
                         "стенду новинок (единственный прокси, коррелирующий с LB)")
    ap.add_argument("--judge", default="artifacts/ce_rubert_large_val_pred.npy",
                    help="сторонняя модель-судья для hard-slice при отборе чекпоинта")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--swap_aug", action="store_true",
                    help="случайно менять порядок товаров в паре при обучении")
    ap.add_argument("--rank_w", type=float, default=0.0,
                    help="вес попарного ранжирующего члена (0.1-0.3); включать "
                         "вместе с --cat_batch, иначе сравнения идут между "
                         "категориями, которых в метрике нет")
    ap.add_argument("--rank_warmup", type=int, default=1500,
                    help="шагов чистого BCE перед включением ранжирующего члена")
    ap.add_argument("--cat_batch", action="store_true",
                    help="батч целиком из одной категории")
    ap.add_argument("--gate_w", type=float, default=0.0,
                    help="вес recall-loss на позитивах ниже cutoff каскада; "
                         "требует --cat_batch")
    ap.add_argument("--gate_fraction", type=float, default=0.75,
                    help="top-доля покатегорийного каскада для gate-loss и "
                         "cascade-aware отбора checkpoint")
    ap.add_argument("--cascade_partner", default="",
                    help="val prediction .npy второй модели: при выборе checkpoint "
                         "оценивать фактический каскад, а не gate solo")
    ap.add_argument("--cascade_gate", default="",
                    help="val prediction .npy фиксированного gate: кандидат является "
                         "второй моделью и пересчитывает отобранную top-долю")
    ap.add_argument("--cats", default="",
                    help="обучать только на этих категориях (через запятую): "
                         "специалист для слабых категорий")
    ap.add_argument("--ordinal", action="store_true",
                    help="порядковая голова по числу голосов: 9 порогов k>=1..9, "
                         "скор = P(k>=9) — ровно мишень теста")
    ap.add_argument("--teacher", default="",
                    help="parquet (id1,id2,teacher) с логитами ансамбля")
    ap.add_argument("--distill_w", type=float, default=0.0,
                    help="вес margin-MSE к логитам учителя (0.3-0.5)")
    ap.add_argument("--distill_correct_only", action="store_true",
                    help="не копировать teacher-margin <= 0 на gold pos-neg "
                         "парах: там teacher противоречит истинной метке")
    ap.add_argument("--llm_labels", default="",
                    help="parquet (id1,id2,target) с мягкими метками LLM-судьи")
    ap.add_argument("--llm_w", type=float, default=0.3,
                    help="вес метки судьи в смеси с ручной")
    ap.add_argument("--fgm", type=float, default=0.0,
                    help="adversarial training (FGM), epsilon; 0.5-1.0 типично")
    ap.add_argument("--fgm_warmup", type=int, default=500,
                    help="шагов обычного обучения перед включением FGM")
    ap.add_argument("--fill_brand", action="store_true",
                    help="бренд из названия по словарю, если его нет в атрибутах")
    ap.add_argument("--aligned_pri2", nargs="?", const="1", default="",
                    help="порядок полей по замеру: '1' — общий, 'bycat' — свой "
                         "для каждой категории (бренд решает в Одежде и не решает "
                         "в Электронике)")
    ap.add_argument("--aligned_cat", action="store_true",
                    help="в выровненной подаче приписывать категорию к имени")
    ap.add_argument("--aligned", action="store_true",
                    help="выровненная подача пары (src/pair_text.py) вместо "
                         "двух склеенных JSON-строк")
    ap.add_argument("--mlm_w", type=float, default=0.0,
                    help="вес MLM-лосса на фазе предобучения (0 = выключено)")
    ap.add_argument("--neg_hardness", default=None,
                    help="НЕ ИСПОЛЬЗОВАТЬ, пока score_train.py не даёт настоящий OOF: "
                         "на in-sample скорах отберёт недоученные пары, а не трудные")
    ap.add_argument("--neg_keep_top", type=float, default=1.0,
                    help="какую долю самых трудных негативов оставить в обучении")
    ap.add_argument("--train_labels", default=None,
                    help="npy-мишень по строкам matches.parquet: на train-парах "
                         "учиться против неё вместо человеческой метки. val не "
                         "затрагивается, чтобы метрика оставалась сравнимой")
    ap.add_argument("--train_keep", default=None,
                    help="npy-маска по строкам matches.parquet: обучаться только "
                         "на train-парах, где маска истинна. Перевзвешивание малой "
                         "доли пула через --train_weights оказалось no-op по "
                         "построению (рефреш видит 16.4%% пула, поэтому 2.5%% "
                         "перевзвешенных пар задевают ~1200 примеров из 48000), "
                         "поэтому для смены обучающего сигнала нужен именно отбор")
    ap.add_argument("--use_all_data", action="store_true",
                    help="обучать на всех ручных парах, включая val; метрика на val "
                         "после этого недостоверна — только для финального сабмита")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--eval_batch", type=int, default=96)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--eval_every", type=int, default=2000)
    ap.add_argument("--ft_eval_every", type=int, default=1500,
                    help="шаг замера в фазе дообучения; отдельно от предобучения, "
                         "где частые замеры съедают часы")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    if args.dual_head and not args.joint:
        ap.error("--dual_head требует --joint")
    if args.eval_before_pretrain and (not args.pretrain or not args.ckpt_every):
        ap.error("--eval_before_pretrain требует --pretrain и --ckpt_every")
    if args.pretrain_only and not args.pretrain:
        ap.error("--pretrain_only требует --pretrain")
    if args.pretrain_targets and not args.pretrain:
        ap.error("--pretrain_targets требует --pretrain")
    if args.pretrain_targets and not os.path.isfile(args.pretrain_targets):
        ap.error(f"нет --pretrain_targets: {args.pretrain_targets}")
    if args.pretrain_targets and (args.ordinal or args.joint
                                  or args.llm_binarize > 0):
        ap.error("--pretrain_targets несовместим с --ordinal, --joint и "
                 "--llm_binarize")
    if args.eval_before_pretrain and args.use_all_data:
        ap.error("--eval_before_pretrain несовместим с --use_all_data: val входит в train")
    if args.eval_before_finetune and args.use_all_data:
        ap.error("--eval_before_finetune несовместим с --use_all_data: val входит в train")
    if args.gate_w > 0 and not args.cat_batch:
        ap.error("--gate_w требует --cat_batch")
    if args.rdrop_w < 0:
        ap.error("--rdrop_w не может быть отрицательным")
    if args.rdrop_w > 0 and (args.ordinal or args.fgm > 0):
        ap.error("--rdrop_w пока несовместим с --ordinal и --fgm")
    if args.dropout_override >= 1:
        ap.error("--dropout_override должен быть меньше 1")
    if args.train_labels and not os.path.isfile(args.train_labels):
        ap.error(f"нет --train_labels: {args.train_labels}")
    if args.train_keep and not os.path.isfile(args.train_keep):
        ap.error(f"нет --train_keep: {args.train_keep}")
    if args.train_weights and not os.path.isfile(args.train_weights):
        ap.error(f"нет --train_weights: {args.train_weights}")
    incompatible_weights = (args.teacher or args.joint or args.pretrain_only
                            or args.ordinal or args.use_all_data or args.llm_labels
                            or args.neg_hardness)
    if args.train_weights and incompatible_weights:
        ap.error("--train_weights поддержан только для отдельного binary gold-phase "
                 "без teacher/joint/pretrain_only/ordinal/use_all_data/llm_labels/"
                 "neg_hardness")
    if not 0 < args.gate_fraction <= 1:
        ap.error("--gate_fraction должен быть в интервале (0, 1]")
    if args.cascade_partner and not os.path.isfile(args.cascade_partner):
        ap.error(f"нет --cascade_partner: {args.cascade_partner}")
    if args.cascade_gate and not os.path.isfile(args.cascade_gate):
        ap.error(f"нет --cascade_gate: {args.cascade_gate}")
    if args.cascade_partner and args.cascade_gate:
        ap.error("--cascade_partner и --cascade_gate взаимоисключающие")
    tag = args.tag or args.model.split("/")[-1]
    if args.ckpt_every and not args.ckpt_dir:
        args.ckpt_dir = f"{args.out}/ce_{tag}_ckpt"

    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    t0 = time.perf_counter()
    matches = pd.read_parquet(f"{args.data}/matches.parquet")
    items = pd.read_parquet(f"{args.data}/items_human.parquet")
    cat = items.set_index("id").category
    pair_cat = matches.id1.map(cat).values

    if args.aligned:
        from pair_text import item_attrs, pair_text, _PRI_V2
        rep = item_attrs(items, with_category=args.aligned_cat,
                         fill_brand=args.fill_brand)
        pri = "bycat" if args.aligned_pri2 == "bycat" else (
              _PRI_V2 if args.aligned_pri2 else None)
        empty = ("", {})
        pairs = [pair_text(rep.get(i1, empty), rep.get(i2, empty), pri=pri)
                 for i1, i2 in zip(matches.id1.tolist(), matches.id2.tolist())]
        ta = [p[0] for p in pairs]
        tb = [p[1] for p in pairs]
        del pairs, rep
    else:
        id2txt = dict(zip(items["id"].tolist(),
                          item_texts(items, args.value_max, order=args.order).tolist()))
        ta = [id2txt.get(i, "") for i in matches.id1.tolist()]
        tb = [id2txt.get(i, "") for i in matches.id2.tolist()]
        del id2txt
    y = matches.target.values.astype(np.float32)

    # Метки второго аннотатора (gemma-4-31B, мягкие из logprobs) подмешиваются
    # к ручным. Не замена: под instance-dependent шумом AP-оптимальный ранкер —
    # это шумная апостериорная вероятность, а не «истина», поэтому уводить метку
    # к правильному ответу вредно (это подтвердил провал v21). Но gemma
    # декоррелирована с нашими моделями (0.728 при пороге 0.80) и выигрывает
    # в Обуви, поэтому её мнение — дополнительный сигнал, а не исправление.
    if args.llm_labels and os.path.exists(args.llm_labels):
        lab = pd.read_parquet(args.llm_labels)
        key = {(a, b): v for a, b, v in
               zip(lab.id1.values, lab.id2.values, lab.target.values)}
        hit = 0
        for r, (a, b) in enumerate(zip(matches.id1.values, matches.id2.values)):
            v = key.get((a, b))
            if v is not None:
                y[r] = (1 - args.llm_w) * y[r] + args.llm_w * v
                hit += 1
        print(f"[{time.perf_counter()-t0:.0f}s] метки gemma подмешаны к {hit} парам "
              f"с весом {args.llm_w}", flush=True)

    # логиты ансамбля-учителя для дистилляции: студент учится воспроизводить
    # зазоры между парами, а не саму метку
    teach = None
    if args.teacher and os.path.exists(args.teacher):
        td = pd.read_parquet(args.teacher)
        tk = {(a, b): v for a, b, v in
              zip(td.id1.values, td.id2.values, td.teacher.values)}
        teach = np.array([tk.get((a, b), np.nan) for a, b in
                          zip(matches.id1.values, matches.id2.values)], dtype=np.float32)
        hit = int(np.isfinite(teach).sum())
        teach = np.nan_to_num(teach, nan=0.0)
        print(f"[{time.perf_counter()-t0:.0f}s] логиты учителя: {hit} пар из "
              f"{len(matches)}", flush=True)

    is_val = make_split(matches, pd.Series(pair_cat))
    sample_weight = None
    if args.train_weights:
        sample_weight = validate_train_weights(
            np.load(args.train_weights, mmap_mode="r"), len(matches))
        train_hard = (~is_val) & (sample_weight > 1)
        hard_min = sample_weight[train_hard].min() if train_hard.any() else 1.0
        hard_max = sample_weight[train_hard].max() if train_hard.any() else 1.0
        print(f"[{time.perf_counter()-t0:.0f}s] внешние train weights: "
              f"{train_hard.sum()} train-пар, веса "
              f"{hard_min:.3f}..{hard_max:.3f}", flush=True)

    # Обучение на подмножестве категорий: метрика macro, поэтому слабая категория
    # весит столько же, сколько сильная. Одежда (PR-AUC 0.588) и Обувь (0.605)
    # дают +0.022 macro, если поднять их до среднего 0.82 — вчетверо больше
    # текущего отставания. В этих категориях имя не идентифицирует товар
    # (89.6% пар с именем <=3 токенов), решают атрибуты, и общая модель,
    # обученная в основном на других категориях, здесь недоучена.
    if args.cats:
        want = [c.strip() for c in args.cats.split(",")]
        sub = np.isin(pair_cat.astype(str), want)
        print(f"[{time.perf_counter()-t0:.0f}s] только категории {want}: "
              f"{sub.sum()} пар из {len(sub)}", flush=True)
        # тексты построены выше и должны быть обрезаны тем же срезом, иначе
        # индексы разъезжаются и модель учится на чужих парах
        idx = np.where(sub)[0]
        ta = [ta[i] for i in idx]
        tb = [tb[i] for i in idx]
        if teach is not None:
            teach = teach[sub]
        if sample_weight is not None:
            sample_weight = sample_weight[sub]
        matches, y, pair_cat, is_val = (matches[sub].reset_index(drop=True), y[sub],
                                        pair_cat[sub], is_val[sub])
    items_ids = pd.DataFrame({"id": items["id"].values})   # нужен стенду новинок
    del items
    print(f"[{time.perf_counter()-t0:.0f}s] data ready; val={is_val.mean():.3f}", flush=True)

    device = "cuda"
    # trust_remote_code нужен реранкерам Alibaba-NLP (кастомная архитектура
    # New*), но native EuroBERT работает только без bundled remote config.
    trust = not args.no_trust_remote_code
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=trust)
    source_config = AutoConfig.from_pretrained(
        args.model, trust_remote_code=trust)
    if args.dropout_override >= 0:
        touched = override_dropout_fields(source_config, args.dropout_override)
        values = ", ".join(f"{name}={getattr(source_config, name):g}"
                           for name in touched)
        print(f"[{time.perf_counter()-t0:.0f}s] dropout override: {values}",
              flush=True)
    saved_score_mode = getattr(source_config, "ozon_score_mode", "")
    distribution_head = bool(args.pretrain_targets) or saved_score_mode == "p9_logodds"
    # ordinal: 9 порогов k>=1..9; distribution: 10 состояний k=0..9.
    n_out = 10 if distribution_head else (9 if args.ordinal else 1)
    source_config.num_labels = n_out
    if distribution_head:
        source_config.ozon_score_mode = "p9_logodds"
        source_config.ozon_target_bins = 10
    load_kwargs = {
        "config": source_config,
        "dtype": torch.float32,
        "trust_remote_code": trust,
        "ignore_mismatched_sizes": bool(args.pretrain_targets) or args.ordinal,
    }
    if args.attn_implementation:
        load_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, **load_kwargs).to(device)
    if args.dual_head:
        if n_out != 1:
            raise ValueError("--dual_head поддерживает только scalar classifier")
        if not hasattr(model, "classifier"):
            raise ValueError("--dual_head требует classifier head")
        model.human_classifier = copy.deepcopy(model.classifier).to(device)
        model.config.ozon_aux_human_head = True
        print(f"[{time.perf_counter()-t0:.0f}s] dual-head: production classifier="
              "consensus, auxiliary classifier=human", flush=True)
    if hasattr(model.config, "classifier_pooling"):
        print(f"[{time.perf_counter()-t0:.0f}s] classifier_pooling="
              f"{model.config.classifier_pooling}", flush=True)
    freeze_bottom_layers(model, args.freeze_bottom)

    def col(b):
        return collate(b, tok, args.max_len, teacher_col=teach is not None)

    tr_idx = np.where(~is_val)[0]
    va_idx = np.where(is_val)[0]
    ds_va = PairDS([ta[i] for i in va_idx], [tb[i] for i in va_idx], y[va_idx])
    dl_va = DataLoader(ds_va, batch_size=args.eval_batch, shuffle=False, collate_fn=col,
                       num_workers=args.workers, pin_memory=True)

    # чекпоинт выбираем по той же метрике, по которой принимаем решения:
    # hard-slice при тестовой prevalence, а не полная val при 0.263 — это
    # разные величины, и пик по шагам у них не обязан совпадать
    hard_mask = None
    if args.judge and os.path.exists(args.judge):
        from metrics import hard_slice
        hard_mask = hard_slice(y[va_idx], pair_cat[va_idx], np.load(args.judge))
        print(f"[{time.perf_counter()-t0:.0f}s] отбор чекпоинта по hard-slice "
              f"({hard_mask.sum()} пар, судья {os.path.basename(args.judge)})", flush=True)

    # Стенд новинок: пары, где оба товара далеки от обучающего каталога.
    # hard_slice антикоррелирует с лидербордом (r=-0.20), а эта метрика даёт
    # r=+0.99 внутри рабочего семейства моделей — по ней и отбираем чекпоинт.
    novel_mask = None
    if args.novelty and os.path.exists(args.novelty):
        from metrics import novelty_mask as _nm
        novel_mask = _nm(matches, va_idx, items_ids, sim_path=args.novelty)
        print(f"[{time.perf_counter()-t0:.0f}s] отбор чекпоинта по стенду новинок "
              f"({novel_mask.sum()} пар из {len(va_idx)})", flush=True)

    cascade_reference = None
    candidate_role = None
    cascade_path = args.cascade_partner or args.cascade_gate
    if cascade_path:
        cascade_reference = np.load(cascade_path).astype(np.float32, copy=False)
        if len(cascade_reference) != len(va_idx):
            raise ValueError(
                f"cascade reference: {len(cascade_reference)} предсказаний, "
                f"ожидалось {len(va_idx)}")
        candidate_role = "scorer" if args.cascade_gate else "gate"
        print(f"[{time.perf_counter()-t0:.0f}s] cascade-aware отбор checkpoint: "
              f"candidate={candidate_role}, "
              f"reference={os.path.basename(cascade_path)}, "
              f"top-{100 * args.gate_fraction:.0f}%", flush=True)

    def evaluate():
        p = predict(model, dl_va, device)
        full = macro_pr_auc(y[va_idx], p, pair_cat[va_idx])[0]
        parts = [f"full {full:.4f}"]
        best_of = full
        if hard_mask is not None:
            hard = macro_pr_auc(y[va_idx][hard_mask], p[hard_mask],
                                pair_cat[va_idx][hard_mask])[0]
            parts.append(f"hard {hard:.4f}")
            best_of = hard
        if novel_mask is not None:
            from metrics import novelty_macro
            nov = novelty_macro(y[va_idx], p, pair_cat[va_idx], novel_mask)
            parts.append(f"новинки {nov:.4f}")
            best_of = nov          # приоритет у стенда новинок
            if cascade_reference is not None:
                cp = candidate_cascade_score(
                    p, pair_cat[va_idx], args.gate_fraction,
                    partner=cascade_reference if candidate_role == "gate" else None,
                    gate=cascade_reference if candidate_role == "scorer" else None)
                cascade_nov = novelty_macro(
                    y[va_idx], cp, pair_cat[va_idx], novel_mask)
                parts.append(f"каскад {cascade_nov:.4f}")
                best_of = cascade_nov
        print("      " + " | ".join(parts), flush=True)
        return best_of

    if args.pretrain:
        # args.pretrain — префикс пары файлов <prefix>_pairs / <prefix>_items
        llm = pd.read_parquet(f"{args.pretrain}_pairs.parquet")
        llm_items = pd.read_parquet(f"{args.pretrain}_items.parquet")
        llm_texts = llm_items["text"].tolist()
        pos = pd.Series(np.arange(len(llm_items), dtype=np.int64),
                        index=llm_items["id"].values)
        del llm_items
        i1 = pos.reindex(llm.id1.values).values
        i2 = pos.reindex(llm.id2.values).values
        ok = ~(pd.isna(i1) | pd.isna(i2))
        target_table = None
        if args.pretrain_targets:
            if "target_code" not in llm.columns:
                raise ValueError(
                    "--pretrain_targets требует target_code в pair parquet")
            target_table = np.load(args.pretrain_targets, mmap_mode="r")
            if (target_table.ndim != 2 or target_table.shape[1] != 10
                    or target_table.dtype != np.float32
                    or not np.isfinite(target_table).all()
                    or (target_table < 0).any()
                    or not np.allclose(target_table.sum(axis=1), 1.0, atol=1e-6)):
                raise ValueError("pretrain target table должна быть корректной float32 [M,10]")
            w_y = llm.target_code.to_numpy(dtype=np.uint32)[ok]
            if len(w_y) and int(w_y.max()) >= len(target_table):
                raise ValueError("target_code выходит за границы pretrain target table")
            print(f"[{time.perf_counter()-t0:.0f}s] full-distribution targets: "
                  f"{len(target_table):,} уникальных строк, "
                  f"{len(w_y):,} pair-кодов", flush=True)
        else:
            w_y = llm.target.values.astype(np.float32)[ok]
        if args.llm_binarize > 0:
            # 7/9 и 8/9 — это 15.6% потока (956k пар), и по устройству теста
            # (prevalence при бинаризации 9/9 = 0.1094 против 0.1113 в тесте)
            # они НЕГАТИВЫ. Мягкая метка 0.78/0.89 учит выталкивать их в голову
            # ранжирования, где решается 46% метрики.
            n_soft = int(((w_y > 0) & (w_y < args.llm_binarize)).sum())
            w_y = (w_y >= args.llm_binarize).astype(np.float32)
            print(f"[{time.perf_counter()-t0:.0f}s] бинаризация по {args.llm_binarize}: "
                  f"{n_soft} пар из серой зоны стали негативами, "
                  f"prevalence {w_y.mean():.4f}", flush=True)

        if args.joint:
            gi1, gi2 = i1[ok].astype(np.int64), i2[ok].astype(np.int64)
            off, ng = len(llm_texts), len(tr_idx)
            texts = llm_texts + [ta[i] for i in tr_idx] + [tb[i] for i in tr_idx]
            plan, reps = build_plan(len(w_y), ng, args.gold_frac)
            ds_j = MixedDS(texts,
                           np.concatenate([gi1, off + np.arange(ng)]),
                           np.concatenate([gi2, off + ng + np.arange(ng)]),
                           np.concatenate([w_y, y[tr_idx]]),
                           np.concatenate([np.full(len(w_y), args.weak_w, np.float32),
                                           np.ones(ng, np.float32)]),
                           plan, n_weak=len(w_y) if args.dual_head else None)
            dl_j = DataLoader(ds_j, batch_size=args.batch, shuffle=True, collate_fn=col,
                              num_workers=args.workers, pin_memory=True, drop_last=True,
                              persistent_workers=args.workers > 0)
            steps = args.joint_steps or len(dl_j)
            print(f"[{time.perf_counter()-t0:.0f}s] joint: {len(w_y)} weak + {ng} gold "
                  f"(gold x{reps} → доля {reps*ng/len(plan):.3f}), {steps} шагов",
                  flush=True)
            run_phase("joint", model, dl_j, steps, args.lr, args, evaluate, t0)
            model.save_pretrained(f"{args.out}/ce_{tag}")
            tok.save_pretrained(f"{args.out}/ce_{tag}")
            save_input_marker(f"{args.out}/ce_{tag}", args)
            p = predict(model, dl_va, device)
            np.save(f"{args.out}/ce_{tag}_val_pred.npy", p)
            macro, per_cat = macro_pr_auc(y[va_idx], p, pair_cat[va_idx])
            print(f"\nFINAL macro PR-AUC = {macro:.4f}")
            return

        pw = None
        if args.cat_weight > 0:
            # Метрика усредняет 20 категорий поровну, а LLM-разметка перекошена в
            # 64.5 раза (Продукты питания 28.8%, Ювелирные изделия 0.45%). Ручное
            # дообучение сбалансировано по построению (разброс 1.35x), а
            # предобучение — нет. Категория лежит в начале текста карточки.
            i1o, i2o = i1[ok].astype(np.int64), i2[ok].astype(np.int64)
            cats = np.array([llm_texts[x].split(" ; ", 1)[0] for x in i1o])
            uniq, inv, cnt = np.unique(cats, return_inverse=True, return_counts=True)
            w_cat = (cnt.max() / cnt) ** args.cat_weight
            pw = (w_cat[inv] / w_cat[inv].mean()).astype(np.float32)
            print(f"[{time.perf_counter()-t0:.0f}s] покатегорийные веса (степень "
                  f"{args.cat_weight}): {len(uniq)} категорий, перекос данных "
                  f"{cnt.max()/cnt.min():.1f}x, вес от {pw.min():.2f} до {pw.max():.2f}",
                  flush=True)
        ds_pt = LookupPairDS(
            llm_texts, i1[ok].astype(np.int64), i2[ok].astype(np.int64),
            w_y, pw, target_table=target_table, swap=args.swap_aug,
            seed=args.seed)
        del pos, i1, i2
        if args.len_bucket:
            # длина пары в символах: токенизировать 5.75M пар заранее дороже выигрыша
            tl = np.fromiter((len(llm_texts[x]) + len(llm_texts[y])
                              for x, y in zip(ds_pt.i1, ds_pt.i2)),
                             dtype=np.int32, count=len(ds_pt))
            bs_pt = LengthBucketSampler(tl, args.batch, seed=args.seed)
            print(f"[{time.perf_counter()-t0:.0f}s] бакеты по длине: {len(bs_pt)} батчей "
                  f"по {args.batch}", flush=True)
            dl_pt = DataLoader(ds_pt, batch_sampler=bs_pt, collate_fn=col,
                               num_workers=args.workers, pin_memory=True,
                               persistent_workers=args.workers > 0)
        else:
            dl_pt = DataLoader(ds_pt, batch_size=args.batch, shuffle=True, collate_fn=col,
                               num_workers=args.workers, pin_memory=True, drop_last=True,
                               persistent_workers=args.workers > 0)
        steps = args.pretrain_steps or len(dl_pt)
        print(f"[{time.perf_counter()-t0:.0f}s] pretrain on {len(ds_pt)} llm pairs, "
              f"{steps} steps", flush=True)
        mlm_head = None
        if args.mlm_w > 0:
            from transformers import AutoModelForMaskedLM
            mlm_kwargs = {"trust_remote_code": trust}
            if args.attn_implementation:
                mlm_kwargs["attn_implementation"] = args.attn_implementation
            mlm_model = AutoModelForMaskedLM.from_pretrained(
                args.model, **mlm_kwargs).to(device)
            # общий энкодер: MLM учит представления, cross-encoder — решение.
            # ВАЖНО: присваивать через base_model_prefix, а не `.base_model` —
            # последнее property без сеттера, присваивание уходит в
            # nn.Module.__setattr__ и создаёт ОТДЕЛЬНЫЙ энкодер, из-за чего
            # градиент MLM в cross-encoder не течёт вовсе (тихий no-op ценой 2x)
            setattr(mlm_model, mlm_model.base_model_prefix,
                    getattr(model, model.base_model_prefix))
            mlm_head = (mlm_model, tok)
            print(f"[{time.perf_counter()-t0:.0f}s] MLM-голова подключена "
                  f"(вес {args.mlm_w})", flush=True)
        if args.ckpt_every:
            os.makedirs(args.ckpt_dir, exist_ok=True)
            tok.save_pretrained(args.ckpt_dir)   # чтобы папку можно было подать в --model
        run_phase("pretrain", model, dl_pt, steps, args.pretrain_lr, args, evaluate, t0,
                  mlm_head=mlm_head)
        if args.mlm_w > 0:
            del mlm_head, mlm_model
            torch.cuda.empty_cache()
        del dl_pt, ds_pt, llm, llm_texts, target_table
        if args.save_after_pretrain or args.pretrain_only:
            # LLM-пары ближе к тесту по распределению (19.5% позитивов против
            # 26.3% в ручной разметке), поэтому модель без дообучения на ручных
            # стоит проверить отдельно — локальная валидация здесь смещена
            model.save_pretrained(f"{args.out}/ce_{tag}_pretrained")
            tok.save_pretrained(f"{args.out}/ce_{tag}_pretrained")
            save_input_marker(f"{args.out}/ce_{tag}_pretrained", args)
            print(f"[{time.perf_counter()-t0:.0f}s] сохранена модель после предобучения",
                  flush=True)
        if args.pretrain_only:
            target = f"{args.out}/ce_{tag}_pretrained_val_pred.npy"
            p = predict(model, dl_va, device)
            np.save(target, p)
            macro, _ = macro_pr_auc(y[va_idx], p, pair_cat[va_idx])
            print(f"[{time.perf_counter()-t0:.0f}s] pretrain-only: "
                  f"val prediction -> {target}; full {macro:.4f}", flush=True)
            return

    if args.use_all_data:
        tr_idx = np.arange(len(y))
        print(f"[{time.perf_counter()-t0:.0f}s] обучаемся на всех {len(tr_idx)} парах "
              f"(val входит в train — метрика ниже недостоверна)", flush=True)

    if args.train_labels:
        y = apply_train_labels(y, tr_idx, np.load(args.train_labels), len(y))
        print(f"[{time.perf_counter()-t0:.0f}s] мишень train подменена из "
              f"{os.path.basename(args.train_labels)}: "
              f"prevalence {y[tr_idx].mean():.4f}", flush=True)

    if args.train_keep:
        before = len(tr_idx)
        tr_idx = select_train_keep(tr_idx, np.load(args.train_keep), len(y))
        print(f"[{time.perf_counter()-t0:.0f}s] отбор по --train_keep: "
              f"{before} -> {len(tr_idx)} пар, "
              f"prevalence {y[tr_idx].mean():.4f}", flush=True)

    if args.neg_hardness and args.neg_keep_top < 1.0:
        # две трети негативов модель отбрасывает ниже 5-го процентиля позитивов —
        # на них учиться нечему, а в тесте (retrieval) таких пар нет вовсе
        s = np.load(args.neg_hardness)
        keep = np.zeros(len(y), dtype=bool)
        keep[tr_idx] = y[tr_idx] == 1
        for c in np.unique(pair_cat):  # порог внутри категории
            neg = tr_idx[(y[tr_idx] == 0) & (pair_cat[tr_idx] == c)]
            if len(neg) == 0:
                continue
            thr = np.quantile(s[neg], 1 - args.neg_keep_top)
            keep[neg[s[neg] >= thr]] = True
        tr_idx = np.where(keep)[0]
        print(f"[{time.perf_counter()-t0:.0f}s] оставлено {len(tr_idx)} пар "
              f"(трудных негативов {args.neg_keep_top:.0%}), "
              f"prevalence {y[tr_idx].mean():.4f}", flush=True)

    if args.train_prevalence > 0:
        # модель учится при 26% позитивов, а применяется при 11% — выравниваем
        from metrics import downsample_positives
        keep = downsample_positives(y[tr_idx], pair_cat[tr_idx], args.train_prevalence)
        tr_idx = tr_idx[keep]
        print(f"[{time.perf_counter()-t0:.0f}s] train проредили до "
              f"prevalence {y[tr_idx].mean():.4f}, осталось {len(tr_idx)} пар", flush=True)

    ds_tr = PairDS([ta[i] for i in tr_idx], [tb[i] for i in tr_idx], y[tr_idx],
                   swap=args.swap_aug, seed=args.seed,
                   teacher=teach[tr_idx] if teach is not None else None,
                   sample_weight=(sample_weight[tr_idx]
                                  if sample_weight is not None else None))
    if args.cat_batch:
        bs = CatBatchSampler(pair_cat[tr_idx], args.batch, seed=args.seed)
        print(f"[{time.perf_counter()-t0:.0f}s] категорийные батчи: {len(bs)} батчей "
              f"по {args.batch}", flush=True)
        dl_tr = DataLoader(ds_tr, batch_sampler=bs, collate_fn=col,
                           num_workers=args.workers, pin_memory=True,
                           persistent_workers=args.workers > 0)
    else:
        dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True, collate_fn=col,
                           num_workers=args.workers, pin_memory=True, drop_last=True,
                           persistent_workers=args.workers > 0)
    steps = optimizer_steps_for_epochs(len(dl_tr), args.epochs, args.accum)
    print(f"[{time.perf_counter()-t0:.0f}s] finetune on {len(ds_tr)} human pairs, "
          f"{steps} steps", flush=True)
    run_phase("finetune", model, dl_tr, steps, args.lr, args, evaluate, t0)

    p = predict(model, dl_va, device)
    macro, per_cat = macro_pr_auc(y[va_idx], p, pair_cat[va_idx])
    print(f"\nFINAL macro PR-AUC = {macro:.4f}")
    for c, s in sorted(per_cat.items(), key=lambda kv: kv[1]):
        print(f"  {c:26s} {s:.4f}")

    model.save_pretrained(f"{args.out}/ce_{tag}")
    tok.save_pretrained(f"{args.out}/ce_{tag}")
    save_input_marker(f"{args.out}/ce_{tag}", args)
    np.save(f"{args.out}/ce_{tag}_val_pred.npy", p)
    print(f"[{time.perf_counter()-t0:.0f}s] saved to {args.out}/ce_{tag}")


if __name__ == "__main__":
    main()
