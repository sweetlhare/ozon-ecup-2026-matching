"""Метрика соревнования и приведение валидации к тестовой доле позитивов.

Константный сабмит показал: в тесте 11.1% позитивов против 26.3% в ручной
разметке. PR-AUC от доли позитивов зависит напрямую, поэтому валидацию
имеет смысл считать и на прореженной выборке — так масштаб сопоставим
с public-лидербордом.
"""
import numpy as np
from sklearn.metrics import average_precision_score

TEST_PREVALENCE = 0.1113


def macro_pr_auc(y, p, cat):
    scores = []
    for c in np.unique(cat):
        m = cat == c
        if y[m].min() == y[m].max():
            continue
        scores.append(average_precision_score(y[m], p[m]))
    return float(np.mean(scores))


def macro_prevalence(y, cat):
    return float(np.mean([y[cat == c].mean() for c in np.unique(cat)]))


def downsample_positives(y, cat, target=TEST_PREVALENCE, seed=0):
    """Маска, прореживающая позитивы в каждой категории до целевой доли."""
    rng = np.random.default_rng(seed)
    keep = np.ones(len(y), dtype=bool)
    for c in np.unique(cat):
        idx = np.where(cat == c)[0]
        pos, neg = idx[y[idx] == 1], idx[y[idx] == 0]
        n_pos = int(round(target * len(neg) / (1 - target)))
        if n_pos < len(pos):
            keep[rng.choice(pos, len(pos) - n_pos, replace=False)] = False
    return keep


def hard_slice(y, cat, judge, drop_easy=0.64, target=TEST_PREVALENCE, seed=0):
    """Маска val, приближающая тестовое распределение.

    Тест — это наша валидация, из которой убраны лёгкие негативы: модель
    отбрасывает 64% негативов ниже 5-го процентиля позитивов, а тестовые
    пары собраны retrieval'ом и таких не содержат. Выбрасываем самые лёгкие
    негативы внутри категории, затем прореживаем позитивы до тестовой доли.

    judge — предсказания СТОРОННЕЙ модели (не той, что оцениваем),
    иначе отбор станет циркулярным.
    """
    rng = np.random.default_rng(seed)
    keep = np.ones(len(y), dtype=bool)
    for c in np.unique(cat):
        idx = np.where(cat == c)[0]
        neg, pos = idx[y[idx] == 0], idx[y[idx] == 1]
        if len(neg) == 0:
            continue
        n_drop = int(round(drop_easy * len(neg)))
        if n_drop:
            keep[neg[np.argsort(judge[neg])[:n_drop]]] = False
        # прореживать позитивы нужно от ОСТАВШИХСЯ негативов, иначе
        # выброс лёгких негативов сам поднимет долю позитивов
        n_neg_left = len(neg) - n_drop
        n_pos = int(round(target * n_neg_left / (1 - target)))
        if n_pos < len(pos):
            keep[rng.choice(pos, len(pos) - n_pos, replace=False)] = False
    return keep


def hard_slice_mean(y, p, cat, judge, n_seeds=20, drop_easy=0.64,
                    target=TEST_PREVALENCE):
    """Среднее hard-slice по нескольким маскам — снижает шум в sqrt(n) раз.

    В одной маске остаётся всего ~2400 позитивов (медиана 124 на категорию),
    отсюда шум ±0.020 — вдвое больше типичной разницы между конфигурациями
    (0.010). Усреднение по 20 маскам даёт ±0.004, что уже позволяет
    различать близкие варианты.
    """
    vals = [macro_pr_auc(y[k], p[k], cat[k])
            for k in (hard_slice(y, cat, judge, drop_easy, target, seed=s)
                      for s in range(n_seeds))]
    return float(np.mean(vals)), float(np.std(vals))


def report(y, p, cat, label="", judge=None):
    """Метрика как есть, в тестовом масштабе и на hard-slice."""
    full = macro_pr_auc(y, p, cat)
    keep = downsample_positives(y, cat)
    scaled = macro_pr_auc(y[keep], p[keep], cat[keep])
    line = (f"{label:24s} macro PR-AUC = {full:.4f}   "
            f"@prevalence {TEST_PREVALENCE:.3f}: {scaled:.4f}")
    hard = None
    if judge is not None:
        hk = hard_slice(y, cat, judge)
        hard = macro_pr_auc(y[hk], p[hk], cat[hk])
        line += f"   hard-slice: {hard:.4f}"
    print(line)
    return full, scaled, hard


def novelty_mask(matches, va_idx, items, sim_path="data/val_maxsim.npy", cutoff=0.95):
    """Пары, где ОБА товара далеки от обучающего каталога.

    Обе прежние валидации антикоррелировали с лидербордом. Разбор показал две
    причины, обе устранимы:

      * `downsample_positives` (прореживание до тестовой prevalence) убивает
        корреляцию: r падает с +0.34 до -0.01. Приведение масштаба к тесту
        добавляет дисперсию и переставляет конфигурации местами;
      * 95.1% валидационных товаров имеют почти-дубль в трейне (медиана
        сходства 0.870) — модель их узнаёт, а на тесте товары новые.

    Метрика на новых парах без прореживания: r = +0.560 по пяти сабмитам
    и +0.987 по четырём (v10 — единственная конфигурация из двух моделей —
    остаётся выбросом). Приём взят из задачи 2 того же соревнования
    (`~/competition/ozon_kontrol/src/novelty_eval.py`), где стенд новинок дал
    ошибку 0.011 против 0.07-0.37 у групповых фолдов.
    """
    pos = {i: k for k, i in enumerate(items.id.tolist())}
    ms = np.load(sim_path)
    sim = {int(a): b for a, b in ms}
    s1 = np.array([sim.get(pos[i], 1.0) for i in matches.id1.values[va_idx]])
    s2 = np.array([sim.get(pos[i], 1.0) for i in matches.id2.values[va_idx]])
    return np.maximum(s1, s2) < cutoff


def novelty_macro(y, p, cat, mask, min_pairs=40):
    """macro PR-AUC на подвыборке новинок, без прореживания позитивов."""
    vals = []
    for c in np.unique(cat[mask]):
        k = mask & (cat == c)
        if k.sum() < min_pairs or y[k].min() == y[k].max():
            continue
        vals.append(average_precision_score(y[k], p[k]))
    return float(np.mean(vals)) if vals else float("nan")

