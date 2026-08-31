"""Построение текста товара.

Значения атрибутов обрезаются: длинные поля («комплектация», «состав»,
описания) занимают до 29% бюджета токенов, а по разметке они шумные —
совпадают у негативов чаще, чем у позитивов.
"""
import json
import os

import pandas as pd

# Порядок полей внутри карточки. Обрезка работает ПОПАРНО (longest_first по паре),
# а не покарточно: при max_len 384 режется 20.1% пар и 13.3% токенов, тогда как
# карточек длиннее 384 всего 2.9%. В 79.6% обрезанных пар гибнет поле, которое
# есть у ОБЕИХ карточек, то есть было бы сравнимо — это 15.6% всех пар.
# При этом порядок ключей JSON антикоррелирует с полезностью: бренд и артикул
# стоят первыми (средняя позиция 0.08 и 0.12) и всегда выживают, а вес, размер,
# ширина и объём — последними (0.81, 0.75, 0.71, 0.62), и именно у них ненулевой
# остаточный сигнал при зафиксированном скоре модели (+0.025, +0.027, +0.012,
# +0.025). Сортировка снижает долю пар с потерей общего поля с 16.5% до 6.7%,
# ничего не выбрасывая и не меняя длину.
_HERE = os.path.dirname(os.path.abspath(__file__))
try:
    from attr_canon import CANON as _CANON
except Exception:  # noqa: BLE001
    _CANON = {}
try:
    _PRI_BY_CAT = {c: {k: i for i, k in enumerate(ks)}
                   for c, ks in json.load(
                       open(os.path.join(_HERE, "pri_by_cat.json"), encoding="utf-8")).items()}
except Exception:  # noqa: BLE001
    _PRI_BY_CAT = {}
# запасной глобальный порядок для категорий и ключей вне pri_by_cat
_PRI_GLOBAL = {k: i for i, k in enumerate([
    "размер", "вес", "объем", "высота", "ширина", "длина", "глубина", "диаметр",
    "цвет", "артикул", "партномер", "модель", "бренд", "тип", "материал",
    "страна", "пол", "сезон", "количество", "комплектация", "состав"])}


def _order_key(cat_pri, k, i):
    """Ключи сортируются: приоритет категории -> глобальный -> исходный порядок.

    Имя ключа сперва канонизируется: в данных «партномер (артикул
    производителя)», «цвет товара», «российский размер» — то же, что «артикул»,
    «цвет», «размер», и без приведения важные поля не находились в приоритетах
    и уезжали в конец, то есть ровно под обрезку.
    """
    ck = k.strip().lower()
    ck = _CANON.get(ck, ck)
    if ck in cat_pri:
        return (0, cat_pri[ck], i)
    if ck in _PRI_GLOBAL:
        return (1, _PRI_GLOBAL[ck], i)
    return (2, 0, i)


def item_texts(items: pd.DataFrame, value_max: int = 0,
               order: bool = False) -> pd.Series:
    """value_max=0 — сырая JSON-строка (быстро, векторно);
    value_max>0 — разбор JSON с обрезкой каждого значения;
    order=True — поля в порядке измеренной силы, а не в порядке ключей JSON.

    Категория идёт первой: она задаёт, какое поле решает исход (в обуви размер,
    в ювелирке вставка), а метрика считается покатегорийно.
    """
    name = items["name"].astype(str)
    if "category" in items.columns:
        name = items["category"].astype(str) + " ; " + name
    if order:
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
            cat_pri = _PRI_BY_CAT.get(c, {})
            items_ = sorted(enumerate(d.items()),
                            key=lambda p: _order_key(cat_pri, p[1][0], p[0]))
            vs = (f"{k}: {str(v)[:value_max] if value_max > 0 else v}"
                  for _, (k, v) in items_)
            out.append(n + " ; " + "; ".join(vs))
        return pd.Series(out, index=items.index)
    if value_max <= 0:
        attrs = (items["attributes"].astype(str)
                 .str.replace(r'^\{"|"\}$', "", regex=True)
                 .str.replace(r'","', "; ", regex=True)
                 .str.replace(r'":"', ": ", regex=True))
        return name + " ; " + attrs

    out = []
    for n, a in zip(name.tolist(), items["attributes"].astype(str).tolist()):
        try:
            d = json.loads(a)
            if not isinstance(d, dict):
                d = {}
        except Exception:  # noqa: BLE001
            d = {}
        out.append(n + " ; " + "; ".join(f"{k}: {str(v)[:value_max]}" for k, v in d.items()))
    return pd.Series(out, index=items.index)

