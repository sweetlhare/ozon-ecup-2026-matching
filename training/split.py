"""Train/val сплит по компонентам связности товаров.

Товары почти не переиспользуются между парами (711k уникальных на 366k пар),
но связные компоненты всё же есть — режем по ним, чтобы не было утечки.
"""
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def make_split(matches: pd.DataFrame, cat: pd.Series, val_frac: float = 0.2,
               seed: int = 42) -> np.ndarray:
    """Возвращает булеву маску: True = val."""
    ids = pd.unique(pd.concat([matches.id1, matches.id2], ignore_index=True))
    pos = pd.Series(np.arange(len(ids)), index=ids)
    a = pos[matches.id1.values].values
    b = pos[matches.id2.values].values

    g = coo_matrix((np.ones(len(a)), (a, b)), shape=(len(ids), len(ids)))
    _, labels = connected_components(g, directed=False)
    comp = labels[a]  # компонента пары

    # стратификация по категории: внутри категории делим компоненты
    rng = np.random.default_rng(seed)
    is_val = np.zeros(len(matches), dtype=bool)
    df = pd.DataFrame({"comp": comp, "cat": cat.values})
    for _, idx in df.groupby("cat").groups.items():
        idx = np.asarray(idx)
        comps = pd.unique(df.comp.values[idx])
        pick = rng.random(len(comps)) < val_frac
        is_val[idx] = np.isin(df.comp.values[idx], comps[pick])
    return is_val

