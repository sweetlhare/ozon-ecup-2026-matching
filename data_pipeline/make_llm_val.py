"""Валидация из LLM-разметки — прямой аналог теста, без подогнанных параметров.

hard_slice воспроизводит тест через параметр drop_easy=0.64, подобранный по трём
сабмитам. Уровень метрики целиком определяется этим параметром (при 0.50 выходит
0.541, при 0.85 — 0.350), поэтому доверять можно только порядку конфигураций.

llm_val устроен иначе: берём hold-out прямо из matches_llm, который порождён тем
же процессом, что тест. Совпадают и структура (целые списки кандидатов), и функция
разметки (метка = 9/9 голосов, prevalence 0.1094 против 0.1113 в тесте), и
сложность негативов. Подгонять нечего.

Товары не должны пересекаться с обучающим потоком, иначе замеряем запоминание:
режем по компонентам связности, как в split.py.
"""
import argparse
import time

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from text import item_texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--val_frac", type=float, default=0.05,
                    help="доля компонент связности, уходящих в валидацию")
    ap.add_argument("--out_prefix", default="data/llmval")
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    t0 = time.perf_counter()
    ml = pd.read_parquet(f"{args.data}/matches_llm.parquet")
    print(f"[{time.perf_counter()-t0:.0f}s] всего пар: {len(ml)}", flush=True)

    # компоненты связности: якорь и все его кандидаты должны попасть в один набор
    ids = pd.unique(pd.concat([ml.id1, ml.id2], ignore_index=True))
    pos = pd.Series(np.arange(len(ids)), index=ids)
    a, b = pos[ml.id1.values].values, pos[ml.id2.values].values
    g = coo_matrix((np.ones(len(a)), (a, b)), shape=(len(ids), len(ids)))
    _, comp = connected_components(g, directed=False)
    pair_comp = comp[a]

    # Делим КОМПОНЕНТЫ на обучение и валидацию. Брать «остаток» после отбора
    # обучающего потока нельзя: `--candidates` забирает все компоненты, где есть
    # хотя бы один позитив, и в остатке prevalence падает до 0.0015.
    rng = np.random.default_rng(args.seed)
    n_comp = comp.max() + 1
    is_val_comp = rng.random(n_comp) < args.val_frac
    sel = is_val_comp[pair_comp]
    sub = ml[sel].reset_index(drop=True)
    np.save(f"{args.out_prefix}_comp_mask.npy", is_val_comp)
    print(f"[{time.perf_counter()-t0:.0f}s] val: {len(sub)} пар из "
          f"{int(is_val_comp.sum())} компонент ({100*sel.mean():.1f}% пар)", flush=True)
    # список товаров валидации — чтобы обучающий поток мог их исключить
    np.save(f"{args.out_prefix}_items_ids.npy",
            pd.unique(pd.concat([sub.id1, sub.id2], ignore_index=True)))

    need = set(pd.concat([sub.id1, sub.id2], ignore_index=True).tolist())
    parts = []
    for batch in ds.dataset(f"{args.data}/items.parquet", format="parquet").to_batches(
            columns=["id", "name", "attributes", "category"], batch_size=200_000):
        df = batch.to_pandas()
        df = df[df["id"].isin(need)]
        if len(df):
            parts.append(pd.DataFrame({"id": df["id"].values,
                                       "text": item_texts(df).values,
                                       "category": df["category"].values}))
    items = pd.concat(parts, ignore_index=True)
    del parts

    # метка как в тесте: единогласное решение 9 голосов
    sub["label"] = (sub.target.values >= 1.0).astype(np.float32)
    cat = dict(zip(items.id, items.category))
    sub["category"] = [cat.get(i, "?") for i in sub.id1]
    prev = float(np.mean([sub.label[sub.category == c].mean()
                          for c in sub.category.unique()]))
    print(f"[{time.perf_counter()-t0:.0f}s] macro-prevalence {prev:.4f} "
          f"(в тесте 0.1113)", flush=True)

    items.to_parquet(f"{args.out_prefix}_items.parquet", index=False)
    sub.to_parquet(f"{args.out_prefix}_pairs.parquet", index=False)
    print(f"[{time.perf_counter()-t0:.0f}s] сохранено {args.out_prefix}_{{pairs,items}}",
          flush=True)


if __name__ == "__main__":
    main()

