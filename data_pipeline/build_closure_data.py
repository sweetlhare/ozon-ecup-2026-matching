#!/usr/bin/env python3
"""Build every closure corpus used before the two final training runs.

Inputs are only the four parquet files published by the organizers.  The
builder deliberately works with item ids and real catalogue texts: it never
edits attributes or invents a product description.
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

try:
    from .calibrate_llm_targets import target_to_k
    from .text import item_texts
except ImportError:  # direct script execution
    from calibrate_llm_targets import target_to_k
    from text import item_texts


SEED = 20260818
SAFE_CATEGORIES = (
    "Автотовары", "Аптека", "Бытовая техника", "Бытовая химия",
    "Галантерея и аксессуары", "Детские товары", "Дом и сад",
    "Канцелярские товары", "Красота и гигиена", "Музыкальные инструменты",
    "Обувь", "Одежда", "Продукты питания", "Спорт и отдых",
    "Строительство и ремонт", "Товары для животных", "Электроника",
)
GAP_CATEGORIES = ("Мебель", "Хобби и творчество", "Ювелирные изделия")
PAIR_DTYPE = np.dtype([("lo", "<i8"), ("hi", "<i8")])


def pair_records(left, right):
    left, right = np.asarray(left, np.int64), np.asarray(right, np.int64)
    out = np.empty(left.shape, dtype=PAIR_DTYPE)
    out["lo"], out["hi"] = np.minimum(left, right), np.maximum(left, right)
    return out


def stable_hash(left, right, seed=SEED):
    lo = np.minimum(np.asarray(left, np.uint64), np.asarray(right, np.uint64))
    hi = np.maximum(np.asarray(left, np.uint64), np.asarray(right, np.uint64))
    with np.errstate(over="ignore"):
        value = (lo * np.uint64(0x9E3779B185EBCA87)
                 ^ hi * np.uint64(0xC2B2AE3D27D4EB4F) ^ np.uint64(seed))
        value ^= value >> np.uint64(30)
        value *= np.uint64(0xBF58476D1CE4E5B9)
        value ^= value >> np.uint64(27)
        value *= np.uint64(0x94D049BB133111EB)
        value ^= value >> np.uint64(31)
    return value


def ordered_items(items_path, selected_ids, batch_size=200_000, sort_ids=True):
    """Render ORDERED texts for selected ids without materializing 13M texts."""
    selected_ids = np.unique(np.asarray(selected_ids, np.int64))
    selected_index = pd.Index(selected_ids)
    chunks = []
    dataset = ds.dataset(str(items_path), format="parquet")
    for batch in dataset.to_batches(
            columns=["id", "name", "attributes", "category"],
            batch_size=batch_size):
        frame = batch.to_pandas()
        frame = frame[selected_index.get_indexer(frame.id.to_numpy(np.int64)) >= 0]
        if len(frame):
            chunks.append(pd.DataFrame({
                "id": frame.id.to_numpy(np.int64),
                "text": item_texts(frame, order=True).to_numpy(),
            }))
    if not chunks:
        raise RuntimeError("selected items are absent from items.parquet")
    result = pd.concat(chunks, ignore_index=True).drop_duplicates("id")
    if sort_ids:
        result = result.sort_values("id", kind="stable")
    result = result.reset_index(drop=True)
    missing = np.setdiff1d(selected_ids, result.id.to_numpy())
    if len(missing):
        raise RuntimeError(f"missing {len(missing)} selected catalogue items")
    return result


def closure_candidates(matches, category_of, min_size=3, max_size=4):
    """Unobserved edges implied by unanimous k=9 connected components."""
    k = target_to_k(matches.target.to_numpy())
    ids = pd.unique(np.concatenate([matches.id1, matches.id2]))
    code = pd.Series(np.arange(len(ids), dtype=np.int64), index=ids)
    left = code.reindex(matches.id1).to_numpy(np.int64)
    right = code.reindex(matches.id2).to_numpy(np.int64)
    observed = set(zip(np.minimum(left, right), np.maximum(left, right)))
    positive = k == 9
    graph = coo_matrix(
        (np.ones(int(positive.sum()), np.uint8), (left[positive], right[positive])),
        shape=(len(ids), len(ids)))
    n_components, component = connected_components(graph, directed=False)
    size = np.bincount(component, minlength=n_components)
    order = np.argsort(component, kind="stable")
    bounds = np.searchsorted(component[order], np.arange(n_components + 1))
    rows = []
    for label in np.where((size >= min_size) & (size <= max_size))[0]:
        members = order[bounds[label]:bounds[label + 1]]
        for a, b in combinations(members.tolist(), 2):
            lo, hi = min(a, b), max(a, b)
            if (lo, hi) not in observed:
                id1, id2 = min(ids[a], ids[b]), max(ids[a], ids[b])
                category = category_of.get(id1)
                if str(category) != str(category_of.get(id2)):
                    raise RuntimeError("closure component crosses categories")
                rows.append((id1, id2, str(category),
                             f"k9_closure_component_{min_size}_{max_size}"))
    return pd.DataFrame(rows, columns=["id1", "id2", "category", "source"])


def consensus_negatives(matches, rendered, category_of, min_annotations=2):
    """Unseen endpoint substitutions for repeated exact inputs, all with k=0."""
    k = target_to_k(matches.target.to_numpy())
    text_code, unique_text = pd.factorize(rendered.text.fillna(""), sort=False)
    item_ids = rendered.id.to_numpy(np.int64)
    code_of = pd.Series(text_code.astype(np.int64), index=item_ids)
    left_text = code_of.reindex(matches.id1).to_numpy()
    right_text = code_of.reindex(matches.id2).to_numpy()
    present = ~(pd.isna(left_text) | pd.isna(right_text))
    k = k[present]
    left_text = left_text[present].astype(np.int64)
    right_text = right_text[present].astype(np.int64)
    text_order = np.argsort(text_code, kind="stable")
    text_bounds = np.searchsorted(text_code[text_order],
                                  np.arange(len(unique_text) + 1))
    lo_text, hi_text = np.minimum(left_text, right_text), np.maximum(left_text, right_text)
    base = len(unique_text)
    keys = lo_text * np.int64(base) + hi_text
    _, group, group_size = np.unique(keys, return_inverse=True, return_counts=True)
    group_zero = np.bincount(group, weights=k == 0,
                             minlength=len(group_size)).astype(np.int64)
    group_order = np.argsort(group, kind="stable")
    group_bounds = np.searchsorted(group[group_order], np.arange(len(group_size) + 1))
    observed = np.unique(pair_records(matches.id1, matches.id2))

    def unseen_pair(text_lo, text_hi):
        ids_lo = np.sort(item_ids[text_order[text_bounds[text_lo]:text_bounds[text_lo + 1]]])
        ids_hi = np.sort(item_ids[text_order[text_bounds[text_hi]:text_bounds[text_hi + 1]]])
        candidates = (combinations(ids_lo.tolist(), 2) if text_lo == text_hi
                      else ((int(a), int(b)) for a in ids_lo for b in ids_hi))
        for a, b in candidates:
            if a == b:
                continue
            candidate = np.array((min(a, b), max(a, b)), dtype=PAIR_DTYPE)[()]
            pos = int(np.searchsorted(observed, candidate))
            if pos == len(observed) or observed[pos] != candidate:
                return min(a, b), max(a, b)
        return None

    rows = []
    eligible = np.where((group_size >= min_annotations)
                        & (group_zero == group_size))[0]
    for label in eligible:
        first = group_order[group_bounds[label]]
        pair = unseen_pair(int(lo_text[first]), int(hi_text[first]))
        if pair is None:
            continue
        category = category_of.get(pair[0])
        if str(category) != str(category_of.get(pair[1])):
            raise RuntimeError("exact-input group crosses categories")
        rows.append((*pair, str(category), "repeat_exact_input_all_k0_counterfactual"))
    return pd.DataFrame(rows, columns=["id1", "id2", "category", "source"])


def deduplicate_inputs(positive, negative, text_of):
    combined = pd.concat([
        positive.assign(_label=1), negative.assign(_label=0)
    ], ignore_index=True)
    left = text_of.reindex(combined.id1).to_numpy()
    right = text_of.reindex(combined.id2).to_numpy()
    codes, unique = pd.factorize(np.concatenate([left, right]), sort=False)
    n = len(combined)
    combined["_input"] = (np.minimum(codes[:n], codes[n:]) * len(unique)
                           + np.maximum(codes[:n], codes[n:]))
    pos = np.unique(combined.loc[combined._label == 1, "_input"])
    neg = np.unique(combined.loc[combined._label == 0, "_input"])
    conflict = np.intersect1d(pos, neg, assume_unique=True)
    combined = combined[~combined._input.isin(conflict)].copy()
    combined["_hash"] = stable_hash(combined.id1, combined.id2)
    combined = combined.sort_values(
        ["_label", "_input", "_hash", "id1", "id2"], kind="stable")
    combined = combined.drop_duplicates(["_label", "_input"], keep="first")
    columns = ["id1", "id2", "category", "source"]
    return (combined.loc[combined._label == 1, columns].reset_index(drop=True),
            combined.loc[combined._label == 0, columns].reset_index(drop=True))


def select_pool(positive, negative, per_category, allow_short=False):
    parts = []
    for category in SAFE_CATEGORIES:
        for frame, target in ((positive, 1.0), (negative, 0.0)):
            pool = frame[frame.category == category].drop_duplicates(["id1", "id2"]).copy()
            if len(pool) < per_category and not allow_short:
                raise RuntimeError(f"not enough target={target:g} rows in {category}")
            take = min(len(pool), per_category)
            pool["_hash"] = stable_hash(pool.id1, pool.id2)
            pool = pool.sort_values(["_hash", "id1", "id2"], kind="stable").head(take)
            pool["target"] = pool["label"] = np.float32(target)
            parts.append(pool)
    selected = pd.concat(parts, ignore_index=True)
    selected["_cat"] = pd.Categorical(selected.category, SAFE_CATEGORIES, ordered=True)
    return (selected.sort_values(["_cat", "target", "_hash", "id1", "id2"],
                                 ascending=[True, False, True, True, True], kind="stable")
            .drop(columns=["_cat", "_hash"]).reset_index(drop=True))


def write_dataset(pairs, rendered, prefix, sort_items=True):
    prefix = Path(prefix)
    ids = np.unique(np.concatenate([pairs.id1, pairs.id2]))
    items = rendered[rendered.id.isin(ids)]
    if sort_items:
        items = items.sort_values("id", kind="stable")
    if len(items) != len(ids):
        raise RuntimeError(f"{prefix}: incomplete item coverage")
    pairs[["id1", "id2", "target", "label", "category", "source"]].to_parquet(
        f"{prefix}_pairs.parquet", index=False)
    items[["id", "text"]].to_parquet(f"{prefix}_items.parquet", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data"))
    args = parser.parse_args()
    data = args.data
    matches = pd.read_parquet(data / "matches_llm.parquet",
                              columns=["id1", "id2", "target"])
    human = pd.read_parquet(data / "matches.parquet", columns=["id1", "id2"])
    categories = pd.read_parquet(data / "items.parquet", columns=["id", "category"])
    category_of = categories.set_index("id").category
    del categories

    positive = closure_candidates(matches, category_of)
    llm_train_items = pd.read_parquet(data / "llmfull_ord_items.parquet",
                                      columns=["id", "text"])
    negative = consensus_negatives(matches, llm_train_items, category_of)
    del llm_train_items
    candidate_ids = np.unique(np.concatenate([
        positive.id1, positive.id2, negative.id1, negative.id2]))
    candidate_items = ordered_items(data / "items.parquet", candidate_ids)
    text_of = candidate_items.set_index("id").text
    positive, negative = deduplicate_inputs(positive, negative, text_of)

    consensus = select_pool(positive, negative, 600)
    big = select_pool(positive, negative, 4_000, allow_short=True)
    observed = np.unique(pair_records(matches.id1, matches.id2))
    human_ids = np.unique(np.concatenate([human.id1, human.id2]))
    for name, frame in (("closure_consensus", consensus), ("closure_big", big)):
        if np.intersect1d(pair_records(frame.id1, frame.id2), observed).size:
            raise RuntimeError(f"{name}: overlap with organizer LLM pairs")
        if np.intersect1d(np.unique(np.concatenate([frame.id1, frame.id2])), human_ids).size:
            raise RuntimeError(f"{name}: overlap with organizer human items")
        write_dataset(frame, candidate_items, data / name)

    consensus_ids = np.unique(np.concatenate([consensus.id1, consensus.id2]))
    consensus_items = (candidate_items[candidate_items.id.isin(consensus_ids)]
                       .sort_values("id", kind="stable").reset_index(drop=True))

    # Preserve the historical dense-code orientation and RNG order exactly.
    ids = pd.unique(np.concatenate([matches.id1.to_numpy(), matches.id2.to_numpy()]))
    code = pd.Series(np.arange(len(ids), dtype=np.int64), index=ids)
    left = code.reindex(matches.id1).to_numpy(np.int64)
    right = code.reindex(matches.id2).to_numpy(np.int64)
    k = target_to_k(matches.target.to_numpy())
    positive_edges = np.where(k == 9)[0]
    n_components, component = connected_components(
        coo_matrix((np.ones(len(positive_edges), np.int8),
                    (left[positive_edges], right[positive_edges])),
                   shape=(len(ids), len(ids))),
        directed=False)
    component_size = np.bincount(component, minlength=n_components)
    observed_dense = set(zip(np.minimum(left, right), np.maximum(left, right)))
    node_category = category_of.reindex(ids).to_numpy()
    k = target_to_k(matches.target.to_numpy())
    gap_rows = []
    rng = np.random.default_rng(20260820)
    for category in GAP_CATEGORIES:
        in_category = np.where(node_category == category)[0]
        allowed = np.zeros(len(ids), dtype=bool)
        allowed[in_category] = True
        labels = np.unique(component[in_category])
        labels = labels[(component_size[labels] >= 3)
                        & (component_size[labels] <= 8)]
        candidates = []
        for label in labels:
            members = np.where((component == label) & allowed)[0]
            for a, b in combinations(members.tolist(), 2):
                lo, hi = min(a, b), max(a, b)
                if (lo, hi) not in observed_dense:
                    candidates.append((lo, hi))
        candidates = list(dict.fromkeys(candidates))
        chosen = [candidates[i] for i in
                  rng.choice(len(candidates), 600, replace=False)]
        for a, b in chosen:
            gap_rows.append((ids[a], ids[b], 1.0, 1.0, category,
                             "k9_closure_gapcat_3_8"))
        negative_edges = np.where((k == 0) & allowed[left] & allowed[right])[0]
        for position in rng.choice(len(negative_edges), 600, replace=False):
            row = negative_edges[position]
            gap_rows.append((matches.id1.iloc[row], matches.id2.iloc[row],
                             0.0, 0.0, category, "observed_k0_gapcat"))
    gap = pd.DataFrame(
        gap_rows,
        columns=["id1", "id2", "target", "label", "category", "source"])
    gap = gap.sort_values(["id1", "id2"], kind="stable").reset_index(drop=True)
    gap_ids = np.unique(np.concatenate([gap.id1, gap.id2]))
    gap_items = ordered_items(data / "items.parquet", gap_ids, sort_ids=False)
    write_dataset(gap, gap_items, data / "closure_gapcats", sort_items=False)

    all20 = pd.concat([consensus, gap], ignore_index=True)
    all20 = all20.sort_values(["id1", "id2"], kind="stable").reset_index(drop=True)
    all20_items = (pd.concat([consensus_items, gap_items], ignore_index=True)
                   .drop_duplicates("id").reset_index(drop=True))
    all20[["id1", "id2", "target", "label", "category", "source"]].to_parquet(
        data / "closure_all20_pairs.parquet", index=False)
    all20_items[["id", "text"]].to_parquet(
        data / "closure_all20_items.parquet", index=False)
    print(f"closure_consensus={len(consensus):,}; closure_big={len(big):,}; "
          f"closure_gapcats={len(gap):,}; closure_all20={len(all20):,}")


if __name__ == "__main__":
    main()
