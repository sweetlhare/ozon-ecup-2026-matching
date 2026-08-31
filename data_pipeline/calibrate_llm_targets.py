"""Калибровка soft target k/9 по повторным разметкам одинаковых текстовых пар.

Тестовая метка равна 1{k=9}, но weak-pretrain использует k/9. Для одинакового
входа, размеченного несколько раз, другая разметка даёт честную оценку
P(k'=9 | наблюдаемый k). Собственная метка строки исключается (leave-one-out),
поэтому k=9 не подтверждает сам себя.

Пример::

    python src/calibrate_llm_targets.py \
      --pairs data/matches_llm.parquet \
      --items data/llmquota_items.parquet \
      --allow-missing-items \
      --apply data/llmquota_ord_pairs.parquet \
      --out data/llm_k9_calibration.json \
      --out-pairs data/llmquota_cal_ord_pairs.parquet
"""
import argparse
import json
import os

import numpy as np
import pandas as pd


_RAW_IDENTITY_COLUMNS = ["category", "name", "attributes"]


def target_to_k(target):
    target = np.asarray(target, dtype=np.float64)
    if not np.isfinite(target).all():
        raise ValueError("target должен содержать только конечные значения")
    k = np.rint(target * 9).astype(np.int8)
    if ((k < 0) | (k > 9)).any():
        raise ValueError("target должен лежать в [0, 1]")
    error = np.abs(target - k / 9)
    if error.max(initial=0.0) > 1e-4:
        raise ValueError("target должен быть кратен 1/9")
    return k


def factorize_item_content(items):
    """Код одинакового входа без склейки многогигабайтных строк.

    Готовый ``id,text`` используется напрямую. Для полного исходного каталога
    текстов ещё нет, поэтому берём строгую идентичность тройки category/name/
    attributes. Два независимых 64-битных хеша сохраняют память и делают риск
    ложного совпадения пренебрежимо малым.
    """
    if "text" in items.columns:
        codes, unique = pd.factorize(items.text.fillna(""), sort=False)
        return codes.astype(np.int64), len(unique), "text"

    missing = [column for column in _RAW_IDENTITY_COLUMNS
               if column not in items.columns]
    if missing:
        raise ValueError(
            "items должен содержать id,text или id,category,name,attributes; "
            f"нет колонок {missing}")
    identity = items[_RAW_IDENTITY_COLUMNS].fillna("")
    h1 = pd.util.hash_pandas_object(
        identity, index=False, hash_key="0123456789abcdef").to_numpy()
    h2 = pd.util.hash_pandas_object(
        identity, index=False, hash_key="fedcba9876543210").to_numpy()
    keys = pd.MultiIndex.from_arrays([h1, h2])
    codes, unique = pd.factorize(keys, sort=False)
    return codes.astype(np.int64), len(unique), "raw-fields-double-hash"


def leave_one_out_table(left_code, right_code, k):
    """Вернуть статистику k -> P(другая метка той же текстовой пары равна 9)."""
    left_code = np.asarray(left_code, dtype=np.int64)
    right_code = np.asarray(right_code, dtype=np.int64)
    k = np.asarray(k, dtype=np.int8)
    if not (len(left_code) == len(right_code) == len(k)):
        raise ValueError("left_code, right_code и k должны иметь одинаковую длину")
    if len(k) == 0:
        raise ValueError("пустой набор пар")

    lo = np.minimum(left_code, right_code)
    hi = np.maximum(left_code, right_code)
    base = int(max(lo.max(), hi.max())) + 1
    key = lo * np.int64(base) + hi
    _, group, group_size = np.unique(key, return_inverse=True, return_counts=True)
    group_positive = np.bincount(group, weights=(k == 9).astype(np.int64))

    size = group_size[group]
    positive = group_positive[group]
    repeated = size > 1
    rows = []
    for value in range(10):
        selected = repeated & (k == value)
        other_labels = int((size[selected] - 1).sum())
        other_positive = float(
            (positive[selected] - (value == 9)).sum())
        # Условие — случайная наблюдаемая строка с данным k, а не случайная
        # пара из двух разметок. Иначе группа размера n получает вес n(n-1) и
        # несколько массовых пустых карточек полностью определяют таблицу.
        if selected.any():
            per_observation = (
                positive[selected] - (value == 9)
            ) / (size[selected] - 1)
            probability = float(per_observation.mean())
        else:
            probability = None
        rows.append({
            "k": value,
            "observations": int(selected.sum()),
            "other_labels": other_labels,
            "other_positive": int(round(other_positive)),
            "probability": probability,
        })
    return rows


def category_leave_one_out_table(left_code, right_code, k, category, alpha=100.0):
    """P(other k=9 | k, category), сглаженная к глобальной таблице.

    ``alpha`` — эквивалентное число глобальных наблюдений для каждой ячейки.
    Его значение 100 выбрано только по group-level OOF, без просмотра val/test.
    """
    left_code = np.asarray(left_code, dtype=np.int64)
    right_code = np.asarray(right_code, dtype=np.int64)
    k = np.asarray(k, dtype=np.int8)
    category = np.asarray(category).astype(str)
    if not (len(left_code) == len(right_code) == len(k) == len(category)):
        raise ValueError("left_code, right_code, k и category должны быть одной длины")
    if len(k) == 0:
        raise ValueError("пустой набор пар")
    if alpha <= 0:
        raise ValueError("category alpha должен быть положительным")

    lo = np.minimum(left_code, right_code)
    hi = np.maximum(left_code, right_code)
    base = int(max(lo.max(), hi.max())) + 1
    key = lo * np.int64(base) + hi
    _, group, group_size = np.unique(key, return_inverse=True, return_counts=True)
    group_positive = np.bincount(group, weights=(k == 9).astype(np.int8))
    size = group_size[group]
    repeated = size > 1
    q = ((group_positive[group[repeated]] - (k[repeated] == 9)) /
         (size[repeated] - 1))
    repeated_k = k[repeated].astype(np.int64)

    global_count = np.bincount(repeated_k, minlength=10).astype(np.float64)
    global_total = np.bincount(repeated_k, weights=q, minlength=10)
    global_mapping = np.full(10, q.mean(), dtype=np.float64)
    np.divide(global_total, global_count, out=global_mapping,
              where=global_count > 0)

    category_code, names = pd.factorize(category, sort=True)
    n_categories = len(names)
    joint = category_code[repeated] * 10 + repeated_k
    count = np.bincount(joint, minlength=n_categories * 10).astype(np.float64)
    total = np.bincount(joint, weights=q, minlength=n_categories * 10)
    prior = np.tile(global_mapping, n_categories)
    probability = (total + alpha * prior) / (count + alpha)

    rows = []
    for code, name in enumerate(names):
        offset = code * 10
        rows.append({
            "category": str(name),
            "observations": int(count[offset:offset + 10].sum()),
            "table": [{
                "k": value,
                "observations": int(count[offset + value]),
                "probability": float(probability[offset + value]),
            } for value in range(10)],
        })
    return rows, global_mapping


def atomic_json(data, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp-{os.getpid()}"
    try:
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def atomic_parquet(frame, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp-{os.getpid()}"
    try:
        frame.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", required=True,
                        help="полный parquet (id1,id2,target) с повторными метками")
    parser.add_argument(
        "--items", required=True,
        help="parquet-каталог: id,text либо id,category,name,attributes")
    parser.add_argument(
        "--allow-missing-items", action="store_true",
        help="исключить из оценки пары без текста; число обязательно пишется в JSON")
    parser.add_argument("--category-items", default="",
                        help="parquet id,category: включить category-калибровку")
    parser.add_argument("--category-alpha", type=float, default=100.0,
                        help="OOF-выбранная сила сглаживания к глобальной таблице")
    parser.add_argument("--apply", default="",
                        help="parquet, в котором заменить target по таблице")
    parser.add_argument("--out", required=True, help="JSON с таблицей калибровки")
    parser.add_argument("--out-pairs", default="",
                        help="новый parquet с откалиброванным target")
    args = parser.parse_args()
    if bool(args.apply) != bool(args.out_pairs):
        parser.error("--apply и --out-pairs задаются вместе")

    pairs = pd.read_parquet(args.pairs, columns=["id1", "id2", "target"])
    n_pairs_source = len(pairs)
    items = pd.read_parquet(args.items)
    if "id" not in items.columns:
        raise ValueError("в items отсутствует колонка id")
    text_code, n_unique_texts, identity = factorize_item_content(items)
    n_items = len(items)
    code_of = pd.Series(text_code.astype(np.int64), index=items.id.values)
    left = code_of.reindex(pairs.id1.values).to_numpy()
    right = code_of.reindex(pairs.id2.values).to_numpy()
    missing_mask = pd.isna(left) | pd.isna(right)
    missing_references = int(pd.isna(left).sum() + pd.isna(right).sum())
    if missing_mask.any():
        if not args.allow_missing_items:
            raise ValueError(
                f"в items отсутствуют {missing_references} ссылок из pairs")
        keep = ~missing_mask
        pairs = pairs.loc[keep].reset_index(drop=True)
        left, right = left[keep], right[keep]
        print(f"пропущено пар без полного текста: {(~keep).sum():,}; "
              f"отсутствующих ссылок: {missing_references:,}", flush=True)
    del items, text_code, code_of

    k = target_to_k(pairs.target.values)
    table = leave_one_out_table(left.astype(np.int64), right.astype(np.int64), k)
    category_rows = None
    category_of = None
    if args.category_items:
        category_items = pd.read_parquet(
            args.category_items, columns=["id", "category"])
        category_of = pd.Series(
            category_items.category.values, index=category_items.id.values)
        category = category_of.reindex(pairs.id1.values).to_numpy()
        if pd.isna(category).any():
            raise ValueError(
                f"нет категории для {int(pd.isna(category).sum())} calibration-пар")
        category_rows, category_global = category_leave_one_out_table(
            left.astype(np.int64), right.astype(np.int64), k, category,
            alpha=args.category_alpha)
        table_global = np.asarray([row["probability"] for row in table])
        if not np.allclose(category_global, table_global, atol=1e-12):
            raise RuntimeError("глобальная таблица расходится между оценщиками")
    payload = {
        "pairs": os.path.abspath(args.pairs),
        "items": os.path.abspath(args.items),
        "n_pairs_source": n_pairs_source,
        "n_pairs": len(pairs),
        "n_pairs_skipped_missing_items": n_pairs_source - len(pairs),
        "n_missing_item_references": missing_references,
        "n_items": n_items,
        "n_unique_texts": n_unique_texts,
        "item_identity": identity,
        "estimator": "mean per-observation leave-one-out P(other k=9)",
        "table": table,
    }
    if category_rows is not None:
        payload.update({
            "category_items": os.path.abspath(args.category_items),
            "category_alpha": args.category_alpha,
            "category_table": category_rows,
        })
    atomic_json(payload, args.out)

    print(f"pairs={len(pairs):,}; items={n_items:,}; "
          f"unique_texts={n_unique_texts:,}; identity={identity}")
    for row in table:
        probability = (f"{row['probability']:.6f}"
                       if row["probability"] is not None else "n/a")
        print(f"k={row['k']}: p9={probability}; "
              f"observations={row['observations']:,}; "
              f"other_labels={row['other_labels']:,}")
    print(f"calibration -> {args.out}")
    if category_rows is not None:
        print(f"category calibration: {len(category_rows)} categories; "
              f"alpha={args.category_alpha:g}")

    if args.apply:
        source = pd.read_parquet(args.apply)
        source_k = target_to_k(source.target.values)
        mapping = np.asarray([
            np.nan if row["probability"] is None else row["probability"]
            for row in table
        ], dtype=np.float32)
        if not np.isfinite(mapping[source_k]).all():
            absent = sorted(set(source_k[~np.isfinite(mapping[source_k])].tolist()))
            raise ValueError(f"нет калибровки для k={absent}")
        source = source.copy()
        target = mapping[source_k]
        if category_rows is not None:
            source_category = category_of.reindex(source.id1.values).to_numpy()
            if pd.isna(source_category).any():
                raise ValueError(
                    f"нет категории для {int(pd.isna(source_category).sum())} apply-пар")
            source_category = source_category.astype(str)
            for row in category_rows:
                selected = source_category == row["category"]
                category_mapping = np.asarray(
                    [cell["probability"] for cell in row["table"]],
                    dtype=np.float32)
                target[selected] = category_mapping[source_k[selected]]
        source["target"] = target
        atomic_parquet(source, args.out_pairs)
        print(f"calibrated pairs={len(source):,}; prevalence={source.target.mean():.6f} "
              f"-> {args.out_pairs}")


if __name__ == "__main__":
    main()

