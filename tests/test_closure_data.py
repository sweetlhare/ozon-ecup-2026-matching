import numpy as np
import pandas as pd

from data_pipeline.build_closure_data import (
    closure_candidates,
    pair_records,
    select_pool,
    stable_hash,
)


def test_pair_records_are_unordered_and_collision_free():
    got = pair_records([9, 1], [2, 7])
    assert got.tolist() == [(2, 9), (1, 7)]


def test_closure_candidates_complete_unobserved_triangle_edge():
    matches = pd.DataFrame({
        "id1": [10, 11], "id2": [11, 12], "target": [1.0, 1.0],
    })
    categories = pd.Series({10: "A", 11: "A", 12: "A"})
    got = closure_candidates(matches, categories)
    assert got[["id1", "id2"]].values.tolist() == [[10, 12]]


def test_select_pool_uses_stable_hash_not_input_order():
    base = pd.DataFrame({
        "id1": [1, 2, 3], "id2": [11, 12, 13],
        "category": ["Автотовары"] * 3, "source": ["x"] * 3,
    })
    empty = base.iloc[:0]
    # Fill every other category so the production selector can enforce its grid.
    from data_pipeline.build_closure_data import SAFE_CATEGORIES
    positives, negatives = [], []
    for offset, category in enumerate(SAFE_CATEGORIES):
        frame = base.copy()
        frame.id1 += offset * 100
        frame.id2 += offset * 100
        frame.category = category
        positives.append(frame)
        negatives.append(frame.assign(id1=frame.id1 + 50, id2=frame.id2 + 50))
    a = select_pool(pd.concat(positives), pd.concat(negatives), 2)
    b = select_pool(pd.concat(positives).sample(frac=1, random_state=3),
                    pd.concat(negatives).sample(frac=1, random_state=4), 2)
    pd.testing.assert_frame_equal(a, b)
    assert np.array_equal(stable_hash([1], [11]), stable_hash([11], [1]))
