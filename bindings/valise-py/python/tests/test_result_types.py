# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Typed result-object ergonomics: Stored equality (with vectors), SearchResult
slicing, repr round-trips, and put_many caller-side length/rank guards.

These cover the P3 DX polish:
- Stored.__eq__ must NOT raise on records carrying ndarrays.
- SearchResult slicing returns a (chainable) SearchResult, not list[Hit].
- reprs of the typed results are non-crashing and readable.
- put_many surfaces shape/length mismatches as plain ValueError.
"""

import numpy as np
import pytest

import valise
from valise.types import Hit, SearchResult, Stats, Stored

from conftest import DIM, make_hybrid, matrix, unit_vec


# ---- Stored equality / hashing (the latent ndarray bug) --------------------


def _stored(key="a", text="t", vectors=None):
    return Stored(
        key=key,
        collection="kb",
        created_at=0,
        text=text,
        vectors=vectors if vectors is not None else [],
    )


def test_stored_eq_with_identical_vectors_does_not_raise():
    v = np.ones(4, dtype=np.float32)
    a = _stored(vectors=[("dense", v.copy())])
    b = _stored(vectors=[("dense", v.copy())])
    # Must be True, and must NOT raise ValueError (the array-truthiness bug).
    assert a == b


def test_stored_eq_differing_vectors_is_false():
    a = _stored(vectors=[("dense", np.ones(4, dtype=np.float32))])
    b = _stored(vectors=[("dense", np.zeros(4, dtype=np.float32))])
    assert a != b


def test_stored_eq_differing_vector_shape_is_false():
    a = _stored(vectors=[("dense", np.ones(4, dtype=np.float32))])
    b = _stored(vectors=[("dense", np.ones(8, dtype=np.float32))])
    assert a != b


def test_stored_eq_differing_vector_name_is_false():
    v = np.ones(4, dtype=np.float32)
    a = _stored(vectors=[("dense", v.copy())])
    b = _stored(vectors=[("other", v.copy())])
    assert a != b


def test_stored_eq_differing_text_is_false():
    a = _stored(text="one")
    b = _stored(text="two")
    assert a != b


def test_stored_eq_empty_vectors_still_works():
    assert _stored(vectors=[]) == _stored(vectors=[])


def test_stored_eq_wrong_type_returns_not_implemented():
    # __eq__ returns NotImplemented for non-Stored, so == falls back to identity.
    assert (_stored() == 42) is False


def test_stored_is_unhashable():
    with pytest.raises(TypeError):
        hash(_stored(vectors=[("dense", np.ones(4, dtype=np.float32))]))
    with pytest.raises(TypeError):
        {_stored()}  # noqa: B018


def test_stored_roundtrip_eq_through_engine(store):
    make_hybrid(store)
    with store.writer() as w:
        w.put(
            "kb",
            "d",
            valise.Record().text("body", "quantum").vector("dense", unit_vec(1)),
        )
        w.commit()
    a = store.get("kb", "d")
    b = store.get("kb", "d")
    assert a is not None and b is not None
    assert len(a.vectors) == 1
    # Two reads of the same record (each carrying a real ndarray) compare equal
    # without raising the array-truthiness ValueError.
    assert a == b


# ---- SearchResult slicing / repr -------------------------------------------


def _search_result():
    keys = ["a", "b", "c", "d", "e"]
    scores = np.array([5.0, 4.0, 3.0, 2.0, 1.0], dtype=np.float32)
    return SearchResult(keys, scores, "kb")


def test_searchresult_slice_returns_searchresult():
    r = _search_result()
    sl = r[1:3]
    assert isinstance(sl, SearchResult)
    assert sl.keys == ["b", "c"]
    assert sl.collection == "kb"


def test_searchresult_slice_scores_are_float32_ndarray():
    r = _search_result()
    sl = r[:2]
    assert isinstance(sl.scores, np.ndarray)
    assert sl.scores.dtype == np.float32
    assert list(sl.scores) == [5.0, 4.0]


def test_searchresult_slice_is_iterable_into_hits():
    r = _search_result()
    hits = list(r[:2])
    assert all(isinstance(h, Hit) for h in hits)
    assert [h.key for h in hits] == ["a", "b"]


def test_searchresult_int_index_returns_hit():
    r = _search_result()
    h = r[0]
    assert isinstance(h, Hit)
    assert h.key == "a"
    assert h.score == 5.0
    assert h.collection == "kb"


def test_searchresult_negative_index():
    r = _search_result()
    assert r[-1].key == "e"


def test_searchresult_out_of_range_raises():
    r = _search_result()
    with pytest.raises(IndexError):
        _ = r[99]


def test_searchresult_repr_previews_hits():
    r = _search_result()
    s = repr(r)
    assert "SearchResult" in s
    assert "collection='kb'" in s
    assert "n=5" in s
    assert "..." in s  # more than 3 hits -> truncated preview
    assert "'a'" in s


def test_searchresult_repr_small_no_ellipsis():
    keys = ["a", "b"]
    scores = np.array([2.0, 1.0], dtype=np.float32)
    s = repr(SearchResult(keys, scores, "kb"))
    assert "..." not in s
    assert "n=2" in s


# ---- repr round-trips for the scalar dataclasses ---------------------------


def test_hit_repr_contains_fields():
    s = repr(Hit("a", 1.5, "kb"))
    assert "Hit" in s
    assert "key='a'" in s
    assert "collection='kb'" in s


def test_stats_repr_contains_fields():
    st = Stats(
        frames_total=10,
        frames_active=8,
        vectors_total=5,
        vectors_active=4,
        file_bytes=1024,
        tombstone_ratio=0.2,
    )
    s = repr(st)
    assert "Stats" in s
    assert "frames_total=10" in s
    # scalar dataclass eq stays correct
    assert st == Stats(10, 8, 5, 4, 1024, 0.2)


def test_stored_repr_with_vectors_does_not_crash():
    s = repr(_stored(vectors=[("dense", np.ones(4, dtype=np.float32))]))
    assert "Stored" in s
    assert "key='a'" in s


# ---- put_many facade-side length / rank guards -----------------------------


def test_put_many_non_2d_raises_value_error(store):
    make_hybrid(store)
    with store.writer() as w:
        with pytest.raises(ValueError, match="2-D"):
            w.put_many("kb", ["a"], np.ones(DIM, dtype=np.float32))


def test_put_many_row_count_mismatch_raises_value_error(store):
    make_hybrid(store)
    with store.writer() as w:
        with pytest.raises(ValueError, match="len\\(keys\\)="):
            w.put_many("kb", ["a", "b"], matrix(3))


def test_put_many_texts_length_mismatch_raises_value_error(store):
    make_hybrid(store)
    with store.writer() as w:
        with pytest.raises(ValueError, match="len\\(texts\\)="):
            w.put_many("kb", ["a", "b"], matrix(2), texts=["only-one"])


def test_put_many_valid_lengths_pass(store):
    make_hybrid(store)
    with store.writer() as w:
        w.put_many("kb", ["a", "b"], matrix(2))
        w.commit()
    assert store.get("kb", "a") is not None
    assert store.get("kb", "b") is not None
