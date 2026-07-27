# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Search parity — text, vector-nearest, hybrid RRF (summing channels),
weighted fusion, recency range/halflife, collection isolation, get_many.
Mirrors tests/db_search.rs with top-1 / count assertions only.
"""

import numpy as np

import valise

from conftest import DIM, make_hybrid, matrix, unit_vec
from valise import Bm25, HalfLife, Range, Rerank, Rrf, Weighted


def test_text_search_ranks_match_first(store):
    make_hybrid(store)
    docs = {
        "q": "quantum entanglement physics",
        "b": "banana bread recipe",
        "c": "the weather today is sunny",
    }
    with store.writer() as w:
        for k, text in docs.items():
            w.put("kb", k, valise.Record().text("body", text).vector("dense", unit_vec(ord(k))))
        w.commit()
    res = store.search("kb", valise.Search().text("body", "quantum", Bm25()).top_k(3))
    assert res.keys[0] == "q"


def test_vector_nearest_first(store):
    make_hybrid(store)
    with store.writer() as w:
        for i in range(20):
            w.put(
                "kb",
                i,  # int keys
                valise.Record().text("body", "doc").vector("dense", unit_vec(500 + i)),
            )
        w.commit()
    q = unit_vec(507)
    res = store.search(
        "kb", valise.Search().vector("dense", q, Rerank.ACCURATE).top_k(3)
    )
    assert res.keys[0] == 7


def test_hybrid_rrf_sums_channels(store):
    make_hybrid(store)
    q = unit_vec(900)
    x_vec = np.ascontiguousarray(q.copy())
    x_vec[0] += 0.05  # X: near the query but not identical (vector rank ~1)
    y_vec = np.ascontiguousarray(q.copy())  # Y: identical (vector rank 0)

    with store.writer() as w:
        # X is in BOTH channels (text term + near vector).
        w.put("kb", "X", valise.Record().text("body", "quantum").vector("dense", x_vec))
        # Y is the nearest vector but lacks the text term.
        w.put("kb", "Y", valise.Record().text("body", "banana").vector("dense", y_vec))
        for i in range(20):
            w.put(
                "kb",
                f"f{i}",
                valise.Record().text("body", "filler").vector("dense", unit_vec(i)),
            )
        w.commit()

    res = store.search(
        "kb",
        valise.Search()
        .text("body", "quantum", Bm25())
        .vector("dense", q, Rerank.ACCURATE)
        .fuse(Rrf(60))
        .top_k(5),
    )
    # Summing RRF ranks the both-channel hit first; averaging would pick Y.
    assert res.keys[0] == "X"


def test_weighted_fusion_runs(store):
    make_hybrid(store)
    q = unit_vec(33)
    with store.writer() as w:
        w.put(
            "kb",
            "hit",
            valise.Record().text("body", "relevant quantum doc").vector("dense", q),
        )
        for i in range(15):
            w.put(
                "kb",
                f"f{i}",
                valise.Record().text("body", "other").vector("dense", unit_vec(i)),
            )
        w.commit()
    res = store.search(
        "kb",
        valise.Search()
        .text("body", "quantum", Bm25())
        .vector("dense", q, Rerank.ACCURATE)
        .fuse(Weighted(0.5, 0.5))
        .top_k(5),
    )
    assert res.keys[0] == "hit"


def test_recency_range_hard_filters(store):
    make_hybrid(store)
    with store.writer() as w:
        w.put("kb", "old", valise.Record().text("body", "doc").vector("dense", unit_vec(1)).at(100))
        w.put("kb", "mid", valise.Record().text("body", "doc").vector("dense", unit_vec(2)).at(200))
        w.put("kb", "new", valise.Record().text("body", "doc").vector("dense", unit_vec(3)).at(300))
        w.commit()
    res = store.search(
        "kb",
        valise.Search().text("body", "doc", Bm25()).recency(Range(150, 250)).top_k(10),
    )
    assert len(res) == 1
    assert res.keys[0] == "mid"


def test_recency_halflife_favors_newer(store):
    make_hybrid(store)
    new_ts = 100 * 86400
    with store.writer() as w:
        w.put("kb", "old", valise.Record().text("body", "doc").vector("dense", unit_vec(1)).at(0))
        w.put("kb", "new", valise.Record().text("body", "doc").vector("dense", unit_vec(2)).at(new_ts))
        w.commit()
    res = store.search(
        "kb",
        valise.Search()
        .text("body", "doc", Bm25())
        .recency(HalfLife(30))
        .now(new_ts)
        .top_k(10),
    )
    assert res.keys[0] == "new"


def test_collection_isolation(tmp_path):
    s = valise.Store.create(str(tmp_path / "iso.vls"))
    en = s.define_space("en", valise.Text())  # shared text space
    s.collection("a", valise.Schema().text("body", valise.Text(space=en)))
    s.collection("b", valise.Schema().text("body", valise.Text(space=en)))
    with s.writer() as w:
        w.put("a", "a1", valise.Record().text("body", "shared quantum text"))
        w.put("b", "b1", valise.Record().text("body", "shared quantum text"))
        w.commit()
    res = s.search("a", valise.Search().text("body", "quantum", Bm25()).top_k(10))
    assert len(res) == 1
    assert res.keys[0] == "a1"


def test_get_many_batches(store):
    make_hybrid(store)
    with store.writer() as w:
        w.put("kb", "a", valise.Record().text("body", "alpha").vector("dense", unit_vec(1)))
        w.put("kb", "b", valise.Record().text("body", "beta").vector("dense", unit_vec(2)))
        w.commit()
    r = store.reader()
    got = r.get_many("kb", ["a", "missing", "b"])
    assert len(got) == 3
    assert got[1] is None
    assert got[0].text == "alpha"
    assert got[2].text == "beta"
