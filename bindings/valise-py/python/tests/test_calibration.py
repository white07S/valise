# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Calibration via commit (Auto) and at declaration (Now).

Before any commit with staged vectors, a vector search raises
NotCalibratedError naming the field's auto space. After committing a batch,
the same search succeeds. Now(sample) calibrates eagerly, so a search works
(returning nothing) before any commit.
"""

import numpy as np
import pytest

import valise

from conftest import DIM, make_hybrid, matrix, unit_vec
from valise import Rerank


def test_deferred_calibration_on_commit(store):
    make_hybrid(store, deferred_sample=8)
    q = unit_vec(0)

    # Uncalibrated: a vector search must raise NotCalibratedError and name
    # the auto space of the field.
    with pytest.raises(valise.NotCalibratedError) as exc:
        store.search("kb", valise.Search().vector("dense", q, Rerank.FAST).top_k(3))
    assert "not calibrated" in str(exc.value)
    assert "~auto/kb/dense" in str(exc.value)

    # Ingest enough vectors to calibrate, then commit.
    with store.writer() as w:
        for i in range(12):
            w.put(
                "kb",
                f"d{i}",
                valise.Record().text("body", "doc").vector("dense", unit_vec(100 + i)),
            )
        w.commit()

    # Now the same search succeeds and returns hits.
    res = store.search(
        "kb", valise.Search().vector("dense", unit_vec(105), Rerank.ACCURATE).top_k(3)
    )
    keys, scores = res.keys, res.scores
    assert len(keys) > 0
    assert scores.dtype == np.float32
    assert len(keys) == len(scores)


def test_eager_calibration_with_now(store):
    """Now(sample) registers the codec at declaration: no NotCalibratedError
    even before the first commit (the search just finds nothing)."""
    store.collection(
        "kb",
        valise.Schema()
        .text("body")
        .vector("dense", valise.Vector(dim=DIM, calibrate=valise.Now(matrix(32, seed=3)))),
    )
    res = store.search("kb", valise.Search().vector("dense", unit_vec(1)).top_k(3))
    assert len(res) == 0

    with store.writer() as w:
        for i in range(4):
            w.put(
                "kb",
                f"d{i}",
                valise.Record().text("body", "doc").vector("dense", unit_vec(200 + i)),
            )
        w.commit()
    res = store.search("kb", valise.Search().vector("dense", unit_vec(202)).top_k(3))
    assert res.keys[0] == "d2"
