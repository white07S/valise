# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Per-field codec choice reaches the engine.

There is no codec-family introspection on the public surface (by design), so
these tests assert the behavioral contract: a custom Qam operating point, the
UPQ family, and explicit Upq cells/design all declare cleanly, calibrate at
the first vector commit, and answer vector searches with the expected
neighbor — and the choice keeps working after reopen (persisted schema doc).
"""

import gc

import numpy as np
import pytest

import valise

from conftest import DIM, make_hybrid, unit_vec


def _ingest(store, n=16, seed0=500):
    with store.writer() as w:
        for i in range(n):
            w.put(
                "kb",
                f"d{i}",
                valise.Record().text("body", "doc").vector("dense", unit_vec(seed0 + i)),
            )
        w.commit()


def _self_match(store, seed0=500, probe=7):
    res = store.search(
        "kb", valise.Search().vector("dense", unit_vec(seed0 + probe)).top_k(3)
    )
    assert res.keys[0] == f"d{probe}"
    assert np.all(np.isfinite(res.scores))


@pytest.mark.parametrize(
    "codec",
    [
        None,  # Rust default: Qam() == QAM(5, 6)
        valise.Qam(),  # explicit default
        valise.Qam(amp_bits=6, phase_bits=8),  # custom operating point
        valise.Upq(),  # family switch, default 2048 cells
        valise.Upq(cells=1024),  # explicit cells (power of two)
        valise.Upq(cells=1024, design=valise.Design.RAYLEIGH),  # explicit design
    ],
    ids=["default", "qam", "qam-6-8", "upq", "upq-1024", "upq-rayleigh"],
)
def test_codec_choice_calibrates_and_searches(store, codec):
    make_hybrid(store, codec=codec)
    _ingest(store)
    _self_match(store)


def test_qam_bits_survive_reopen(tmp_path):
    path = str(tmp_path / "qam.vls")
    s = valise.Store.create(path)
    make_hybrid(s, codec=valise.Qam(amp_bits=6, phase_bits=8))
    _ingest(s)
    del s
    gc.collect()
    s2 = valise.Store.open(path)
    _self_match(s2)


def test_upq_cells_survive_reopen(tmp_path):
    path = str(tmp_path / "upq.vls")
    s = valise.Store.create(path)
    make_hybrid(s, codec=valise.Upq(cells=1024))
    _ingest(s)
    del s
    gc.collect()
    s2 = valise.Store.open(path)
    _self_match(s2)


def test_shared_space_carries_codec(tmp_path):
    """define_space owns codec+calibrate; bindings just reference it."""
    s = valise.Store.create(str(tmp_path / "shared.vls"))
    emb = s.define_space(
        "emb", valise.Vector(dim=DIM, codec=valise.Upq(), calibrate=valise.Auto(sample=8))
    )
    s.collection(
        "kb", valise.Schema().text("body").vector("dense", valise.Vector(space=emb))
    )
    _ingest(s)
    _self_match(s)
