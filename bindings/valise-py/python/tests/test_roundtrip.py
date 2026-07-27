# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Round-trip parity: put/get, update-by-key, delete, put_auto, multi-vector,
persistence across reopen (schema restored from file — NO re-declare).
Mirrors tests/db_layer.rs.

QAM is lossy, so vector VALUES are never asserted equal to the input — only
field name, dimension, and finiteness.
"""

import gc

import numpy as np

import valise

from conftest import DIM, make_hybrid, unit_vec


def test_put_get_text_only(tmp_path):
    s = valise.Store.create(str(tmp_path / "t.vls"))
    s.collection("notes", valise.Schema().text("body"))
    with s.writer() as w:
        w.put("notes", "d", valise.Record().text("body", "hello world"))
        w.commit()
    got = s.get("notes", "d")
    assert got is not None
    assert got.text == "hello world"
    assert got.vectors == []


def test_put_get_vector(store):
    make_hybrid(store)
    with store.writer() as w:
        w.put(
            "kb",
            "d",
            valise.Record().text("body", "quantum").vector("dense", unit_vec(1)),
        )
        w.commit()  # calibrates the deferred space
    got = store.get("kb", "d")
    assert got is not None
    assert len(got.vectors) == 1
    name, vec = got.vectors[0]
    assert name == "dense"
    assert len(vec) == DIM
    assert np.all(np.isfinite(vec))  # lossy codec: only finiteness, not values


def test_update_by_key_remaps(store):
    make_hybrid(store)
    with store.writer() as w:
        w.put(
            "kb",
            "d",
            valise.Record().text("body", "first").vector("dense", unit_vec(1)),
        )
        w.commit()
    # Update by the same key with a text-only record: latest text wins and the
    # vector is dropped because the field was omitted on the update.
    with store.writer() as w:
        w.put("kb", "d", valise.Record().text("body", "second"))
        w.commit()
    got = store.get("kb", "d")
    assert got is not None
    assert got.text == "second"
    assert got.vectors == []


def test_delete(store):
    make_hybrid(store)
    with store.writer() as w:
        w.put("kb", "d", valise.Record().text("body", "x").vector("dense", unit_vec(2)))
        w.commit()
    with store.writer() as w:
        assert w.delete("kb", "d") is True
        assert w.delete("kb", "missing") is False
        w.commit()
    assert store.get("kb", "d") is None


def test_put_auto_generates_key(store):
    make_hybrid(store)
    with store.writer() as w:
        key = w.put_auto(
            "kb", valise.Record().text("body", "auto").vector("dense", unit_vec(3))
        )
        w.commit()
    assert key is not None
    got = store.get("kb", key)
    assert got is not None
    assert got.text == "auto"


def test_multi_vector_fields(tmp_path):
    s = valise.Store.create(str(tmp_path / "t.vls"))
    s.collection(
        "media",
        valise.Schema()
        .vector("text", valise.Vector(dim=DIM, calibrate=valise.Auto(sample=8)))
        .vector("image", valise.Vector(dim=DIM, calibrate=valise.Auto(sample=8))),
    )
    with s.writer() as w:
        w.put(
            "media",
            "m",
            valise.Record().vector("text", unit_vec(10)).vector("image", unit_vec(20)),
        )
        w.commit()
    got = s.get("media", "m")
    assert got is not None
    names = {name for name, _ in got.vectors}
    assert names == {"text", "image"}


def test_int_dim_shorthand(tmp_path):
    """Schema().vector(name, 64) is shorthand for Vector(dim=64)."""
    s = valise.Store.create(str(tmp_path / "t.vls"))
    s.collection("v", valise.Schema().vector("dense", DIM))
    with s.writer() as w:
        w.put("v", "d", valise.Record().vector("dense", unit_vec(5)))
        w.commit()
    assert s.get("v", "d") is not None


def test_persists_across_reopen_no_redeclare(tmp_path):
    path = str(tmp_path / "reopen.vls")
    s = valise.Store.create(path)
    make_hybrid(s)
    with s.writer() as w:
        w.put(
            "kb",
            "survivor",
            valise.Record().text("body", "durable").vector("dense", unit_vec(7)),
        )
        w.commit()
    # Drop the first store handle, then reopen the same file.
    del s
    gc.collect()

    # Schemas are persisted in-file: reopen just works (no re-declaration).
    s2 = valise.Store.open(path)
    got = s2.get("kb", "survivor")
    assert got is not None
    assert got.text == "durable"
