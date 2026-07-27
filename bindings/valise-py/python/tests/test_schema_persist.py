"""Schema persistence parity (mirrors tests/db_schema_persist.rs).

Schemas are persisted in-file as hidden ~valise.schema doc frames that ride the
next commit. Reopen restores them: get/search just work with NO re-declare,
the deferred codec choice (family + operating point) survives reopen, and the
hidden doc store never leaks into collections() or stats().
"""

import gc

import numpy as np
import pytest

import valise

from conftest import DIM, hybrid_schema, make_hybrid, unit_vec


class _Box:
    """Holds the store so dropping the box releases the ONLY reference.

    `Store.open` consults an inode-keyed registry of live Database handles:
    if the first store is still referenced anywhere, "reopen" returns the
    same live in-memory handle and nothing is read back from disk. Tests
    must therefore never bind the store to a bare local they keep across
    the reopen — they hold a `_Box` and `_reopen(box, path)` empties it
    before opening.
    """

    def __init__(self, store):
        self.store = store

    def __getattr__(self, name):  # delegate Store methods
        return getattr(object.__getattribute__(self, "store"), name)


def _open_boxed(path):
    return _Box(valise.Store.create(path))


def _reopen(box, path):
    box.store = None
    gc.collect()
    return _Box(valise.Store.open(path))


def test_reopen_just_works_text_and_vector(tmp_path):
    path = str(tmp_path / "p.vls")
    s = _open_boxed(path)
    make_hybrid(s)
    with s.writer() as w:
        for i in range(10):
            w.put(
                "kb",
                f"d{i}",
                valise.Record().text("body", f"quantum doc {i}").vector(
                    "dense", unit_vec(100 + i)
                ),
            )
        w.commit()

    s2 = _reopen(s, path)
    # Reads, text search, and vector search all work without re-declaring.
    assert s2.get("kb", "d3").text == "quantum doc 3"
    t = s2.search("kb", valise.Search().text("body", "quantum").top_k(3))
    assert len(t) > 0
    v = s2.search("kb", valise.Search().vector("dense", unit_vec(105)).top_k(3))
    assert v.keys[0] == "d5"
    # The restored schema accepts an identical re-declare as a no-op.
    s2.collection("kb", hybrid_schema())


def test_reopen_restores_deferred_upq_codec(tmp_path):
    """A not-yet-calibrated UPQ choice survives reopen via the schema doc:
    the field calibrates with the declared family at the first vector commit
    AFTER reopening."""
    path = str(tmp_path / "upq.vls")
    s = _open_boxed(path)
    s.collection(
        "kb",
        valise.Schema()
        .text("body")
        .vector(
            "dense",
            valise.Vector(dim=DIM, codec=valise.Upq(), calibrate=valise.Auto(sample=8)),
        ),
    )
    # Persist the schema doc with a text-only commit; the codec stays deferred.
    with s.writer() as w:
        w.put("kb", "t0", valise.Record().text("body", "schema carrier"))
        w.commit()

    s2 = _reopen(s, path)
    # Still deferred after reopen: vector search names the auto space.
    with pytest.raises(valise.NotCalibratedError) as exc:
        s2.search("kb", valise.Search().vector("dense", unit_vec(0)).top_k(3))
    assert "~auto/kb/dense" in str(exc.value)

    # First vector commit after reopen calibrates the *declared* (UPQ) codec.
    with s2.writer() as w:
        for i in range(12):
            w.put(
                "kb",
                f"d{i}",
                valise.Record().text("body", "doc").vector("dense", unit_vec(300 + i)),
            )
        w.commit()
    res = s2.search("kb", valise.Search().vector("dense", unit_vec(307)).top_k(3))
    assert res.keys[0] == "d7"

    # And the calibrated state itself persists across another reopen.
    s3 = _reopen(s2, path)
    res = s3.search("kb", valise.Search().vector("dense", unit_vec(307)).top_k(3))
    assert res.keys[0] == "d7"


def test_declare_without_commit_persists_nothing(tmp_path):
    """Schema durability is stage-only: a declare with no commit leaves
    nothing in the file (reopen comes back empty)."""
    path = str(tmp_path / "stage.vls")
    s = _open_boxed(path)
    s.collection("kb", valise.Schema().text("body"))
    s2 = _reopen(s, path)
    assert s2.collections() == []


def test_hidden_schema_store_excluded_from_listing_and_stats(tmp_path):
    path = str(tmp_path / "hidden.vls")
    s = _open_boxed(path)
    make_hybrid(s)
    with s.writer() as w:
        w.put("kb", "a", valise.Record().text("body", "x").vector("dense", unit_vec(1)))
        w.put("kb", "b", valise.Record().text("body", "y").vector("dense", unit_vec(2)))
        w.commit()
    assert [c.name for c in s.collections()] == ["kb"]
    # stats() counts user data only: 2 user frames, not 2 + the schema doc.
    st = s.stats()
    assert st.frames_total == 2
    assert st.frames_active == 2

    s2 = _reopen(s, path)
    assert [c.name for c in s2.collections()] == ["kb"]


def test_spaces_lists_auto_spaces_flagged(store):
    make_hybrid(store)
    emb = store.define_space("emb", valise.Vector(dim=DIM))
    assert emb.is_vector
    infos = {i.name: i for i in store.spaces()}
    # One auto space per inline field, named ~auto/{collection}/{field}.
    assert infos["~auto/kb/body"].auto and infos["~auto/kb/body"].is_text
    assert infos["~auto/kb/dense"].auto and infos["~auto/kb/dense"].is_vector
    assert infos["~auto/kb/dense"].dim == DIM
    assert not infos["emb"].auto
    # Lookup by name returns a bindable handle.
    assert store.space("emb").name == "emb"
    assert store.space("missing") is None


def test_divergent_redeclare_after_reopen(tmp_path):
    path = str(tmp_path / "div.vls")
    s = _open_boxed(path)
    make_hybrid(s)
    with s.writer() as w:
        w.put("kb", "a", valise.Record().text("body", "x").vector("dense", unit_vec(1)))
        w.commit()
    s2 = _reopen(s, path)
    with pytest.raises(valise.SchemaMismatchError):
        s2.collection(
            "kb",
            valise.Schema().text("body", valise.Text(lang=valise.Lang.RAW)).vector(
                "dense", valise.Vector(dim=DIM, calibrate=valise.Auto(sample=8))
            ),
        )


def test_compact_preserves_schema_and_search(tmp_path):
    """Compaction streams schema docs through; text + vector search keep
    working on the compacted file and after another reopen."""
    path = str(tmp_path / "comp.vls")
    s = _open_boxed(path)
    make_hybrid(s)
    with s.writer() as w:
        for i in range(10):
            w.put(
                "kb",
                f"d{i}",
                valise.Record().text("body", "quantum doc").vector(
                    "dense", unit_vec(400 + i)
                ),
            )
        w.commit()
    with s.writer() as w:
        assert w.delete("kb", "d0") is True
        w.commit()
    report = s.compact()
    assert report.compacted
    assert len(s.search("kb", valise.Search().text("body", "quantum").top_k(5))) > 0
    assert s.search(
        "kb", valise.Search().vector("dense", unit_vec(404)).top_k(3)
    ).keys[0] == "d4"

    s2 = _reopen(s, path)
    assert s2.get("kb", "d4").text == "quantum doc"
    assert np.all(np.isfinite(s2.search(
        "kb", valise.Search().vector("dense", unit_vec(404)).top_k(3)
    ).scores))
