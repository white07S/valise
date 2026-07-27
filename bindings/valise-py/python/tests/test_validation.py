"""Validation / error-surface parity. Engine + binding-local rejections map
to valise.ValidationError, EXCEPT float64-at-the-numpy-boundary which rust-numpy
raises as a Python TypeError before the binding body runs — those accept
(TypeError, valise.ValidationError). New in the simple-API surface:
'~'-reserved names, SchemaMismatchError on divergent redeclare, and
NotCalibratedError naming the auto space.
"""

import numpy as np
import pytest

import valise

from conftest import DIM, make_hybrid, unit_vec
from valise import Bm25, Metric


def test_dim_mismatch(store):
    make_hybrid(store)
    bad = unit_vec(1, dim=DIM + 1)  # record vector dim != space dim
    with store.writer() as w:
        with pytest.raises(valise.ValidationError) as exc:
            w.put("kb", "d", valise.Record().vector("dense", bad))
    assert "dim" in str(exc.value)


def test_float64_rejected_record(store):
    make_hybrid(store)
    bad = np.ones(DIM, dtype=np.float64)
    with store.writer() as w:
        with pytest.raises((TypeError, valise.ValidationError)):
            w.put("kb", "d", valise.Record().vector("dense", bad))


def test_float64_rejected_put_many(store):
    make_hybrid(store)
    bad = np.ones((3, DIM), dtype=np.float64)
    with store.writer() as w:
        with pytest.raises((TypeError, valise.ValidationError)):
            w.put_many("kb", ["a", "b", "c"], bad)


def test_noncontiguous_rejected_put_many(store):
    make_hybrid(store)
    noncontig = np.random.rand(3, 128).astype(np.float32)[:, ::2]
    assert not noncontig.flags["C_CONTIGUOUS"]
    with store.writer() as w:
        with pytest.raises(valise.ValidationError):
            w.put_many("kb", ["x", "y", "z"], noncontig)


def test_bad_dim_rejected(store):
    # The engine raises Error::Unsupported for a non-conforming dim, which the
    # binding classifies as UnsupportedError (a subclass of ValiseError). Inline
    # specs surface it at declaration (when the auto space is defined).
    with pytest.raises(valise.UnsupportedError) as exc:
        store.collection(
            "bad", valise.Schema().vector("dense", valise.Vector(dim=100))
        )  # not a multiple of 64
    assert "multiple of 64" in str(exc.value)


def test_bool_key_rejected(store):
    make_hybrid(store)
    with store.writer() as w:
        with pytest.raises(valise.ValidationError) as exc:
            w.put("kb", True, valise.Record().text("body", "x"))
        assert "bool" in str(exc.value)
    with pytest.raises(valise.ValidationError):
        store.get("kb", True)


def test_redefine_divergent_space(store):
    store.define_space("v", valise.Vector(dim=DIM, metric=Metric.COSINE))
    # Idempotent re-definition with the same params is allowed.
    store.define_space("v", valise.Vector(dim=DIM, metric=Metric.COSINE))
    # Divergent metric must be rejected.
    with pytest.raises(valise.ValidationError) as exc:
        store.define_space("v", valise.Vector(dim=DIM, metric=Metric.DOT))
    msg = str(exc.value)
    assert "metric" in msg or "different" in msg


def test_reserved_tilde_names_rejected(store):
    """User collection/space names starting with '~' are reserved."""
    with pytest.raises(valise.ValidationError) as exc:
        store.collection("~mine", valise.Schema().text("body"))
    assert "reserved" in str(exc.value)
    with pytest.raises(valise.ValidationError):
        store.define_space("~auto/sneaky", valise.Text())


def test_reserved_schema_collection_hidden(store):
    """The internal ~valise.schema doc store never shows up in collections()."""
    make_hybrid(store)
    with store.writer() as w:
        w.put("kb", "d", valise.Record().text("body", "x").vector("dense", unit_vec(1)))
        w.commit()
    names = [c.name for c in store.collections()]
    assert names == ["kb"]


def test_divergent_redeclare_raises_schema_mismatch(store):
    make_hybrid(store)
    # Identical re-declare is a no-op.
    make_hybrid(store)
    # Divergent (text analyzer change) raises SchemaMismatchError naming the
    # collection and the differing field.
    divergent = (
        valise.Schema()
        .text("body", valise.Text(lang=valise.Lang.RAW))
        .vector("dense", valise.Vector(dim=DIM, calibrate=valise.Auto(sample=8)))
    )
    with pytest.raises(valise.SchemaMismatchError) as exc:
        store.collection("kb", divergent)
    assert "kb" in str(exc.value)


def test_vector_search_before_calibration_names_auto_space(store):
    make_hybrid(store)
    with pytest.raises(valise.NotCalibratedError) as exc:
        store.search("kb", valise.Search().vector("dense", unit_vec(0)).top_k(3))
    msg = str(exc.value)
    assert "not calibrated" in msg
    assert "~auto/kb/dense" in msg


def test_shared_space_binding_rejects_codec(store):
    """Codec/calibrate on a shared-space binding is a declaration-time error."""
    emb = store.define_space("emb", valise.Vector(dim=DIM))
    with pytest.raises(valise.ValidationError):
        store.collection(
            "kb2",
            valise.Schema().vector("dense", valise.Vector(space=emb, codec=valise.Upq())),
        )


def test_search_unknown_field(store):
    make_hybrid(store)
    with pytest.raises(valise.ValidationError) as exc:
        store.search("kb", valise.Search().text("nope", "q", Bm25()).top_k(1))
    assert "unknown field" in str(exc.value)


def test_search_text_on_vector_field(store):
    make_hybrid(store)
    with pytest.raises(valise.ValidationError) as exc:
        store.search("kb", valise.Search().text("dense", "q", Bm25()).top_k(1))
    assert "vector field" in str(exc.value)


def test_search_no_channels(store):
    make_hybrid(store)
    with pytest.raises(valise.ValidationError) as exc:
        store.search("kb", valise.Search())
    assert "no channels" in str(exc.value)


def test_strict_enum_metric_rejects_raw_string():
    """Vector takes a Metric enum, not a raw string."""
    with pytest.raises(TypeError, match="metric must be a Metric enum"):
        valise.Vector(dim=DIM, metric="cosine")


def test_vector_requires_exactly_one_of_dim_space(store):
    with pytest.raises(ValueError, match="exactly one"):
        valise.Vector()
    emb = store.define_space("emb", valise.Vector(dim=DIM))
    with pytest.raises(ValueError, match="exactly one"):
        valise.Vector(dim=DIM, space=emb)


def test_partial_qam_bits_rejected(store):
    """Qam mirrors Codec::qam_bits: both bits or neither."""
    with pytest.raises(valise.ValidationError, match="both"):
        store.collection(
            "q",
            valise.Schema().vector("dense", valise.Vector(dim=DIM, codec=valise.Qam(amp_bits=6))),
        )


def test_upq_design_requires_cells(store):
    """Upq mirrors Codec::upq_with: a design needs an explicit cells budget."""
    with pytest.raises(valise.ValidationError, match="cells"):
        store.collection(
            "u",
            valise.Schema().vector(
                "dense", valise.Vector(dim=DIM, codec=valise.Upq(design=valise.Design.RAYLEIGH))
            ),
        )


def test_strict_scorer_rejects_raw_string(store):
    """Search.text takes a typed TextScorer, not a raw string."""
    make_hybrid(store)
    with pytest.raises(TypeError):
        valise.Search().text("body", "q", "bm25")


def test_strict_rerank_rejects_raw_string(store):
    """Search.vector takes a Rerank enum, not a raw string."""
    make_hybrid(store)
    with pytest.raises(TypeError):
        valise.Search().vector("dense", unit_vec(1), "fast")
