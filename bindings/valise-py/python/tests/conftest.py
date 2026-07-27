# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Shared fixtures and helpers for the valise Python parity suite.

ONE source of truth for vector generation and the standard hybrid collection.
All tests import from here; do NOT redefine these helpers per file.

Notes pinned to the binding contract:
- DIM must be a positive multiple of 64 (the engine rejects otherwise). 64 is
  the smallest legal value and is what every test uses.
- Vectors are ALWAYS np.float32 + np.ascontiguousarray. rust-numpy's
  PyReadonlyArray<f32> rejects float64 at the boundary, and non-contiguous
  arrays are rejected by as_slice(); tests rely on float32 + contiguous.
- Auto (deferred) calibration fits the codec at the first commit containing
  vectors for the field; until then a vector search raises
  NotCalibratedError naming the auto space (~auto/{collection}/{field}).
- Schemas are persisted in-file: reopen needs NO re-declaration.
"""

import numpy as np
import pytest

import valise

DIM = 64  # MUST be a multiple of 64.


def unit_vec(seed: int, dim: int = DIM) -> np.ndarray:
    """Deterministic, normalized, C-contiguous float32 vector."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    v /= max(float(np.linalg.norm(v)), 1e-6)
    return np.ascontiguousarray(v, dtype=np.float32)


def matrix(n: int, dim: int = DIM, seed: int = 0) -> np.ndarray:
    """Deterministic [n, dim] block of normalized, C-contiguous float32 rows."""
    rng = np.random.default_rng(seed)
    m = rng.standard_normal((n, dim)).astype(np.float32)
    m /= np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-6)
    return np.ascontiguousarray(m, dtype=np.float32)


@pytest.fixture
def store(tmp_path):
    """A fresh store per test (isolated file under pytest's tmp_path)."""
    return valise.Store.create(str(tmp_path / "t.vls"))


def hybrid_schema(*, deferred_sample=8, dim=DIM, codec=None) -> "valise.Schema":
    """The standard text+vector schema: body (English BM25) + dense (cosine)."""
    return (
        valise.Schema()
        .text("body")
        .vector(
            "dense",
            valise.Vector(dim=dim, codec=codec, calibrate=valise.Auto(sample=deferred_sample)),
        )
    )


def make_hybrid(store, *, deferred_sample=8, dim=DIM, codec=None):
    """Declare the 'kb' collection with body/dense fields (one call —
    inline specs auto-define their private spaces)."""
    store.collection(
        "kb", hybrid_schema(deferred_sample=deferred_sample, dim=dim, codec=codec)
    )


# Re-export the typed query/enum surface so tests can `from conftest import ...`
# or use `valise.<name>` interchangeably. The canonical public surface is `valise.*`.
Bm25 = valise.Bm25
TfidfCosine = valise.TfidfCosine
Rrf = valise.Rrf
Weighted = valise.Weighted
Range = valise.Range
HalfLife = valise.HalfLife
Metric = valise.Metric
Rerank = valise.Rerank
TfMode = valise.TfMode
