"""Declarative collection schemas — the Python mirror of Rust
``Schema`` / ``Text`` / ``Vector`` / ``Codec`` / ``Calibrate``.

One call declares a collection::

    store.collection("notes", Schema().text("body").vector("embedding", Vector(dim=768)))

Field specs are plain typed value objects. Defaults live in **Rust**: every
parameter left as ``None`` is resolved by the native layer through the core
constructors (``Text::english()``, ``Codec::qam()`` = QAM(5, 6),
``Calibrate::auto()`` = 50 000-vector sample at first commit,
``Metric::Cosine``), so this module never duplicates a default value.

Codec choice is per-field and progressive: say nothing and get the
production QAM codec; pass ``Vector(dim=768, codec=Upq())`` to switch
family; pass ``Qam(amp_bits=8, phase_bits=8)`` to move the operating point.

The Rust-only escape hatch ``Codec::from_params`` (prebuilt engine
``CodecParams``) is intentionally not mirrored — it requires engine-level
parameter blobs that only the Rust API can construct (`store.raw()` tier).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

import numpy as np

from . import _native
from ._enums import Design, Lang, Metric

__all__ = [
    "Schema",
    "Text",
    "Vector",
    "Qam",
    "Upq",
    "Codec",
    "Auto",
    "Now",
    "Calibrate",
    "Space",
]


class Space:
    """An opaque handle to a defined space, returned by
    :meth:`valise.Store.define_space` / :meth:`valise.Store.space`. Bind it in a
    schema with ``Text(space=...)`` or ``Vector(space=...)``.
    """

    def __init__(self, native: "_native.Space") -> None:
        self._n = native

    @property
    def name(self) -> str:
        """The space's name."""
        return self._n.name

    @property
    def is_text(self) -> bool:
        return self._n.is_text

    @property
    def is_vector(self) -> bool:
        return self._n.is_vector

    def __repr__(self) -> str:
        return repr(self._n)


# --------------------------------------------------------------------------- #
# Codec choice (mirror of Rust Codec constructors)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Qam:
    """QAM Lloyd-Max codec.

    ``Qam()`` is the production default operating point (Rust
    ``Codec::qam()`` = 5 amplitude + 6 phase bits). Pass **both**
    ``amp_bits`` and ``phase_bits`` to move it (Rust ``Codec::qam_bits``);
    higher bits trade storage for recall.

    Args:
        amp_bits: Bits per amplitude index, or ``None`` for the Rust default.
        phase_bits: Bits per phase index, or ``None`` for the Rust default.
    """

    amp_bits: Optional[int] = None
    phase_bits: Optional[int] = None


@dataclass(frozen=True)
class Upq:
    """UPQ (unrestricted polar quantization) codec.

    ``Upq()`` is the default operating point (Rust ``Codec::upq()`` =
    2048 cells, empirical ring design). ``Upq(cells=...)`` mirrors
    ``Codec::upq_cells``; a ``design`` requires an explicit ``cells``
    budget (mirror of ``Codec::upq_with``).

    Args:
        cells: Total joint cells per complex pair (power of two), or ``None``
            for the Rust default.
        design: Ring-design source (:class:`valise.Design`), or ``None`` for
            empirical.
    """

    cells: Optional[int] = None
    design: Optional[Design] = None

    def __post_init__(self) -> None:
        if self.design is not None and not isinstance(self.design, Design):
            raise TypeError(
                f"design must be a Design enum, got {type(self.design).__name__}"
            )


#: A typed per-field codec choice.
Codec = Union[Qam, Upq]


# --------------------------------------------------------------------------- #
# Calibration choice (mirror of Rust Calibrate constructors)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Auto:
    """Calibrate at the first commit that contains vectors for the field.

    ``Auto()`` is the Rust default (``Calibrate::auto()``: fit from up to
    50 000 staged vectors). ``Auto(sample=n)`` mirrors
    ``Calibrate::auto_sample(n)``.

    Args:
        sample: Sample-size cap, or ``None`` for the Rust default.
    """

    sample: Optional[int] = None


class Now:
    """Calibrate eagerly at declaration from a representative sample
    (mirror of Rust ``Calibrate::now``).

    Args:
        vectors: A 2-D ``[n, dim]`` array of sample vectors. Coerced to
            C-contiguous ``float32``.
    """

    def __init__(self, vectors: np.ndarray) -> None:
        arr = np.ascontiguousarray(vectors, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"Now expects a 2-D [n, dim] sample array, got {arr.ndim}-D")
        self.vectors = arr

    def __repr__(self) -> str:
        n, dim = self.vectors.shape
        return f"Now(vectors=<{n}x{dim} float32>)"


#: A typed calibration choice.
Calibrate = Union[Auto, Now]


# --------------------------------------------------------------------------- #
# Field specs
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Text:
    """Text field spec: an inline analyzer choice or a shared-space binding.

    ``Text()`` is the Rust default (``Text::english()``: NFKC, case-fold,
    Porter2 stemming, Lucene stopwords). ``Text(lang=Lang.RAW)`` is the
    pass-through whitespace tokenizer. ``Text(space=...)`` binds a shared
    space from :meth:`valise.Store.define_space` (mutually exclusive with
    ``lang``).
    """

    lang: Optional[Lang] = None
    space: Optional[Space] = None

    def __post_init__(self) -> None:
        if self.lang is not None and not isinstance(self.lang, Lang):
            raise TypeError(f"lang must be a Lang enum, got {type(self.lang).__name__}")
        if self.space is not None and not isinstance(self.space, Space):
            raise TypeError(
                f"space must be a Space handle, got {type(self.space).__name__}"
            )
        if self.lang is not None and self.space is not None:
            raise ValueError("Text: lang and space are mutually exclusive")


class Vector:
    """Vector field spec: an inline ``(dim, metric, codec, calibrate)``
    choice or a shared-space binding. Exactly one of ``dim``/``space``.

    Inline defaults are the Rust ones: cosine metric, ``Qam()`` codec,
    ``Auto()`` calibration. ``dim`` must be a positive multiple of 64
    (checked at declaration). On a shared binding (``space=...``) the
    configuration arguments are rejected at declaration time —
    metric/codec/calibration belong to the space and are configured at
    :meth:`valise.Store.define_space`.

    Args:
        dim: The vector dimension (inline spec).
        metric: Distance metric, or ``None`` for cosine.
        codec: :class:`Qam` / :class:`Upq`, or ``None`` for ``Qam()``.
        calibrate: :class:`Auto` / :class:`Now`, or ``None`` for ``Auto()``.
        space: A shared vector :class:`Space` (instead of ``dim``).
    """

    def __init__(
        self,
        dim: Optional[int] = None,
        *,
        metric: Optional[Metric] = None,
        codec: Optional[Codec] = None,
        calibrate: Optional[Calibrate] = None,
        space: Optional[Space] = None,
    ) -> None:
        if (dim is None) == (space is None):
            raise ValueError("Vector: exactly one of dim/space is required")
        if metric is not None and not isinstance(metric, Metric):
            raise TypeError(f"metric must be a Metric enum, got {type(metric).__name__}")
        if codec is not None and not isinstance(codec, (Qam, Upq)):
            raise TypeError(f"codec must be Qam or Upq, got {type(codec).__name__}")
        if calibrate is not None and not isinstance(calibrate, (Auto, Now)):
            raise TypeError(
                f"calibrate must be Auto or Now, got {type(calibrate).__name__}"
            )
        if space is not None and not isinstance(space, Space):
            raise TypeError(f"space must be a Space handle, got {type(space).__name__}")
        self.dim = dim
        self.metric = metric
        self.codec = codec
        self.calibrate = calibrate
        self.space = space

    def __repr__(self) -> str:
        if self.space is not None:
            return f"Vector(space={self.space!r})"
        parts = [f"dim={self.dim}"]
        for name in ("metric", "codec", "calibrate"):
            v = getattr(self, name)
            if v is not None:
                parts.append(f"{name}={v!r}")
        return f"Vector({', '.join(parts)})"


# --------------------------------------------------------------------------- #
# Native lowering (shared by Schema and Store.define_space)
# --------------------------------------------------------------------------- #
def _codec_kwargs(codec: Optional[Codec]) -> dict[str, Any]:
    if codec is None:
        return {}
    if isinstance(codec, Qam):
        return {
            "codec_family": "qam",
            "qam_amp_bits": codec.amp_bits,
            "qam_phase_bits": codec.phase_bits,
        }
    if isinstance(codec, Upq):
        return {
            "codec_family": "upq",
            "upq_cells": codec.cells,
            "upq_design": None if codec.design is None else codec.design.value,
        }
    raise TypeError(f"codec must be Qam or Upq, got {type(codec).__name__}")


def _calibrate_kwargs(calibrate: Optional[Calibrate]) -> dict[str, Any]:
    if calibrate is None:
        return {}
    if isinstance(calibrate, Auto):
        return {"cal_mode": "auto", "cal_sample": calibrate.sample}
    if isinstance(calibrate, Now):
        return {"cal_mode": "now", "cal_vectors": calibrate.vectors}
    raise TypeError(f"calibrate must be Auto or Now, got {type(calibrate).__name__}")


def _vector_kwargs(spec: Vector) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if spec.space is not None:
        kwargs["space"] = spec.space._n
    else:
        kwargs["dim"] = spec.dim
    if spec.metric is not None:
        kwargs["metric"] = spec.metric.value
    kwargs.update(_codec_kwargs(spec.codec))
    kwargs.update(_calibrate_kwargs(spec.calibrate))
    return kwargs


def _text_kwargs(spec: Text) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if spec.lang is not None:
        kwargs["lang"] = spec.lang.value
    if spec.space is not None:
        kwargs["space"] = spec.space._n
    return kwargs


# --------------------------------------------------------------------------- #
# Schema builder
# --------------------------------------------------------------------------- #
class Schema:
    """The set of named fields a collection's records may fill.

    Built fluently and bound by :meth:`valise.Store.collection`::

        Schema().text("body").vector("embedding", Vector(dim=768))

    ``vector(name, 768)`` is accepted as an int shorthand for
    ``vector(name, Vector(dim=768))``.
    """

    def __init__(self) -> None:
        self._fields: List[Tuple[str, Union[Text, Vector]]] = []

    def text(self, name: str, spec: Optional[Text] = None) -> "Schema":
        """Add a text field. ``spec=None`` is the Rust default
        (``Text::english()``). At most one text field per collection in v1.
        """
        if spec is None:
            spec = Text()
        if not isinstance(spec, Text):
            raise TypeError(f"spec must be a Text, got {type(spec).__name__}")
        self._fields.append((name, spec))
        return self

    def vector(self, name: str, spec: Union[Vector, int]) -> "Schema":
        """Add a vector field. The dimension is mandatory, so the spec is
        always explicit: ``Vector(dim=768)``, the int shorthand ``768``, or a
        shared binding ``Vector(space=...)``.
        """
        if isinstance(spec, bool) or not isinstance(spec, (Vector, int)):
            raise TypeError(f"spec must be a Vector or int dim, got {type(spec).__name__}")
        if isinstance(spec, int):
            spec = Vector(dim=spec)
        self._fields.append((name, spec))
        return self

    def _native(self) -> "_native.Schema":
        """Lower to the native builder (one native call per field)."""
        ns = _native.Schema()
        for name, spec in self._fields:
            if isinstance(spec, Text):
                ns.text(name, **_text_kwargs(spec))
            else:
                ns.vector(name, **_vector_kwargs(spec))
        return ns
