# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Typed query builders for the valise facade.

Defines the strict, typed value objects for text scoring, fusion, and recency,
plus the :class:`Search` and :class:`Record` facades. Every builder method
forwards to the owned native builder in **exactly one** native call; the
facade never re-walks a query in Python.

Defaults live in Rust: scorer/fusion parameters left as ``None`` resolve to
the core constructor defaults (``TextScorer::bm25()`` = ``k1=1.2, b=0.75``,
``Fusion::rrf(60)``) in the native layer. Where Rust offers method pairs
(``text``/``text_with``, ``vector``/``vector_with``), Python uses default
arguments instead — the concepts match, the idiom is per-language.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from . import _native
from ._enums import Rerank, TfMode

__all__ = [
    "Bm25",
    "TfidfCosine",
    "TfidfCosineApprox",
    "CountCosine",
    "CountCosineApprox",
    "Dice",
    "Overlap",
    "Containment",
    "TextScorer",
    "Rrf",
    "Weighted",
    "Fusion",
    "Range",
    "HalfLife",
    "RrfChannel",
    "Recency",
    "Search",
    "Record",
]


# --------------------------------------------------------------------------- #
# Text scorers (typed; carry their own parameters)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Bm25:
    """Okapi BM25 lexical scorer — the default of :meth:`Search.text`.

    Args:
        k1: Term-frequency saturation, or ``None`` for the Rust default (1.2).
        b: Length normalization, or ``None`` for the Rust default (0.75).
    """

    k1: Optional[float] = None
    b: Optional[float] = None


@dataclass(frozen=True)
class TfidfCosine:
    """TF-IDF cosine-similarity lexical scorer.

    Args:
        tf_mode: Term-frequency weighting mode.
    """

    tf_mode: TfMode = TfMode.LOG


@dataclass(frozen=True)
class TfidfCosineApprox:
    """Cheaper ``sqrt(doclen)``-normalized TF-IDF cosine approximation.

    Args:
        tf_mode: Term-frequency weighting mode.
    """

    tf_mode: TfMode = TfMode.LOG


@dataclass(frozen=True)
class CountCosine:
    """Raw-count cosine-similarity lexical scorer."""


@dataclass(frozen=True)
class CountCosineApprox:
    """Cheaper count-cosine approximation."""


@dataclass(frozen=True)
class Dice:
    """Sørensen–Dice set-overlap scorer."""


@dataclass(frozen=True)
class Overlap:
    """Overlap-coefficient set scorer."""


@dataclass(frozen=True)
class Containment:
    """Containment (asymmetric overlap) set scorer."""


#: A typed text scorer (mirror of Rust ``TextScorer``).
TextScorer = Union[
    Bm25,
    TfidfCosine,
    TfidfCosineApprox,
    CountCosine,
    CountCosineApprox,
    Dice,
    Overlap,
    Containment,
]


# --------------------------------------------------------------------------- #
# Fusion strategies
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Rrf:
    """Reciprocal-rank fusion — the default with ≥2 channels.

    Args:
        k: The RRF rank constant, or ``None`` for the Rust default (60).
    """

    k: Optional[int] = None


@dataclass(frozen=True)
class Weighted:
    """Linear weighted fusion of the text and vector channels (each channel
    min-max normalized to ``[0, 1]`` first).

    Args:
        text: Weight applied to the text channel.
        vector: Weight applied to every vector channel.
    """

    text: float
    vector: float


#: A typed fusion strategy.
Fusion = Union[Rrf, Weighted]


# --------------------------------------------------------------------------- #
# Recency strategies
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Range:
    """Hard recency filter to the **inclusive** ``[from_, to]`` unix-second
    window (candidates outside it are pruned before fusion).

    Args:
        from_: Inclusive lower bound (unix seconds). Named ``from_`` because
            ``from`` is a Python keyword.
        to: Inclusive upper bound (unix seconds).
    """

    from_: int
    to: int


@dataclass(frozen=True)
class HalfLife:
    """Exponential recency decay with the given half-life (soft post-fusion
    multiplier ``score * 0.5 ** (age / half_life)``).

    Args:
        days: The decay half-life in days.
    """

    days: float


@dataclass(frozen=True)
class RrfChannel:
    """Recency as an additional RRF channel (candidates ranked newest first,
    contributing ``weight / (k + rank + 1)``). Requires :class:`Rrf` fusion.

    Args:
        half_life_days: The recency half-life in days.
        weight: The channel weight.
    """

    half_life_days: float
    weight: float


#: A typed recency strategy (mirror of Rust ``Recency``).
Recency = Union[Range, HalfLife, RrfChannel]


# --------------------------------------------------------------------------- #
# Search facade
# --------------------------------------------------------------------------- #
class Search:
    """Typed hybrid-search query builder.

    Wraps a single owned native ``Search`` builder. Each builder method makes
    exactly one native call into that owned state and returns ``self`` for
    chaining; the facade never re-walks the query in Python.

    Defaults (all applied by Rust): BM25 text scorer, ``ACCURATE`` vector
    rerank, RRF(60) fusion, ``top_k=10``.
    """

    def __init__(self) -> None:
        self._s = _native.Search()

    def text(
        self,
        field: str,
        query: str,
        scorer: TextScorer = Bm25(),
    ) -> "Search":
        """Add a lexical (text) channel.

        Args:
            field: The text field to query.
            query: The query string.
            scorer: A typed :data:`TextScorer` (default :class:`Bm25`).
                Raw strings are not accepted.

        Returns:
            ``self``, for chaining.
        """
        if isinstance(scorer, Bm25):
            self._s.text(field, query, "bm25", scorer.k1, scorer.b)
        elif isinstance(scorer, TfidfCosine):
            self._s.text(field, query, "tfidf_cosine", None, None, scorer.tf_mode.value)
        elif isinstance(scorer, TfidfCosineApprox):
            self._s.text(
                field, query, "tfidf_cosine_approx", None, None, scorer.tf_mode.value
            )
        elif isinstance(scorer, CountCosine):
            self._s.text(field, query, "count_cosine")
        elif isinstance(scorer, CountCosineApprox):
            self._s.text(field, query, "count_cosine_approx")
        elif isinstance(scorer, Dice):
            self._s.text(field, query, "dice")
        elif isinstance(scorer, Overlap):
            self._s.text(field, query, "overlap")
        elif isinstance(scorer, Containment):
            self._s.text(field, query, "containment")
        else:
            raise TypeError(
                f"scorer must be a TextScorer, got {type(scorer).__name__}"
            )
        return self

    def vector(
        self,
        field: str,
        query: np.ndarray,
        rerank: Rerank = Rerank.ACCURATE,
    ) -> "Search":
        """Add a vector (nearest-neighbor) channel. May be called multiple
        times for multi-vector / multimodal search; each becomes a fusion
        channel.

        Args:
            field: The vector field to query.
            query: A 1-D ``float32`` ndarray (borrowed zero-copy by native).
            rerank: Rerank fidelity (default ``Rerank.ACCURATE``, mirroring
                Rust).

        Returns:
            ``self``, for chaining.
        """
        if not isinstance(rerank, Rerank):
            raise TypeError(
                f"rerank must be a Rerank enum, got {type(rerank).__name__}"
            )
        self._s.vector(field, query, rerank.value)
        return self

    def fuse(self, fusion: Fusion) -> "Search":
        """Set the channel-fusion strategy.

        Args:
            fusion: A typed :data:`Fusion` (:class:`Rrf` or :class:`Weighted`).

        Returns:
            ``self``, for chaining.
        """
        if isinstance(fusion, Rrf):
            self._s.fuse_rrf(fusion.k)
        elif isinstance(fusion, Weighted):
            self._s.fuse_weighted(fusion.text, fusion.vector)
        else:
            raise TypeError(
                f"fusion must be a Fusion, got {type(fusion).__name__}"
            )
        return self

    def recency(self, recency: Recency) -> "Search":
        """Set the recency strategy.

        Args:
            recency: A typed :data:`Recency` (:class:`Range`,
                :class:`HalfLife`, or :class:`RrfChannel`).

        Returns:
            ``self``, for chaining.
        """
        if isinstance(recency, Range):
            self._s.recency_range(recency.from_, recency.to)
        elif isinstance(recency, HalfLife):
            self._s.recency_half_life(recency.days)
        elif isinstance(recency, RrfChannel):
            self._s.recency_rrf_channel(recency.half_life_days, recency.weight)
        else:
            raise TypeError(
                f"recency must be a Recency, got {type(recency).__name__}"
            )
        return self

    def top_k(self, k: int) -> "Search":
        """Limit the result set to the top ``k`` hits."""
        self._s.top_k(k)
        return self

    def now(self, unix_secs: int) -> "Search":
        """Set the reference 'now' (unix seconds) for recency decay."""
        self._s.now(unix_secs)
        return self


def _native_search(search: Search) -> "_native.Search":
    """Return the owned native ``Search`` builder without copying.

    ``Search`` is not ``Clone`` in core; the native builder reassembles its
    components on ``build()``. Pass this through directly to store/reader
    search — never reconstruct it in Python.
    """
    return search._s


# --------------------------------------------------------------------------- #
# Record facade
# --------------------------------------------------------------------------- #
class Record:
    """Typed record builder.

    Accumulates text/vector fields and metadata for a single record. Each
    method forwards one native call and returns ``self`` for chaining.
    """

    def __init__(self) -> None:
        self._n = _native.Record()

    def text(self, field: str, value: str) -> "Record":
        """Set the value of a text field."""
        self._n.text(field, value)
        return self

    def vector(self, field: str, values: np.ndarray) -> "Record":
        """Set a vector field (``values`` borrowed zero-copy by native)."""
        self._n.vector(field, values)
        return self

    def at(self, unix_secs: int) -> "Record":
        """Set the record's ``created_at`` timestamp (unix seconds)."""
        self._n.at(unix_secs)
        return self

    def child_of(self, parent: Union[str, int, bytes]) -> "Record":
        """Link this record as a child of ``parent``."""
        self._n.child_of(parent)
        return self
