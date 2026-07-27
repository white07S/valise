# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Typed result objects for the valise facade.

These wrap the native return shapes (lists, ndarrays, dicts, native ``Stored``)
into typed dataclasses and a lazy :class:`SearchResult`. The heavy data — the
``scores`` ndarray and the moved vector ndarrays — is never copied; only small
per-accessed-hit scalar conversions happen.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, List, Optional, Union

import numpy as np

__all__ = [
    "Key",
    "Hit",
    "SearchResult",
    "Stats",
    "CompactReport",
    "Stored",
    "SpaceInfo",
    "CollectionInfo",
]

#: A record key. Accepted as ``str``, ``int``, or ``bytes``.
Key = Union[str, int, bytes]


@dataclass(frozen=True)
class Hit:
    """A single search hit.

    Attributes:
        key: The record key.
        score: The fused relevance score.
        collection: The collection the hit belongs to.
    """

    key: Key
    score: float
    collection: str


@dataclass(frozen=True)
class SpaceInfo:
    """One row of :meth:`valise.Store.spaces`.

    Attributes:
        name: The space name. Auto spaces are named
            ``~auto/{collection}/{field}``.
        dim: The vector dimension, or ``None`` for a text space.
        auto: ``True`` for the private spaces auto-defined by inline field
            specs.
    """

    name: str
    dim: Optional[int]
    auto: bool

    @property
    def is_vector(self) -> bool:
        return self.dim is not None

    @property
    def is_text(self) -> bool:
        return self.dim is None


@dataclass(frozen=True)
class CollectionInfo:
    """One row of :meth:`valise.Store.collections` (the hidden ``~valise.schema``
    store is never listed).

    Attributes:
        name: The collection name.
        id: The engine collection id.
    """

    name: str
    id: int


class SearchResult:
    """The result of a search: parallel ``keys`` and ``scores``.

    Built from the native ``(keys: list, scores: np.ndarray[float32])`` tuple
    in one native call. Both are held as-is (zero new allocation); the
    ``scores`` ndarray is the moved array, never copied. :class:`Hit` objects
    are materialized lazily — on indexing or iteration — so there is no eager
    per-hit loop over the result set.

    For a single-collection :meth:`valise.Store.search`, ``collection`` is that
    collection's name. For :meth:`valise.Store.search_view`, hits span
    partitions: ``collection`` is ``None`` and the per-hit names are carried
    on each :class:`Hit` (and by :attr:`collections`).
    """

    __slots__ = ("_keys", "_scores", "_collection")

    def __init__(
        self,
        keys: List[Key],
        scores: np.ndarray,
        collection: Union[str, List[str]],
    ) -> None:
        self._keys = keys
        self._scores = scores
        self._collection = collection

    @property
    def keys(self) -> List[Key]:
        """The result keys (the native list, returned as-is)."""
        return self._keys

    @property
    def scores(self) -> np.ndarray:
        """The result scores (the moved ``float32`` ndarray, returned as-is)."""
        return self._scores

    @property
    def collection(self) -> Optional[str]:
        """The single collection these hits belong to, or ``None`` for a
        multi-collection (view) result — use :attr:`collections` then."""
        return self._collection if isinstance(self._collection, str) else None

    @property
    def collections(self) -> List[str]:
        """Per-hit collection names (uniform for a single-collection search)."""
        if isinstance(self._collection, str):
            return [self._collection] * len(self._keys)
        return list(self._collection)

    def _coll_at(self, i: int) -> str:
        c = self._collection
        return c if isinstance(c, str) else c[i]

    def __len__(self) -> int:
        return len(self._keys)

    def __getitem__(
        self, i: Union[int, slice]
    ) -> Union[Hit, "SearchResult"]:
        if isinstance(i, slice):
            # A slice stays a SearchResult: chainable, re-iterable, and the
            # sliced scores are a numpy *view* (zero-copy) — not a list[Hit].
            coll = (
                self._collection
                if isinstance(self._collection, str)
                else self._collection[i]
            )
            return SearchResult(self._keys[i], self._scores[i], coll)
        if isinstance(i, int) and i < 0:
            i += len(self._keys)
        return Hit(self._keys[i], float(self._scores[i]), self._coll_at(i))

    def __iter__(self) -> Iterator[Hit]:
        for idx, (k, s) in enumerate(zip(self._keys, self._scores)):
            yield Hit(k, float(s), self._coll_at(idx))

    def __repr__(self) -> str:
        n = len(self)
        head = ", ".join(
            f"({k!r}, {float(s):.4g})"
            for k, s in zip(self._keys[:3], self._scores[:3])
        )
        more = ", ..." if n > 3 else ""
        if isinstance(self._collection, str):
            scope = f"collection={self._collection!r}"
        else:
            scope = f"collections={sorted(set(self._collection))!r}"
        return f"SearchResult({scope}, n={n}, hits=[{head}{more}])"


@dataclass(frozen=True)
class Stats:
    """Store-level statistics (user data only — the hidden ``~valise.schema``
    doc frames are excluded).

    Attributes:
        frames_total: Total frames ever written.
        frames_active: Live (non-tombstoned) frames.
        vectors_total: Total vectors ever written.
        vectors_active: Live vectors.
        file_bytes: On-disk file size in bytes.
        tombstone_ratio: Fraction of frames that are tombstones.
    """

    frames_total: int
    frames_active: int
    vectors_total: int
    vectors_active: int
    file_bytes: int
    tombstone_ratio: float

    def needs_compaction(self, threshold: float) -> bool:
        """Whether :attr:`tombstone_ratio` meets/exceeds ``threshold``
        (e.g. ``0.3``). Mirrors Rust ``StoreStats::needs_compaction``."""
        return self.tombstone_ratio >= threshold

    @classmethod
    def _from_native(cls, d: dict[str, Any]) -> "Stats":
        return cls(
            frames_total=d["frames_total"],
            frames_active=d["frames_active"],
            vectors_total=d["vectors_total"],
            vectors_active=d["vectors_active"],
            file_bytes=d["file_bytes"],
            tombstone_ratio=d["tombstone_ratio"],
        )


@dataclass(frozen=True)
class CompactReport:
    """The result of a compaction.

    Attributes:
        compacted: Whether a rewrite actually occurred.
        frames_before: Frame count before compaction.
        frames_after: Frame count after compaction.
        vectors_before: Vector count before compaction.
        vectors_after: Vector count after compaction.
        bytes_before: File size before compaction.
        bytes_after: File size after compaction.
    """

    compacted: bool
    frames_before: int
    frames_after: int
    vectors_before: int
    vectors_after: int
    bytes_before: int
    bytes_after: int

    @classmethod
    def _from_native(cls, d: dict[str, Any]) -> "CompactReport":
        return cls(
            compacted=d["compacted"],
            frames_before=d["frames_before"],
            frames_after=d["frames_after"],
            vectors_before=d["vectors_before"],
            vectors_after=d["vectors_after"],
            bytes_before=d["bytes_before"],
            bytes_after=d["bytes_after"],
        )


@dataclass(frozen=True, eq=False)
class Stored:
    """A stored record as returned by ``get`` / ``get_many``.

    Attributes:
        key: The record key.
        collection: The owning collection.
        created_at: The record timestamp (unix seconds).
        text: The text field value, if any.
        vectors: An ordered list of ``(field_name, ndarray)`` pairs. The
            ndarrays are the moved arrays from native — not copied.

    Equality is structural (``np.array_equal`` over each vector, so shape and
    dtype count). Because it carries ndarrays, :class:`Stored` is unhashable —
    like the ``list`` / ``ndarray`` payloads it holds.

    Example:
        >>> import os, tempfile
        >>> from valise import Store, Schema, Record
        >>> path = os.path.join(tempfile.mkdtemp(), "demo.vls")
        >>> store = Store.open(path)
        >>> store.collection("kb", Schema().text("body"))
        >>> with store.writer() as w:
        ...     w.put("kb", "a", Record().text("body", "the quick brown fox"))
        ...     _ = w.commit()
        >>> got = store.reader().get("kb", "a")
        >>> got.text
        'the quick brown fox'
        >>> got.created_at >= 0
        True
        >>> got == store.reader().get("kb", "a")
        True
    """

    key: Key
    collection: str
    created_at: int
    text: Optional[str]
    vectors: list[tuple[str, np.ndarray]]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Stored):
            return NotImplemented
        if (self.key, self.collection, self.created_at, self.text) != (
            other.key,
            other.collection,
            other.created_at,
            other.text,
        ):
            return False
        if len(self.vectors) != len(other.vectors):
            return False
        for (an, av), (bn, bv) in zip(self.vectors, other.vectors):
            if an != bn or not np.array_equal(av, bv):
                return False
        return True

    # ndarray-bearing payload -> not hashable (matches list/ndarray semantics).
    __hash__ = None  # type: ignore[assignment]

    @classmethod
    def _from_native(cls, ns: Any) -> "Stored":
        return cls(
            key=ns.key,
            collection=ns.collection,
            created_at=ns.created_at,
            text=ns.text,
            vectors=ns.vectors,
        )
