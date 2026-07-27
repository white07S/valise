"""The single-writer facade.

Delegates lifecycle (``__enter__`` / ``__exit__`` / ``close``) to the native
writer so the single-writer lock and crash-safety semantics are preserved
exactly. ``__exit__`` always calls ``close()`` (releasing the lock via native
Drop) and returns ``False`` to propagate exceptions.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

import numpy as np

from . import _native
from .partition import Partitioned
from .query import Record
from .types import Key

__all__ = ["Writer"]


class Writer:
    """A serialized writer holding the single-writer lock until closed."""

    def __init__(self, native: "_native.Writer") -> None:
        self._n = native

    def put(self, coll: str, key: Key, record: Record) -> None:
        """Stage a record under ``key`` (replacing any existing one)."""
        self._n.put(coll, key, record._n)

    def put_auto(self, coll: str, record: Record) -> Key:
        """Stage a record under an auto-generated key; returns the key."""
        return self._n.put_auto(coll, record._n)

    def put_into(self, partitioned: Partitioned, key: Key, record: Record) -> None:
        """Stage a record into a :class:`valise.Partitioned` logical collection.

        The physical partition is derived from the record's ``created_at``
        (set via :meth:`valise.Record.at`; defaults to now).
        """
        if not isinstance(partitioned, Partitioned):
            raise TypeError(
                f"partitioned must be a Partitioned, got {type(partitioned).__name__}"
            )
        self._n.put_into(partitioned._n, key, record._n)

    def put_many(
        self,
        coll: str,
        keys: list[Key],
        vectors: np.ndarray,
        field: str = "dense",
        texts: Optional[list[str]] = None,
        text_field: str = "body",
    ) -> None:
        """Bulk-stage ``len(keys)`` records in one native call.

        ``vectors`` is a 2-D ``float32`` ndarray ``[N, dim]`` borrowed zero-copy
        by native; there is no per-row Python loop.

        Raises:
            ValueError: if ``vectors`` is not 2-D, or if its row count does not
                match ``len(keys)``, or if ``texts`` is given with a length
                other than ``len(keys)``. (dtype / contiguity are validated by
                the native layer.)
        """
        if vectors.ndim != 2:
            raise ValueError(
                f"put_many expects a 2-D [N, dim] vectors array, "
                f"got {vectors.ndim}-D"
            )
        if vectors.shape[0] != len(keys):
            raise ValueError(
                f"put_many: len(keys)={len(keys)} but vectors has "
                f"{vectors.shape[0]} rows"
            )
        if texts is not None and len(texts) != len(keys):
            raise ValueError(
                f"put_many: len(texts)={len(texts)} != len(keys)={len(keys)}"
            )
        self._n.put_many(coll, keys, vectors, field, texts, text_field)

    def delete(self, coll: str, key: Key) -> bool:
        """Tombstone ``key``; returns ``True`` if it existed."""
        return self._n.delete(coll, key)

    def commit(self) -> int:
        """Commit staged mutations; returns the new generation."""
        return self._n.commit()

    def close(self) -> None:
        """Release the single-writer lock (drops the native writer)."""
        self._n.close()

    def __enter__(self) -> "Writer":
        return self

    def __exit__(self, *exc: Any) -> Literal[False]:
        # Always release the lock, on every exit path. Returning False
        # propagates any in-flight exception.
        self._n.close()
        return False
