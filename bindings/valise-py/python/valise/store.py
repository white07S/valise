# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""The Store facade — the application entry point.

Owns collection declaration (:class:`valise.Schema`), shared spaces,
partitioning, writer/reader acquisition, point lookups, search, compaction,
and stats. Enums convert to their native string via ``enum.value``; results
return as typed dataclasses.

Schemas are **persisted in-file**: after :meth:`Store.collection` and a
commit, a later :meth:`Store.open` restores every collection's schema —
reopen needs no re-declaration.
"""

from __future__ import annotations

import os
from enum import Enum
from typing import Optional, TypeVar, Union

from . import _native
from ._enums import Durability
from .partition import Partition, Partitioned, View
from .query import Search, _native_search
from .reader import Reader
from .schema import Schema, Space, Text, Vector, _text_kwargs, _vector_kwargs
from .types import (
    CollectionInfo,
    CompactReport,
    Key,
    SearchResult,
    SpaceInfo,
    Stats,
    Stored,
)
from .writer import Writer

__all__ = ["Store"]

_PathLike = Union[str, "os.PathLike[str]"]


_E = TypeVar("_E", bound=Enum)


def _enum_value(value: object, enum_cls: type[_E], argname: str) -> str:
    """Return ``value.value`` after asserting ``value`` is an ``enum_cls``.

    Gives the same clear ``TypeError`` the query builders raise for a raw string,
    instead of an opaque ``AttributeError`` from reaching for ``.value``.
    """
    if not isinstance(value, enum_cls):
        raise TypeError(
            f"{argname} must be a {enum_cls.__name__} enum, got {type(value).__name__}"
        )
    return str(value.value)


class Store:
    """A single-file Valise store.

    :meth:`open` is open-or-create; :meth:`create` fails if the path exists.
    Declare collections with :meth:`collection`, then acquire a
    :class:`Writer` to ingest and a :class:`Reader` to query.

    Example::

        store = Store.open("kb.vls")
        store.collection("notes", Schema().text("body").vector("embedding", Vector(dim=768)))
        with store.writer() as w:
            w.put("notes", "doc-1", Record().text("body", "...").vector("embedding", emb))
            w.commit()
        hits = store.search("notes", Search().text("body", "...").vector("embedding", emb))
    """

    def __init__(self, native: "_native.Store") -> None:
        self._n = native

    @classmethod
    def open(
        cls,
        path: _PathLike,
        durability: Durability = Durability.BUFFERED,
    ) -> "Store":
        """Open an existing store, creating it if absent (open-or-create)."""
        return cls(
            _native.Store.open(
                os.fspath(path), _enum_value(durability, Durability, "durability")
            )
        )

    @classmethod
    def create(
        cls,
        path: _PathLike,
        durability: Durability = Durability.BUFFERED,
    ) -> "Store":
        """Create a fresh store at ``path``, failing if it already exists."""
        return cls(
            _native.Store.create(
                os.fspath(path), _enum_value(durability, Durability, "durability")
            )
        )

    # ---- Schema ----

    def collection(self, name: str, schema: Schema) -> None:
        """Create-or-open ``name``, bind ``schema``, and persist the schema
        in-file (it rides the next commit) — a later :meth:`open` restores it,
        so reopen needs **no re-declaration**.

        Re-declaring is a no-op when identical and accepted when purely
        additive; anything else raises :class:`valise.SchemaMismatchError`.
        Names starting with ``~`` are reserved and rejected.
        """
        if not isinstance(schema, Schema):
            raise TypeError(f"schema must be a Schema, got {type(schema).__name__}")
        self._n.collection(name, schema._native())

    def define_space(self, name: str, spec: Union[Text, Vector]) -> Space:
        """Define (or return the existing) shared space from a field spec —
        ``Text()`` / ``Vector(dim=768, codec=Upq())`` / … — then bind it in
        any schema with ``Text(space=...)`` / ``Vector(space=...)``.

        Re-defining an existing name is a no-op for an identical spec and an
        error for a divergent one.
        """
        if isinstance(spec, Text):
            if spec.space is not None:
                raise ValueError("define_space: spec must be inline (not space-bound)")
            return Space(self._n.define_space_text(name, **_text_kwargs(spec)))
        if isinstance(spec, Vector):
            if spec.space is not None:
                raise ValueError("define_space: spec must be inline (not space-bound)")
            return Space(self._n.define_space_vector(name, **_vector_kwargs(spec)))
        raise TypeError(f"spec must be a Text or Vector, got {type(spec).__name__}")

    def space(self, name: str) -> Optional[Space]:
        """Look up a previously defined space by name, or ``None``."""
        ns = self._n.space(name)
        return None if ns is None else Space(ns)

    def spaces(self) -> list[SpaceInfo]:
        """All defined spaces — shared and ``~auto/{collection}/{field}`` ones
        (the latter flagged ``auto``) — sorted by name."""
        return [
            SpaceInfo(name=n, dim=d, auto=a) for (n, d, a) in self._n.spaces()
        ]

    def collections(self) -> list[CollectionInfo]:
        """All user collections (the hidden ``~valise.schema`` store is never
        listed)."""
        return [CollectionInfo(name=n, id=i) for (n, i) in self._n.collections()]

    def partitioned(self, base: str, schema: Schema, by: Partition) -> Partitioned:
        """Create-or-open a logical partitioned collection: records routed by
        ``by`` into physical collections (``"{base}:{period}"``) that share
        one resolved schema."""
        if not isinstance(schema, Schema):
            raise TypeError(f"schema must be a Schema, got {type(schema).__name__}")
        return Partitioned(
            self._n.partitioned(
                base, schema._native(), _enum_value(by, Partition, "by")
            )
        )

    # ---- Handles ----

    def reader(self) -> Reader:
        """Acquire a :class:`Reader` (live, last-committed-wins per operation\n        — not a frozen snapshot)."""
        return Reader(self._n.reader())

    def writer(self) -> Writer:
        """Acquire the single :class:`Writer` (blocks until available)."""
        return Writer(self._n.writer())

    # ---- Read ----

    def get(self, coll: str, key: Key) -> Optional[Stored]:
        """Fetch a single record, or ``None`` if absent."""
        ns = self._n.get(coll, key)
        return None if ns is None else Stored._from_native(ns)

    def search(self, coll: str, search: Search) -> SearchResult:
        """Run a search; returns a :class:`SearchResult` in one native call."""
        keys, scores = self._n.search(coll, _native_search(search))
        return SearchResult(keys, scores, coll)

    def search_view(self, view: View, search: Search) -> SearchResult:
        """Search across a partitioned :class:`View` as one globally-fused
        query; hits carry their per-hit partition collection."""
        if not isinstance(view, View):
            raise TypeError(f"view must be a View, got {type(view).__name__}")
        keys, scores, colls = self._n.search_view(view._n, _native_search(search))
        return SearchResult(keys, scores, colls)

    # ---- Maintenance ----

    def compact(self, recalibrate: bool = False) -> CompactReport:
        """Compact the store; returns a typed :class:`CompactReport`."""
        return CompactReport._from_native(self._n.compact(recalibrate))

    def stats(self) -> Stats:
        """Return store-level :class:`Stats` (user data only)."""
        return Stats._from_native(self._n.stats())
