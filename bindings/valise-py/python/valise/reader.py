# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""The reader facade — point lookups and search over a committed snapshot."""

from __future__ import annotations

from typing import Optional

from . import _native
from .partition import View
from .query import Search, _native_search
from .types import Key, SearchResult, Stored

__all__ = ["Reader"]


class Reader:
    """A snapshot reader: ``get`` / ``get_many`` / ``search`` / ``search_view``."""

    def __init__(self, native: "_native.Reader") -> None:
        self._n = native

    def get(self, coll: str, key: Key) -> Optional[Stored]:
        """Fetch a single record, or ``None`` if absent."""
        ns = self._n.get(coll, key)
        return None if ns is None else Stored._from_native(ns)

    def get_many(self, coll: str, keys: list[Key]) -> list[Optional[Stored]]:
        """Fetch many records in one native call.

        Returns a list aligned with ``keys``; missing entries are ``None``. The
        list comprehension wraps already-built native ``Stored`` objects — it
        is not a per-record native round-trip.
        """
        return [
            None if ns is None else Stored._from_native(ns)
            for ns in self._n.get_many(coll, keys)
        ]

    def search(self, coll: str, search: Search) -> SearchResult:
        """Run a search; returns a :class:`SearchResult` in one native call.

        Example:
            >>> import os, tempfile
            >>> from valise import Store, Schema, Record, Search
            >>> path = os.path.join(tempfile.mkdtemp(), "demo.vls")
            >>> store = Store.open(path)
            >>> store.collection("kb", Schema().text("body"))
            >>> with store.writer() as w:
            ...     w.put("kb", "a", Record().text("body", "the quick fox"))
            ...     w.put("kb", "b", Record().text("body", "a lazy dog"))
            ...     _ = w.commit()
            >>> result = store.reader().search(
            ...     "kb", Search().text("body", "fox").top_k(5)
            ... )
            >>> result[0].key
            'a'
            >>> len(result)
            1
            >>> result.scores.dtype
            dtype('float32')
        """
        keys, scores = self._n.search(coll, _native_search(search))
        return SearchResult(keys, scores, coll)

    def search_view(self, view: View, search: Search) -> SearchResult:
        """Search across a partitioned :class:`valise.View` as ONE globally-fused
        multi-collection query; hits carry their per-hit partition collection.
        """
        if not isinstance(view, View):
            raise TypeError(f"view must be a View, got {type(view).__name__}")
        keys, scores, colls = self._n.search_view(view._n, _native_search(search))
        return SearchResult(keys, scores, colls)
