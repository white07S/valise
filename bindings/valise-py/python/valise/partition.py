# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Partitioning + views — time-routed logical collections.

A :class:`Partitioned` (from :meth:`valise.Store.partitioned`) routes records
into physical collections named ``"{base}:{period}"`` that share one schema
(hence one set of file-global spaces — no per-partition codec duplication).
A :class:`View` selects a window of those partitions for a fan-out-free
:meth:`valise.Store.search_view` (one multi-collection query, one global
fusion). :meth:`Partitioned.forget_before` is the cheap whole-window soft
forget (tombstones; bytes are reclaimed at :meth:`valise.Store.compact`).

The Rust ``Partition::Custom`` (an arbitrary routing closure) is not
mirrored — the router runs inside the engine write path and cannot call back
into Python. ``BY_DAY`` / ``BY_MONTH`` cover the time-based cases.
"""

from __future__ import annotations

import enum
from typing import List, Optional, Sequence, Union

from . import _native

__all__ = ["Partition", "Window", "Partitioned", "View"]


class Partition(enum.Enum):
    """How records are routed to physical partitions (mirror of Rust
    ``Partition::ByDay`` / ``ByMonth``)."""

    BY_DAY = "day"
    BY_MONTH = "month"


class Window:
    """A window of partitions for a view. Construct via the factory methods:
    :meth:`last_days`, :meth:`range`, :meth:`partitions`.
    """

    __slots__ = ("_kind", "_days", "_from", "_to", "_parts")

    _kind: str
    _days: Optional[int]
    _from: Optional[int]
    _to: Optional[int]
    _parts: Optional[list[str]]

    def __init__(self) -> None:
        raise TypeError(
            "use Window.last_days(...), Window.range(...), or Window.partitions(...)"
        )

    @classmethod
    def _make(
        cls,
        kind: str,
        days: Optional[int] = None,
        from_: Optional[int] = None,
        to: Optional[int] = None,
        parts: Optional[list[str]] = None,
    ) -> "Window":
        w = object.__new__(cls)
        w._kind = kind
        w._days = days
        w._from = from_
        w._to = to
        w._parts = parts
        return w

    @classmethod
    def last_days(cls, days: int) -> "Window":
        """Partitions intersecting ``[now - days, now]``."""
        return cls._make("last_days", days=days)

    @classmethod
    def range(cls, from_: int, to: int) -> "Window":
        """Partitions intersecting ``[from_, to]`` (inclusive unix seconds)."""
        return cls._make("range", from_=from_, to=to)

    @classmethod
    def partitions(cls, suffixes: Sequence[str]) -> "Window":
        """Explicit period suffixes (e.g. ``"2026-05-25"``), resolved against
        the existing ``"{base}:{suffix}"`` collections."""
        return cls._make("partitions", parts=list(suffixes))

    def __repr__(self) -> str:
        if self._kind == "last_days":
            return f"Window.last_days({self._days})"
        if self._kind == "range":
            return f"Window.range({self._from}, {self._to})"
        return f"Window.partitions({self._parts!r})"


class View:
    """A resolved set of physical partition collections. Pass to
    :meth:`valise.Store.search_view` / :meth:`valise.Reader.search_view`."""

    def __init__(self, native: "_native.View") -> None:
        self._n = native

    @property
    def collections(self) -> List[str]:
        """The physical collection names in this view."""
        return self._n.collections()

    def __repr__(self) -> str:
        return repr(self._n)


class Partitioned:
    """A logical partitioned collection (cheap handle; see module docs)."""

    def __init__(self, native: "_native.Partitioned") -> None:
        self._n = native

    def view(self, window: Window, now: Optional[int] = None) -> View:
        """Resolve the partitions intersecting ``window`` to a :class:`View`.

        ``now`` (unix seconds) overrides the reference time of
        ``Window.last_days`` for deterministic results; it is ignored by the
        other window kinds.
        """
        if not isinstance(window, Window):
            raise TypeError(f"window must be a Window, got {type(window).__name__}")
        if window._kind == "last_days":
            assert window._days is not None  # set by Window.last_days
            return View(self._n.view_last_days(window._days, now))
        if window._kind == "range":
            assert window._from is not None and window._to is not None
            return View(self._n.view_range(window._from, window._to))
        assert window._parts is not None  # set by Window.partitions
        return View(self._n.view_partitions(window._parts))

    def all(self) -> View:
        """A view over every existing partition."""
        return View(self._n.all())

    def forget_before(self, cutoff: int) -> int:
        """Soft-forget: tombstone every record in partitions whose entire
        period precedes ``cutoff`` (unix seconds). Returns the number of
        frames tombstoned. Bytes are reclaimed at compaction.

        Acquires the single writer internally — do **not** call while holding
        an open :class:`valise.Writer`.
        """
        return self._n.forget_before(cutoff)
