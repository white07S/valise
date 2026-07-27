"""valise — Python bindings for the Valise single-file archive.

A fully typed Python facade over the Rust ``valise`` application API, with
the same nouns, defaults, and capabilities (one concept, two languages). The
public classes (:class:`Store`, :class:`Schema`, :class:`Record`,
:class:`Search`, :class:`Reader`, :class:`Writer`, :class:`Partitioned`) are
pure Python and forward to the native layer in a single call, returning typed
dataclasses instead of raw tuples and dicts.

The public API is **strict**: field specs, codecs, scorers, fusion, rerank,
durability, and language are selected with typed value objects and enums —
raw strings are not accepted. Defaults live in Rust: parameters left unset
resolve to the core defaults (English BM25 text, cosine QAM(5, 6) vectors
with auto-calibration, ``ACCURATE`` rerank, RRF fusion). The native classes
remain reachable as ``valise._native`` for debugging, but are not the public
surface.

Example:
    One call declares a collection; schemas are persisted in-file, so a
    reopen needs no re-declaration (mirror of the Rust quickstart).

    >>> import os, tempfile
    >>> from valise import Store, Schema, Record, Search
    >>>
    >>> path = os.path.join(tempfile.mkdtemp(), "kb.vls")
    >>> store = Store.open(path)                      # open-or-create
    >>> store.collection("notes", Schema().text("body"))
    >>>
    >>> with store.writer() as w:
    ...     w.put("notes", "a", Record().text("body", "the quick brown fox"))
    ...     w.put("notes", "b", Record().text("body", "a lazy sleeping dog"))
    ...     _ = w.commit()
    >>>
    >>> result = store.search("notes", Search().text("body", "fox").top_k(5))
    >>> result.keys
    ['a']
    >>>
    >>> store2 = Store.open(path)                     # second open: schema restored
    >>> store2.get("notes", "a").text
    'the quick brown fox'

    For vectors, add ``.vector("embedding", Vector(dim=768))`` to the schema
    (cosine, auto-calibrating QAM by default; pass ``codec=Upq()`` or
    ``Qam(amp_bits=..., phase_bits=...)`` to change it) and
    ``.vector("embedding", emb)`` to records and searches.
"""

from __future__ import annotations

from . import _native
from ._enums import Design, Durability, Lang, Metric, Rerank, TfMode
from .errors import (
    BusyError,
    CorruptionError,
    NotCalibratedError,
    ValiseError,
    SchemaMismatchError,
    UnsupportedError,
    ValidationError,
)
from .partition import Partition, Partitioned, View, Window
from .query import (
    Bm25,
    Containment,
    CountCosine,
    CountCosineApprox,
    Dice,
    HalfLife,
    Overlap,
    Range,
    Record,
    Rrf,
    RrfChannel,
    Search,
    TfidfCosine,
    TfidfCosineApprox,
    Weighted,
)
from .reader import Reader
from .schema import Auto, Now, Qam, Schema, Space, Text, Upq, Vector
from .store import Store
from .types import (
    CollectionInfo,
    CompactReport,
    Hit,
    SearchResult,
    SpaceInfo,
    Stats,
    Stored,
)
from .writer import Writer

format_version = _native.format_version
__version__ = _native.format_version()

__all__ = [
    # Core facades
    "Store",
    "Reader",
    "Writer",
    "Schema",
    "Record",
    "Search",
    # Schema field specs + codec/calibration choices
    "Text",
    "Vector",
    "Qam",
    "Upq",
    "Auto",
    "Now",
    "Space",
    # Partitioning
    "Partition",
    "Window",
    "Partitioned",
    "View",
    # Result types
    "Hit",
    "SearchResult",
    "Stats",
    "CompactReport",
    "Stored",
    "SpaceInfo",
    "CollectionInfo",
    # Enums
    "Metric",
    "Lang",
    "Durability",
    "Rerank",
    "TfMode",
    "Design",
    # Typed text scorers
    "Bm25",
    "TfidfCosine",
    "TfidfCosineApprox",
    "CountCosine",
    "CountCosineApprox",
    "Dice",
    "Overlap",
    "Containment",
    # Typed fusion
    "Rrf",
    "Weighted",
    # Typed recency
    "Range",
    "HalfLife",
    "RrfChannel",
    # Exceptions
    "ValiseError",
    "CorruptionError",
    "BusyError",
    "UnsupportedError",
    "ValidationError",
    "NotCalibratedError",
    "SchemaMismatchError",
    # Misc
    "format_version",
    "__version__",
]
