# valise

`valise` is the Python binding for **Valise** — a
single-file, append-only, crash-safe, multi-collection archive for AI and
retrieval workloads. One file packages raw payloads, a frame catalog, lexical
retrieval (BM25 / TF-IDF) with canonical statistics, quantized vector storage
(QAM / UPQ codecs), hybrid search, collection filters, and a footer TOC that is
the authoritative root of the active snapshot.

The Python surface is a **fully typed facade** over the Rust `valise` engine,
mirroring the Rust application API 1:1 — same nouns, same defaults (see
`docs/PARITY.md` in the engine repo). The public classes (`Store`, `Schema`,
`Record`, `Search`, `Reader`, `Writer`) are pure Python and forward to the
native layer in a **single call**, returning typed dataclasses (`Hit`,
`SearchResult`, `Stats`, `CompactReport`, `Stored`). The native classes remain
reachable as `valise._native` for debugging, but are not the public surface.

The API is **strict**: schemas, codecs, scorers, fusion, rerank, durability,
and language are selected with typed value objects and enums — raw strings are
not accepted.

## Install

```bash
pip install valise
```

## A minimal runnable example

One call declares the collection; text-only needs no embeddings and therefore
no calibration step.

```python
import os, tempfile
from valise import Store, Schema, Record, Search

path = os.path.join(tempfile.mkdtemp(), "demo.vls")
store = Store.open(path)                       # open-or-create
store.collection("kb", Schema().text("body"))  # English BM25 (default)

with store.writer() as w:
    w.put("kb", "a", Record().text("body", "the quick brown fox"))
    w.put("kb", "b", Record().text("body", "a lazy sleeping dog"))
    w.commit()

result = store.search("kb", Search().text("body", "fox").top_k(5))
print(result.keys)  # -> ['a']
```

Reopen just works: the schema is persisted in the file, so a second process
calls `Store.open(path)` and reads — no re-declaration.

For the full hybrid (text + vector) flow, see the [Quickstart](quickstart.md).

!!! note "Single source of truth for runnable examples"
    The load-bearing runnable example lives as a docstring in the `valise` package
    and is verified under `pytest --doctest-modules`. The Markdown snippets in
    these docs are illustrative; the docstring example is the tested one.
