# valise

**Retrieval in one file.**

Your RAG prototype works. Now ship it — and suddenly the corpus isn't a thing
you have, it's a thing you *operate*. A vector database container. An index
directory that has to travel with the documents and stay consistent with them.
A rebuild step in CI. The most valuable artifact you built is the one piece you
can't hand to anyone.

SQLite solved this for relational data. **Valise does it for retrieval**:
documents, a BM25 index, compressed vectors, and the schema describing them, in
one `.vls` file you copy, version, and query in place.

```bash
pip install valise
```

```python
import numpy as np
from valise import Store, Schema, Record, Search, Vector

store = Store.open("kb.vls")                      # open-or-create
store.collection("notes", Schema()
    .text("body")                                 # English BM25 by default
    .vector("dense", Vector(dim=768)))            # cosine, auto-calibrated codec

with store.writer() as w:
    w.put("notes", "doc-1",
          Record().text("body", "portable retrieval capsules")
                  .vector("dense", embedding))    # float32 ndarray
    w.commit()                                    # the durability point

hits = store.search("notes", Search()
    .text("body", "retrieval capsule")
    .vector("dense", query)
    .top_k(10))

print(hits.keys)      # ['doc-1']
print(hits.scores)    # float32 ndarray
```

The schema lives **in the file**. A later run, another process, or another
machine calls `Store.open("kb.vls")` and searches immediately — nothing to
re-declare, nothing to migrate.

Note that leaving the `with` block releases the writer lock; it does **not**
commit. `commit()` is explicit, and it is the only durability point.

## Copy it while it's being written

That's the test that matters. Take a copy of a live corpus mid-write, move it
to a machine with a different OS and instruction set, open it there. On a
171,000-document hybrid corpus:

| | **Valise** | Tantivy + USearch | SQLite + FTS5 + vec |
|---|---:|---:|---:|
| Mid-write copies that opened correctly | **50 / 50** | 4 / 50 | 0 / 50 |
| Artifact | **239 MB, 1 file** | 677 MB, 62 files | 865 MB |

Top-10 results come back **identical** across macOS/ARM and Linux/x86-64, from
the same file, with no reconfiguration.

## There is no index to build

Every ANN library builds a graph first, and pays again whenever the corpus
changes. Valise builds nothing: vectors are stored only as quantized codes, and
the candidate-search sketch is derived from those same codes at open — in
memory, never written.

At 100k × 768d against recall-matched baselines: **0.50 s** to ingest and
commit, versus 50 s for FAISS HNSW, 91 s for USearch, 139 s for hnswlib.
**90–310× faster to build**, and nothing on disk but your data.

The bill: the scan is linear, so a mature HNSW answers individual queries
faster. If you build once and serve a billion queries over a static corpus, use
a graph. If your corpus changes, travels, or there are thousands of small ones,
this trade is the right way round.

Valise **does not generate embeddings** — bring your own model.

## What you get

- **Hybrid search.** Lexical and vector channels fused at query time, with
  reciprocal-rank fusion as the default. Text scorers include BM25, TF-IDF
  cosine, count cosine, approximate cosine variants, and Dice / overlap /
  containment.
- **Compact vectors.** QAM Lloyd-Max and UPQ polar codecs at ~5.5 bits per
  dimension, with NEON and AVX2 kernels. There is no persisted HNSW graph or
  IVF index — nothing to rebuild, and nothing on disk but the capsule.
- **Crash safety.** Commits are footer-rooted and atomic: after a crash a
  reader sees the previous committed state or the new one, never a mixture.
  Segment payloads carry BLAKE3 checksums.
- **Time partitions**, tombstones with explicit compaction, and recency as
  either a ranking signal or a hard filter.
- **A typed surface.** Full type hints, checked under `mypy --strict`, with
  strict enums rather than magic strings. Vectors cross the FFI boundary
  zero-copy.

## Batch ingest

`put_many` takes a C-contiguous `float32` `[N, dim]` array and ingests the
whole batch in one native call, rather than a per-row Python loop:

```python
vectors = np.ascontiguousarray(model.encode(bodies), dtype=np.float32)
with store.writer() as w:
    w.put_many("notes", keys, vectors, texts=bodies)
    w.commit()
```

## Reading everything back

Your data is never locked in. `keys()` plus `get()` walks a whole
collection:

```python
r = store.reader()
for key in r.keys("notes"):
    print(r.get("notes", key).text)
```

The `valise` command-line tool (`cargo install valise`) does the same from a
shell, including `valise export kb.vls > kb.jsonl`.

## Requirements

Python 3.9 or newer, and NumPy. Wheels are published for **Linux** and
**macOS** on x86-64 and aarch64.

**Windows is not supported yet** — the commit protocol relies on positional
file IO and fcntl advisory locks, and the Windows equivalents are not
implemented.

## Status

Pre-1.0 and under active development. Below 1.0, minor versions may break
both the API and the on-disk format.

## Links

- [Documentation](https://white07S.github.io/valise) — quickstart, concepts,
  full API reference
- [Source](https://github.com/white07S/valise) — the Rust engine, the on-disk
  format specification, and the benchmark methodology
- [Rust crate](https://crates.io/crates/valise)

Licensed under [MPL-2.0](https://github.com/white07S/valise/blob/main/LICENSE).
