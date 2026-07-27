# valise

**Retrieval in one file.** Text search, vector search, and your documents
packed into a single portable `.vls` file — no server, no sidecar index
directory, no external vector database.

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

## Why

Valise is for corpora that need to *move*: between agents, devices, eval
runs, customer environments, air-gapped deployments. What you ship is a
complete, queryable artifact rather than a service to stand up.

It is **not** a hosted vector database or a distributed search cluster, and
**it does not generate embeddings** — bring your own model. For live,
multi-tenant, continuously-updated corpora, reach for Qdrant, LanceDB,
Milvus, or an SQLite extension instead.

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
