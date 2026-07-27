<div align="center">

# valise

**Retrieval in one file.** Text search, vector search, and your documents —
packed into a single portable `.vls` file. No server, no sidecar index
directory, no external vector database.

[![CI](https://github.com/white07S/valise/actions/workflows/ci.yml/badge.svg)](https://github.com/white07S/valise/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/valise.svg)](https://crates.io/crates/valise)
[![docs.rs](https://docs.rs/valise/badge.svg)](https://docs.rs/valise)
[![PyPI](https://img.shields.io/pypi/v/valise.svg)](https://pypi.org/project/valise/)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

</div>

---

## Install

```bash
cargo add valise                    # Rust library
cargo install valise                # `valise` command-line tool
pip install valise                  # Python bindings
```

## Quickstart

```rust
use valise::prelude::*;

let store = Store::open("kb.vls")?;                 // open-or-create
store.collection("notes", Schema::new()
    .text("body")                                   // English BM25 default
    .vector("dense", Vector::dim(768)))?;           // cosine + QAM default

let mut w = store.writer();
w.put(
    "notes",
    "doc-1",
    Record::new()
        .text("body", "portable retrieval capsules")
        .vector("dense", &embedding),
)?;
w.commit()?;                                        // durability point

let hits = store.search("notes", Search::new()
    .text("body", "retrieval capsule")
    .vector("dense", &query)
    .top_k(10))?;
```

Schemas are persisted in the file. A later run — or a different process, or
a different machine — calls `Store::open("kb.vls")` and searches immediately,
with no schema re-declaration and nothing to migrate.

```bash
cargo run --example quickstart      # runs the above end to end, no data needed
```

<details>
<summary><b>The same thing in Python</b></summary>

```python
import numpy as np
from valise import Store, Schema, Vector, Record, Search

store = Store.open("kb.vls")
store.collection("notes", Schema().text("body").vector("dense", Vector(dim=768)))

with store.writer() as w:
    w.put("notes", "doc-1",
          Record().text("body", "portable retrieval capsules")
                  .vector("dense", np.asarray(embedding, dtype=np.float32)))
    w.commit()   # the durability point — leaving the block only releases the lock

hits = store.search("notes",
    Search().text("body", "retrieval capsule").vector("dense", query).top_k(10))
```

For a batch, `w.put_many(coll, keys, vectors, texts=...)` ingests a
C-contiguous `float32` `[N, dim]` array in one native call instead of a
per-row Python loop. See the
[Python quickstart](bindings/valise-py/docs/quickstart.md).

</details>

## Inspect a capsule from the shell

```console
$ valise info kb.vls
kb.vls
  size            11.0 KiB
  records         4 active / 4 total
  vectors         4 active / 4 total
  tombstones      0.0%
  collections     1
    - notes

$ valise search kb.vls notes "vector quantization" --top-k 3
  1. polar-q                          0.0164
  2. capsule                          0.0161

$ valise export kb.vls > kb.jsonl        # every record, as JSON Lines
```

`info`, `search`, `get`, and `export` all take `--json`. Your data is never
locked in: `export` walks the whole capsule with no library code.

## Why Valise

AI applications increasingly need context that can *move* — between agents,
devices, eval runs, customer environments, and air-gapped deployments.
Vector databases are strong for live shared services, but they are awkward
when the unit you want to distribute is a complete, queryable knowledge
artifact.

Reach for Valise when a corpus should be:

- **self-contained** — documents, text index, vectors, schemas, and metadata
  in one file
- **local-first** — query with no hosted service and no Docker sidecar
- **reproducible** — ship the identical retrieval artifact to tests, users,
  agents, or customer sites
- **crash-safe** — append-only commits rooted at a footer TOC, fuzzed against
  torn writes
- **hybrid** — lexical and vector search in the same embedded runtime

### What it is not

Valise is not a hosted vector database, a distributed search cluster, or an
embedding model. **It stores vectors; it does not produce them** — bring your
own model.

For live, multi-tenant, continuously updated production corpora, use Qdrant,
LanceDB, Milvus, Pinecone, Weaviate, Chroma, or an SQLite extension. Use
Valise when the retrieval corpus itself should be portable, inspectable,
reproducible, or embedded directly into an application.

## Retrieval model

Valise owns its retrieval primitives rather than embedding an opaque search
engine blob.

**Text** — BM25; TF-IDF cosine and count cosine; approximate cosine variants;
Dice, overlap, and containment set scorers; canonical term dictionaries,
postings, and document statistics.

**Vectors** — QAM Lloyd-Max polar codec (default `(5, 6)` bits); UPQ polar
codec family, opt-in per field; an in-memory sign-sketch candidate scan
derived from the stored codes; codec-specific rerank paths. There is no
persisted HNSW graph or IVF index — and therefore no index to rebuild, and
nothing on disk but the capsule.

Hybrid search fuses the two channels at query time, with RRF as the
application-layer default.

## API levels

| Level | Entry point | Use it for |
|---|---|---|
| Application | `valise::prelude` / `db::Store` | keyed records, schemas, text/vector/hybrid search, partitions, compaction |
| Engine | `ValiseFile` / `Database` | explicit catalog registration, frame and vector primitives, raw format work |
| Python | `valise` on PyPI | typed Python facade over the same application concepts |

Start at the application layer. The engine is for extending the format or
embedding Valise inside another storage system.

## Status

Pre-1.0 and under active development.

- **Format** — on-disk line v2.4 (`FORMAT_MAJOR = 2`, `FORMAT_MINOR = 3`).
  Byte layout is pinned by a golden-hash test; changes are documented in
  [MIGRATION.md](MIGRATION.md).
- **Concurrency** — N readers and a single leader-batched writer, coordinated
  across processes through an in-header region with mmap snapshots.
- **Stability** — while the version is below 1.0, minor versions may break
  both the API and the on-disk format.

## Documentation

| Document | Contents |
|---|---|
| [docs/FORMAT.md](docs/FORMAT.md) | On-disk format, commit and recovery protocol |
| [docs/SIMPLE_API_SPEC.md](docs/SIMPLE_API_SPEC.md) | Application API contract |
| [docs/VECTOR_SEARCH.md](docs/VECTOR_SEARCH.md) | Vector-search design, recall/latency envelope, failed experiments |
| [docs/EXTENDING.md](docs/EXTENDING.md) | Adding scorers or codec families |
| [docs/CONCURRENCY.md](docs/CONCURRENCY.md) | Reader/writer model and decision log |
| [docs/PARITY.md](docs/PARITY.md) | Rust ↔ Python surface parity |
| [bench/REPRODUCE.md](bench/REPRODUCE.md) | How to regenerate every published number |

## Repository layout

| Path | Contents |
|---|---|
| `src/db/` | Application layer: `Store`, schema registry, identity, records, search, partitions, compaction |
| `src/file/` | Engine lifecycle: catalogs, segments, codecs, text indexing, vector search, TOC IO |
| `src/format/` | On-disk codecs and persisted structs |
| `src/codec/` | Vector codec families and SIMD kernels |
| `src/retrieval/` | Lexical scorers, fusion, top-k, sign-sketch scan |
| `src/concurrency/` | Readers, writer pipeline, snapshots, coordination region |
| `bindings/valise-py/` | Python package and PyO3 bindings |
| `bench/` | Benchmark harnesses, peer comparisons, reproducibility notes |

## Building from source

```bash
cargo build
cargo test
cargo clippy --all-targets
cargo fmt --all --check
```

Python bindings:

```bash
cd bindings/valise-py
uv venv .venv && uv pip install --python .venv maturin
source .venv/bin/activate && maturin develop
pytest -q python/tests
```

## Contributing

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md) for the
build steps, the review bar, and the format invariants a change must not
break. Security issues go through
[private vulnerability reporting](https://github.com/white07S/valise/security/advisories/new),
not the public tracker; see [SECURITY.md](SECURITY.md).

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))

at your option.

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in this work by you, as defined in the Apache-2.0
license, shall be dual licensed as above, without any additional terms or
conditions.
