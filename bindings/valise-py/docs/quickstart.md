# Quickstart

Valise keeps documents, their embeddings, and a keyword index in **one file** and
answers **hybrid** (keyword + vector) search over them. If you haven't yet, skim
the [data model](concepts/model.md) — schemas, collections, and records in one
diagram.

The [home page](index.md) has the minimal text-only example. This page adds a
vector channel and fuses the two.

## 1. Declare a collection

One call declares the fields. A text field defaults to the English BM25
analyzer; a vector field needs its dimension (your embedding model's width)
and defaults to cosine distance with an auto-calibrating QAM codec.

```python
import numpy as np
from valise import Store, Schema, Vector, Record, Search, Rrf

store = Store.open("library.vls")     # open-or-create, one file, no sidecars
store.collection("docs", Schema()
    .text("body")                     # English BM25 (default)
    .vector("dense", Vector(dim=64))) # cosine, auto-calibrating QAM(5,6)
```

`Schema().vector("dense", 64)` is shorthand for `Vector(dim=64)`. The dim must
be a positive multiple of 64.

## 2. Ingest documents with their embeddings

In real use the vectors come from **your** embedding model (sentence-transformers,
OpenAI, …) — Valise stores and searches them, it does not produce them.

`put_many` takes a C-contiguous `float32` `[N, dim]` array and ingests the whole
batch in a single native call (no per-row Python loop).

```python
bodies = [
    "Rust gives memory safety without a garbage collector.",
    "Python is a dynamic language popular for data science.",
    "Go uses goroutines for lightweight concurrency.",
]
vectors = np.ascontiguousarray(your_model.encode(bodies), dtype=np.float32)  # [3, 64]

with store.writer() as w:
    w.put_many("docs", ["rust", "python", "go"], vectors, texts=bodies)
    w.commit()   # commit is the durability barrier + the first calibration
```

The first commit that contains vectors fits the codec from the staged sample
(`Auto`, up to 50 000 vectors by default). Searching the vector field before
that commit raises `NotCalibratedError`.

## 3. Hybrid search

Combine a lexical channel and a vector channel; with two or more channels they
are fused with reciprocal-rank fusion (`Rrf(60)`) by default:

```python
result = store.search(
    "docs",
    Search()
    .text("body", "memory safety")          # scorer defaults to Bm25()
    .vector("dense", query_vec)             # rerank defaults to Rerank.ACCURATE
    .top_k(3),
)

for hit in result:
    print(hit.key, hit.score)
# the lexical channel alone already ranks 'rust' first for "memory safety"
```

`search` returns a [`SearchResult`](concepts/search.md): iterate it for
`Hit(key, score)`, or use `result.keys` / `result.scores` (a moved `float32`
array) for the columnar form.

## 4. Reopen — no re-declaration

The schema (fields, codec, calibration choice) is persisted in the file. A
second process, or a later run, just opens and reads:

```python
store = Store.open("library.vls")
store.get("docs", "rust").text          # just works
```

Re-declaring the same schema is an idempotent no-op; a *divergent* re-declare
(changed field, codec, or calibration) raises `SchemaMismatchError`.

## Choosing a codec (optional)

Quantization is per vector field. The default `Qam()` (QAM Lloyd-Max, 5+6
bits) is the production operating point; `Upq()` (unrestricted polar
quantization) is the higher-accuracy alternative:

```python
from valise import Qam, Upq

Schema().vector("dense", Vector(dim=768, codec=Upq()))          # 2048 cells
Schema().vector("dense", Vector(dim=768, codec=Qam(8, 8)))      # more bits, more recall
```

## Good to know

These are covered in [Concepts](concepts/model.md); the short version:

- **Commit before searching a vector field** — an auto-calibrating field fits
  its codec at the first commit containing vectors; until then vector search
  raises `NotCalibratedError`. See [Schemas & Spaces](concepts/spaces.md).
- **Vector scores are approximate** — the codec is lossy, so rank on keys, not
  on exact score equality. See [Search & Fusion](concepts/search.md).
- **Time-partitioned data** — `store.partitioned(...)` routes records into
  per-day/per-month collections with windowed views and `forget_before`. See
  [Partitions](concepts/partitions.md).
