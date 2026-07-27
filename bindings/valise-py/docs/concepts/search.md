# Search & Fusion

The [`Search`](../reference.md) builder assembles a hybrid query — a lexical
(text) channel, one or more vector channels, or both — and a fusion strategy
that combines them. Every builder method forwards exactly one native call and
returns `self` for chaining; the facade never re-walks the query in Python.

```python
from valise import Search

q = (
    Search()
    .text("body", "quick fox")        # scorer defaults to Bm25(k1=1.2, b=0.75)
    .vector("dense", query_vec)       # rerank defaults to Rerank.ACCURATE
    .top_k(10)                        # default 10
)
result = store.search("kb", q)
```

Defaults live in the Rust core (the single source of truth): BM25 text
scoring, accurate vector rerank, and `Rrf(60)` fusion when two or more
channels are present. Override any of them explicitly:

```python
from valise import Search, Bm25, Rerank, Rrf, Weighted

Search().text("body", "quick fox", Bm25(k1=0.9, b=0.4))
Search().vector("dense", query_vec, Rerank.FAST)   # keep quantized scores
Search().text(...).vector(...).fuse(Weighted(text=0.3, vector=0.7))
```

`.vector(...)` may be called multiple times for multi-vector / multimodal
search; each call becomes a fusion channel. At most one text channel in v1.

## Strict typed value objects — raw strings are rejected

The text scorer, fusion strategy, and rerank precision are **typed value
objects / enums**. Passing a raw string raises `TypeError`.

**Text scorers** (the `TextScorer` union — referenced inline here, not a public
class): `Bm25(k1, b)`, `TfidfCosine(tf_mode)`, `CountCosine()`, `Dice()`,
`Overlap()`, `Containment()`.

**Fusion** (the `Fusion` union): `Rrf(k)` for reciprocal-rank fusion, or
`Weighted(text, vector)` for a linear weighted blend of the two channel kinds.

**Rerank precision**: the `Rerank.FAST` / `Rerank.ACCURATE` enum on the vector
channel. `ACCURATE` (the default) reranks the candidate set with full-precision
vectors; `FAST` keeps the quantized scores.

```python
# Correct — typed objects:
Search().text("body", "fox", Bm25()).fuse(Rrf(60))

# Rejected — raw strings raise TypeError:
# Search().text("body", "fox", "bm25").fuse("rrf")
```

Searching a vector field before its first vector commit raises
`NotCalibratedError` — see [Schemas & Spaces](spaces.md).

## SearchResult

`search` returns a [`SearchResult`](../reference.md): parallel `keys` and
`scores` built from the native `(keys, scores)` tuple in one native call. The
`scores` ndarray is the moved `float32` array, never copied.

```python
len(result)        # number of hits
result.keys        # list of keys (native list, as-is)
result.scores      # float32 ndarray (moved, zero-copy)
result[0]          # a Hit(key, score, collection)
result[:3]         # `SearchResult` (sliceable; slicing returns another `SearchResult`) (slicing supported)
for hit in result: # lazy iteration; Hit objects materialized on access
    print(hit.key, hit.score)
```

Each [`Hit`](../reference.md) is a frozen dataclass with `key`, `score`, and
`collection`. Higher score is better. Hits are materialized lazily on
indexing/iteration — there is no eager per-hit loop over the result set.

!!! note "Scores are not reproducible"
    When a vector channel is present, fused scores depend on the lossy codec
    and an RNG, so exact score values are not reproducible. Assert on counts and
    keys, not on score equality.
