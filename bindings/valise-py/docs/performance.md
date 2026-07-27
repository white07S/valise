# Performance

The binding is **thin**. The boundary crossing itself is cheap (~50–70 ns per
PyO3 call); the cost that matters is copies, per-call allocation, and crossing
too often. The facade is designed around one hard guardrail:

> **Every public operation forwards to the native engine in a single native
> call.** `put_many` and `search` are never turned into a Python-side per-record
> loop. The `float32` bulk-in path stays zero-copy (a contiguous 2-D borrow);
> search scores come back as a moved `float32` ndarray, never a list of objects.

Because Valise's own operations are µs–ms, a well-built binding lands at
**0.01 %–1 %** overhead on everything except high-cardinality per-record ingest
— which is exactly why bulk ingest goes through `put_many`.

## Per-operation budget

The engine costs below are measured end-to-end; the binding overhead is the
achievable per-call cost.

| Operation | Valise engine cost (measured) | Binding overhead (achievable) | Ratio | Optimize? |
|---|---|---|---|---|
| Vector search (768-d, k=10) | ~1,200 µs p50 | 1 call (~50 ns) + 0-copy borrow in + columnar out | **<0.1 %** | No |
| Text / BM25 search | ~130 µs p50 | ~50 ns call + query-string copy + columnar out | **<1 %** | No |
| `get` by key | ~1–10 µs | ~50 ns + payload **move** (not copy) | <1 % | Only the return path |
| `commit` | ~4 ms (`F_FULLFSYNC`) | ~50 ns call | **~0.001 %** | No |
| `put` **one** record | ~1.4 µs (vec) / ~67 µs (text) | ~50 ns + arg parse + 3 KB copy (~300 ns) | 5–30 % on the cheap vector case | **Yes — via batching, not per-call** |
| `put_many` **N** records | N × engine cost | **one** crossing total + one 2-D borrow | **<1 %** amortized | This *is* the optimization |

## The two cliffs to avoid

1. **Per-record FFI loops.** Ingesting 100k vectors via 100k `put()` calls costs
   ~5 ms of crossings plus ~30 ms of per-row 3 KB copies. One
   `put_many(ndarray[N, dim])` crosses once and borrows the contiguous slice.
2. **Copying the f32 hot path at scale.** 3 KB copies are ~300 ns each — trivial
   once, but 1M vectors × 300 ns is ~300 ms of pure copy. The bulk path borrows
   the 2-D buffer; only the single async query vector is copied.

The budgets above are the contract. They are enforced by the benchmark in
`bindings/valise-py/bench/`, with `examples/binding_overhead.rs` as the
direct-Rust mirror that isolates FFI cost from engine cost.

## A note on docs as a single source of truth

The load-bearing runnable examples live as **docstrings** in the `valise` package
and are tested under `pytest --doctest-modules python/valise`. mkdocstrings renders
those same docstrings into the [API Reference](reference.md), so the rendered
examples and the tested examples are the same text. The Markdown snippets in the
concept pages are illustrative and are not executed by the doctest gate.
