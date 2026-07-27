# Rust ↔ Python parity — one concept per row

The application surface (`valise::prelude` ↔ `import valise`) is mirrored
1:1: same nouns, same defaults, same capabilities. Defaults live in **Rust
only** — Python passes `None`/sentinels and the native layer applies them.
Contract: `docs/SIMPLE_API_SPEC.md`. Rows whose Python side is marked
Both columns are implemented in the Rust/PyO3 binding and exercised by the
local parity suite. Published wheel availability is a release-process concern,
not a difference in the surface contract.

## Store lifecycle

| Concept | Rust | Python | Notes |
|---|---|---|---|
| Open (open-or-create) | `Store::open(path)?` | `Store.open(path)` | Default `Durability::Buffered`; `commit()` is the durability barrier. |
| Open with options | `Store::open_with(path, StoreOptions { durability })?` | `Store.open(path, durability=...)` | |
| Create (fail-if-exists) | `Store::create(path)?` / `create_with` | `Store.create(path, durability=...)` | |
| Durability | `Durability::{Buffered, SyncAll, FullSync}` | `Durability.{BUFFERED, SYNC_ALL, FULL_SYNC}` (`FSYNC` renamed `FULL_SYNC`, spec §6) | Store default is `Buffered`. |
| Reopen | schema persisted in-file — `Store::open` then read/search; `collection()` re-declare is an idempotent no-op | same | No re-declaration. Exception: a shared **deferred** vector space is not doc-reconstructable (spec §9c). |
| List collections | `store.collections() -> Vec<CollectionInfo>` | `store.collections() -> list[CollectionInfo]` | Never shows the reserved `~valise.schema`. |
| Engine escape hatch | `store.raw() -> Arc<Database>` | `valise._native` (debugging only) | Intentional asymmetry: Python exposes no engine API. |

## Schema declaration

| Concept | Rust | Python | Notes |
|---|---|---|---|
| Declare collection | `store.collection(name, Schema)? -> Collection` | `store.collection(name, Schema)` | Create-or-open + persist schema. Identical redeclare = no-op; additive = accepted; divergent = `SchemaMismatch`. |
| Schema | `Schema::new().text(name)` `.text_with(name, Text)` `.vector(name, Vector)` | `Schema().text(name, spec=None)` `.vector(name, Vector(...) \| int)` | Python uses an optional arg instead of `text_with` (spec §9e); `vector(name, 768)` int shorthand is Python-only sugar. |
| Text spec | `Text::english()` / `Text::raw()` / `Text::space(&Space)` | `Text(lang=None, space=None)` — `lang=None` resolves to English in Rust | Default analyzer: English (NFKC, case-fold, Porter2, stopwords). |
| Vector spec | `Vector::dim(u32)` `.metric(Metric)` `.codec(Codec)` `.calibrate(Calibrate)` / `Vector::space(&Space)` | `Vector(dim=None, *, metric=None, codec=None, calibrate=None, space=None)` — `None` fields resolve to the Rust defaults (cosine / `qam()` / `auto()`) | `dim` must be a positive multiple of 64. Exactly one of dim/space. Codec/calibrate/metric on a shared binding is a declaration-time error. |
| Metric | `Metric::{Cosine, Dot, L2}` | `Metric.{COSINE, DOT, L2}` | Default `Cosine`. |
| Codec | `Codec::qam()` / `qam_bits(a, p)` / `upq()` / `upq_cells(c)` / `upq_with(c, UpqDesign)` / `from_params(CodecParams)` | `Qam()` / `Qam(8, 8)` (both bits or neither) / `Upq()` / `Upq(cells=4096)` / `Upq(2048, Design.RAYLEIGH)` — omitted fields resolve in Rust | Default `qam()` = QAM(5, 6). `UpqDesign::{Empirical, Rayleigh}` ↔ `Design.{EMPIRICAL, RAYLEIGH}`. `from_params` is Rust-only (engine escape hatch). |
| Calibrate | `Calibrate::auto()` / `auto_sample(n)` / `now(Vec<Vec<f32>>)` | `Auto()` / `Auto(sample=8)` / `Now(ndarray)` — `Auto()` resolves to the Rust 50 000 default | Default `auto()` = fit at first vector commit from ≤ 50 000 staged vectors. `now` fits eagerly at declaration. |

## Spaces (shared tier)

| Concept | Rust | Python | Notes |
|---|---|---|---|
| Define shared space | `store.define_space(name, impl Into<FieldSpec>)? -> Space` | `store.define_space(name, Text(...) \| Vector(...))` | Idempotent by name; divergent re-define errors. |
| Bind in a schema | `Text::space(&s)` / `Vector::space(&s)` | `Text(space=s)` / `Vector(space=s)` | |
| Look up / list | `store.space(name) -> Option<Space>`; `store.spaces() -> Vec<SpaceInfo>` | `store.space(name)`; `store.spaces()` | `spaces()` lists shared **and** `~auto/{coll}/{field}` spaces (auto flagged). |

## Records and keys

| Concept | Rust | Python | Notes |
|---|---|---|---|
| Key | `Key::{Str, U64, Bytes}` — `impl Into<Key>` everywhere | `str \| int \| bytes` | Booleans / negative ints rejected in Python. |
| Record | `Record::new().text(f, &str).vector(f, &[f32]).at(unix_secs).child_of(Key)` | `Record().text(f, str).vector(f, float32 ndarray).at(unix_secs).child_of(key)` | Vectors borrowed zero-copy across FFI. `created_at` must be non-decreasing in commit order. |
| Read back | `Stored { key, collection, created_at, text, vectors }` | `Stored` dataclass, same fields | Vector round-trips are lossy (quantized) — never assert exact equality. |

## Writing

| Concept | Rust | Python | Notes |
|---|---|---|---|
| Writer | `store.writer() -> Writer` (also `writer_owned()` / `try_writer_owned()`) | `store.writer()` context manager | Single serialized writer; blocks until prior writer drops. Owned/try variants are Rust-only (FFI plumbing uses them internally). |
| Put / delete | `w.put(coll, key, Record)?` / `w.put_auto` / `w.delete(coll, key)?` | `w.put(coll, key, Record)` / `w.delete(coll, key)` | |
| Commit | `w.commit()? -> CommitOutcome` | `w.commit()` | The durability barrier **and** the first-calibration point for `auto` vector fields. |
| Bulk ingest | `w.bulk() -> Bulk` (group-commit loader) | `w.put_many(coll, keys, ndarray[N, dim], texts=...)` | Intentional asymmetry: `put_many` is Python-side sugar (one FFI crossing, zero-copy 2-D borrow) ≈ Rust `bulk()`. |
| Partition write | `w.put_into(&Partitioned, key, Record)?` | `w.put_into(partitioned, key, Record())` | Routes by the record `created_at` into `{base}:{period}`. |

## Search

| Concept | Rust | Python | Notes |
|---|---|---|---|
| Run a search | `store.search(coll, Search)? -> SearchResult` (or `reader().search`) | `store.search(coll, Search)` / `store.reader().search(...)` | |
| Text channel | `Search::new().text(field, query)` / `.text_with(field, query, TextScorer)` | `Search().text(field, query, scorer=Bm25())` | Default scorer `TextScorer::bm25()` = BM25(k1=1.2, b=0.75). One text channel per query in v1. |
| Text scorers | `TextScorer::bm25()` / `bm25_with(k1, b)` / `tfidf_cosine(TfMode)` / `tfidf_cosine_approx(TfMode)` / `count_cosine()` / `count_cosine_approx()` / `dice()` / `overlap()` / `containment()` | `Bm25(k1=1.2, b=0.75)` / `TfidfCosine(tf_mode)` / `TfidfCosineApprox(tf_mode)` / `CountCosine()` / `CountCosineApprox()` / `Dice()` / `Overlap()` / `Containment()` | Python `tf_mode` defaults to `LOG` (convenience; Rust requires it explicitly). |
| Vector channel | `.vector(field, &[f32])` / `.vector_with(field, &[f32], Rerank)` | `.vector(field, ndarray, rerank=Rerank.ACCURATE)` | Default `Rerank::Accurate` in **both** (Python's old `FAST` default is gone, spec §6). Repeatable for multi-vector search. |
| Rerank | `Rerank::{Fast, Accurate}` | `Rerank.{FAST, ACCURATE}` | `Accurate` = full reconstruction rerank of sign-sketch survivors. |
| Fusion | `.fuse(Fusion::rrf(k))` / `Fusion::weighted(text, vector)` | `.fuse(Rrf(k))` / `Weighted(text, vector)` | Default `Rrf { k: 60 }`; no-op for single-channel queries. |
| Recency | `.recency(Recency::range(from, to))` / `Recency::half_life(days)` / `Recency::rrf_channel(half_life_days, weight)` | `.recency(Range(from_, to))` / `HalfLife(days)` / `RrfChannel(half_life_days, weight)` | `Range` is **inclusive** `[from, to]`. `from_` trailing underscore: `from` is a keyword. |
| Pin "now" | `.now(unix_secs)` | `.now(unix_secs)` | For deterministic decay; defaults to wall clock. |
| Top-k | `.top_k(k)` | `.top_k(k)` | Default 10. |
| Result | `SearchResult` — `Deref<Target = [Hit]>`, `IntoIterator`, `.into_hits()` | `SearchResult` — `len` / index / slice / iterate; `.keys`, `.scores` (moved float32 ndarray) | `Hit { key, score, collection }`; higher score = better. Columnar `.keys`/`.scores` is Python-only sugar. |

## Partitions and views

| Concept | Rust | Python | Notes |
|---|---|---|---|
| Partitioned collection | `store.partitioned(base, Schema, Partition)? -> Partitioned` | `store.partitioned(base, Schema, Partition)` | Physical collections `{base}:{period}` share one schema (and the base's `~auto/{base}/{field}` spaces). |
| Routing | `Partition::{ByDay, ByMonth, Custom(fn)}` | `Partition.{BY_DAY, BY_MONTH}` | `Custom` is Rust-only (closures cannot cross the FFI boundary). | |
| Views | `p.view(Window)` / `p.view_as_of(Window, now)` / `p.all()` → `View` | `p.view(...)` / `p.all()` | `Window::{LastDays(n), Range { from, to }, Partitions(vec)}`. |
| Search a view | `store.search_view(&View, Search)?` | `store.search_view(view, Search)` | Global fusion across partitions. |
| Soft retention | `p.forget_before(cutoff)? -> usize` | `p.forget_before(cutoff)` | Tombstones whole time partitions; reclaim with `compact`. |

## Maintenance

| Concept | Rust | Python | Notes |
|---|---|---|---|
| Compact | `store.compact(CompactOptions)? -> CompactReport` | `store.compact(recalibrate=False)` → `CompactReport` | Rewrite live set + atomic swap; explicit only. `CompactOptions { recalibrate }` flattens to the Python kwarg. |
| Stats | `store.stats() -> StoreStats` (`tombstone_ratio()`, `needs_compaction(t)`) | `store.stats()` → `Stats` (`tombstone_ratio` field, `needs_compaction(t)`) | Counts cover user data only (no `~valise.schema` frames). |

## Errors

| Concept | Rust | Python | Notes |
|---|---|---|---|
| Not calibrated | `Error::NotCalibrated { space }` | `NotCalibratedError` | Vector search before the first vector commit; names the space (`~auto/{coll}/{field}` for inline fields). Fix: commit a batch containing vectors. |
| Schema mismatch | `Error::SchemaMismatch { collection, detail }` | `SchemaMismatchError` | Divergent redeclare (including a codec/calibrate change) with field-level detail. |
| Everything else | `Error` (thiserror) / `Result<T>` | `valise` exception hierarchy | |
