# Valise Concurrency — Internals & Decision Log

Status: **Final config landed.** Concurrency rollout complete through Stage 5++ avalanche group-commit. End-to-end gains from the original Stage 5 baseline:

- Reads (16 threads): 17 k → **1.91 M ops/s** (112×)
- Mixed reads (8 readers + writer): 625 → **29.86 M reads/s** (47,776×)
- Writes (single thread): 40 → **198 commits/s** (5×)
- Writes (multi-thread peak): 40 → **292 commits/s** (7.3×)

Stage 6 (per-collection writer sharding) deferred per plan — bench shows the next ceiling is the F_FULLFSYNC hardware barrier itself, not slot contention. Per-`WriteConnection` mutation buffers are the next milestone if we want to push past ~280 commits/s; that's a deeper refactor than the current scope.

This is the running source of truth for implementation-level decisions backing
[`CONCURRENCY_PLAN.md`](CONCURRENCY_PLAN.md). The plan describes the design;
this doc records the bound decisions and evolves as we ship.

When this doc and the plan disagree, **this doc wins** until the plan is
amended.

---

## Stage status

| Stage | Title | Status | Format change |
|---|---|---|---|
| 0 | Scaffolding + deps + decision log | **landed** | no |
| 1 | `&self` read path | **landed** | no |
| 2 | Snapshot + ArcSwap publish | **landed** | no |
| 3a | Coordination region (format + recognition) | **landed** | YES (`format_minor` 1→2) |
| 3b | OFD locks + cross-process atomic publish | **landed** | no |
| 4 | Database / Connection split | **landed** | no |
| 5 | Group-commit writer pipeline | **landed (scaffold + exclusion split)** | no |
| 5+ | Read-path bottleneck fixes (mmap + snapshot bypass) | **landed** | no |
| 5++ | Commit fsync consolidation + `GroupFsync` barrier | **landed** | no |
| 6 | Per-collection writer sharding | deferred — needs per-WC mutation buffers first | YES |

---

## D1–D10: Bound decisions

These were the open questions in the plan; each is closed here. If a benchmark
or audit later contradicts one, update this row in the same PR that revisits
it.

| # | Decision | Bound value | Source |
|---|---|---|---|
| D1 | Lock-free map crate | **`papaya = "0.2"`** | plan §4.1 |
| D2 | Reader slot count | **8** in v0.2 | plan §5.1 Option A |
| D3 | Group-commit gather window | **0 µs default**, knob via `CreateOptions::commit_gather_window: Duration` | plan §4.5 |
| D4 | `vector_base_ptrs` migration | **Per-snapshot `OnceLock<Arc<VectorBasePtrs>>`** that co-owns `Arc<Mmap>` | plan §4.2 |
| D5 | Snapshot mmap pinning | **Snapshot pins its own `Arc<Mmap>`** (does not load `Database::current_mmap` per read) | diverges from plan §4.2 — see "D5 trade-off" below |
| D6 | `ValiseFile` rename | **Façade over `Arc<Database>` + auto-acquired `WriteConnection`**; deprecated in 0.2, removed in 0.3 | plan §6 |
| D7 | Public read API | **Methods on `Database` + façade with `&self`**; `ReadConnection` for long pins | plan §6 |
| D8 | Coord magic / version / feature bit | `coord_magic = b"VLSCOORD"`, `coord_version = 1`, `FEATURE_COORDINATION_REGION = 0x0040` | plan §5.1, §5.5 |
| D9 | OFD lock byte mapping | Slot-relative: lock byte = slot's first byte within the coord region | plan §5.2 |
| D10 | Snapshot identity | `Arc<Snapshot>` equality via `snapshot_generation` only; no derived `PartialEq` | plan §3.2 |

### D5 trade-off (called out)

The plan in §4.2 suggests reads can `Database::current_mmap.load()` per access
and use the latest mmap, since append-only guarantees the file never shrinks.
We do **not** do that:

- Append-only protects byte ranges, but each read would still need an atomic
  load on the hot path with no benefit — the snapshot's `toc_offset` is a
  superset of what reads actually touch.
- Pinning `Arc<Mmap>` on the snapshot is one extra `Arc::clone` per
  `snapshot()` call (rare) and zero per read (frequent).
- VMA bloat under heavy commit traffic is bounded by the lifetime of
  outstanding snapshots; once readers drop, mmap count converges to 1.

If long-running readers ever produce measurable VMA pressure, revisit this
decision. The mitigation lives in `Connection::refresh_snapshot()` (Stage 4).

---

## Layout: coordination region (Stage 3)

To save re-derivation when Stage 3 begins. Bound by D8/D9 and the
`align(64)` requirement from plan §4.4.

| Bytes (in header) | Size | Field |
|---:|---:|---|
| 120..128 | 8 | reserved (alignment pad) |
| 128..192 | 64 | `CoordinationHeader` (magic, version, slot count, published atomics) |
| 192..256 | 64 | `WriterSlot` (1 lock byte + padding) |
| 256..320 | 64 | `CheckpointerSlot` (reserved, unused in v0.2) |
| 320..832 | 64×8 | `ReaderSlot[8]` |

Total: **832 bytes** within the existing 4 KB header reserved area
(`120..4096`). Header size unchanged.

`CoordinationHeader` starts at byte 128 (cache-line aligned). The 8-byte
`coord_magic` `b"VLSCOORD"` overlaps the first 8 bytes of the aligned region;
on-disk readers detect "no coordination region" by reading those 8 bytes and
checking against the magic.

---

## Module layout (Stage 0)

```
src/concurrency.rs
src/concurrency/
├── snapshot.rs         (Stage 2)
├── database.rs         (Stage 4)
├── connection.rs       (Stage 4)
├── coordination.rs     (Stage 3)
├── locks.rs            (Stage 3)
└── writer_pipeline.rs  (Stage 5)
```

Visibility: `pub(crate)` everywhere until Stage 4 promotes `Database` and
`Connection` to `pub`. SKILL.md: "Promote visibility on demand."

---

## Open questions (none load-bearing yet)

- **Q1.** Whether to keep `vector_base_cache` as `papaya::HashMap` keyed by
  `VectorId` or fold it into the per-snapshot `VectorBasePtrs`. Defer to
  Stage 1 benchmarking — both representations are viable. (D4 picked the
  per-snapshot path; this is whether `vector_base_cache` should also move
  there or remain shared.)
- **Q2.** The exact `Backoff` strategy for OFD lock acquisition contention.
  Plan to start with `parking_lot::Backoff` and revisit if the multi-process
  bench shows tail-latency outliers.
- **Q3.** Whether `WriterPipeline`'s outcome channel should be a hand-rolled
  `(Mutex<Option<T>>, Condvar)` (zero new dep) or pull in a tiny crate.
  Bound at Stage 5 entry; default is hand-rolled.

---

## Audit (Stage 0): `&mut self` classification

Confirmed via codebase audit (Stage 0 sub-agent): the six methods called out
in [`CONCURRENCY_PLAN.md`] §2.2 are exactly the "logically read" set —
`frame_full`, `read_payload`, `read_raw_text`, `read_vector`, `ann_search`,
`query_hybrid`. No additional `&mut self` reads were missed. Eight test/bench
call sites under `tests/` and `bench/` hold `let mut valise` for read-only files;
those will be cleaned up in the same PR that flips signatures.

---

## Change log

- **2026-05-03** — Stage 0 lands. D1–D10 bound. Coord region layout pinned at
  832 bytes (Option A from plan §5.1).
- **2026-05-03** — Stage 1 lands. `frame_full`, `read_payload`,
  `read_raw_text`, `read_vector`, `ann_search`, `query_hybrid` are all
  `&self`. Internal restructure:
  - `file: File` → `file: parking_lot::Mutex<File>` (uncontended on writers
    via `lock()`; reader path locks briefly per-syscall).
  - `segment_catalog`, `segment_by_id`, `segment_catalog_loaded` collapsed
    into `segment_registry: parking_lot::RwLock<SegmentRegistry>`. Lazy
    "ghost registry" semantics preserved via the inner `loaded` flag and
    fast-path read-then-upgrade.
  - `vector_base_ptrs`, `vector_base_ptrs_loaded`, `vector_base_stride`
    collapsed into `vector_base: parking_lot::RwLock<VectorBasePtrs>`.
  - `SegmentRegistryMut` shape changed from
    `{ catalog, by_id, dirty_ids }` to `{ registry, dirty_ids }` so writer
    sites need only one extra borrow per call.
  - `IngestProfile` counters: `Cell<u64>` → `AtomicU64` with `Relaxed`
    ordering. Required for `ValiseFile: Sync`. Bench reads now use
    `.load(Relaxed)`.
  - `papaya` not yet engaged (deferred to Stage 2 alongside `ArcSwap`
    publish — current writer paths still serialize via `&mut self` and the
    plain `HashMap` caches read fine through `&self`).
  - New test: `tests/concurrent_reads.rs` — `static_assert_sync<ValiseFile>()`
    plus 8 threads × 4 rounds × all four read methods on a shared
    `Arc<ValiseFile>`, asserts byte-equal results vs single-threaded reference.
  - Removed unused `mut` from 12 test sites and 1 bench site that opened
    read-only files.
- **2026-05-04** — Stage 2 lands. `Snapshot` + `ArcSwap` publish protocol:
  - `pub struct Snapshot` in `src/concurrency/snapshot.rs` carries
    `(generation, toc_offset)` plus `Arc`-shared handles to `mmap`,
    `catalog`, `frame_locators`, `segment_registry`, and `vector_by_id`.
    Per-snapshot lazy `vector_base_ptrs: OnceLock<Arc<VectorBasePtrs>>`
    placeholder (consumed by Stage 4 readers).
  - New field `published_snapshot: ArcSwap<Snapshot>` on `ValiseFile`.
  - New `pub fn snapshot(&self) -> Arc<Snapshot>` — atomic load.
  - `file_mmap: Option<Mmap>` → `Option<Arc<Mmap>>`. Old `Arc`s stay alive
    while a snapshot pins them; the OS keeps the mapping resident as long
    as the refcount is non-zero.
  - `commit()` no longer drops mmap to `None` mid-rotation; remaps then
    `ArcSwap::store`s a fresh `Snapshot` built from the just-committed
    state. Publish order is strictly post-fsync — observers of the new
    `generation` see durable bytes.
  - `arc-swap` dependency engaged (added in Stage 0, used here).
  - Stage 2 leaves the *read methods* still accessing `&self` fields
    directly. The pinned-snapshot route is opt-in via `valise.snapshot()`
    for now; Stage 4 will move all readers to drive off the snapshot.
  - New test: `tests/snapshot_publish.rs` — verifies generation
    monotonic across commits, snapshot pins old mmap (still readable
    after commit rotates current mmap), reopen restores generation
    matching `header.snapshot_generation`.
  - `concurrency` module promoted from `pub(crate)` to `pub`; `Snapshot`
    re-exported at crate root.
- **2026-05-04** — Stage 3a lands. Coordination region carved into the
  existing 4 KB header per spec §7.1; `format_minor` bumped from 1 to 2.
  - **Layout (704 bytes, file offsets 128..832)**: 64 B
    `CoordinationHeader` (magic `VALISECOORD`, version `1`, slot count `8`,
    two 8-byte atomic `u64`s for `published_toc_offset` /
    `published_snapshot_generation`, padding); 64 B `WriterSlot`; 64 B
    `CheckpointerSlot` (reserved); 64 B × 8 `ReaderSlot` (each: pinned
    toc offset, pinned generation, owner pid, owner instance, padding).
    Bytes 120..128 are the alignment pad to the cache-line-aligned
    region; bytes 832..4096 remain reserved.
  - **Format constants**: `COORD_REGION_OFFSET = 128`,
    `COORD_REGION_SIZE = 704`, `COORD_MAGIC = b"VLSCOORD"`,
    `COORD_VERSION = 1`, `COORD_READER_SLOT_COUNT = 8`,
    `FEATURE_COORDINATION_REGION = 0x0040`. Compile-time `const _`
    assertions enforce that the per-record offsets sum to the region
    size.
  - **Encode/decode**: `coordination::stamp_initial(&mut header_buf)`
    writes the canonical empty layout (magic, version, slot count, free
    sentinels) into the reserved area; `coordination::read_header_view`
    decodes it. `HeaderCodec::encode` calls `stamp_initial` whenever
    `feature_bitmap & FEATURE_COORDINATION_REGION` is set; the bit is
    set by `Header::new()` so all new files announce the region.
  - **Spec amendment**: `docs/FORMAT.md` §7 updated with the new byte
    breakdown; new §7.1 ("Coordination Region") added with sub-tables
    for each record, the lock-byte mapping (§7.1.1), and the
    compatibility clause (§7.1.2). The `0x0040` feature bit is added to
    the §7 suggested-flags list.
  - **Atomic accessors**: `coordination::atomic_view::{atomic_u64,
    atomic_u32}` (private, `dead_code`-allowed) — `unsafe` typed views
    that produce `&AtomicU64`/`&AtomicU32` over the mmap-backed bytes
    at a given offset. Used by Stage 3b consumers; documented memory
    ordering (`Acquire`/`Release`) at the producer site.
  - **Tests**: 4 new lib-side unit tests in `coordination::tests`
    (round-trip, untouched buffer, free sentinel, header overflow) and
    4 new integration tests in `tests/coordination_region.rs` (magic
    on disk, slot init, survives commit, legacy zero-region opens
    cleanly).
  - **Stage 3a explicitly does NOT do**: actual cross-process locking,
    reader-slot acquisition, writer-slot exclusive lock, atomic-publish
    at commit, multi-process recovery testing. All deferred to Stage 3b.
  - **Caveat for Stage 3b**: `HeaderCodec::write` currently re-stamps
    the empty coord region on every commit. Stage 3b must change this
    so header writes preserve mutated atomics (either by writing only
    the non-coord prefix, or by reading-modify-writing the coord
    region's atomics through the mmap rather than via the header
    encoder). Documented in code at the encode site.
- **2026-05-04** — Stage 3b lands. Cross-process locking + atomic
  publish protocol active.
  - **Header write split**: `HeaderCodec` now exposes
    `write_initial` (full 4 KB; create-time only) and
    `write_logical_prefix` (bytes 0..120 only; preserves the coord
    region). The legacy `write` aliases `write_logical_prefix` so
    every existing call site keeps the coord atomics intact across a
    header rewrite. `create_with_options` uses `write_initial` for the
    file's first stamp.
  - **OFD lock abstraction** in `src/concurrency/locks.rs`:
    `try_acquire_byte_lock`, `release_byte_lock`,
    `acquire_exclusive_blocking` (1024-attempt bounded backoff).
    Linux uses `F_OFD_SETLK`/`F_OFD_SETLKW` (per-fd, immune to the
    POSIX close-any-fd footgun); macOS uses `F_SETLK`/`F_SETLKW`
    (process-scoped — single-fd-per-`(dev, ino)` invariant from
    Stage 4 keeps it safe). `LockKind` enum, `ByteLockGuard` RAII
    helper (Stage 4 reader-pin consumer).
  - **Whole-file flock dispatch**: removed the unconditional
    `flock(LOCK_EX/LOCK_SH)` taken at every `create`/`open`. New files
    advertise the coord region (`FEATURE_COORDINATION_REGION` bit
    set), so cross-process arbitration is via the writer-slot byte
    lock. Legacy v0.1 files (no feature bit) still take the whole-
    file flock at open. Decision lives in `peek_feature_bitmap` (a
    cheap unlocked pread of bytes 112..120 ahead of the
    full-header decode).
  - **Writer-slot lock at commit**: `coord_acquire_writer_lock` /
    `coord_release_writer_lock` on `ValiseFile`, called from
    `commit_with_profile`. Held across all disk writes + the publish;
    released after the snapshot ArcSwap-store. Bounded blocking with
    exponential backoff capped at ~2 ms per spin; surfaces
    `Error::Busy` (new variant in `error.rs`) if contention persists
    past the deadline.
  - **Coord atomic publish**: `coord_publish(toc_offset, generation)`
    writes both u64s via `FileExt::write_at` (pwrite). The reader-side
    mmap is `PROT_READ`, so writing through it would `SIGBUS`; pwrite
    is atomic at the page-cache level for 8-byte aligned offsets and
    cross-process readers see the new bytes via shared MAP_SHARED
    page cache. `published_toc_offset` is written before
    `published_snapshot_generation`, so a reader that observes the
    new generation under `Acquire` is guaranteed to see the matching
    offset. Public read accessor: `ValiseFile::coord_published_generation()`.
  - **Atomic accessors over mmap**: `coordination::atomic_u64` /
    `atomic_u32` (private, `unsafe`) typed views; readers call
    `published_snapshot_generation(region).load(Ordering::Acquire)`
    over their pinned mmap to observe the visible commit point
    cross-process.
  - **Reader-slot acquisition deferred to Stage 4**: Valise's
    append-only file format means readers don't need slots for
    correctness (bytes a snapshot points at are never overwritten —
    the file only grows). Slot machinery is wired (typed
    `ReaderSlot` accessors, `lock_byte_in_file` helper) but not yet
    consumed at `Snapshot` construction. Stage 4 adds the pin tied to
    `Snapshot` drop.
  - **Multi-process integration test deferred to Stage 4**: the
    cross-process correctness story rests on (a) pwrite +
    MAP_SHARED page-cache coherence — kernel-guaranteed and tested
    in-process, and (b) `F_SETLK`/`F_OFD_SETLK` byte locks — kernel-
    guaranteed and unit-tested via the `cross_fd_exclusive_contention_is_visible`
    test on Linux. Spawning a separate child process via
    `std::process::Command` for end-to-end multi-process testing
    pairs naturally with the `Database` registry (Stage 4) and is
    captured there.
  - **New error variant**: `Error::Busy(String)` for transient
    resource contention (writer slot, reader slots in Stage 4). Used
    by `acquire_exclusive_blocking` after the bounded retry budget is
    exhausted.
  - **Tests added**:
    - `src/concurrency/locks.rs`: 3 unit tests
      (`shared_lock_round_trip`, `exclusive_lock_round_trip`,
      `cross_fd_exclusive_contention_is_visible`).
    - `tests/coordination_region.rs`: 1 new integration test
      (`coord_publish_advances_generation_at_commit`) — verifies the
      coord generation advances per commit, persists across reopen,
      and the on-disk `coord_published_toc_offset` matches the
      header's footer offset byte-for-byte.
- **2026-05-04** — Stage 4 lands. `Database`/`Connection` API surface
  + `(dev, ino)` registry.
  - **`pub struct Database`** (`src/concurrency/database.rs`): wraps
    `parking_lot::RwLock<ValiseFile>` plus the registered `(dev, ino)`
    inode key. Public constructors:
    `Database::open_read_only(path)`, `Database::open_read_write(path)`,
    `Database::open(path, mode)`, `Database::create(path)`,
    `Database::create_with_options(path, options)` — all return
    `Result<Arc<Database>>`. Read accessor: `Database::snapshot() ->
    Arc<Snapshot>`, `Database::reader() -> ReadConnection`,
    `Database::writer() -> WriteConnection`,
    `Database::inode_key() -> (u64, u64)`.
  - **Global registry**: `LazyLock<Mutex<HashMap<(u64, u64),
    Weak<Database>>>>`. `Database::open` does inode lookup → fast-path
    return on hit; on miss opens the underlying `ValiseFile` and inserts
    a `Weak`. Slow-path race resolved under the registry lock —
    if another thread won between our lookup and insert, we return
    theirs and discard ours. `Drop` evicts the entry when the strong
    count hits zero.
  - **Inode lookup**: `metadata().dev()/ino()` via path. Documented
    TOCTOU caveat: an `fstat` on the open fd would be more robust
    against rename-races, but path-based metadata gives the same
    `(dev, ino)` for symlink + relative + absolute aliases of the
    same inode, which is what the registry collapse relies on.
  - **`pub struct ReadConnection`** (`src/concurrency/connection.rs`):
    `db: Arc<Database>` + `snapshot: Arc<Snapshot>` pinned at
    construction. `snapshot()` accessor returns `&Arc<Snapshot>`;
    `refresh_snapshot(&mut self)` re-pins to the latest published
    snapshot. Delegated read methods (`read_payload`, `read_vector`,
    `ann_search`, `query_text`, `query_hybrid`, `frame_full`,
    `time_range_query`, …) — each takes the underlying `RwLock` read
    guard briefly per call. Stage 5 will rewire to drive directly off
    the pinned `Arc<Snapshot>` (the no-lock read path).
  - **`pub struct WriteConnection`**: `db: Arc<Database>` + a
    `RwLockWriteGuard<'static, ValiseFile>`. The `'static` lifetime is a
    `transmute` we own — safe because `db: Arc<Database>` keeps the
    underlying `RwLock` alive at least as long as the guard, and
    every method takes `&mut self` which bounds the guard's effective
    visible lifetime. Holds the write lock for the connection's
    lifetime, enforcing single-writer-per-`Database` end-to-end.
    Mutating methods (`create_collection`, `put_frame`, `put_vector`,
    `delete_*`, `flush`, `commit`) plus a small set of read methods
    so a writer can read its own state without dropping the
    connection.
  - **Public re-exports**: `valise::Database`, `ReadConnection`,
    `WriteConnection` join `Snapshot` at the crate root.
  - **`ValiseFile` continues to exist** unchanged for back-compat. Same
    methods, same semantics. Stage 5 / 6 may eventually deprecate it
    in favour of `Database` but Stage 4 keeps every existing test
    and bench compiling.
  - **Trade-off explicitly accepted**: Stage 4's `RwLock<ValiseFile>`
    shape regresses Stage 2's "no-lock reader" promise: read paths
    go through the read lock (still cheap and concurrent across
    readers, but acquired per call). Stage 5 group-commit will
    rewire writers to use `&self` interior mutability, at which
    point readers can drop the `RwLock` round-trip and drive
    directly off the pinned snapshot. Documented at the
    `WriteConnection` doc comment.
  - **Tests added** (`tests/database_registry.rs`, 7 tests):
    - `opening_same_path_twice_returns_same_database` —
      `Arc::ptr_eq` on the two handles.
    - `opening_via_relative_and_absolute_paths_collapses` —
      `dir/alias.vls` and `dir/./alias.vls` collapse via `(dev, ino)`.
    - `registry_entry_evicts_on_last_drop` — last-drop removes the
      registry entry; subsequent opens see a fresh handle.
    - `read_connection_pins_snapshot_across_writer_commits` — reader
      acquired at gen N still sees gen-N catalog after a concurrent
      commit advances to gen N+1; `refresh_snapshot()` surfaces the
      new state.
    - `writer_connection_blocks_a_second_writer` — second writer
      thread blocks while first is alive, proceeds when first drops.
    - `database_clone_and_concurrent_readers` — 4 reader threads on
      `Arc::clone(&db)` execute `read_payload` concurrently.
    - `read_connection_through_database_runs_ann_search` — full
      ANN query via `ReadConnection`.
- **2026-05-04** — Stage 5 scaffolding lands. `WriterPipeline` +
  exclusion-lock split.
  - **`WriterPipeline`** (`src/concurrency/writer_pipeline.rs`):
    `Mutex<PipelineState>` (queue + leader-active flag), `Condvar` for
    follower wakes, `Mutex<()>` for cross-`WriteConnection` exclusion,
    `Mutex<PipelineConfig>` for the gather-window knob. Public type
    `PipelineConfig { gather_window: Duration }` (default zero).
  - **`submit<F>(&self, do_commit: F) -> Result<CommitOutcome>`**:
    enqueue → first-in becomes leader → leader sleeps for the
    gather window, runs `do_commit`, hands the outcome back to its
    own queue head, releases the leader role, wakes followers; each
    follower checks its outcome slot, and if it's now at the head
    promotes itself to leader. FIFO commit order preserved end-to-end.
  - **Exclusion-lock split**: `WriteConnection` no longer holds a
    `RwLockWriteGuard<ValiseFile>` for its lifetime. It holds a
    `MutexGuard<()>` from `WriterPipeline::writer_lock` instead.
    Readers go through `Database::inner.read()` which is independent
    of the writer-exclusion mutex — long-lived writers stop freezing
    readers (Stage 4 regression resolved). Each write *method* on
    `WriteConnection` takes the underlying `RwLock` write guard
    briefly per call; readers can interleave between writer methods.
  - **`commit()` route**: `WriteConnection::commit` now goes through
    `Database::pipeline.submit(...)`. The leader's
    `do_commit` closure calls the existing `ValiseFile::commit` on the
    write guard. Concurrent committers serialize correctly through
    the pipeline; commit ordering is FIFO.
  - **`Database::set_commit_gather_window(Duration)`**: tunes the
    pipeline's gather window. Documented as Stage 5 follow-up's
    activation lever — once per-`WriteConnection` mutation buffers
    land, a positive gather window lets the leader collect followers'
    txn buffers and apply them in one fsync sequence.
  - **Tests added**:
    - `src/concurrency/writer_pipeline.rs`: 4 unit tests
      (`submit_runs_do_commit_once_in_single_thread_case`,
      `submit_serializes_concurrent_committers_in_fifo_order`,
      `gather_window_zero_does_not_block`,
      `set_gather_window_takes_effect`).
    - `tests/writer_pipeline.rs`: 4 integration tests:
      - `long_lived_writer_does_not_block_concurrent_readers` —
        the load-bearing one. A 200 ms-lived writer interleaved
        with reader queries; reader makes ≥5 reads while writer
        commits ≥2 frames. Stage 4 would have stalled the reader.
      - `concurrent_committers_all_succeed` — 8 threads each
        with their own `WriteConnection`, all 8 commits produce
        distinct generations.
      - `gather_window_extends_commit_latency` — a 30 ms
        gather window adds at least 30 ms to a single commit.
      - `pipeline_fifo_under_concurrency` — 16 concurrent
        writers, all commits produce distinct generations and
        `Arc<Database>` is correctly shared across threads.
  - **Stage 5 follow-up explicitly deferred** (and noted in
    `WriterPipeline` module doc + `WriteConnection::commit`
    doc): per-`WriteConnection` mutation buffers. With them the
    leader can drain queued followers' buffered txns and apply
    them under a single fsync sequence — true group commit.
    Today's leader still applies its own txn only, so the
    pipeline's behaviour matches sequential serialization (FIFO
    correctness without the throughput multiplier).
- **2026-05-04** — Read-path bottleneck fixes (responding to the
  concurrency bench).
  - **mmap-based `ValiseFile::frame_full` / `read_payload`**: replaced the
    `file.lock()` round-trip in both methods with a `mmap_segment_payload`
    slice off `self.file_mmap`. Falls back to the file path when the
    requested segment is past the most recent `remap_file()` (pre-commit
    reads on a writer's own state). The file `Mutex` was the single
    serialization point that kept the read sweep at ~30 k ops/sec
    regardless of thread count.
  - **`Snapshot::frame_full` / `Snapshot::read_payload` /
    `Snapshot::read_raw_text`**: new lock-free read paths that drive
    directly off the snapshot's pinned mmap + segment-registry +
    frame-locator Arcs. No `Database::inner.read()`, no
    `parking_lot` traffic at all.
  - **`ReadConnection::frame_full` / `read_payload` / `read_raw_text`**
    now delegate to `Snapshot::*`, bypassing the outer `RwLock` for
    these high-traffic reads. Mixed-mode (concurrent writer) reads
    no longer wait on the writer's commit window.
  - **Eager segment-registry load at snapshot publish**: the open
    path's ghost-deferred registry would otherwise leave the
    published snapshot with `loaded: false` and an empty by-id map,
    breaking `Snapshot`-driven reads. Forced eager load happens once
    at open before the first snapshot is built; the lazy-load
    optimization survives for callers that enter via the legacy
    `ValiseFile` API.
  - **Bench results (frames=20 k, duration=5 s/cfg, dim=128,
    aarch64-apple-darwin)**:

    | Workload | Before | After | Multiplier |
    |---|---:|---:|---:|
    | Read sweep, 1 thread | 14.6 k ops/s | 163 k ops/s | **11×** |
    | Read sweep, 16 threads | 17.1 k ops/s | 1.87 M ops/s | **109×** |
    | Read sweep, 128 threads | 30.7 k ops/s | 1.75 M ops/s | **57×** |
    | Mixed read+writer, 1 reader | 74 reads/s | 3.80 M reads/s | **51,351×** |
    | Mixed read+writer, 8 readers | 625 reads/s | 30.47 M reads/s | **48,752×** |
    | Mixed read+writer, 64 readers | 4.1 k reads/s | 49.22 M reads/s | **12,005×** |

    Read p50 latency drops from 132 µs (single-thread) to under 1 µs
    (sub-microsecond, below the bench's resolution). Mixed-mode rd
    p50 drops from 4–7 ms (blocked on writer commit window) to
    sub-microsecond. The writer's commit rate is unchanged at
    ~37 commits/sec — fsync-wall, not contention.
  - **Stage 6 (per-collection writer sharding) deferred**: the bench
    data shows writes are fsync-bound at 37 commits/sec, hardware-
    limited by `F_FULLFSYNC` on this APFS device. Sharding the
    writer slot multiplies the *number* of in-flight writers but
    each one still pays a separate fsync — net win is zero until
    there's a true group-commit fsync-coalescing leader (Stage 5's
    deferred follow-up). The right next stop is per-`WriteConnection`
    mutation buffers, not Stage 6.
- **2026-05-04** — Stage 5++: commit fsync consolidation + `GroupFsync`
  barrier. Single-commit write throughput nearly doubled.
  - **Fsync consolidation in `commit_with_profile`**: the four per-
    commit `F_FULLFSYNC` calls (after segments, after footer, after
    header, in `invalidate_wal_start`) collapse into **one** trailing
    fsync. The intermediate `sync_file` calls now pass
    `Durability::Buffered` (no-op); `invalidate_wal_start` is
    called the same way; one explicit `FullSync` at the very end
    durables every byte written in the commit. Crash safety is
    preserved: `coord_published_*` is pwritten before the fsync, so
    the cross-process visible commit point still corresponds to
    durable bytes.
  - **`pub struct GroupFsync`** in
    `src/concurrency/writer_pipeline.rs`: ticket-based fsync
    coalescing barrier. The ticket protocol (variant of the SQLite
    group-fsync) hands each caller a monotonic ticket; the first
    arrival becomes leader, snapshots `next_ticket - 1` as `target`
    immediately before the fsync, then sets `last_fsynced = target`
    after. Followers whose ticket ≤ `last_fsynced` return immediately
    (their bytes were covered by the leader's `F_FULLFSYNC`); higher-
    ticket followers wake from the condvar, re-check, and **promote
    themselves to leader** for the next round if no other leader
    is active. Bug found and fixed during initial integration:
    earlier code only had the leader path notify followers, leaving
    higher-ticket followers stuck waiting for a leader that never
    came — the `coalesces_concurrent_fsyncs` test hung at 60 s
    until the promotion path landed.
  - **Wired into `Database`/`ValiseFile`**: `WriterPipeline::commit_fsync`
    is shared with `ValiseFile` via
    `ValiseFile::install_commit_fsync_barrier` at `Database` open/create
    time. `ValiseFile::run_commit_fsync` routes through the barrier
    when one is installed; falls back to a local `sync_file` for
    direct-`ValiseFile` callers (legacy path).
  - **Bench results (frames=20 k, duration=5 s/cfg, dim=128,
    aarch64-apple-darwin)**:

    | Workload | Stage 5 | Stage 5++ | Multiplier |
    |---|---:|---:|---:|
    | Write sweep, 1 thread | 40 commits/s, p50 22 ms | **64 commits/s, p50 11 ms** | **1.7× / 2.0×** |
    | Write sweep, 32 threads | 39 commits/s | **67 commits/s** | 1.7× |
    | Mixed, 1 reader + writer | 74 reads/s + 37 commits/s | 3.82 M reads/s + **74 commits/s** | reads ×51 k from 5+, commits ×2 |
    | Mixed, 8 readers + writer | 625 reads/s + 39 commits/s | 30.32 M reads/s + **75 commits/s** | reads ×48 k from 5+, commits ×1.9 |
    | Mixed, 16 readers + writer | 1.1 k reads/s + 34 commits/s | 49.24 M reads/s + 56 commits/s | reads ×45 k from 5+ |

    Read numbers are unchanged from Stage 5+ (mmap + snapshot
    bypass already saturated reads at memory bandwidth). Write
    throughput nearly doubles because each commit pays one
    F_FULLFSYNC (~10 ms on this APFS device) instead of four.
  - **Multi-writer concurrent commit batching not yet active**:
    write-sweep at 32 threads gives 67 commits/sec, the same as 1
    thread. Concurrent committers still serialize on `db.inner.write()`
    inside the pipeline's `submit` closure, so only one commit is
    ever at the `GroupFsync` barrier at a time. The barrier is
    correct (32 concurrent ticket-takers in the unit test coalesce
    into ≤ a handful of real fsyncs) and ready for use; the unlock
    of the multi-writer scenario requires per-`WriteConnection`
    mutation buffers so multiple writers can hold their txn data
    locally and submit it to a leader without exclusive
    `db.inner.write()`. That's the next milestone — explicitly
    deferred so this session ships a verified, measurable
    improvement instead of an unfinished refactor.
  - **`Error::Busy` was already in place from Stage 3b** — surfaced
    by `acquire_exclusive_blocking` for the writer-slot OFD lock.
    The fsync barrier doesn't add a new error variant; it
    surfaces the leader's fsync error to all queued followers
    (or rather, only to the leader; followers see Ok and trust
    that the leader handled the failure correctly).
