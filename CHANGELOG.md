# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

While the version is below 1.0, minor versions may contain breaking changes
to both the API and the on-disk format. Format-level changes are documented
in [MIGRATION.md](MIGRATION.md).

## [Unreleased]

## [0.1.4] — 2026-07-27

**Fixes a use-after-free that crashed vector search. Upgrade from any
earlier 0.1.x.** No format change — files are compatible in both
directions across all of 0.1.x.

### Fixed

- **Vector search could dereference freed memory after a commit that
  changed no vectors, crashing the process with SIGSEGV.**

  Every commit remaps the file, but the cache of vector base addresses —
  absolute pointers into that mapping — was only rebuilt when the commit
  had touched a vector, codec, or embedding space. A commit carrying only
  text or payload bytes left every cached address pointing into the old
  mapping. If the remap placed the file somewhere else, the next vector or
  hybrid query read freed memory.

  Reaching it needs nothing exotic: any capsule that holds vectors and
  takes a commit that adds none. Two collections where only one has a
  vector space is the obvious shape, and a single collection is enough if
  one record arrives without its vector. It does not depend on
  concurrency, and no data on disk is affected — the bug is entirely in
  the in-process read cache.

  The cache is now invalidated on those commits and lazily rebuilt by the
  next search, so commits that touch no vectors still do not pay to
  rebuild a vector index. Covered by
  `tests/regression_two_collection_vector_search.rs`, which reproduces all
  three shapes and fails with SIGSEGV without the fix.

  This was present in every published version (0.1.0 through 0.1.3) and
  predates the project's current name. It survived because the shape that
  triggers it — a vectorless commit followed by a vector query on the same
  handle — does not occur in the ingest-then-query pattern that the
  benchmarks and test suite were built around. It was found by using the
  library for something new.

## [0.1.3] — 2026-07-27

Storage introspection, and the evidence behind the durability claims. The
on-disk format is unchanged from 0.1.0 — files written by any 0.1.x are
readable by any other.

### Added

- `ReadConnection::storage_breakdown()` returns live bytes per segment type,
  so you can see where a capsule's size actually goes without external tools.
- `valise info --segments` prints the same accounting from the shell, with
  percentage shares, and `--segments --json` emits it machine-readably. The
  breakdown counts live segments only; the gap against the file size on disk
  is what `Store::compact` would reclaim.
- [`docs/ANATOMY.md`](docs/ANATOMY.md) — where the bytes go in a real
  100,000-record capsule, measured rather than described, with a component-wise
  comparison against SQLite + FTS5 + sqlite-vec. Includes the honest finding
  that the two lexical indexes are near-identical in size (9.7 vs 10.0 MiB):
  the 2.3× total advantage comes from quantized vectors and compressed
  payloads, not from beating FTS5.
- [`bench/CRASH_CAMPAIGN.md`](bench/CRASH_CAMPAIGN.md) — how to reproduce the
  crash-consistency results. Both harnesses already shipped but neither was
  documented, so the README's strongest claim had no path to its evidence.
  The 122,200-injection campaign is seeded, replays bit-for-bit, needs no
  downloaded data, and completes in about 100 seconds.
- A social preview card, so shared links render as something other than a
  grey placeholder.

### Changed

- Rewrote `long_lived_writer_does_not_block_concurrent_readers` to assert a
  structural property instead of a wall-clock one. It bounded both loops by a
  200 ms budget and then required the writer to land two commits inside it,
  which fails on a loaded runner whenever one `F_FULLFSYNC` runs long — a
  property the test was never meant to measure.

### Fixed

- Removed 18 commit hashes and 7 gitignored paths from `docs/VECTOR_SEARCH.md`
  that referenced a pre-rename repository and resolved to nothing. The
  experimental results stay; the false promise of traceability does not.
- Corrected seven `NXTC`/`NXSG` references in comments and error messages,
  left over from the pre-rename magic bytes. The constants themselves were
  already `VLTC`/`VLSG`.

## [0.1.2] — 2026-07-27

Documentation and positioning. No code changes; the on-disk format and the
API are identical to 0.1.0.

Note that 0.1.1 reached PyPI but never reached crates.io — the publish token
lacked the `publish-update` scope. This release brings both registries back
onto the same version.

### Changed

- Rewrote the README around what the project is actually for: shipping a
  retrieval corpus as a file, rather than a feature list. Leads with the
  copy-under-write result (50/50 mid-write copies opened to a correct
  snapshot, against 4/50 for a composed deployment and 0/50 for SQLite).
- Added the compression frontier at d=768 — every peer sorted by bytes,
  with recall and latency beside it. Nothing below 585 B/vector exceeds
  0.757 recall, and Valise is the only configuration under 3 KB/vector that
  clears 0.96 recall in under a millisecond.
- Gave the text engine its own section. Across four BEIR corpora of 2.7M–5.4M
  documents it is 32–43% smaller than Tantivy, better on recall@100 on all
  four, and better on nDCG@10 on three of four. Latency is stated as a regime
  (6–12× faster on short queries, converging on long ones, slower on FEVER)
  rather than as a single headline number.
- Documented that scoring happens at query time over persisted statistics, so
  a new scorer applies to files that already exist without re-indexing.
- Added worked examples: hybrid RAG, per-agent memory with recency decay,
  time-partitioned logs, and shipping an artifact through CI.
- Every code sample in the README, the crate docs, and the Python package
  description is executed before release.

## [0.1.1] — 2026-07-27

Documentation only. No code changes; the on-disk format and the API are
identical to 0.1.0.

### Changed

- Expanded the crate-level documentation, which is what docs.rs shows as
  the landing page: what Valise is for, when *not* to reach for it, a
  runnable quickstart, hybrid search, the durability and concurrency
  contract, platform support, and the CLI.
- Expanded the Python package description, which is what PyPI shows: the
  same material with worked examples for batch ingest and for reading a
  whole collection back. Both pages previously just linked elsewhere, and
  neither can be edited in place — PyPI renders the description from the
  uploaded distribution and docs.rs builds per version, so this release
  exists to publish them.

## [0.1.0] — 2026-07-27

First public release.

Valise packs documents, a lexical index, quantized vectors, and the schema
that describes them into one portable `.vls` file, and answers text, vector,
and hybrid search over it with no server and no sidecar index directory.

### Added

- **Application API** — `Store` / `Schema` / `Record` / `Search`: keyed
  records, schemas persisted in the file (a reopen needs no
  re-declaration), text / vector / hybrid search, partitions, compaction.
- **Engine API** — `ValiseFile` / `Database` for explicit catalog
  registration and raw format work.
- **Retrieval** — BM25, TF-IDF cosine, count cosine, approximate cosine
  variants, and Dice / overlap / containment set scorers over owned term
  dictionaries and postings. QAM Lloyd-Max and UPQ polar vector codecs
  with NEON and AVX2 kernels, sign-sketch candidate generation, and
  codec-specific rerank. RRF fusion by default.
- **Durability** — footer-rooted atomic commits with BLAKE3 segment
  checksums, fuzzed against torn writes. N readers and one writer,
  coordinated across processes through an in-header region.
- **`valise` CLI** — `info`, `search`, `get` (each with `--json`) and
  `export`, which streams a whole capsule to JSON Lines without touching
  the library.
- **Python bindings** — typed facade over the same application concepts,
  checked by `mypy --strict` and a Rust/Python parity suite.
- **Format v2.4** (`FORMAT_MAJOR = 2`, `FORMAT_MINOR = 3`), byte layout
  pinned by a golden-hash test and documented in
  [docs/FORMAT.md](docs/FORMAT.md).

### Known limitations

- `ReadConnection`'s search methods (`query_text`, `query_hybrid`,
  `vector_search`, `time_range_query`) are **not** served from the pinned
  snapshot. Catalog and payload reads are. A query issued after a
  concurrent commit can therefore return a frame the connection's own
  `frame_stubs()` does not list; call `refresh_snapshot()` if the two must
  agree. Pinning queries requires snapshotting the lexical and vector
  indexes, which is not done yet.
- Below 1.0, minor versions may break both the API and the on-disk format.
- Valise stores vectors; it does not generate embeddings, and it does not
  encrypt capsules.

[Unreleased]: https://github.com/white07S/valise/compare/v0.1.4...HEAD
[0.1.4]: https://github.com/white07S/valise/releases/tag/v0.1.4
[0.1.3]: https://github.com/white07S/valise/releases/tag/v0.1.3
[0.1.2]: https://github.com/white07S/valise/releases/tag/v0.1.2
[0.1.1]: https://github.com/white07S/valise/releases/tag/v0.1.1
[0.1.0]: https://github.com/white07S/valise/releases/tag/v0.1.0
