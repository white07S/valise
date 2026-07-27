# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

While the version is below 1.0, minor versions may contain breaking changes
to both the API and the on-disk format. Format-level changes are documented
in [MIGRATION.md](MIGRATION.md).

## [Unreleased]

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

[Unreleased]: https://github.com/white07S/valise/compare/v0.1.2...HEAD
[0.1.2]: https://github.com/white07S/valise/releases/tag/v0.1.2
[0.1.1]: https://github.com/white07S/valise/releases/tag/v0.1.1
[0.1.0]: https://github.com/white07S/valise/releases/tag/v0.1.0
