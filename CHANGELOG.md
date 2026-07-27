# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

While the version is below 1.0, minor versions may contain breaking changes
to both the API and the on-disk format. Format-level changes are documented
in [MIGRATION.md](MIGRATION.md).

## [Unreleased]

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

[Unreleased]: https://github.com/white07S/valise/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/white07S/valise/releases/tag/v0.1.0
