# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

While the version is below 1.0, minor versions may contain breaking changes
to both the API and the on-disk format. Format-level changes are documented
in [MIGRATION.md](MIGRATION.md).

## [Unreleased]

### Added

- `valise` command-line tool with `info`, `search`, `get`, and `export`
  subcommands. Built by the default `cli` feature; library-only consumers
  can turn it off with `default-features = false`.
- `Reader::keys` — enumerate every committed key in a collection. Pairs
  with `Reader::get_many` to walk a capsule end to end, which is what
  `valise export` does.
- Continuous integration: formatting, clippy, tests, a no-default-features
  library build, an MSRV check, and a packaging check on Linux, plus a
  build and test pass on Apple silicon.

### Changed

- Minimum supported Rust version is now declared as 1.87.

### Removed

- Seven unused dependencies: `ed25519-dalek`, `base64`, `sha2`,
  `crc32fast`, `lz4_flex`, `once_cell`, and `papaya`. None were referenced
  from `src/`.

[Unreleased]: https://github.com/white07S/valise/commits/main
