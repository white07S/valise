# Contributing to Valise

Thanks for taking the time. This document covers how to build the project,
what the review bar is, and the few invariants that are not negotiable
because they are baked into the file format.

## Getting set up

```bash
git clone https://github.com/white07S/valise
cd valise
cargo build
cargo test
```

The toolchain is pinned in `rust-toolchain.toml` (stable). The minimum
supported Rust version is declared as `rust-version` in `Cargo.toml` and is
verified by CI — if you use a newer language feature, bump it in the same
pull request.

For the Python bindings:

```bash
cd bindings/valise-py
uv venv .venv && uv pip install --python .venv maturin
source .venv/bin/activate
maturin develop
uv pip install -e '.[docs,test]'
pytest -q python/tests
```

## Before you open a pull request

CI runs these; running them locally first is faster than a round trip.

```bash
cargo fmt --all --check
cargo clippy --all-targets
cargo test --all-targets
cargo build --no-default-features   # the library must build without the CLI
```

## Architecture in one screen

```
src/
├── lib.rs          Public API root; re-exports both API levels
├── db/             Application layer — Store, Schema, Record, Search
├── file/           Engine — ValiseFile, the single-writer file API
├── format/         On-disk structures: header, TOC, segments, catalog
├── codec/          Vector quantization (QAM Lloyd-Max, UPQ) + SIMD kernels
├── retrieval/      Lexical and vector scoring
├── concurrency/    N readers + 1 writer, snapshots, coordination region
├── text/           Analyzers: normalization, tokenization, stemming
└── io/             Low-level file primitives
```

Two API levels, and the distinction matters for review:

- **Application layer** (`db::Store`, `prelude`) — what applications and the
  Python bindings use. Changes here need a Python parity story.
- **Engine layer** (`ValiseFile`, `Database`) — catalog registration, frame
  and vector primitives. Reach for it when extending the format.

## Invariants

These are the rules that a change must not break. Most are enforced by
tests; all of them are enforced by review.

1. **Single file, no sidecars.** A capsule is one file. No lock files, no
   index directories, no temp files left behind. `tests/create_contract.rs`
   checks this.
2. **Crash safety.** Commits are footer-rooted and atomic: after any crash
   a reader sees either the previous committed state or the new one, never
   a mixture. `tests/crash_consistency.rs` fuzzes this with torn writes.
3. **Format changes are versioned.** `tests/golden_format_v2.rs` pins a
   BLAKE3 hash of a deterministic fixture. If your change flips that hash,
   the on-disk layout moved: confirm it was intentional, bump the format
   version, document it in `MIGRATION.md`, and only then re-generate the
   hash (the test prints the new one when it fails).
4. **Valise stores vectors; it does not produce them.** Embedding
   generation is out of scope. Do not add a model runtime as a dependency.
5. **Don't grow `src/file.rs`.** It is the engine entry point and is
   already split into `src/file/`. New engine behavior goes in a focused
   submodule.
6. **Don't add a dependency without justifying it.** Say in the pull
   request why the standard library or an existing dependency is not
   enough. The dependency tree is a feature.

## Style

- Follow the surrounding code. Match its naming, comment density, and idiom.
- Comments explain *why*, not *what*. If the code needs a comment to say
  what it does, the code is the thing to change.
- `thiserror` for error types, `tracing` for logging.
- Public items get doc comments. The crate denies `print_stdout`,
  `print_stderr`, and `dbg_macro` — diagnostics go through `tracing`, and
  the opt-in profiling output goes through the `prof_eprintln!` macro.

## Benchmarks

`bench/` is a separate crate and is not published. It pulls in peer engines
(tantivy, usearch, hnsw_rs) for head-to-head comparison. `bench/REPRODUCE.md`
documents how to regenerate every number, including the corpora to download.
Benchmark data, caches, and results are gitignored — they are all
regenerable, and they are large.

If you change a codec or a scoring path, include before/after numbers from
the relevant bench in the pull request.

## Reporting bugs

Open an issue with the smallest reproduction you can manage. For anything
involving data loss or corruption, please include the platform, filesystem,
and whether the process was killed mid-commit — that narrows it enormously.

Security issues go to the process in [SECURITY.md](SECURITY.md), not to the
public tracker.
