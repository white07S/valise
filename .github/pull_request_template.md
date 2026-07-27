## What this changes

<!-- One or two sentences. Link the issue if there is one. -->

## Why

<!-- The problem being solved, not the diff. -->

## Checklist

- [ ] `cargo fmt --all --check` passes
- [ ] `cargo clippy --all-targets` reports no errors
- [ ] `cargo test --all-targets` passes
- [ ] `cargo build --no-default-features` still builds the library

## Format impact

- [ ] This change does **not** alter the on-disk layout
- [ ] This change **does** alter the on-disk layout, and I have:
  - [ ] bumped the format version
  - [ ] documented it in `MIGRATION.md`
  - [ ] re-generated the hash in `tests/golden_format_v2.rs`

<!-- If tests/golden_format_v2.rs failed, that is the signal the layout
     moved. Do not paste the new hash in without confirming the change was
     intentional. -->

## Performance impact

<!-- If you touched a codec, a scoring path, or the commit path, include
     before/after numbers from the relevant bench. Delete this section if
     the change cannot affect performance. -->

## Python parity

<!-- Application-layer (db::Store) changes usually need a matching change
     in bindings/valise-py. Delete if not applicable. -->

- [ ] Not applicable — this does not touch the application layer
- [ ] Python binding updated, `pytest -q python/tests` passes
