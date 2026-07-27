# Valise File Format Status

Status: **v2.4 current implementation status**. This document is a compact
format map and source-of-truth index, not yet the full prose specification.

Current code line:

- `FORMAT_MAJOR = 2`
- `FORMAT_MINOR = 3`
- user-facing name: **Valise**

Valise is a single-file, append-only, crash-safe, multi-collection archive format
for AI and retrieval workloads. It is designed to package source payloads,
canonical lexical retrieval primitives, quantized vectors, schemas, filters,
time indexes, and recovery metadata into one `.vls` file.

## Source of Truth

Until the full prose spec is rewritten, the operational contract lives in these
files:

- [`src/file/commit.rs`](../src/file/commit.rs) — the commit protocol, with
  the crash-safety contract as a doc comment on `commit_phase_writes`.
- [`src/file/lifecycle.rs`](../src/file/lifecycle.rs) — open and recovery,
  including the scan-back path (`resolve_footer_state`).
- [`CONTRIBUTING.md`](../CONTRIBUTING.md) — architecture map and workflow
  rules.
- [`src/format.rs`](../src/format.rs) — active `FORMAT_MAJOR` /
  `FORMAT_MINOR` constants and format-module root.
- [`src/format/header.rs`](../src/format/header.rs) — fixed 4 KiB header.
- [`src/format/toc.rs`](../src/format/toc.rs) — self-validating TOC footer.
- [`src/format/segment.rs`](../src/format/segment.rs) — generic segment
  header and segment type registry.
- [`src/format/catalog.rs`](../src/format/catalog.rs) and
  [`src/format/catalog_codec.rs`](../src/format/catalog_codec.rs) — catalog
  descriptors, deltas, and ID allocation.
- [`tests/golden_format_v2.rs`](../tests/golden_format_v2.rs) — deterministic
  fixture hash that catches accidental byte-layout drift.

For a concrete byte-level walkthrough — a real 100,000-record capsule broken
down segment by segment, and compared against the same corpus in SQLite — see
[ANATOMY.md](ANATOMY.md).

If code and this page disagree, treat the code and golden tests as the active
contract, then update this page in the same change.

Note that source comments cite section numbers (`spec §12.2`, `§14.4`, and so
on) from the full prose specification, which is not published yet. Those
numbers have no anchor on this page — follow the file references above
instead.

## File Shape

```text
[ Header (4 KiB) ]
[ Segments (append-only) ]
[ TOC footer (active snapshot root) ]
```

The header stores the active `footer_offset`. Segments are appended; existing
persisted records are not modified in place. The TOC footer is the authoritative
root of the active snapshot.

Valise does **not** use a WAL in the current v2 line.

## Commit and Recovery

The v2 commit protocol is:

1. Take the cross-process writer lock in the coordination region.
2. Flush staged payloads and vector batches, then append the new segments
   to EOF.
3. Encode and append a TOC footer.
4. Publish `(footer_offset, snapshot_generation)` to the coordination
   region. This happens **before** the header write, so a reader that
   observes the coordination region never sees an offset the file cannot
   satisfy.
5. Rewrite the header's 120-byte logical prefix, updating `footer_offset`
   and `snapshot_generation`. The aligned 8-byte `footer_offset` store
   within that write is the atomic commit switch.
6. One `F_FULLFSYNC` for the whole commit. Per-step writes are forced to
   buffered mode so four barriers collapse into one; the configured
   durability mode governs writes *outside* commit, not within it.

Recovery, on open:

1. Read the header, seek to `header.footer_offset`, decode the TOC footer.
2. Validate the embedded BLAKE3 body checksum and full-footer checksum.
3. Check the footer against the create-contract digest at header byte 832,
   which anchors the footer to this file's lineage and rejects a footer
   that belongs to a different file.
4. Cross-check the header's `snapshot_generation` against the footer's. A
   torn 120-byte header rewrite shows up here as a disagreement.
5. Rebuild the in-memory view from the catalog chains the TOC references.

**If any of that fails** — footer missing, torn, checksum-invalid, past
EOF, generation-inconsistent, or one of its snapshot roots will not load —
recovery scans the whole file for candidate `VLTC` footers and selects the
highest-generation footer that validates completely. There is no stored
previous-footer pointer and no replay log: the anchor is re-derived from
the file itself. Decoy `VLTC` bytes inside payload data fail the
double-checksum test and are skipped.

## What v2.4 Ships

Storage and lifecycle:

- fixed 4 KiB header
- append-only segments
- catalog deltas
- TOC footer as the sole commit point
- BLAKE3 segment payload checksums
- BLAKE3 TOC self-checksums
- coordination region for process-shared access
- schema persistence through an internal `~valise.schema` collection

Text:

- canonical term dictionary
- frame-id-sorted postings
- document statistics
- BM25
- TF-IDF cosine and count cosine
- approximate cosine variants
- Dice, overlap, and containment set scorers

Vectors:

- `QamLloydMax` codec family
- `Upq` codec family
- vector data segments
- in-memory sign-sketch candidate scan derived from stored codes at file open
- codec-specific rerank paths

Application layer:

- keyed records
- persisted collection schemas
- text/vector/hybrid search
- RRF and weighted fusion
- time partitions and views
- tombstones and explicit compaction

## Vector Search Surface

Current vector search is **not** based on a persisted ANN graph, HNSW index, IVF
index, CSR vote index, or sidecar file.

At file open, Valise derives a dense sign-sketch from the persisted vector codec
bytes. For a space that has one — QAM(5,6) configurations and UPQ — the query
flow is:

1. Pack the query into the codec's query representation.
2. Run a sign-sketch Hamming scan over active vectors.
3. Keep a bounded candidate set controlled by `channel_k`.
4. Rerank candidates through the codec-specific path.
5. Optionally run a full reconstruction rerank for the accurate path.

A space with no sketch — any other codec configuration, or an empty space —
falls back to a full brute-force scan through the primary codec. That is
correct but linear in the corpus, so it is worth knowing which path a given
space is on (`src/file/vector_search.rs`).

See [`docs/VECTOR_SEARCH.md`](VECTOR_SEARCH.md) for the current benchmark
envelope, limits, and experiment history.

## Text Retrieval Surface

Valise persists text primitives, not an opaque Tantivy/Lucene index. The same
canonical postings and document statistics support multiple query-time scorers.

Application-level search currently exposes one text channel per query and one
or more vector channels, fused at the DB layer. Exact Jaccard exists at the
engine/retrieval level but is not part of the v1 application `Search` builder;
use Dice/Overlap/Containment in that surface.

## Compatibility

The format is pre-1.0. `tests/golden_format_v2.rs` pins a BLAKE3 hash of a
deterministic fixture, so any change to the byte layout is caught and has to
be deliberate — but below 1.0 a `FORMAT_MINOR` bump may reject older files at
open rather than migrating them. Every such change is recorded in
[MIGRATION.md](../MIGRATION.md).

Treat capsules as rebuildable from source data until a 1.0 stability policy
is declared.

## Known Limits

- The format prose spec is not complete.
- The CLI covers inspection and export (`info`, `search`, `get`,
  `export`); the library and Python package are the main surfaces for
  building on.
- The application schema surface is intentionally v1-shaped.
- Hard-delete / crypto-shred semantics are deferred; compaction reclaims bytes
  but is not a compliance erase primitive.
- Embedding generation is out of scope; Valise stores vectors and their contracts.
- Distributed serving, sharding, hosted multi-tenancy, and network protocols are
  out of scope for the file format.
