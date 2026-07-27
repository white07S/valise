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

- [`CONTRIBUTING.md`](../CONTRIBUTING.md) — architecture map, workflow rules,
  and the commit/recovery protocol.
- [`src/format.rs`](../src/format.rs) — active `FORMAT_MAJOR` /
  `FORMAT_MINOR` constants and format-module root.
- [`src/format/header.rs`](../src/format/header.rs) — fixed 4 KB header.
- [`src/format/toc.rs`](../src/format/toc.rs) — self-validating TOC footer.
- [`src/format/segment.rs`](../src/format/segment.rs) — generic segment
  header and segment type registry.
- [`src/format/catalog.rs`](../src/format/catalog.rs) and
  [`src/format/catalog_codec.rs`](../src/format/catalog_codec.rs) — catalog
  descriptors, deltas, and ID allocation.
- [`tests/golden_format_v2.rs`](../tests/golden_format_v2.rs) — deterministic
  fixture hash that catches accidental byte-layout drift.

If code and this page disagree, treat the code and golden tests as the active
contract, then update this page in the same change.

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

1. Encode new segments and append them to EOF.
2. Encode and append a TOC footer.
3. Atomically rewrite the aligned 8-byte `header.footer_offset`.
4. Flush according to the configured durability mode.

Recovery is:

1. Read the header.
2. Seek to `header.footer_offset`.
3. Decode the TOC footer.
4. Validate the embedded BLAKE3 body checksum and full-footer checksum.
5. Rebuild the in-memory view from the catalog chains referenced by the TOC.

If a crash leaves only a prefix of the new commit on disk, recovery falls back
to the previously published footer offset. There is no replay step.

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
bytes. Query flow is:

1. Pack the query into the codec's query representation.
2. Run a sign-sketch Hamming scan over active vectors.
3. Keep a bounded candidate set controlled by `channel_k`.
4. Rerank candidates through the codec-specific path.
5. Optionally run a full reconstruction rerank for the accurate path.

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

The format is still a private pre-v1 development line. The current
implementation and golden-format tests are the compatibility target for this
repository. Treat externally distributed files as rebuildable until a v1 format
stability policy is declared.

## Known Limits

- The format prose spec is not complete.
- The CLI is minimal; library and Python package are the main surfaces.
- The application schema surface is intentionally v1-shaped.
- Hard-delete / crypto-shred semantics are deferred; compaction reclaims bytes
  but is not a compliance erase primitive.
- Embedding generation is out of scope; Valise stores vectors and their contracts.
- Distributed serving, sharding, hosted multi-tenancy, and network protocols are
  out of scope for the file format.
