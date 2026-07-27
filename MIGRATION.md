# Valise format migration

**If you are starting with Valise today, you do not need this document.**
Every version below predates the first public release; there are no
published files in any earlier format, so there is nothing to migrate from.
It is kept because it records why the format looks the way it does, and it
sets the precedent for how format changes will be documented from here on.

While the version is below 1.0, a `FORMAT_MINOR` bump may reject older
files at open rather than migrating them. That will be stated here, in
[CHANGELOG.md](CHANGELOG.md), and in the release notes each time.

## Versions

| Version | `FORMAT_MAJOR` | `FORMAT_MINOR` | What it shipped |
|---|---|---|---|
| v0.1 | (pre-release) | — | per-row WAL durability, per-call fsync per mutation |
| v0.2 | 1 | — | group-commit (one WAL entry per `commit()`), avalanche fsync, redundant `Header.toc_checksum` |
| v2 | 2 | 0 | WAL eliminated; TOC footer is the sole commit point via atomic 8-byte `header.footer_offset` rewrite |
| v2.1 | 2 | 0 | Inner-segment wire optimizations (see below). `FORMAT_MAJOR` unchanged — the header layout is identical to v2; what moved is the contents of individual segment payloads. v2.0 files do not open under a v2.1 binary. |
| v2.2 | 2 | 1 | ANN profile burial. Catalog/TOC shape change: dropped `TocFooterBody.ann_profile_catalog_root`, `TocFooterBody.vector_ann_roots`, `EmbeddingSpaceDesc.ann_profile_id`, `IdAllocator.next_ann_profile_id`, `SegmentType::{AnnProfileCatalog, VectorAnn}`, `CatalogTableKind::AnnProfile`, and the `AnnProfileDesc` / `AnnEngine` / `AnnParams` types. `FORMAT_MINOR` bumped 0 → 1 so the header validator rejects v2.0/v2.1 files at open instead of letting bincode fail mid-decode of the TOC footer body. |
| v2.3 | 2 | 2 | **Vote-profile burial.** Vector search switched from the persisted CSR vote index to an in-memory **sign-sketch** (1 bit/dim, derived for free from QAM phase codes at file-open) + **QAM-sliding** rerank — no commit-time vector build, no persisted vote segment. Dropped: `TocFooterBody.vote_profile_catalog_root`, `CatalogSnapshot.vote_profiles`, `EmbeddingSpaceDesc.vote_profile_id`, `IdAllocator.next_vote_profile_id`, `SegmentType::{VoteProfileCatalog, VectorVoteIndex}`, `CatalogTableKind::VoteProfile`, and the `VoteProfileDesc` / `VoteProfileId` types. Also deleted: `retrieval/vote.rs`, `file/vote_io.rs`, `format/vote_index_segment.rs`, `auto_tier_at_commit`, the `resolve_vote_index` / `vote_index_cache` query path, and all `VALISE_VOTE_*` env knobs. `FORMAT_MINOR` bumped 1 → 2; v2.0–v2.2 files rejected at header validate. The runtime numbers: ~3× smaller `.vls` (617 vs 746 B/vec at dim 768), single-thread p50 1061µs at recall 0.97 (vs the old ~10-core p50 1182µs at recall 0.90 — sketch single-thread already beats vote multi-core, with higher recall). |
| v2.4 (current) | 2 | 3 | **UPQ codec family.** Adds `CodecFamily::Upq` (ordinal 2) — unrestricted polar quantization: one joint cell index per complex pair over `M` amplitude rings with power-of-two per-ring phase counts at the same 11 bits/pair as QAM(5,6). New params blob `VLUP` (`format/upq_params.rs`), registration via `register_codec_upq*`, search via sketch scan + decoded-i8 contiguous rerank (`file/upq_search.rs`). Measured on Cohere d=768: +0.6 recall pts accurate / +3.2 fast and ~2× faster queries vs QAM(5,6) at identical disk bytes. `FORMAT_MINOR` bumped 2 → 3; pre-v2.4 files rejected at header validate (regenerable bench corpora; no production migration tooling needed). QAM(5,6) remains the default codec — UPQ is opt-in per embedding space. |

## v2 → v2.1: inner-segment wire optimizations

Five storage-side wire changes, none of them altering `FORMAT_MAJOR` or the header layout. They all sit inside segment payloads and the catalog encode/decode paths.

1. **`FrameDesc.payload_checksum` removed.** The per-frame 32-byte BLAKE3 hash was redundant with the segment-level `SegmentHeader.payload_checksum` and the registry's `SegmentRef.checksum`. Wire change: the FrameCatalog encodes one fewer field per row.
2. **`TimeIndexSegment` v1 → v2 (columnar, magic still `VLTI`, version bumped to 2).** Row-of-`(u64 frame_id, u32 collection_id, i64 created_at, i64 updated_at)` replaced with frame_id varint deltas, RLE collection_id, delta-of-delta created_at, zigzag-offset updated_at. Quora-shape: 28 → 3 B/row.
3. **Batched `Payload` segments + zstd compression.** `put_frame` no longer emits one segment per call. Payloads buffer in memory up to `PAYLOAD_BATCH_FLUSH_BYTES` (4 MiB), then flush as a single zstd-level-3 compressed Payload segment. `PayloadRef.bytes_offset` / `bytes_length` slice the decompressed view. Quora-shape: 522 931 segments → 8 segments, 68.9 MiB → 11.1 MiB.
4. **`DocStatsSegment` v1 → v2 (columnar, magic `VLTS`, version 2).** Row-of-fixed-width-fields replaced with columnar layout: frame_id varint deltas, collection_id RLE, packed `field_lengths` with per-column u8/u16/u32 width tag, varint total/unique term counts. Quora-shape: 28 → 3.99 B/row.
5. **`CollectionFilterSegment` v1 → v2 (magic `VLCF`, version 2).** Now selects per-side between the old sorted-postings stream (`encoding = 0`) and a new run-encoded stream (`encoding = 1`) of `(start, length)` pairs, picking the cheaper at encode time. Single-collection builds (every BEIR corpus) collapse the entire 522 k-entry frame_id list into one run record. Quora-shape: 510.8 KiB → 105 B.
6. **`FrameCatalog` v1 (bincode-per-row) → v2 (`VLF2` magic, columnar).** Replaces the generic catalog envelope for the Frame table only. Other catalogs (TextSpace, Codec, etc.) still use the v1 bincode envelope. Wire: frame_id varint deltas; RLE for `collection_id` / `role` / `status` / `payload_encoding`; delta-of-delta `created_at`; zigzag-offset `updated_at`; per-column presence-list + packed values for the 7 sparse `Option<…>` columns; per-column varint streams for `payload_ref.{segment_id, bytes_offset, bytes_length}`. The v1 ghost-catalog "slim decode + lazy heavy load" path is preserved as a fallback for legacy bincode chains; v2 chains are decoded eagerly at open because the columnar block isn't randomly seekable. Quora-shape: 41.71 → 7.41 B/row.
7. **`PostingsSegment` v2 → v3 (tf-exception encoding).** Each posting's `(gap, term_freq)` collapses into one varint `(gap << 1) | (tf > 1 ? 1 : 0)`. When the flag bit is set, the actual `tf` is pulled from a parallel exception stream that follows the gap stream. `tf = 1` (95 %+ of postings on short-text corpora) costs zero bytes. Quora-shape postings: 19.66 → 16.16 MiB.

Cumulative Quora BM25 build (522 931 docs): **186.99 MiB → 36.06 MiB (−80.7 %)**, **375 B/doc → 72 B/doc**, retrieval scores bit-identical, BM25 build_s 4.00 → 2.03 s (−49 %), open_s 0.276 → 0.345 s (+25 %; eager v2 FrameCatalog decode trade-off), p50 query latency unchanged (0.670 → 0.677 ms).

## v0.2 → v2: what changed

- The embedded WAL region is **gone**. Header bytes 40..72 (formerly `wal_offset`, `wal_size`, `wal_checkpoint_seq`, `wal_head_seq`) are reserved-zero. Header bytes 80..112 (formerly `toc_checksum`) are reserved-zero.
- The 64 MB minimum WAL pre-allocation is gone — a fresh empty file is ~9 KB instead of ~64 MB.
- Mid-ingest auto-checkpoints (the WAL-pressure trigger that caused the auto-tier and BM25 corruption bugs) cannot fire because there is no WAL pressure to monitor. The user-explicit `commit()` is the only commit path.
- Recovery is now: read header → seek to `footer_offset` → validate the TOC footer's embedded self-checksum → done. No replay, no invalidation. The TOC has carried two embedded checksums (`body_checksum` over the body bytes, `footer_checksum` over header fields plus body) since v0.2; the only thing that changed in v2 is that the redundant `Header.toc_checksum` cross-check is gone, since the embedded checksums are sufficient.

## Why v0.2 files don't open

A v0.2 binary writes `format_major = 1`. A v2 binary's `Header::validate` rejects anything other than `format_major = 2` with:

```
unsupported format major version: 1 (binary expects 2). v1 (pre-WAL-elimination) files are not readable; see MIGRATION.md.
```

This is intentional. The two formats differ in:
1. Where segments start (v0.2: after the WAL region; v2: immediately after the 4 KB header).
2. Whether the WAL region exists at all.
3. What header fields exist.

A naive "ignore the version mismatch and keep going" reader would seek to the wrong byte ranges. Hard fail surfaces the issue at open() rather than corrupting recovered state.

## Why v2.0 files don't open under a v2.1 binary

Same `format_major = 2` but the inner segment payloads have moved. A v2.1 reader trying to walk a v2.0 file will fail loudly with one of:

```
TimeIndexSegment version mismatch: expected 2, got 1
DocStatsSegment version mismatch: expected 2, got 1
CollectionFilterSegment version mismatch: expected 2, got 1
PostingsSegment version mismatch: expected 3, got 2
FrameCatalog magic mismatch: expected VLF2, got [56 4c 43 43]  (VLCC = v1 bincode catalog)
```

These are all rejection paths inside the per-segment codec — the file header itself opens fine. Same migration story as v0.2 → v2: there is no in-place migration; rebuild from source (option A) or write a one-shot bridge (option B).

## Why v2.0/v2.1 files don't open under a v2.2 binary

v2.2 bumped `FORMAT_MINOR` from 0 to 1 and changed the bincode shape of `TocFooterBody` and `EmbeddingSpaceDesc` (ANN profile burial). The v2.2 header validator catches this at open:

```
unsupported format minor version: 0 (binary expects 1). v2.0 and v2.1 files
are not readable by a v2.2 binary; see MIGRATION.md.
```

The rejection happens before any catalog read, so bincode never sees a struct it can't decode. Rebuild from source data (option A in the next section) is the only migration path — every prior v2.x file holds either an empty ANN catalog chain (post-HNSW-removal binaries) or an HNSW chain we don't support anymore.

## Migration path

There is no in-place migration. The two formats have incompatible byte layouts; you'd be rewriting the whole file. Two practical options:

### Option A: rebuild from source data (recommended)

If you have the corpus and embeddings stored elsewhere (vector store, blob store, source documents), re-ingest into a fresh v2 file. This is the cleanest path and gives you a chance to update calibrations, codec choices, or auto-promotion thresholds.

```rust
let mut valise = ValiseFile::create_with_options(&new_path, CreateOptions::default())?;
// re-register codecs, embedding spaces, ANN profiles, fusion profiles
// re-ingest frames, vectors, text
valise.commit()?;
```

### Option B: write a one-shot bridge tool (only if needed)

If the corpus is not externally available, a bridge tool would:
1. Open the v0.2 file with a v0.2-compatible reader (kept in a maintenance branch — `git tag v0.2-final` before merging the v2 changes).
2. Walk every collection, frame, vector, and text index in the catalog.
3. Re-create them in a fresh v2 file via the public API.
4. `commit()`.

Valise does **not** ship this tool today. It would only be needed by a deployment that has v0.2 files it can't regenerate, and we have none. If you need one, the implementation is well under a day of work — the public API is preserved across the version boundary.

## What stays unchanged across v0.2 → v2

- Public types and method names (`ValiseFile`, `CreateOptions`, `PutFrame`, `PutVector`, etc.). The differences are in the deleted `wal_size` / `wal_mode` fields on `CreateOptions`, the deleted `wal_used_bytes()` / `wal_size()` accessors on `ValiseFile`, and the deleted `checkpoint_seq` field on `CommitOutcome`. Code that didn't touch those compiles unchanged.
- TOC footer body shape (catalog references, segment registry roots).
- Segment formats (vector data, ANN, vote-index, text indexes).
- Codec parameter encoding.
- Coord region layout (cross-process arbitration).
- Create-time contract digest at byte 832.

## Verifying a file's version

```console
$ xxd -l 8 /path/to/file.vls
00000000: 564c 5300 0200 0300                      VLS.....
#         ^^^^^^^^^ magic "VLS\0"
#                   ^^^^ format_major = 2 (LE u16)
#                        ^^^^ format_minor = 3 (LE u16)  -> v2.4
```
