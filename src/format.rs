// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! On-disk Valise structures.
//!
//! This module mirrors `docs/FORMAT.md`; keep format-level changes traceable
//! to a spec section.

pub mod catalog;
pub mod catalog_codec;
pub mod collection_filter;
pub mod create_contract;
pub mod doc_stats;
pub mod dtype;
pub mod frame_catalog_codec;
pub mod header;
pub mod payload;
pub mod postings;
pub mod qam_lloyd_max_params;
pub mod registry;
pub mod segment;
pub mod term_dictionary;
pub mod text;
pub mod time_index;
pub mod time_index_segment;
pub mod toc;
pub mod upq_params;
pub mod vector;

/// Format major version.
///
///   - **v1 (FORMAT_MAJOR = 1):** v1 consolidation. Embedded WAL,
///     `Header.toc_checksum`, group-commit recovery via WAL replay.
///   - **v2 (FORMAT_MAJOR = 2, current):** Phase 4 WAL elimination.
///     The embedded WAL region is gone. `Header.toc_checksum` is gone
///     (the TOC footer is fully self-validating). Recovery is: read
///     header → seek to `footer_offset` → validate the TOC's embedded
///     self-checksum → done. The byte ranges 40..72 (former
///     `wal_offset` / `wal_size` / `wal_checkpoint_seq` / `wal_head_seq`)
///     and 80..112 (former `toc_checksum`) are now reserved-zero.
///
/// v1 files are NOT readable by a v2 binary; the version mismatch is
/// rejected at `Header::validate`. See `MIGRATION.md` for migration.
pub const FORMAT_MAJOR: u16 = 2;

/// Format minor version. Bumped on TOC-body / catalog-descriptor
/// shape changes that don't move segment boundaries.
///
///   - **0 (v2.0):** original v2 layout.
///   - **1 (v2.2, current):** ANN profile burial. Dropped
///     `TocFooterBody.ann_profile_catalog_root`,
///     `EmbeddingSpaceDesc.ann_profile_id`, and the `AnnProfile*`
///     descriptor types. Old v2.0/v2.1 files do not open under a v2.2
///     binary — the bincode decoders for `TocFooterBody` and the
///     catalog deltas detect the missing fields and fail loudly. See
///     `MIGRATION.md`.
///   * v2.3 (`FORMAT_MINOR = 2`): vote-profile burial. The CSR vote
///     index is gone (vector search is now sign-sketch + QAM-sliding
///     rerank, derived in-memory at open). Dropped: `TocFooterBody.
///     vote_profile_catalog_root`, `CatalogSnapshot.vote_profiles`,
///     `EmbeddingSpaceDesc.vote_profile_id`, `IdAllocator.next_vote_
///     profile_id`, `SegmentType::{VoteProfileCatalog, VectorVoteIndex}`,
///     `CatalogTableKind::VoteProfile`, and the `VoteProfileDesc` /
///     `VoteProfileId` types. Same rejection mechanism as v2.2.
///   * v2.4 (`FORMAT_MINOR = 3`): UPQ codec family. Adds
///     `CodecFamily::Upq` (ordinal 2; one 11-bit-class joint polar
///     cell index per complex pair, see `format::upq_params`) as a
///     sibling of `QamLloydMax`. Older readers reject the new enum
///     discriminant per spec §20; the exact-minor header check keeps
///     the failure loud and early. See `MIGRATION.md`.
pub const FORMAT_MINOR: u16 = 3;
pub const HEADER_SIZE: usize = 4096;

// ---- Coordination region (spec §7.1, Stage 3) ------------------------------
//
// Carved out of the header's reserved area starting at byte 128 (cache-line
// aligned). 704 bytes total = 64 (header) + 64 (writer) + 64 (checkpointer)
// + 64 × 8 (reader slots). Bytes 120..128 are the alignment pad; bytes
// 832..4096 remain reserved for future use.

/// First byte of the coordination region within the file header.
pub const COORD_REGION_OFFSET: usize = 128;
/// Total size of the coordination region in bytes (header + writer slot +
/// checkpointer slot + 8 reader slots, each 64 bytes).
pub const COORD_REGION_SIZE: usize = 704;
/// Byte offset of the writer slot within the coord region.
pub const COORD_WRITER_SLOT_OFFSET: usize = 64;
/// Byte offset of the checkpointer slot within the coord region.
pub const COORD_CHECKPOINTER_SLOT_OFFSET: usize = 128;
/// Byte offset of the first reader slot within the coord region.
pub const COORD_READER_SLOTS_OFFSET: usize = 192;
/// Cache-line size on every supported target (aarch64-apple-darwin, x86_64).
pub const COORD_CACHE_LINE: usize = 64;
/// Number of reader slots in v0.2. Bumpable in a later minor revision —
/// the slot table is sized by `coord_reader_slot_count` in the region's
/// own header, not by a constant in the consumer.
pub const COORD_READER_SLOT_COUNT: u32 = 8;
/// 8-byte magic at the head of the coord region. Readers detect "no
/// coord region" by comparing the first 8 bytes against this value
/// (or by checking the `feature_bitmap` bit, which is the canonical
/// signal).
pub const COORD_MAGIC: [u8; 8] = *b"VLSCOORD";
/// Coordination region wire version. v0.2 ships `1`; layout changes in
/// future minors bump this independently of `FORMAT_MINOR`.
pub const COORD_VERSION: u32 = 1;

/// Feature bit set in `Header::feature_bitmap` when the file includes a
/// valid coordination region (Stage 3 onward).
pub const FEATURE_COORDINATION_REGION: u64 = 0x0040;

/// Feature bit set when the file carries a `CreateContractV1` in its TOC
/// footer body and a 32-byte BLAKE3 digest in the header reserved area
/// at [`CREATE_CONTRACT_DIGEST_OFFSET`]. Phase 1 of the consolidation
/// plan. Files written by older binaries leave the bit clear; readers
/// synthesize a default in-memory contract for those.
pub const FEATURE_CREATE_CONTRACT: u64 = 0x0080;

/// Byte offset of the 32-byte create-contract digest within the file
/// header's post-coord reserved area. Cache-line aligned; the entire
/// digest lives outside the coordination region (`128..832`) so commit-
/// time `write_logical_prefix` calls (which only rewrite bytes 0..120)
/// preserve it untouched, matching the contract's immutable-after-create
/// semantics.
pub const CREATE_CONTRACT_DIGEST_OFFSET: usize = 832;
/// Length of the create-contract digest in bytes (BLAKE3-256).
pub const CREATE_CONTRACT_DIGEST_LEN: usize = BLAKE3_LEN;

pub const BLAKE3_LEN: usize = 32;

pub type Checksum = [u8; BLAKE3_LEN];

#[derive(
    Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Hash, serde::Deserialize, serde::Serialize,
)]
pub struct CollectionId(pub u32);

#[derive(
    Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Hash, serde::Deserialize, serde::Serialize,
)]
pub struct FrameId(pub u64);

#[derive(
    Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Hash, serde::Deserialize, serde::Serialize,
)]
pub struct TextSpaceId(pub u32);

#[derive(
    Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Hash, serde::Deserialize, serde::Serialize,
)]
pub struct AnalyzerId(pub u32);

#[derive(
    Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Hash, serde::Deserialize, serde::Serialize,
)]
pub struct FieldSchemaId(pub u32);

#[derive(
    Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Hash, serde::Deserialize, serde::Serialize,
)]
pub struct RetrievalProfileId(pub u32);

#[derive(
    Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Hash, serde::Deserialize, serde::Serialize,
)]
pub struct EmbeddingSpaceId(pub u32);

#[derive(
    Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Hash, serde::Deserialize, serde::Serialize,
)]
pub struct CodecId(pub u32);

#[derive(
    Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Hash, serde::Deserialize, serde::Serialize,
)]
pub struct FusionProfileId(pub u32);

#[derive(
    Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Hash, serde::Deserialize, serde::Serialize,
)]
pub struct VectorId(pub u64);

#[derive(
    Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Hash, serde::Deserialize, serde::Serialize,
)]
pub struct TermId(pub u32);

#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    Eq,
    Ord,
    PartialEq,
    PartialOrd,
    Hash,
    serde::Deserialize,
    serde::Serialize,
)]
pub struct SegmentId(pub u64);

pub(crate) fn bincode_config() -> impl bincode::config::Config {
    bincode::config::standard()
        .with_little_endian()
        .with_variable_int_encoding()
}

pub(crate) fn blake3_checksum(bytes: &[u8]) -> Checksum {
    *blake3::hash(bytes).as_bytes()
}
