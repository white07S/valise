// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! High-level single-file Valise lifecycle.

pub(crate) use std::{
    collections::{HashMap, HashSet},
    fs::{File, OpenOptions as FsOpenOptions},
    path::{Path, PathBuf},
};

pub(crate) use std::sync::Arc;

pub(crate) use arc_swap::ArcSwap;
pub(crate) use parking_lot::{Mutex, RwLock};

pub(crate) use crate::concurrency::snapshot::Snapshot;

pub(crate) mod catalog_io;
pub(crate) mod codec_io;
mod filter_io;
pub(crate) mod query_admission;
pub(crate) mod query_arena;
pub(crate) mod segment_io;
pub(crate) mod text_indexing;
mod time;
mod time_index_io;
mod toc_io;
mod upq_search;
mod vector_search;

pub(in crate::file) use catalog_io::{
    dirty_records, read_catalog_snapshot, read_segment_catalog, upsert_collection, upsert_frame,
    upsert_vector,
};
pub(in crate::file) use codec_io::build_codec_cache;
pub(in crate::file) use segment_io::{
    SegmentRegistry, SegmentRegistryMut, append_registered_segment, append_segment_at_end,
    build_segment_map, legacy_offset_segment_ref, read_payload_ref, register_segment,
    segment_catalog_entry, segment_ref_at_end, verify_payload_wire_bytes,
};
pub(in crate::file) use time::current_unix_timestamp;
pub(in crate::file) use toc_io::{
    FooterCandidate, read_toc_footer, scan_for_footer_candidates, write_toc_footer_at_end,
};
pub(crate) use vector_search::VectorBasePtrs;
pub(in crate::file) use vector_search::build_vector_base_ptrs;

pub(crate) use uuid::Uuid;

pub(crate) use crate::{
    codec::VectorCodec,
    error::{Error, Result},
    format::{
        AnalyzerId, CodecId, CollectionId, EmbeddingSpaceId, FieldSchemaId, FrameId,
        FusionProfileId, RetrievalProfileId, SegmentId, TextSpaceId, VectorId,
        catalog::{
            AnalyzerDesc, CatalogSnapshot, CodecDesc, CodecFamily, CollectionDesc,
            EmbeddingSpaceDesc, FieldSchemaDesc, FrameDesc, FrameRole, FrameStatus, FrameStub,
            FusionProfileDesc, PayloadEncoding, RetrievalProfileDesc, TextSpaceDesc, VectorDesc,
            VectorMetric, VectorStatus,
        },
        catalog_codec::{CatalogTableKind, encode_catalog_delta},
        create_contract::CreateContractV1,
        dtype::DtypeSet,
        header::{Header, HeaderCodec},
        payload::PayloadRef,
        registry::IdAllocator,
        segment::{SEGMENT_HEADER_SIZE, SegmentHeaderCodec, SegmentRef, SegmentType},
        toc::{TocFooter, TocFooterBody, TocFooterCodec},
        vector::{VectorDataSegmentReader, VectorDataSegmentWriter},
    },
    io::{Durability, sync_file, sync_parent_dir},
    text::analyzer::Analyzer,
};

pub(crate) use text_indexing::{
    PendingFrameIndex, TextSpaceState, build_flush_output, rebuild_text_space_state,
    text_index_root_mut,
};

pub(in crate::file) use catalog_io::{
    upsert_analyzer, upsert_codec, upsert_embedding_space, upsert_field_schema,
    upsert_fusion_profile, upsert_retrieval_profile, upsert_text_space,
};
pub(in crate::file) use segment_io::read_segment_payload_typed;

mod api_types;
mod profile;
mod query_profile;
mod query_types;

pub use api_types::{
    AutoPromote, CreateOptions, EmbeddingSpaceSpec, OpenMode, PutFrame, PutVector,
    ReadVectorResult, Reconstruct, TextMode, VectorContract, VectorFidelity,
};
pub use profile::{CommitOutcome, CommitProfile, IngestProfile, TextIndexBuildProfile};
pub use query_profile::{
    QueryProfile, VectorOpenProfile, last_query_profile, last_vector_open_profile,
};
pub(crate) use query_types::DEFAULT_SKETCH_CANDIDATE_BUDGET;
pub use query_types::{
    HybridHit, HybridQuery, HybridTextChannel, HybridVectorChannel, QueryAlgorithm,
    SketchScanTimings, TextQuery, TimeQuery, VectorHit, VectorSearchQuery, VoteSearchTrace,
};

pub struct ValiseFile {
    path: PathBuf,
    /// File handle wrapped in a parking_lot `Mutex` so the read path can
    /// take `&self` (Stage 1 of `docs/CONCURRENCY_PLAN.md`). Writers under
    /// `&mut self` call `lock()` uncontended; readers under `&self` acquire
    /// briefly per syscall. The mmap below covers committed bytes for
    /// zero-copy hot paths so the lock is only held during catalog/payload
    /// chain reads.
    file: Mutex<File>,
    /// Read-only mmap covering the file from offset 0 to its size at the
    /// last `(re)map_file` call. Refreshed after `open` and at the end of
    /// every `commit` so the active TOC's segments are addressable as
    /// zero-copy byte slices. Hot read paths (`vector_search`, `read_vector`,
    /// rerank) prefer slicing into this mmap over `File` reads — see
    /// `mmap_segment_payload`. Writes still go through `self.file.lock()`
    /// and may extend the file past `file_mmap.len()`; callers that need
    /// to read those bytes before the next commit must use the file path.
    ///
    /// Stage 2: now `Arc`-shared with whatever `Snapshot` is currently
    /// published. Old snapshots may continue to hold an older `Arc<Mmap>`
    /// after this field rotates to a newer mapping at commit.
    file_mmap: Option<Arc<memmap2::Mmap>>,
    /// Last published snapshot. `commit()` builds a new `Snapshot` from
    /// the freshly-written state and atomically `ArcSwap`-stores it here.
    /// Reader callers that need a long-lived consistent view (multi-step
    /// queries, streaming) call [`ValiseFile::snapshot`] to clone the `Arc`.
    /// Initial value is `Snapshot::empty()`.
    published_snapshot: ArcSwap<Snapshot>,
    /// Stage 5++: optional fsync coalescing barrier installed by
    /// `Database`. When present, the per-commit final fsync routes
    /// through it so concurrent committers share one F_FULLFSYNC.
    /// `RwLock` so installation can happen post-construction without
    /// requiring `&mut self`.
    commit_fsync_barrier: RwLock<Option<Arc<crate::concurrency::writer_pipeline::GroupFsync>>>,
    mode: OpenMode,
    durability: Durability,
    header: Header,
    footer: TocFooter,
    /// Phase 1: file-level create-time contract. For new files this is
    /// the contract built from `CreateOptions` and stamped into both
    /// the header digest and the TOC footer body. For pre-bump files
    /// it's a synthesized in-memory copy (`text_enabled = true`,
    /// `allowed_dtypes = F32` only) — see plan §4.4.
    create_contract: CreateContractV1,
    catalog: CatalogSnapshot,
    /// Lazy locators for heavy `FrameDesc` fields, built at open by the
    /// slim catalog decoder. `frame_full(id)` consults this map to slice
    /// the per-frame payload bytes out of the committed catalog segment
    /// without re-reading the entire chain.
    frame_locators: HashMap<FrameId, catalog_io::FrameLocator>,
    /// Naked-pointer vector base cache. Wrapped in `RwLock` so `vector_search`
    /// can take `&self` (Stage 1) and lazy-load on first call. Each
    /// indexed slot holds the absolute address (as `usize`) of the
    /// vector's encoded base record inside the file mmap. Slot 0 and
    /// tombstoned-vector slots are 0 (sentinel). Refreshed at every
    /// commit's mmap remap.
    ///
    /// SAFETY: pointers are valid only until `file_mmap` is remapped, and
    /// every commit remaps. So `commit()` must, after the remap, either
    /// rebuild this cache (when the commit touched vectors) or call
    /// `VectorBasePtrs::invalidate` (when it did not) so the next search
    /// lazy-loads against the new mapping. A commit that does neither
    /// leaves dangling addresses and the next vector search is a
    /// use-after-free — that was a real bug, see
    /// `tests/regression_two_collection_vector_search.rs`.
    vector_base: RwLock<VectorBasePtrs>,
    id_allocator: IdAllocator,
    /// In-memory segment registry. Wrapped in `RwLock` so the read path
    /// can take `&self` and lazy-load if the open path deferred the
    /// catalog decode (ghost registry). Writer paths under `&mut self`
    /// use `RwLock::get_mut()` for uncontended exclusive access.
    segment_registry: RwLock<SegmentRegistry>,
    dirty_segment_ids: HashSet<SegmentId>,
    dirty_collection_ids: HashSet<CollectionId>,
    dirty_frame_ids: HashSet<FrameId>,
    dirty_embedding_space_ids: HashSet<EmbeddingSpaceId>,
    dirty_codec_ids: HashSet<CodecId>,
    dirty_vector_ids: HashSet<VectorId>,
    dirty_text_space_ids: HashSet<TextSpaceId>,
    dirty_analyzer_ids: HashSet<AnalyzerId>,
    dirty_field_schema_ids: HashSet<FieldSchemaId>,
    dirty_retrieval_profile_ids: HashSet<RetrievalProfileId>,
    dirty_fusion_profile_ids: HashSet<FusionProfileId>,
    /// In-memory codec instances cached by `codec_id`, populated at
    /// `register_codec` time and rebuilt at `open()` from the catalog.
    codec_cache: HashMap<CodecId, Box<dyn VectorCodec>>,
    /// Per-`(embedding_space_id, codec_id)` accumulator for vectors put
    /// since the last commit. Each batch carries a pre-allocated segment
    /// id; the data segment is appended at commit time (see
    /// `flush_pending_vector_batches`).
    pending_vector_batches: HashMap<(EmbeddingSpaceId, CodecId), PendingVectorBatch>,
    /// Pending raw payload bytes, concatenated into one buffer. Drained
    /// at `flush_pending_payloads` (called when the buffer crosses
    /// `PAYLOAD_BATCH_FLUSH_BYTES`, or at the top of every commit's
    /// write phase). Replaces the legacy "one Payload segment per
    /// put_frame" model — see CONTRIBUTING.md "Don't grow `src/file.rs`"
    /// (the flush helper lives below; the per-frame entry just records
    /// where the bytes landed in the buffer).
    pending_payload_buf: Vec<u8>,
    /// Per-frame index into `pending_payload_buf`. Each entry will get
    /// its `PayloadRef.segment_id`/`bytes_offset` rewritten at flush
    /// time. Keeps insertion order so the flushed segment lays out
    /// bytes in the order the user called `put_frame`.
    pending_payload_frames: Vec<PendingPayloadFrame>,
    /// `frame_id → index in pending_payload_frames` for the read path.
    /// `read_payload` checks this first; on hit it returns a slice of
    /// `pending_payload_buf` without going to disk.
    pending_payload_by_frame: HashMap<FrameId, usize>,
    /// Pending per-frame identity-key bytes (`PutFrame.uri`), concatenated
    /// into one buffer. Drained alongside payloads in
    /// `flush_pending_payloads` into a single `Metadata` segment; only
    /// frames that supplied a `uri` appear here, so this is usually empty
    /// or small relative to `pending_payload_buf`.
    pending_uri_buf: Vec<u8>,
    pending_uris: Vec<PendingUri>,
    /// Per-phase ingest counters. Populated only when
    /// `ingest_profile_enabled` (set from the `VALISE_INGEST_PROFILE` env var
    /// at construction time). Outside that flag every instrumented site
    /// short-circuits, so production builds pay nothing.
    pub ingest_profile: IngestProfile,
    ingest_profile_enabled: bool,
    /// Per-`text_space_id` pending tokenization buffers. Keyed by
    /// `frame_id` within each text_space. Flushed at commit-time as delta
    /// `TermDict` / `Postings` / `DocStats` segments.
    pending_text_indexes: HashMap<TextSpaceId, HashMap<FrameId, PendingFrameIndex>>,
    /// In-memory canonical text state per text_space, rebuilt at `open()`
    /// from the delta chains.
    text_space_states: HashMap<TextSpaceId, TextSpaceState>,
    /// Cached count of `FrameStatus::Tombstoned` entries in
    /// `catalog.frame_stubs`. Maintained at open time and incremented in
    /// `delete_frame`. Lets the BM25 / TF-IDF / Jaccard read paths
    /// short-circuit the per-query 500k-stub tombstone scan when the
    /// common case (no deletions) holds.
    tombstoned_frame_count: u64,
    /// Cached analyzer per text_space, built at `register_text_space` and
    /// on `open()` from the stored `AnalyzerDesc`.
    analyzer_cache: HashMap<TextSpaceId, Analyzer>,
    /// `VectorId → VectorDesc` lookup over all *committed, active* vectors.
    /// Refreshed at the end of `commit` and at `open`.
    /// Used by `sketch_then_rerank_impl` and the brute-force fallback to
    /// resolve frame_id / collection_id for the top-k winners.
    vector_by_id_cache: HashMap<VectorId, VectorDesc>,
    /// Per-collection (frame_ids, vector_ids) snapshot, populated at open
    /// from `collection_filter_roots` and refreshed at commit when frames
    /// or vectors in a collection change.
    collection_filter_cache: filter_io::CollectionFilterCache,
    /// Global time index, sorted ascending by `created_at`. Refreshed at
    /// commit whenever frames change.
    time_index_cache: time_index_io::TimeIndexCache,
    /// Segment ids whose on-disk (wire) payload bytes have been re-hashed
    /// against the stored BLAKE3 since this handle opened. The two stored
    /// checksum copies (registry `SegmentRef` + segment header) only
    /// vouch for each other — without re-hashing, one flipped bit in a
    /// committed zstd payload stream decompresses "successfully" and
    /// corrupts every payload in its batch. `read_payload_ref` and the
    /// mmap payload read paths (engine + `Snapshot::read_payload`)
    /// verify each Payload segment once per handle; later reads skip
    /// the hash under a cheap read lock (segments are immutable once
    /// committed, so a verified id stays valid for the handle's
    /// lifetime). `Arc`-shared with every published [`Snapshot`] so
    /// lock-free readers reuse the same verification state.
    verified_payload_segments: Arc<RwLock<HashSet<SegmentId>>>,
    dirty: bool,
}

/// In-memory accumulator for a `(embedding_space_id, codec_id)` batch
/// during ingest. **Raw f32 inputs are stored verbatim** — encoding is
/// streamed into `writer` in fixed-size chunks during ingest so peak
/// `raw_values` RAM stays bounded at `VECTOR_CHUNK_SIZE * dim * 4` bytes
/// regardless of corpus size. The trailing partial chunk is encoded at
/// commit time inside `flush_pending_batch`.
///
/// Pre-streaming this struct held every raw f32 until commit, peaking at
/// ~1.2 GiB at 100K rows × 3072 dims and OOM-territory at 1M+. Streaming
/// trades a fixed ~30 ms encode pause every CHUNK rows for unbounded
/// corpus scaling.
/// Soft cap on the in-memory payload buffer between flushes. Once the
/// cumulative buffer crosses this, `put_frame` flushes a Payload
/// segment before returning. 4 MiB is large enough to amortize per-
/// segment overhead (76 B SegmentHeader + ~64 B SegmentRegistry row)
/// down to negligible per-frame, and small enough that no single
/// commit pulls a huge segment into the page cache on open.
pub(crate) const PAYLOAD_BATCH_FLUSH_BYTES: usize = 4 * 1024 * 1024;

#[derive(Debug)]
pub(crate) struct PendingPayloadFrame {
    frame_id: FrameId,
    /// Byte offset of this frame's payload inside `pending_payload_buf`.
    bytes_offset: u64,
    /// Length of this frame's payload bytes.
    bytes_length: u64,
}

/// One buffered `PutFrame.uri`, recording where its bytes sit inside
/// `pending_uri_buf`. Resolved to a `FrameDesc.uri_ref` (a `MetadataRef`
/// into the flushed `Metadata` segment) in `flush_pending_payloads`.
#[derive(Debug)]
pub(crate) struct PendingUri {
    frame_id: FrameId,
    bytes_offset: u64,
    bytes_length: u64,
}

pub(crate) struct PendingVectorBatch {
    embedding_space_id: EmbeddingSpaceId,
    codec_id: CodecId,
    segment_id: SegmentId,
    /// Vector dimension (number of f32 per row). Captured at the first
    /// `put_vector` for this batch from the codec's `dimension()`.
    dim: usize,
    /// Codec metadata captured once so commit doesn't re-query it for
    /// every vector.
    base_bytes: usize,
    /// One `VectorDesc` per call, in `put_vector` order. The
    /// `ordinal_in_segment` field equals the index into this `Vec`.
    vectors: Vec<VectorDesc>,
    /// Flat row-major raw inputs for vectors that haven't been encoded
    /// into `writer` yet. Bounded to less than `VECTOR_CHUNK_SIZE * dim`
    /// f32s after streaming drain — the prior unbounded design held
    /// every raw f32 until commit and OOM'd at 1M+.
    raw_values: Vec<f32>,
    /// Encoded-byte accumulator for vectors already drained from
    /// `raw_values`. `writer.item_count()` tracks how many of `vectors`
    /// have been written; the rest still live in `raw_values`. At commit
    /// time `writer.finish()` produces the final segment payload.
    writer: VectorDataSegmentWriter,
}

/// Vectors per streaming-encode chunk. At dim=3072 a chunk weighs
/// 4096 × 3072 × 4 B = 48 MiB of raw f32. The codec compresses each
/// vector to ~base_bytes_per_vector bytes (2112 B for QAM(5,6) at
/// dim=3072), so the post-encode bytes that stay in `writer` are several×
/// smaller than the raw input would be. Picking 4096 keeps the
/// per-chunk encode-stall under ~50 ms on aarch64.
pub(crate) const VECTOR_CHUNK_SIZE: usize = 4096;

mod accessors;
mod commit;
mod commit_writers;
mod frame_ops;
mod lifecycle;
mod registration;
mod text_ops;
mod vector_ops;

mod internals;
pub(in crate::file) use internals::*;
// These two are reached from outside the `file` module tree (concurrency::{snapshot,connection}),
// so they need crate visibility, broader than the file-subtree glob above.
pub(crate) use internals::{CommitWritePhaseResult, mmap_segment_payload};

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_text_space_desc() -> TextSpaceDesc {
        TextSpaceDesc {
            text_space_id: TextSpaceId(3),
            name: "default".into(),
            analyzer_id: AnalyzerId(5),
            field_schema_id: FieldSchemaId(7),
            default_profile_id: RetrievalProfileId(11),
            flags: 0,
            enabled_retrievers: 0,
        }
    }

    fn sample_analyzer_desc() -> AnalyzerDesc {
        use crate::format::catalog::{
            PunctuationPolicy, Stemming, StopwordsPolicy, Tokenizer, UnicodeNormalization,
        };
        AnalyzerDesc {
            analyzer_id: AnalyzerId(5),
            unicode_normalization: UnicodeNormalization::Nfkc,
            case_fold: true,
            accent_fold: false,
            tokenizer: Tokenizer::UnicodeWords,
            stemming: Stemming::None,
            stopword_set_ref: None,
            stopword_query_only: true,
            stopwords: StopwordsPolicy::None,
            english_possessive_strip: false,
            min_token_len: 2,
            max_token_len: 64,
            shingle_size: 0,
            ngram_min: None,
            ngram_max: None,
            punctuation_policy: PunctuationPolicy::Drop,
        }
    }

    fn sample_field_schema_desc() -> FieldSchemaDesc {
        use crate::format::catalog::{FieldDesc, FieldSource};
        FieldSchemaDesc {
            field_schema_id: FieldSchemaId(7),
            fields: vec![FieldDesc {
                field_id: 1,
                name: "search_text".into(),
                source: FieldSource::SearchText,
                store_positions: false,
                store_term_freq: true,
                store_set_membership: false,
                weight: 1.0,
            }],
        }
    }

    fn sample_retrieval_profile_desc() -> RetrievalProfileDesc {
        use crate::format::catalog::{IdfVariant, RetrievalProfileParams, RetrievalProfileType};
        RetrievalProfileDesc {
            profile_id: RetrievalProfileId(11),
            profile_type: RetrievalProfileType::Bm25,
            params: RetrievalProfileParams::Bm25 {
                k1: 1.2,
                b: 0.75,
                idf_variant: IdfVariant::RobertsonSparckJones,
            },
        }
    }

    #[test]
    fn register_text_side_lifecycle_roundtrip() {
        use tempfile::tempdir;
        let dir = tempdir().unwrap();
        let path = dir.path().join("text-catalogs.vls");

        let (analyzer_id, field_schema_id, profile_id, text_space_id) = {
            let mut valise = ValiseFile::create(&path).unwrap();
            let analyzer_id = valise.register_analyzer(sample_analyzer_desc()).unwrap();
            let field_schema_id = valise
                .register_field_schema(sample_field_schema_desc())
                .unwrap();
            let profile_id = valise
                .register_retrieval_profile(sample_retrieval_profile_desc())
                .unwrap();
            let mut text_space = sample_text_space_desc();
            text_space.analyzer_id = analyzer_id;
            text_space.field_schema_id = field_schema_id;
            text_space.default_profile_id = profile_id;
            let text_space_id = valise.register_text_space(text_space).unwrap();
            valise.commit().unwrap();
            (analyzer_id, field_schema_id, profile_id, text_space_id)
        };

        let valise = ValiseFile::open_read_only(&path).unwrap();
        assert_eq!(valise.analyzers().len(), 1);
        assert_eq!(valise.analyzers()[0].analyzer_id, analyzer_id);
        assert_eq!(valise.field_schemas().len(), 1);
        assert_eq!(valise.field_schemas()[0].field_schema_id, field_schema_id);
        assert_eq!(valise.retrieval_profiles().len(), 1);
        assert_eq!(valise.retrieval_profiles()[0].profile_id, profile_id);
        assert_eq!(valise.text_spaces().len(), 1);
        assert_eq!(valise.text_spaces()[0].text_space_id, text_space_id);
        assert_eq!(valise.text_spaces()[0].analyzer_id, analyzer_id);
        assert_eq!(valise.text_spaces()[0].field_schema_id, field_schema_id);
        assert_eq!(valise.text_spaces()[0].default_profile_id, profile_id);
    }

    #[test]
    fn register_text_space_rejects_unknown_analyzer() {
        use tempfile::tempdir;
        let dir = tempdir().unwrap();
        let path = dir.path().join("bad-analyzer.vls");
        let mut valise = ValiseFile::create(&path).unwrap();
        let mut text_space = sample_text_space_desc();
        text_space.analyzer_id = AnalyzerId(999);
        let err = valise
            .register_text_space(text_space)
            .expect_err("unknown analyzer should be rejected");
        assert!(err.to_string().contains("unknown analyzer_id"));
    }

    #[test]
    fn register_text_space_rejects_unknown_field_schema() {
        use tempfile::tempdir;
        let dir = tempdir().unwrap();
        let path = dir.path().join("bad-fs.vls");
        let mut valise = ValiseFile::create(&path).unwrap();
        let analyzer_id = valise.register_analyzer(sample_analyzer_desc()).unwrap();
        let mut text_space = sample_text_space_desc();
        text_space.analyzer_id = analyzer_id;
        text_space.field_schema_id = FieldSchemaId(999);
        let err = valise
            .register_text_space(text_space)
            .expect_err("unknown field_schema should be rejected");
        assert!(err.to_string().contains("unknown field_schema_id"));
    }

    #[test]
    fn register_text_space_rejects_unknown_profile() {
        use tempfile::tempdir;
        let dir = tempdir().unwrap();
        let path = dir.path().join("bad-profile.vls");
        let mut valise = ValiseFile::create(&path).unwrap();
        let analyzer_id = valise.register_analyzer(sample_analyzer_desc()).unwrap();
        let field_schema_id = valise
            .register_field_schema(sample_field_schema_desc())
            .unwrap();
        let mut text_space = sample_text_space_desc();
        text_space.analyzer_id = analyzer_id;
        text_space.field_schema_id = field_schema_id;
        text_space.default_profile_id = RetrievalProfileId(999);
        let err = valise
            .register_text_space(text_space)
            .expect_err("unknown profile should be rejected");
        assert!(err.to_string().contains("unknown default_profile_id"));
    }

    #[test]
    fn register_methods_allocate_monotonic_ids() {
        use tempfile::tempdir;
        let dir = tempdir().unwrap();
        let path = dir.path().join("ids.vls");
        let mut valise = ValiseFile::create(&path).unwrap();
        let a1 = valise.register_analyzer(sample_analyzer_desc()).unwrap();
        let a2 = valise.register_analyzer(sample_analyzer_desc()).unwrap();
        assert!(a2.0 > a1.0);
        let f1 = valise
            .register_field_schema(sample_field_schema_desc())
            .unwrap();
        let f2 = valise
            .register_field_schema(sample_field_schema_desc())
            .unwrap();
        assert!(f2.0 > f1.0);
    }

    fn sample_fusion_profile_desc() -> FusionProfileDesc {
        use crate::format::catalog::FusionNormalization;
        FusionProfileDesc {
            fusion_profile_id: FusionProfileId(1),
            bm25_weight: 0.6,
            tfidf_weight: 0.0,
            jaccard_weight: 0.0,
            vector_weight: 0.4,
            normalization: FusionNormalization::ZScore,
        }
    }

    #[test]
    fn register_fusion_profiles_persist_across_reopen() {
        use tempfile::tempdir;
        let dir = tempdir().unwrap();
        let path = dir.path().join("fusion.vls");

        let fusion_id = {
            let mut valise = ValiseFile::create(&path).unwrap();
            let fusion_id = valise
                .register_fusion_profile(sample_fusion_profile_desc())
                .unwrap();
            valise.commit().unwrap();
            fusion_id
        };

        let valise = ValiseFile::open_read_only(&path).unwrap();
        assert_eq!(valise.fusion_profiles().len(), 1);
        assert_eq!(valise.fusion_profiles()[0].fusion_profile_id, fusion_id);
    }

    #[test]
    fn register_fusion_profile_rejects_invalid_weights() {
        use crate::format::catalog::FusionNormalization;
        use tempfile::tempdir;
        let dir = tempdir().unwrap();
        let path = dir.path().join("bad-fusion.vls");
        let mut valise = ValiseFile::create(&path).unwrap();
        let bad = FusionProfileDesc {
            fusion_profile_id: FusionProfileId(0),
            bm25_weight: 0.0,
            tfidf_weight: 0.0,
            jaccard_weight: 0.0,
            vector_weight: 0.0,
            normalization: FusionNormalization::ZScore,
        };
        let err = valise
            .register_fusion_profile(bad)
            .expect_err("all-zero weights should be rejected");
        assert!(err.to_string().contains("at least one positive"));
    }

    #[test]
    fn fusion_register_methods_allocate_monotonic_ids() {
        use tempfile::tempdir;
        let dir = tempdir().unwrap();
        let path = dir.path().join("monotonic.vls");
        let mut valise = ValiseFile::create(&path).unwrap();
        let f1 = valise
            .register_fusion_profile(sample_fusion_profile_desc())
            .unwrap();
        let f2 = valise
            .register_fusion_profile(sample_fusion_profile_desc())
            .unwrap();
        assert!(f2.0 > f1.0);
    }
}
