// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Free helper functions and small helper types for the `ValiseFile` engine,
//! factored out of `file.rs` to keep the module root thin. Re-exported into
//! the `file` module tree via `pub(in crate::file) use internals::*`.

use super::*;

#[allow(clippy::too_many_arguments)]
/// Specialized flush for the v2 columnar FrameCatalog wire form. Bypasses
/// `encode_catalog_delta` (which uses bincode per row) and writes one
/// `VLF2` segment per delta. Same chain semantics: each delta records the
/// previous segment as its parent so reload can walk backwards.
pub(in crate::file) fn flush_frame_catalog_columnar(
    file: &mut File,
    id_allocator: &mut IdAllocator,
    segments: SegmentRegistryMut<'_>,
    body_root: &mut Option<SegmentRef>,
    dirty_ids: &HashSet<FrameId>,
    catalog: &[FrameDesc],
) -> Result<()> {
    use crate::format::frame_catalog_codec::{FrameCatalogDelta, encode_frame_catalog};
    if dirty_ids.is_empty() {
        return Ok(());
    }
    let records = dirty_records(catalog, dirty_ids, |f: &FrameDesc| f.frame_id);
    let delta = FrameCatalogDelta {
        previous_root: *body_root,
        frames: records,
    };
    let payload = encode_frame_catalog(&delta)?;
    let item_count = u32::try_from(delta.frames.len())
        .map_err(|_| Error::Format("FrameCatalog delta count overflow".into()))?;
    let root = append_registered_segment(
        file,
        id_allocator,
        segments,
        SegmentType::FrameCatalog,
        item_count,
        &payload,
    )?;
    *body_root = Some(root);
    Ok(())
}

pub(in crate::file) fn flush_catalog_table<T, Id, F>(
    file: &mut File,
    id_allocator: &mut IdAllocator,
    segments: SegmentRegistryMut<'_>,
    body_root: &mut Option<SegmentRef>,
    dirty_ids: &HashSet<Id>,
    catalog: &[T],
    table: CatalogTableKind,
    segment_type: SegmentType,
    id_of: F,
) -> Result<()>
where
    T: Clone + serde::Serialize,
    Id: Eq + Ord + std::hash::Hash + Copy,
    F: Fn(&T) -> Id,
{
    if dirty_ids.is_empty() {
        return Ok(());
    }
    let records = dirty_records(catalog, dirty_ids, id_of);
    let payload = encode_catalog_delta(table, *body_root, &records)?;
    let root = append_registered_segment(
        file,
        id_allocator,
        segments,
        segment_type,
        records.len() as u32,
        &payload,
    )?;
    *body_root = Some(root);
    Ok(())
}

pub(crate) fn build_analyzer_cache(
    catalog: &CatalogSnapshot,
) -> Result<HashMap<TextSpaceId, Analyzer>> {
    let mut cache = HashMap::new();
    for space in &catalog.text_spaces {
        let analyzer_desc = catalog
            .analyzers
            .iter()
            .find(|a| a.analyzer_id == space.analyzer_id)
            .ok_or_else(|| {
                Error::Integrity(format!(
                    "text_space {} references unknown analyzer_id {}",
                    space.text_space_id.0, space.analyzer_id.0
                ))
            })?;
        let analyzer = Analyzer::from_desc(analyzer_desc)?;
        cache.insert(space.text_space_id, analyzer);
    }
    Ok(cache)
}

pub(crate) fn rebuild_all_text_space_states(
    file: &mut File,
    body: &TocFooterBody,
    id_allocator: &mut IdAllocator,
    text_spaces: &[TextSpaceDesc],
) -> Result<HashMap<TextSpaceId, TextSpaceState>> {
    let mut states = HashMap::new();
    for space in text_spaces {
        let root = body
            .text_index_roots
            .iter()
            .find(|r| r.text_space_id == space.text_space_id);
        let state = if let Some(root) = root {
            rebuild_text_space_state(file, root, id_allocator)?
        } else {
            TextSpaceState::default()
        };
        states.insert(space.text_space_id, state);
    }
    Ok(states)
}

/// BLAKE3 digest over a calibration sample, used as the
/// `QamLloydMaxParams.calibration_id` provenance tag. Deterministic in
/// `(dim, sample)`: the same calibration data always yields the same id.
pub(crate) fn calibration_id_from_sample(dim: usize, sample: &[Vec<f32>]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(&(dim as u64).to_le_bytes());
    hasher.update(&(sample.len() as u64).to_le_bytes());
    for v in sample {
        for &x in v {
            hasher.update(&x.to_le_bytes());
        }
    }
    *hasher.finalize().as_bytes()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum TextChannel {
    Bm25,
    Tfidf,
    Jaccard,
}

pub(crate) fn text_algorithm_channel(
    catalog: &CatalogSnapshot,
    algorithm: QueryAlgorithm,
) -> Result<TextChannel> {
    match algorithm {
        QueryAlgorithm::Bm25 { .. } => Ok(TextChannel::Bm25),
        QueryAlgorithm::CountCosine
        | QueryAlgorithm::TfidfCosine { .. }
        | QueryAlgorithm::CountCosineApprox
        | QueryAlgorithm::TfidfCosineApprox { .. } => Ok(TextChannel::Tfidf),
        QueryAlgorithm::Dice | QueryAlgorithm::Overlap | QueryAlgorithm::Containment => {
            Ok(TextChannel::Jaccard)
        }
        QueryAlgorithm::Profile(profile_id) => {
            let profile = catalog
                .retrieval_profiles
                .iter()
                .find(|p| p.profile_id == profile_id)
                .ok_or_else(|| {
                    Error::Format(format!(
                        "query_hybrid: unknown retrieval profile_id {}",
                        profile_id.0
                    ))
                })?;
            match profile.profile_type {
                crate::format::catalog::RetrievalProfileType::Bm25 => Ok(TextChannel::Bm25),
                crate::format::catalog::RetrievalProfileType::Tfidf => Ok(TextChannel::Tfidf),
                crate::format::catalog::RetrievalProfileType::JaccardExact
                | crate::format::catalog::RetrievalProfileType::JaccardWeighted => {
                    Ok(TextChannel::Jaccard)
                }
            }
        }
    }
}

/// f32 distance for reranked hits. Matches the codec convention
/// (smaller = more similar) so reranked scores are directly comparable to
/// the lossy QAM-sliding scores.
pub(crate) fn full_distance(a: &[f32], b: &[f32], metric: VectorMetric) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "rerank dim mismatch");
    match metric {
        VectorMetric::Cosine => {
            let mut dot = 0.0f32;
            let mut na = 0.0f32;
            let mut nb = 0.0f32;
            for (x, y) in a.iter().zip(b.iter()) {
                dot += x * y;
                na += x * x;
                nb += y * y;
            }
            let denom = (na.sqrt() * nb.sqrt()).max(f32::EPSILON);
            1.0 - dot / denom
        }
        VectorMetric::InnerProduct => {
            let mut dot = 0.0f32;
            for (x, y) in a.iter().zip(b.iter()) {
                dot += x * y;
            }
            -dot
        }
        VectorMetric::L2 => {
            let mut sum = 0.0f32;
            for (x, y) in a.iter().zip(b.iter()) {
                let d = x - y;
                sum += d * d;
            }
            sum
        }
    }
}

/// Slice into a file-level mmap covering the on-disk payload bytes of a
/// committed segment (the bytes after the 76-byte segment header).
/// Validates header magic, segment type, segment id, length, and the
/// registry-supplied BLAKE3 checksum equality with the live header. The
/// per-byte BLAKE3 of the payload is NOT recomputed here — that runs once
/// at registration time; this hot path trusts the registry/header equality.
pub(crate) fn mmap_segment_payload(
    mmap: &memmap2::Mmap,
    seg_ref: SegmentRef,
    expected_type: SegmentType,
) -> Result<&[u8]> {
    let start = usize::try_from(seg_ref.offset)
        .map_err(|_| Error::Format("segment offset overflows usize".into()))?;
    let length = usize::try_from(seg_ref.length)
        .map_err(|_| Error::Format("segment length overflows usize".into()))?;
    let end = start
        .checked_add(length)
        .ok_or_else(|| Error::Format("segment range overflow".into()))?;
    if end > mmap.len() {
        return Err(Error::Integrity(format!(
            "mmap_segment_payload: segment {} ({}..{}) past mmap end ({})",
            seg_ref.segment_id.0,
            start,
            end,
            mmap.len()
        )));
    }
    if length < SEGMENT_HEADER_SIZE {
        return Err(Error::Format("segment shorter than its header".into()));
    }
    let segment_bytes = &mmap[start..end];
    let header_bytes: &[u8; SEGMENT_HEADER_SIZE] = segment_bytes[..SEGMENT_HEADER_SIZE]
        .try_into()
        .expect("SEGMENT_HEADER_SIZE bytes split off");
    let header = SegmentHeaderCodec::decode(header_bytes)?;
    if header.segment_type != expected_type {
        return Err(Error::Integrity(format!(
            "mmap_segment_payload: segment {} type mismatch (expected {:?}, got {:?})",
            seg_ref.segment_id.0, expected_type, header.segment_type
        )));
    }
    if header.segment_id != seg_ref.segment_id {
        return Err(Error::Integrity("segment id mismatch".into()));
    }
    if header.payload_checksum != seg_ref.checksum {
        return Err(Error::Integrity(
            "mmap_segment_payload: segment header checksum disagrees with registry".into(),
        ));
    }
    let payload = &segment_bytes[SEGMENT_HEADER_SIZE..];
    if payload.len() as u64 != header.payload_length {
        return Err(Error::Integrity(
            "mmap_segment_payload: payload length mismatch".into(),
        ));
    }
    Ok(payload)
}

/// Build the `VectorId → VectorDesc` lookup for the active vectors in
/// `catalog`. Used at `open()` and at the end of `commit()` to keep the
/// `vector_search` hot path from rebuilding this map per query. Tombstoned
/// vectors are excluded; the sign-sketch index is built only over active
/// vectors, so omitted entries are never produced as candidates.
pub(crate) fn rebuild_vector_by_id_cache(
    catalog: &CatalogSnapshot,
) -> HashMap<VectorId, VectorDesc> {
    let mut map = HashMap::with_capacity(catalog.vectors.len());
    for v in &catalog.vectors {
        if v.status == VectorStatus::Active {
            map.insert(v.vector_id, v.clone());
        }
    }
    map
}

/// Stage 5++ avalanche commit: state captured at the end of phase A
/// (writes complete, fsync still pending) and consumed by phase B
/// (publish). Carrying `captured_header` here lets phase B publish a
/// snapshot reflecting *this* writer's commit even if concurrent
/// writers have already rewritten `self.header` for their own commit.
pub(crate) struct CommitPhaseAState {
    pub(crate) footer: TocFooter,
    pub(crate) captured_header: Header,
    pub(crate) writer_lock_acquired: bool,
    /// `true` if any vector / codec / embedding-space mutation happened
    /// in this commit. Phase B's `vector_by_id_cache` and
    /// `vector_base_ptrs` rebuilds walk every active vector and every
    /// segment they live in — for incremental commits that don't
    /// touch vectors that's a substantial waste. Skipping when this
    /// flag is `false` was the difference between Stage 5++ avalanche
    /// being meaningful and being smothered by linear-in-corpus phase
    /// B work.
    pub(crate) vectors_changed: bool,
}

/// Outcome of `commit_phase_writes` — either the early-return for a
/// no-op commit (no dirty state) or the prepared state for phase B.
pub(crate) enum CommitWritePhaseResult {
    NoOp(CommitOutcome),
    Prepared(CommitPhaseAState),
}

/// Build a `Snapshot` from the current state. Used at open/create and at
/// commit-publish; the caller is responsible for ensuring the inputs are
/// consistent (i.e. all reflect the same generation).
/// `verified_payload_segments` is the handle-level verified-set `Arc`
/// (see `ValiseFile::verified_payload_segments`), shared into the snapshot
/// so lock-free readers reuse — and contribute to — the same one-hash-
/// per-segment verification state.
pub(crate) fn build_snapshot(
    header: &Header,
    file_mmap: Option<Arc<memmap2::Mmap>>,
    catalog: &CatalogSnapshot,
    frame_locators: &HashMap<FrameId, catalog_io::FrameLocator>,
    segment_registry: &SegmentRegistry,
    vector_by_id: &HashMap<VectorId, VectorDesc>,
    verified_payload_segments: Arc<RwLock<HashSet<SegmentId>>>,
) -> Snapshot {
    Snapshot {
        generation: header.snapshot_generation,
        toc_offset: header.footer_offset,
        mmap: file_mmap,
        catalog: Arc::new(catalog.clone()),
        frame_locators: Arc::new(frame_locators.clone()),
        segment_registry: Arc::new(segment_registry.clone()),
        vector_by_id: Arc::new(vector_by_id.clone()),
        vector_base_ptrs: std::sync::OnceLock::new(),
        verified_payload_segments,
    }
}

/// Create a fresh read-only mmap covering the entire current extent of
/// `file`. Returns `None` for empty files (mmap of length 0 is rejected
/// on some platforms). The mapping is independent of the `&mut File`
/// handle used for writes; callers must remap after appends to expose
/// the new bytes.
pub(crate) fn remap_file(file: &File) -> Result<Option<memmap2::Mmap>> {
    let len = file.metadata()?.len();
    if len == 0 {
        return Ok(None);
    }
    // Safety: the file is open for read access; we never expose a `&mut`
    // alias to the mapped region. Concurrent writes to bytes already
    // covered by the mapping (e.g. header rewrites or appended segments)
    // are visible to the mmap because both views share the OS page cache
    // for the same inode.
    let mmap = unsafe { memmap2::Mmap::map(file)? };
    Ok(Some(mmap))
}

pub(crate) fn observe_catalog_ids(catalog: &CatalogSnapshot, id_allocator: &mut IdAllocator) {
    for collection in &catalog.collections {
        id_allocator
            .observe_collection_id(collection.collection_id)
            .expect("BUG: valid collection ID from decoded catalog");
    }
    for frame in &catalog.frames {
        id_allocator
            .observe_frame_id(frame.frame_id)
            .expect("BUG: valid frame ID from decoded catalog");
    }
    for space in &catalog.embedding_spaces {
        id_allocator
            .observe_embedding_space_id(space.embedding_space_id)
            .expect("BUG: valid embedding space ID from decoded catalog");
    }
    for codec in &catalog.codecs {
        id_allocator
            .observe_codec_id(codec.codec_id)
            .expect("BUG: valid codec ID from decoded catalog");
    }
    for vector in &catalog.vectors {
        id_allocator
            .observe_vector_id(vector.vector_id)
            .expect("BUG: valid vector ID from decoded catalog");
    }
    for text_space in &catalog.text_spaces {
        id_allocator
            .observe_text_space_id(text_space.text_space_id)
            .expect("BUG: valid text space ID from decoded catalog");
    }
    for analyzer in &catalog.analyzers {
        id_allocator
            .observe_analyzer_id(analyzer.analyzer_id)
            .expect("BUG: valid analyzer ID from decoded catalog");
    }
    for field_schema in &catalog.field_schemas {
        id_allocator
            .observe_field_schema_id(field_schema.field_schema_id)
            .expect("BUG: valid field schema ID from decoded catalog");
    }
    for profile in &catalog.retrieval_profiles {
        id_allocator
            .observe_retrieval_profile_id(profile.profile_id)
            .expect("BUG: valid retrieval profile ID from decoded catalog");
    }
    for profile in &catalog.fusion_profiles {
        id_allocator
            .observe_fusion_profile_id(profile.fusion_profile_id)
            .expect("BUG: valid fusion profile ID from decoded catalog");
    }
}
