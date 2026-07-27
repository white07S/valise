// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! `ValiseFile` read-only accessors + time/hybrid queries.

use super::*;

impl ValiseFile {
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Borrow the in-memory create-time contract. For new files this is
    /// the contract built from `CreateOptions`; for pre-bump files it
    /// is a synthesized legacy contract (text-enabled, f32-only).
    /// Returned by reference to avoid cloning the 16-byte reserved
    /// area on hot inspection paths.
    #[must_use]
    pub fn create_contract(&self) -> &CreateContractV1 {
        &self.create_contract
    }

    /// Whether text retrieval is enabled for this file.
    #[must_use]
    pub fn text_enabled(&self) -> bool {
        self.create_contract.text_enabled
    }

    /// Dtype whitelist for `register_embedding_space`. Phase 4 readers
    /// use this to validate ingest dtypes.
    #[must_use]
    pub fn allowed_dtypes(&self) -> DtypeSet {
        DtypeSet::from_bits_truncate(self.create_contract.allowed_dtypes)
    }

    /// Hard cap on `EmbeddingSpaceSpec.dimension`.
    #[must_use]
    pub fn max_dim(&self) -> u32 {
        self.create_contract.max_dim
    }

    /// Vestigial auto-promotion thresholds (gated the removed vote index; see
    /// [`AutoPromote`]). Returned for completeness; no runtime path uses them.
    #[must_use]
    pub fn auto_promote(&self) -> AutoPromote {
        AutoPromote {
            non_f8_threshold: self.create_contract.auto_promote_non_f8,
            f8_threshold: self.create_contract.auto_promote_f8,
        }
    }

    /// Override the per-call durability for write operations. See
    /// [`crate::Database::set_durability`].
    pub fn set_durability(&mut self, durability: Durability) {
        self.durability = durability;
    }

    /// Atomic load of the most recently published `Snapshot`. Each
    /// successful `commit()` swaps a new value in via `ArcSwap`; readers
    /// that need a long-lived consistent view (multi-step queries,
    /// streaming, post-commit pinning) clone this `Arc` and use it for
    /// the duration of their work. The pinned snapshot keeps its mmap
    /// and catalog alive even if the file rotates to a newer mapping
    /// underneath. See `docs/CONCURRENCY_PLAN.md` §3.2.
    #[must_use]
    pub fn snapshot(&self) -> Arc<Snapshot> {
        self.published_snapshot.load_full()
    }

    #[must_use]
    pub fn header(&self) -> &Header {
        &self.header
    }

    #[must_use]
    pub fn collections(&self) -> &[CollectionDesc] {
        &self.catalog.collections
    }

    /// Frames that are members of `collection_id`, sorted ascending by id.
    /// Reads from the persisted collection filter (spec §16); pending
    /// (uncommitted) frames are not visible — call `commit()` first.
    /// Returns an empty slice when the collection has no filter segment yet.
    #[must_use]
    pub fn collection_member_frames(&self, collection_id: CollectionId) -> &[FrameId] {
        self.collection_filter_cache
            .get(&collection_id)
            .map(|(frames, _)| frames.as_slice())
            .unwrap_or(&[])
    }

    /// Vectors that are members of `collection_id`, sorted ascending by id.
    /// Same semantics as `collection_member_frames`.
    #[must_use]
    pub fn collection_member_vectors(&self, collection_id: CollectionId) -> &[VectorId] {
        self.collection_filter_cache
            .get(&collection_id)
            .map(|(_, vectors)| vectors.as_slice())
            .unwrap_or(&[])
    }

    /// Time-range query over the persisted time index, spec §17. Returns
    /// frame_ids whose `created_at` falls in `[query.from, query.to]`,
    /// optionally filtered by collection. Pending (uncommitted) frames are
    /// not visible — call `commit()` first.
    #[must_use]
    pub fn time_range_query(&self, query: TimeQuery) -> Vec<FrameId> {
        time_index_io::time_range(
            &self.time_index_cache,
            query.from,
            query.to,
            query.collection_id,
        )
    }

    /// Hybrid retrieval (text + vector → fused), spec §18.3. Runs each
    /// configured channel independently, then fuses scores under the
    /// fusion profile. Returns the top `query.k` hits.
    pub fn query_hybrid(&self, query: HybridQuery) -> Result<Vec<HybridHit>> {
        let profile = self
            .catalog
            .fusion_profiles
            .iter()
            .find(|p| p.fusion_profile_id == query.fusion_profile_id)
            .cloned()
            .ok_or_else(|| {
                Error::Format(format!(
                    "query_hybrid: unknown fusion_profile_id {}",
                    query.fusion_profile_id.0
                ))
            })?;

        let mut bm25_scores: HashMap<FrameId, f32> = HashMap::new();
        let mut tfidf_scores: HashMap<FrameId, f32> = HashMap::new();
        let mut jaccard_scores: HashMap<FrameId, f32> = HashMap::new();
        let mut vector_distances: HashMap<FrameId, f32> = HashMap::new();

        if let Some(text) = &query.text {
            let hits = self.query_text(TextQuery {
                text_space_id: text.text_space_id,
                query: text.query.clone(),
                algorithm: text.algorithm,
                top_k: Some(query.channel_k),
                channel_k: None,
            })?;
            // Categorize hits into the right channel based on the
            // algorithm. Profile-driven algorithms route by their profile
            // type; profile-free algorithms use a fixed channel.
            let channel = text_algorithm_channel(&self.catalog, text.algorithm)?;
            let target = match channel {
                TextChannel::Bm25 => &mut bm25_scores,
                TextChannel::Tfidf => &mut tfidf_scores,
                TextChannel::Jaccard => &mut jaccard_scores,
            };
            for (i, hit) in hits.into_iter().enumerate() {
                if i >= query.channel_k {
                    break;
                }
                target.insert(hit.frame_id, hit.score);
            }
        }

        if let Some(vec_channel) = &query.vector {
            // Route through `vector_search` so both QAM(5,6) sketch spaces
            // and brute-force-fallback spaces work. `ef` maps to `channel_k`
            // (the sketch candidate budget); `None` lets the dispatch use its
            // default (DEFAULT_SKETCH_CANDIDATE_BUDGET).
            let vector_hits = self.vector_search(VectorSearchQuery {
                embedding_space_id: vec_channel.embedding_space_id,
                query: vec_channel.query.clone(),
                k: query.channel_k,
                channel_k: vec_channel.ef,
                collection_filter: None,
                fidelity: vec_channel.fidelity,
            })?;
            for hit in vector_hits {
                vector_distances.insert(hit.frame_id, hit.score);
            }
        }

        let fused = crate::retrieval::fusion::fuse(
            &profile,
            bm25_scores,
            tfidf_scores,
            jaccard_scores,
            vector_distances,
        );
        Ok(fused
            .into_iter()
            .take(query.k)
            .map(|(frame_id, score)| HybridHit { frame_id, score })
            .collect())
    }

    /// Every committed `FrameDesc`, **including tombstoned ones**. The v2
    /// columnar FrameCatalog (`VLF2`) decodes these eagerly at open, so
    /// this is the complete set post-open (the v1 lazy/ghost-catalog path
    /// is gone — v1 files are no longer readable). For just the live set,
    /// use [`Self::active_frames`].
    #[must_use]
    pub fn frames(&self) -> &[FrameDesc] {
        &self.catalog.frames
    }

    /// Iterate every active (non-tombstoned) frame. Compaction primitive:
    /// stream the live frame set into a fresh file, dropping tombstoned
    /// frames' bytes. Complete and committed-state in v2 (see
    /// [`Self::frames`]).
    pub fn active_frames(&self) -> impl Iterator<Item = &FrameDesc> {
        self.catalog
            .frames
            .iter()
            .filter(|f| f.status == FrameStatus::Active)
    }

    /// Slim `(frame_id, status)` view of every active frame, populated at
    /// open by the Ghost-Catalog walk. Stable across the file's lifetime;
    /// the BM25/Tfidf/Jaccard hot paths consume this same slice.
    #[must_use]
    pub fn frame_stubs(&self) -> &[FrameStub] {
        &self.catalog.frame_stubs
    }

    /// Heavy `FrameDesc` accessor — the lazy counterpart to
    /// [`Self::frame_stubs`]. Looks up the frame in the in-memory write
    /// catalog first (post-`put_frame`, pre-commit); on miss, slices the
    /// frame's payload bytes out of the committed catalog segment using
    /// the locator built at open and decodes a single record. O(1) seek
    /// + bincode decode.
    pub fn frame_full(&self, frame_id: FrameId) -> Result<FrameDesc> {
        // Hot path: frame is already heavy in `catalog.frames`.
        if let Ok(idx) = self
            .catalog
            .frames
            .binary_search_by_key(&frame_id.0, |f| f.frame_id.0)
        {
            return Ok(self.catalog.frames[idx].clone());
        }
        let locator =
            self.frame_locators.get(&frame_id).copied().ok_or_else(|| {
                Error::Format(format!("frame_full: unknown frame_id {}", frame_id.0))
            })?;
        // Stage 6 fix: slice the catalog segment from the file mmap
        // instead of taking the file `Mutex` for a seek+read. Eliminates
        // the cross-thread serialization point on the read path —
        // `frame_full` is one of the two ops that benchmarked as the
        // ~30 k ops/sec ceiling under contention. Falls back to file
        // read when the mmap doesn't cover the requested range — that
        // happens for pre-commit reads where the writer extended the
        // file past the last `remap_file()` call.
        let mmap_payload = self
            .file_mmap
            .as_ref()
            .and_then(|m| {
                if locator.catalog_segment.offset + locator.catalog_segment.length <= m.len() as u64
                {
                    Some(mmap_segment_payload(
                        m,
                        locator.catalog_segment,
                        SegmentType::FrameCatalog,
                    ))
                } else {
                    None
                }
            })
            .transpose()?;
        let payload_owned: Vec<u8>;
        let payload: &[u8] = match mmap_payload {
            Some(slice) => slice,
            None => {
                payload_owned = segment_io::read_segment_payload_typed(
                    &mut self.file.lock(),
                    locator.catalog_segment,
                    SegmentType::FrameCatalog,
                )?;
                &payload_owned
            }
        };
        let end = locator
            .payload_offset
            .checked_add(locator.payload_len)
            .ok_or_else(|| Error::Integrity("frame_full: locator offset overflow".into()))?;
        if end > payload.len() {
            return Err(Error::Integrity(format!(
                "frame_full: locator out of range (payload len {}, locator end {end})",
                payload.len()
            )));
        }
        let bytes = &payload[locator.payload_offset..end];
        let (desc, _): (FrameDesc, _) =
            bincode::serde::decode_from_slice(bytes, crate::format::bincode_config())
                .map_err(|e| Error::Serde(e.to_string()))?;
        if desc.frame_id != frame_id {
            return Err(Error::Integrity(format!(
                "frame_full: locator pointed at frame_id {} but expected {}",
                desc.frame_id.0, frame_id.0
            )));
        }
        Ok(desc)
    }

    /// The `created_at` of a frame, by id, without cloning its `FrameDesc`.
    /// Binary-searches the in-memory catalog (frames are fully decoded at open
    /// and kept sorted by id). Cheap accessor for the recency-ranking path.
    pub fn frame_created_at(&self, frame_id: FrameId) -> Result<i64> {
        self.catalog
            .frames
            .binary_search_by_key(&frame_id.0, |f| f.frame_id.0)
            .map(|idx| self.catalog.frames[idx].created_at)
            .map_err(|_| {
                Error::Format(format!("frame_created_at: unknown frame_id {}", frame_id.0))
            })
    }

    /// Read the external identity key/URI persisted for `frame_id`, if
    /// any. Resolves the frame's `uri_ref` (a `MetadataRef` into a
    /// `Metadata` segment), reads the bytes, and validates the per-key
    /// BLAKE3 checksum. Returns `Ok(None)` when the frame carries no
    /// `uri_ref` — including a `uri` still buffered in an uncommitted
    /// batch (it materializes only at flush). Higher layers use this to
    /// rebuild a key → frame identity map at open.
    pub fn read_frame_uri(&self, frame_id: FrameId) -> Result<Option<Vec<u8>>> {
        match self.frame_full(frame_id)?.uri_ref {
            Some(uri_ref) => Ok(Some(self.read_metadata_ref(&uri_ref)?)),
            None => Ok(None),
        }
    }

    /// Resolve a `MetadataRef` to its bytes: locate the owning segment in
    /// the registry, read its payload, slice the referenced range, and
    /// validate the embedded BLAKE3 checksum.
    pub(crate) fn read_metadata_ref(
        &self,
        mref: &crate::format::catalog::MetadataRef,
    ) -> Result<Vec<u8>> {
        self.ensure_segment_registry_loaded()?;
        let seg_ref = self
            .segment_registry
            .read()
            .by_id
            .get(&mref.segment_id)
            .copied()
            .ok_or_else(|| {
                Error::Integrity(format!(
                    "read_metadata_ref: segment {} missing from registry",
                    mref.segment_id.0
                ))
            })?;
        // Fast path: slice the Metadata segment straight out of the file mmap,
        // avoiding the `file.lock()` + full-segment heap copy. Mirrors
        // `frame_full`; falls back to a locked read when the mmap doesn't cover
        // the range (e.g. a pre-commit-extended segment).
        let mmap_payload = self
            .file_mmap
            .as_ref()
            .and_then(|m| {
                if seg_ref.offset + seg_ref.length <= m.len() as u64 {
                    Some(mmap_segment_payload(m, seg_ref, SegmentType::Metadata))
                } else {
                    None
                }
            })
            .transpose()?;
        let payload_owned: Vec<u8>;
        let payload: &[u8] = match mmap_payload {
            Some(slice) => slice,
            None => {
                payload_owned = read_segment_payload_typed(
                    &mut self.file.lock(),
                    seg_ref,
                    SegmentType::Metadata,
                )?;
                &payload_owned
            }
        };
        let start = mref.bytes_offset as usize;
        let end = start
            .checked_add(mref.bytes_length as usize)
            .ok_or_else(|| Error::Integrity("read_metadata_ref: range overflow".into()))?;
        if end > payload.len() {
            return Err(Error::Integrity(format!(
                "read_metadata_ref: range {start}..{end} out of segment payload len {}",
                payload.len()
            )));
        }
        let bytes = payload[start..end].to_vec();
        if crate::format::blake3_checksum(&bytes) != mref.checksum {
            return Err(Error::Integrity(
                "read_metadata_ref: metadata checksum mismatch".into(),
            ));
        }
        Ok(bytes)
    }

    #[must_use]
    pub fn embedding_spaces(&self) -> &[EmbeddingSpaceDesc] {
        &self.catalog.embedding_spaces
    }

    #[must_use]
    pub fn codecs(&self) -> &[CodecDesc] {
        &self.catalog.codecs
    }

    #[must_use]
    pub fn vectors(&self) -> &[VectorDesc] {
        &self.catalog.vectors
    }

    /// Iterate every active (non-tombstoned) vector in `embedding_space_id`.
    /// Per-space compaction primitive: stream the live vectors of one
    /// space so a fresh file can re-encode / recalibrate them. The vector
    /// catalog is fully loaded at open, so this is the complete set.
    pub fn active_vectors(
        &self,
        embedding_space_id: EmbeddingSpaceId,
    ) -> impl Iterator<Item = &VectorDesc> {
        self.catalog.vectors.iter().filter(move |v| {
            v.status == VectorStatus::Active && v.embedding_space_id == embedding_space_id
        })
    }

    #[must_use]
    pub fn text_spaces(&self) -> &[TextSpaceDesc] {
        &self.catalog.text_spaces
    }

    #[must_use]
    pub fn analyzers(&self) -> &[AnalyzerDesc] {
        &self.catalog.analyzers
    }

    #[must_use]
    pub fn field_schemas(&self) -> &[FieldSchemaDesc] {
        &self.catalog.field_schemas
    }

    /// Diagnostic: per-`SegmentType` byte usage of the *live* segments
    /// in the on-disk registry. Each tuple is
    /// `(segment_type, segment_count, total_bytes)` where `total_bytes`
    /// includes the segment header + payload (matches `root.length`).
    /// Tombstoned / superseded segments are excluded; the totals reflect
    /// what readers would actually consume.
    ///
    /// Used by benches to break down `.vls` size by spec section
    /// (TermDictionary / Postings / DocStats / VectorData / …). Always
    /// available; trivial cost (one pass over the in-memory registry).
    #[must_use]
    pub fn storage_breakdown_by_segment_type(
        &self,
    ) -> Vec<(crate::format::segment::SegmentType, u32, u64)> {
        use crate::format::registry::SegmentCatalogStatus;
        use std::collections::HashMap;
        let reg = self.segment_registry.read();
        let mut buckets: HashMap<crate::format::segment::SegmentType, (u32, u64)> = HashMap::new();
        for entry in &reg.catalog {
            if entry.status != SegmentCatalogStatus::Live {
                continue;
            }
            let b = buckets.entry(entry.segment_type).or_insert((0, 0));
            b.0 += 1;
            b.1 += entry.root.length;
        }
        let mut out: Vec<_> = buckets.into_iter().map(|(t, (c, b))| (t, c, b)).collect();
        out.sort_by_key(|e| e.0 as u16);
        out
    }

    #[must_use]
    pub fn retrieval_profiles(&self) -> &[RetrievalProfileDesc] {
        &self.catalog.retrieval_profiles
    }

    #[must_use]
    pub fn fusion_profiles(&self) -> &[FusionProfileDesc] {
        &self.catalog.fusion_profiles
    }
}
