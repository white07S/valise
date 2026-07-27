//! `ValiseFile` vector write/read + frame/payload reads.

use super::*;

impl ValiseFile {
    /// Buffer a vector into the pending batch for its
    /// `(embedding_space_id, codec_id)` pair. The vector is encoded
    /// immediately, but the data segment is written at the next
    /// `commit()`. Reading the vector before commit returns an error.
    pub fn put_vector(&mut self, input: PutVector<'_>) -> Result<VectorId> {
        self.ensure_write()?;
        self.ensure_segment_registry_loaded()?;
        let prof = self.ingest_profile_enabled;
        let _call_start = std::time::Instant::now();

        let _t = std::time::Instant::now();
        // O(log N) binary search; catalog.frames is sorted by frame_id.
        let frame_idx = self
            .catalog
            .frames
            .binary_search_by_key(&input.owner_frame_id.0, |f| f.frame_id.0)
            .map_err(|_| {
                Error::Format(format!(
                    "put_vector: unknown frame_id {}",
                    input.owner_frame_id.0
                ))
            })?;
        let frame = &self.catalog.frames[frame_idx];
        let collection_id = frame.collection_id;
        let frame_active = frame.status == FrameStatus::Active;
        if !frame_active {
            return Err(Error::Format(format!(
                "put_vector: frame {} is not active",
                input.owner_frame_id.0
            )));
        }
        if prof {
            let p = &self.ingest_profile;
            p.vec_frame_lookup_ns.fetch_add(
                _t.elapsed().as_nanos() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
        }

        let _t = std::time::Instant::now();
        let space_idx = self
            .catalog
            .embedding_spaces
            .binary_search_by_key(&input.embedding_space_id.0, |s| s.embedding_space_id.0)
            .map_err(|_| {
                Error::Format(format!(
                    "put_vector: unknown embedding_space_id {}",
                    input.embedding_space_id.0
                ))
            })?;
        let space = &self.catalog.embedding_spaces[space_idx];
        let expected_dim = space.dimension;
        if (input.values.len() as u32) != expected_dim {
            return Err(Error::Format(format!(
                "put_vector: dimension mismatch (space dim = {expected_dim}, input len = {})",
                input.values.len()
            )));
        }
        // f8 ingest would go through a separate raw-bytes writer path,
        // deferred to a follow-up. Reject it here so the type-level shape
        // compiles but callers don't accidentally hit a half-implemented path.
        if space.dtype.is_f8() {
            return Err(Error::Unsupported(
                "put_vector: f8 ingest deferred to a follow-up phase".into(),
            ));
        }
        // Non-f8 spaces always have a primary codec (validated at
        // register_embedding_space). Ingest writes the single primary stream.
        let codec_id = space.primary_codec_id.ok_or_else(|| {
            Error::Integrity(
                "put_vector: non-f8 space has no primary_codec_id (validation should have caught this)"
                    .into(),
            )
        })?;
        if prof {
            let p = &self.ingest_profile;
            p.vec_space_lookup_ns.fetch_add(
                _t.elapsed().as_nanos() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
        }

        let _t = std::time::Instant::now();
        // Codec lookup only — encoding is deferred to commit time so the
        // hot per-call path is O(1) + O(log N) catalog ops. The raw f32
        // input is appended to a flat per-batch buffer; the entire batch
        // encodes in parallel inside `flush_pending_batch`.
        let (dim, base_bytes) = {
            let codec = self.codec_cache.get(&codec_id).ok_or_else(|| {
                Error::Integrity(format!("put_vector: codec {} not in cache", codec_id.0))
            })?;
            (codec.dimension() as usize, codec.base_bytes_per_vector())
        };
        if prof {
            let p = &self.ingest_profile;
            p.vec_codec_get_ns.fetch_add(
                _t.elapsed().as_nanos() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
        }

        let _t = std::time::Instant::now();
        let key = (input.embedding_space_id, codec_id);
        let batch = match self.pending_vector_batches.entry(key) {
            std::collections::hash_map::Entry::Vacant(slot) => {
                let segment_id = self.id_allocator.allocate_segment_id()?;
                slot.insert(PendingVectorBatch {
                    embedding_space_id: input.embedding_space_id,
                    codec_id,
                    segment_id,
                    dim,
                    base_bytes,
                    vectors: Vec::new(),
                    raw_values: Vec::with_capacity(VECTOR_CHUNK_SIZE * dim),
                    writer: VectorDataSegmentWriter::new(
                        input.embedding_space_id,
                        codec_id,
                        base_bytes,
                    ),
                })
            }
            std::collections::hash_map::Entry::Occupied(slot) => slot.into_mut(),
        };

        let vector_id = self.id_allocator.allocate_vector_id()?;
        let ordinal = batch.vectors.len() as u32;
        // Hot path: append raw f32 to the flat row-major buffer and the
        // matching VectorDesc to the order list. No allocation per row
        // beyond the Vec growth amortization.
        batch.raw_values.extend_from_slice(input.values);
        let desc = VectorDesc {
            vector_id,
            owner_frame_id: input.owner_frame_id,
            collection_id,
            embedding_space_id: input.embedding_space_id,
            data_segment_id: batch.segment_id,
            ordinal_in_segment: ordinal,
            status: VectorStatus::Active,
        };
        batch.vectors.push(desc.clone());
        if prof {
            let p = &self.ingest_profile;
            p.vec_pending_append_ns.fetch_add(
                _t.elapsed().as_nanos() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
        }

        let _t = std::time::Instant::now();
        upsert_vector(&mut self.catalog, desc);
        self.dirty_vector_ids.insert(vector_id);
        self.dirty = true;
        if prof {
            let p = &self.ingest_profile;
            p.vec_catalog_upsert_ns.fetch_add(
                _t.elapsed().as_nanos() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
        }

        // Streaming encode: drain full chunks out of `raw_values` into the
        // batch's `writer`, freeing RAM. The `batch` borrow above is
        // released by NLL before this self-method call.
        self.drain_full_vector_chunks(key)?;

        if prof {
            let p = &self.ingest_profile;
            p.vec_total_ns.fetch_add(
                _call_start.elapsed().as_nanos() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            p.vec_calls
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        Ok(vector_id)
    }

    /// If the per-batch `raw_values` buffer holds at least one full
    /// `VECTOR_CHUNK_SIZE`-row chunk, encode the leading full chunks in
    /// parallel via rayon and append the encoded bytes into the batch's
    /// `VectorDataSegmentWriter`. Drains the encoded prefix from
    /// `raw_values`; the trailing partial chunk stays in place for the
    /// next call. Doing the encode mid-ingest distributes what was a
    /// commit-time peak (parallel encode of every vector at once) across
    /// the ingest loop, in exchange for a bounded ~30–50 ms stall every
    /// CHUNK rows. Bounded RAM is the entire point.
    pub(crate) fn drain_full_vector_chunks(
        &mut self,
        key: (EmbeddingSpaceId, CodecId),
    ) -> Result<()> {
        // Phase 1: check threshold and lift the encodable prefix out.
        // Drop the `batch` borrow before touching `self.codec_cache`.
        let (dim, taken_raw, vectors_offset) = {
            let batch = match self.pending_vector_batches.get_mut(&key) {
                Some(b) => b,
                None => return Ok(()),
            };
            let dim = batch.dim;
            let pending = batch.raw_values.len() / dim;
            if pending < VECTOR_CHUNK_SIZE {
                return Ok(());
            }
            let full_count = (pending / VECTOR_CHUNK_SIZE) * VECTOR_CHUNK_SIZE;
            let bytes = full_count * dim;
            let already_written = batch.writer.item_count() as usize;
            let taken_raw: Vec<f32> = batch.raw_values.drain(..bytes).collect();
            (dim, taken_raw, already_written)
        };

        // Phase 2: parallel encode using a fresh immutable borrow on
        // `codec_cache`. No batch borrow held here.
        let n = taken_raw.len() / dim;
        let encoded: Vec<Vec<u8>> = {
            use rayon::prelude::*;
            let codec = self
                .codec_cache
                .get(&key.1)
                .ok_or_else(|| {
                    Error::Integrity(format!(
                        "drain_full_vector_chunks: codec {} missing from cache",
                        key.1.0
                    ))
                })?
                .as_ref();
            (0..n)
                .into_par_iter()
                .map(|i| codec.encode(&taken_raw[i * dim..(i + 1) * dim]))
                .collect::<Result<Vec<_>>>()?
        };

        // Phase 3: append encoded vectors to the batch writer in order.
        let batch = self
            .pending_vector_batches
            .get_mut(&key)
            .expect("BUG: pending vector batch removed during drain");
        for (i, enc) in encoded.into_iter().enumerate() {
            let desc = &batch.vectors[vectors_offset + i];
            let ord = batch.writer.append(desc.vector_id, &enc)?;
            debug_assert_eq!(ord, desc.ordinal_in_segment);
        }
        Ok(())
    }

    /// Read a previously-committed vector. `mode` selects the
    /// representation:
    ///
    /// - [`Reconstruct::StoredBytes`] returns the codec's raw base
    ///   bytes — no decode work, copy-out of the mmap.
    /// - [`Reconstruct::F32Vector`] decodes the bytes through the
    ///   primary codec and returns a `Vec<f32>` of length
    ///   `space.dimension`.
    ///
    /// v1 requires the vector to be committed; pending (uncommitted)
    /// vectors return an error directing the caller to `commit()` first.
    pub fn read_vector(&self, vector_id: VectorId, mode: Reconstruct) -> Result<ReadVectorResult> {
        self.ensure_segment_registry_loaded()?;
        let in_pending = self
            .pending_vector_batches
            .values()
            .any(|b| b.vectors.iter().any(|v| v.vector_id == vector_id));
        if in_pending {
            return Err(Error::Format(format!(
                "read_vector: vector {} not yet committed",
                vector_id.0
            )));
        }
        let desc_idx = self
            .catalog
            .vectors
            .binary_search_by_key(&vector_id.0, |v| v.vector_id.0)
            .map_err(|_| {
                Error::Format(format!("read_vector: unknown vector_id {}", vector_id.0))
            })?;
        let desc = self.catalog.vectors[desc_idx].clone();
        if desc.status != VectorStatus::Active {
            return Err(Error::Format(format!(
                "read_vector: vector {} is tombstoned",
                vector_id.0
            )));
        }

        let codec_id = {
            let idx = self
                .catalog
                .embedding_spaces
                .binary_search_by_key(&desc.embedding_space_id.0, |s| s.embedding_space_id.0)
                .map_err(|_| {
                    Error::Integrity(format!(
                        "read_vector: unknown embedding_space_id {}",
                        desc.embedding_space_id.0
                    ))
                })?;
            self.catalog.embedding_spaces[idx]
                .primary_codec_id
                .ok_or_else(|| {
                    Error::Unsupported(
                        "read_vector: f8 spaces lack a primary codec; raw read lands in Phase 4"
                            .into(),
                    )
                })?
        };
        let segment_ref = self
            .segment_registry
            .read()
            .by_id
            .get(&desc.data_segment_id)
            .copied()
            .ok_or_else(|| {
                Error::Integrity(format!(
                    "read_vector: data segment {} not in registry",
                    desc.data_segment_id.0
                ))
            })?;
        let base_bytes = {
            let codec = self.codec_cache.get(&codec_id).ok_or_else(|| {
                Error::Integrity(format!("read_vector: codec {} not in cache", codec_id.0))
            })?;
            codec.base_bytes_per_vector()
        };

        let mmap_ref = self
            .file_mmap
            .as_ref()
            .ok_or_else(|| Error::Integrity("read_vector: file mmap is not initialized".into()))?;
        let payload = mmap_segment_payload(mmap_ref, segment_ref, SegmentType::VectorData)?;
        let reader = VectorDataSegmentReader::open(payload, base_bytes)?;
        let stored_id = reader.vector_id(desc.ordinal_in_segment)?;
        if stored_id != vector_id {
            return Err(Error::Integrity(format!(
                "read_vector: ordinal {} maps to vector {} but catalog says {}",
                desc.ordinal_in_segment, stored_id.0, vector_id.0
            )));
        }
        let base = reader.base(desc.ordinal_in_segment)?;
        match mode {
            Reconstruct::StoredBytes => Ok(ReadVectorResult::StoredBytes(base.to_vec())),
            Reconstruct::F32Vector => {
                let codec = self
                    .codec_cache
                    .get(&codec_id)
                    .expect("BUG: codec_cache lookup raced");
                Ok(ReadVectorResult::F32Vector(codec.decode_lossy(base)?))
            }
        }
    }

    /// Tombstone a committed vector. v1 rejects deleting an uncommitted
    /// (pending) vector — call `commit()` first.
    pub fn delete_vector(&mut self, vector_id: VectorId) -> Result<()> {
        self.ensure_write()?;
        let in_pending = self
            .pending_vector_batches
            .values()
            .any(|b| b.vectors.iter().any(|v| v.vector_id == vector_id));
        if in_pending {
            return Err(Error::Format(format!(
                "delete_vector: vector {} is uncommitted; call commit() first",
                vector_id.0
            )));
        }
        let active = self
            .catalog
            .vectors
            .iter()
            .any(|v| v.vector_id == vector_id && v.status == VectorStatus::Active);
        if !active {
            return Err(Error::Format(format!(
                "delete_vector: unknown active vector_id {}",
                vector_id.0
            )));
        }
        if let Ok(idx) = self
            .catalog
            .vectors
            .binary_search_by_key(&vector_id.0, |v| v.vector_id.0)
        {
            self.catalog.vectors[idx].status = VectorStatus::Tombstoned;
        }
        self.dirty_vector_ids.insert(vector_id);
        self.dirty = true;
        Ok(())
    }

    pub fn delete_frame(&mut self, frame_id: FrameId) -> Result<()> {
        self.ensure_write()?;
        let frame_active = self
            .catalog
            .frames
            .binary_search_by_key(&frame_id.0, |f| f.frame_id.0)
            .ok()
            .map(|i| self.catalog.frames[i].status == FrameStatus::Active)
            .unwrap_or(false);
        if !frame_active {
            return Err(Error::Format(format!(
                "unknown active frame id: {}",
                frame_id.0
            )));
        }
        let updated_at = current_unix_timestamp()?;
        if let Ok(idx) = self
            .catalog
            .frames
            .binary_search_by_key(&frame_id.0, |f| f.frame_id.0)
        {
            let frame = &mut self.catalog.frames[idx];
            frame.status = FrameStatus::Tombstoned;
            frame.role = FrameRole::Tombstone;
            frame.updated_at = updated_at;
        }
        // Mirror the tombstone into the slim `frame_stubs` index so the
        // retrieval hot path (BM25 / Tfidf / Jaccard tombstone filter)
        // sees the deletion before the next commit.
        if let Ok(idx) = self
            .catalog
            .frame_stubs
            .binary_search_by_key(&frame_id.0, |s| s.frame_id.0)
        {
            if self.catalog.frame_stubs[idx].status != FrameStatus::Tombstoned {
                self.tombstoned_frame_count += 1;
            }
            self.catalog.frame_stubs[idx].status = FrameStatus::Tombstoned;
        }
        self.dirty_frame_ids.insert(frame_id);
        self.dirty = true;
        Ok(())
    }

    /// Read the raw text of a frame as UTF-8, spec §12 (raw text). v1 stores
    /// raw text directly in the frame's `PayloadSegment` (§11) — separate
    /// `NXRT` / `NXRD` segments are reserved for v2 (deduplicated /
    /// dictionary-compressed text). Returns an error if the bytes are not
    /// valid UTF-8.
    pub fn read_raw_text(&self, frame_id: FrameId) -> Result<String> {
        let bytes = self.read_payload(frame_id)?;
        String::from_utf8(bytes).map_err(|err| {
            Error::Format(format!(
                "read_raw_text: frame {} payload is not valid UTF-8: {err}",
                frame_id.0
            ))
        })
    }

    pub fn read_payload(&self, frame_id: FrameId) -> Result<Vec<u8>> {
        // Pending buffer hit: payload was put this commit but the
        // batch hasn't flushed yet. Slice it out of the in-memory
        // buffer without touching the file. This keeps the
        // read-your-writes semantic that callers relied on under the
        // legacy per-frame model.
        if let Some(&idx) = self.pending_payload_by_frame.get(&frame_id) {
            let entry = &self.pending_payload_frames[idx];
            let start = entry.bytes_offset as usize;
            let end = start + entry.bytes_length as usize;
            return Ok(self.pending_payload_buf[start..end].to_vec());
        }
        let frame = self.frame_full(frame_id)?;
        if frame.payload_encoding != PayloadEncoding::Raw {
            return Err(Error::Unsupported(
                "compressed frame payloads are not implemented in v0.1-alpha".into(),
            ));
        }
        let payload_ref = frame
            .payload_ref
            .ok_or_else(|| Error::Format("frame has no materialized payload".into()))?;
        self.ensure_segment_registry_loaded()?;
        let segment = self
            .segment_registry
            .read()
            .by_id
            .get(&payload_ref.segment_id)
            .copied()
            .unwrap_or_else(|| legacy_offset_segment_ref(payload_ref.segment_id));
        // Stage 6 fix: mmap-based slice of the payload segment, then a
        // bounded `bytes_offset..bytes_length` view. Replaces the file
        // `Mutex` round-trip that bottlenecked the read sweep at ~30 k
        // ops/sec. The kernel page cache makes the mmap and any
        // concurrent post-commit reads coherent. Falls back to file
        // read when the mmap doesn't cover the requested range
        // (pre-commit reads, where the writer has extended the file
        // past the most recent `remap_file()`).
        // Peek the segment header's compression flag from the registry
        // catalog. If the payload is compressed we route through the
        // file-based `read_payload_ref` path which does a one-shot
        // zstd decode; the mmap fast path stays for uncompressed
        // Payload segments. Payload reads are cold relative to the
        // BM25 query loop, so the extra read/decompress on compressed
        // segments is acceptable for the ~2.5× on-disk savings.
        let compression = self
            .segment_registry
            .read()
            .catalog
            .binary_search_by_key(&payload_ref.segment_id.0, |e| e.segment_id.0)
            .ok()
            .and_then(|i| {
                let reg = self.segment_registry.read();
                reg.catalog.get(i).map(|e| e.compression)
            })
            .unwrap_or(crate::format::segment::Compression::None);

        let payload = if compression == crate::format::segment::Compression::None {
            match self.file_mmap.as_ref().and_then(|m| {
                if segment.offset + segment.length <= m.len() as u64 {
                    Some(m)
                } else {
                    None
                }
            }) {
                Some(mmap) => {
                    let segment_payload =
                        mmap_segment_payload(mmap, segment, SegmentType::Payload)?;
                    // Integrity: re-hash the wire bytes against the
                    // stored BLAKE3 once per segment per handle — the
                    // header/registry cross-check inside
                    // `mmap_segment_payload` only proves the two STORED
                    // copies agree with each other.
                    verify_payload_wire_bytes(
                        segment.segment_id,
                        &segment.checksum,
                        segment_payload,
                        &self.verified_payload_segments,
                    )?;
                    let end = payload_ref
                        .bytes_offset
                        .checked_add(payload_ref.bytes_length)
                        .ok_or_else(|| Error::Format("payload ref range overflow".into()))?;
                    if end > segment_payload.len() as u64 {
                        return Err(Error::Integrity(
                            "payload ref exceeds segment length".into(),
                        ));
                    }
                    segment_payload[payload_ref.bytes_offset as usize..end as usize].to_vec()
                }
                None => read_payload_ref(
                    &mut self.file.lock(),
                    segment,
                    payload_ref,
                    &self.verified_payload_segments,
                )?,
            }
        } else {
            // Compressed Payload segments take the file-based slow path
            // unconditionally — `read_payload_ref` knows how to
            // decompress and slice (and re-hashes the wire bytes once
            // per segment per open before trusting the zstd stream).
            read_payload_ref(
                &mut self.file.lock(),
                segment,
                payload_ref,
                &self.verified_payload_segments,
            )?
        };
        Ok(payload)
    }
}
