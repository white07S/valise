// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! `ValiseFile` collection + frame write path.

use super::*;

impl ValiseFile {
    pub fn create_collection(&mut self, name: impl Into<String>) -> Result<CollectionId> {
        self.create_collection_at(name, current_unix_timestamp()?)
    }

    /// Like [`Self::create_collection`] but with an explicit
    /// `created_at` timestamp. Used by golden-format tests to keep the
    /// catalog encoding bit-stable across process runs.
    pub fn create_collection_at(
        &mut self,
        name: impl Into<String>,
        created_at: i64,
    ) -> Result<CollectionId> {
        self.ensure_write()?;
        let name = name.into();
        if self
            .catalog
            .collections
            .iter()
            .any(|collection| collection.name == name && collection.flags & 1 == 0)
        {
            return Err(Error::Format(format!("collection already exists: {name}")));
        }

        let collection_id = self.id_allocator.allocate_collection_id()?;
        let collection = CollectionDesc {
            collection_id,
            name,
            created_at,
            default_text_space_id: None,
            default_embedding_space_id: None,
            flags: 0,
            metadata_ref: None,
        };
        upsert_collection(&mut self.catalog, collection);
        self.dirty_collection_ids.insert(collection_id);
        self.dirty = true;
        Ok(collection_id)
    }

    pub fn put_frame(&mut self, input: PutFrame<'_>) -> Result<FrameId> {
        self.ensure_write()?;
        self.ensure_segment_registry_loaded()?;
        let prof = self.ingest_profile_enabled;
        let _call_start = std::time::Instant::now();

        let _t = std::time::Instant::now();
        if !self
            .catalog
            .collections
            .iter()
            .any(|collection| collection.collection_id == input.collection_id)
        {
            return Err(Error::Format(format!(
                "unknown collection id: {}",
                input.collection_id.0
            )));
        }
        if prof {
            let p = &self.ingest_profile;
            p.put_frame_collection_lookup_ns.fetch_add(
                _t.elapsed().as_nanos() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
        }

        let _t = std::time::Instant::now();
        let now = current_unix_timestamp()?;
        let frame_id = self.id_allocator.allocate_frame_id()?;
        if prof {
            let p = &self.ingest_profile;
            p.put_frame_allocate_ids_ns.fetch_add(
                _t.elapsed().as_nanos() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
        }

        // Buffer the raw payload bytes; the segment_id and bytes_offset
        // are filled in at flush time (see `flush_pending_payloads`).
        // `payload_ref = None` is the sentinel for "still in the
        // pending buffer"; the read path checks `pending_payload_by_frame`
        // before consulting the materialized segment registry.
        let _t = std::time::Instant::now();
        let bytes_offset = self.pending_payload_buf.len() as u64;
        let bytes_length = input.payload.len() as u64;
        self.pending_payload_buf.extend_from_slice(input.payload);
        let pending_index = self.pending_payload_frames.len();
        self.pending_payload_frames.push(PendingPayloadFrame {
            frame_id,
            bytes_offset,
            bytes_length,
        });
        self.pending_payload_by_frame
            .insert(frame_id, pending_index);
        // Buffer the identity key/URI the same way as the payload; the
        // `uri_ref` MetadataRef is filled in at flush time (see
        // `flush_pending_payloads`). Stays `None` here, like `payload_ref`.
        if let Some(uri) = input.uri {
            let uri_offset = self.pending_uri_buf.len() as u64;
            self.pending_uri_buf.extend_from_slice(uri);
            self.pending_uris.push(PendingUri {
                frame_id,
                bytes_offset: uri_offset,
                bytes_length: uri.len() as u64,
            });
        }
        let frame = FrameDesc {
            frame_id,
            collection_id: input.collection_id,
            role: input.role,
            created_at: input.created_at.unwrap_or(now),
            updated_at: input.updated_at.unwrap_or(now),
            uri_ref: None,
            title_ref: None,
            payload_ref: None,
            payload_encoding: PayloadEncoding::Raw,
            search_text_ref: None,
            parent_frame_id: input.parent_frame_id,
            chunk_index: input.chunk_index,
            chunk_count: input.chunk_count,
            metadata_ref: None,
            status: FrameStatus::Active,
        };
        if prof {
            let p = &self.ingest_profile;
            p.put_frame_segment_write_ns.fetch_add(
                _t.elapsed().as_nanos() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
        }

        let _t = std::time::Instant::now();
        upsert_frame(&mut self.catalog, frame);
        self.dirty_frame_ids.insert(frame_id);
        self.dirty = true;
        if prof {
            let p = &self.ingest_profile;
            p.put_frame_catalog_upsert_ns.fetch_add(
                _t.elapsed().as_nanos() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
        }

        // Flush early when the buffer crosses the soft cap so a single
        // huge commit doesn't accumulate hundreds of MiB in memory.
        // No fsync — the commit's terminal F_FULLFSYNC will cover
        // these bytes alongside the catalog rewrite. Under Buffered
        // durability this matches the legacy per-frame write profile
        // (one `write_all` per frame, page cache draining at commit).
        if self.pending_payload_buf.len() >= PAYLOAD_BATCH_FLUSH_BYTES {
            self.flush_pending_payloads()?;
        }

        if prof {
            let p = &self.ingest_profile;
            p.put_frame_total_ns.fetch_add(
                _call_start.elapsed().as_nanos() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            p.put_frame_calls
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        Ok(frame_id)
    }

    /// Drain `pending_payload_buf` into a single `Payload` segment, fix
    /// up every pending frame's `payload_ref`, and reflect the result
    /// in the in-memory catalog. Idempotent on empty input.
    ///
    /// The batched bytes are zstd-compressed at level 3 before being
    /// written. Quora-shaped English text compresses ~2.5–3× at this
    /// level; the level was picked as a balance between encode-time
    /// throughput and final ratio (level 1 is ~30% faster but ~10%
    /// fatter; level 6 is ~5% denser but 2× slower). Decompression on
    /// the read path is rare relative to BM25 query traffic.
    pub(crate) fn flush_pending_payloads(&mut self) -> Result<()> {
        if self.pending_payload_frames.is_empty() {
            debug_assert!(self.pending_payload_buf.is_empty());
            return Ok(());
        }
        let segment_id = self.id_allocator.allocate_segment_id()?;
        let item_count = u32::try_from(self.pending_payload_frames.len())
            .map_err(|_| Error::Format("payload batch too large for u32 item_count".into()))?;
        let uncompressed_length = self.pending_payload_buf.len() as u64;
        let compressed = zstd::encode_all(self.pending_payload_buf.as_slice(), 3)
            .map_err(|err| Error::Format(format!("zstd encode failed: {err}")))?;
        let payload_segment = crate::file::segment_io::append_compressed_segment_at_end(
            &mut self.file.lock(),
            segment_id,
            SegmentType::Payload,
            item_count,
            &compressed,
            uncompressed_length,
            crate::format::segment::Compression::Zstd,
        )?;
        sync_file(&self.file.lock(), self.durability)?;
        register_segment(
            SegmentRegistryMut {
                registry: self.segment_registry.get_mut(),
                dirty_ids: &mut self.dirty_segment_ids,
            },
            crate::file::segment_io::compressed_segment_catalog_entry(
                segment_id,
                SegmentType::Payload,
                item_count,
                &compressed,
                uncompressed_length,
                crate::format::segment::Compression::Zstd,
                payload_segment,
            )?,
        );
        // Drain any buffered identity keys into one `Metadata` segment
        // (uncompressed — keys are short and high-entropy, so zstd would
        // only add overhead) and resolve each to a `MetadataRef`. Mirrors
        // the payload batching above; `uri_ref` is set in the same
        // catalog scan below. Usually empty.
        let uri_refs_by_frame: HashMap<FrameId, crate::format::catalog::MetadataRef> =
            if self.pending_uris.is_empty() {
                HashMap::new()
            } else {
                let uri_segment_id = self.id_allocator.allocate_segment_id()?;
                let uri_item_count = u32::try_from(self.pending_uris.len())
                    .map_err(|_| Error::Format("uri batch too large for u32 item_count".into()))?;
                let uri_segment = append_segment_at_end(
                    &mut self.file.lock(),
                    uri_segment_id,
                    SegmentType::Metadata,
                    uri_item_count,
                    &self.pending_uri_buf,
                )?;
                sync_file(&self.file.lock(), self.durability)?;
                register_segment(
                    SegmentRegistryMut {
                        registry: self.segment_registry.get_mut(),
                        dirty_ids: &mut self.dirty_segment_ids,
                    },
                    segment_catalog_entry(
                        uri_segment_id,
                        SegmentType::Metadata,
                        uri_item_count,
                        &self.pending_uri_buf,
                        uri_segment,
                    )?,
                );
                let pending_uris = std::mem::take(&mut self.pending_uris);
                let mut map = HashMap::with_capacity(pending_uris.len());
                for u in &pending_uris {
                    let start = u.bytes_offset as usize;
                    let end = start + u.bytes_length as usize;
                    let checksum =
                        crate::format::blake3_checksum(&self.pending_uri_buf[start..end]);
                    map.insert(
                        u.frame_id,
                        crate::format::catalog::MetadataRef {
                            segment_id: uri_segment_id,
                            bytes_offset: u.bytes_offset,
                            bytes_length: u.bytes_length,
                            checksum,
                        },
                    );
                    self.dirty_frame_ids.insert(u.frame_id);
                }
                map
            };

        // Take ownership so we can re-upsert the FrameDescs without
        // borrowing self twice. Cleared either way. Build a HashMap
        // for O(1) lookup so we can do a single linear pass over
        // `catalog.frames` instead of an O(n²) linear-search-per-
        // pending-frame (which made a Quora-scale ingest balloon to
        // 178 s before this fix).
        let pending = std::mem::take(&mut self.pending_payload_frames);
        let mut refs_by_frame: HashMap<FrameId, (u64, u64)> = HashMap::with_capacity(pending.len());
        for entry in &pending {
            refs_by_frame.insert(entry.frame_id, (entry.bytes_offset, entry.bytes_length));
            self.dirty_frame_ids.insert(entry.frame_id);
        }
        let mut matched = 0usize;
        for frame in &mut self.catalog.frames {
            if let Some(&(offset, length)) = refs_by_frame.get(&frame.frame_id) {
                frame.payload_ref = Some(PayloadRef {
                    segment_id,
                    bytes_offset: offset,
                    bytes_length: length,
                });
                matched += 1;
            }
            if let Some(uri_ref) = uri_refs_by_frame.get(&frame.frame_id) {
                frame.uri_ref = Some(uri_ref.clone());
            }
        }
        if matched != refs_by_frame.len() {
            return Err(Error::Integrity(format!(
                "flush_pending_payloads: matched {matched}/{} pending frames in catalog",
                refs_by_frame.len()
            )));
        }
        self.pending_payload_buf.clear();
        self.pending_payload_by_frame.clear();
        self.pending_uri_buf.clear();
        Ok(())
    }
}
