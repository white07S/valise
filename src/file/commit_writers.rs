// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! `ValiseFile` commit-time segment writers: pending text-index / vector-batch flush
//! and the catalog-segment writer.

use super::*;

impl ValiseFile {
    pub(crate) fn flush_pending_text_indexes_profiled(
        &mut self,
        body: &mut TocFooterBody,
        profile: &mut CommitProfile,
    ) -> Result<()> {
        let text_space_ids: Vec<TextSpaceId> = self.pending_text_indexes.keys().copied().collect();
        for text_space_id in text_space_ids {
            let mut pending = self
                .pending_text_indexes
                .remove(&text_space_id)
                .expect("BUG: pending text index disappeared between key listing and removal");
            if pending.is_empty() {
                continue;
            }

            let (prior_term_dict, prior_postings, prior_doc_stats) = {
                let root = text_index_root_mut(body, text_space_id);
                (
                    root.term_dictionary_root,
                    root.postings_root,
                    root.doc_stats_root,
                )
            };

            let state = self.text_space_states.entry(text_space_id).or_default();
            let t_build = std::time::Instant::now();
            let out = build_flush_output(
                text_space_id,
                &mut pending,
                state,
                prior_term_dict,
                prior_postings,
                prior_doc_stats,
                Some(&mut profile.text_indexes_build_breakdown),
            )?;
            profile.text_indexes_build += t_build.elapsed();

            let t_write = std::time::Instant::now();

            // Term dictionary segment.
            let term_dict_seg_id = self.id_allocator.allocate_segment_id()?;
            let term_dict_ref = append_segment_at_end(
                &mut self.file.lock(),
                term_dict_seg_id,
                SegmentType::TermDictionary,
                out.term_dict_record_count,
                &out.term_dict_segment,
            )?;
            register_segment(
                SegmentRegistryMut {
                    registry: self.segment_registry.get_mut(),
                    dirty_ids: &mut self.dirty_segment_ids,
                },
                segment_catalog_entry(
                    term_dict_seg_id,
                    SegmentType::TermDictionary,
                    out.term_dict_record_count,
                    &out.term_dict_segment,
                    term_dict_ref,
                )?,
            );

            // Postings segment.
            let postings_seg_id = self.id_allocator.allocate_segment_id()?;
            let postings_ref = append_segment_at_end(
                &mut self.file.lock(),
                postings_seg_id,
                SegmentType::Postings,
                out.postings_record_count,
                &out.postings_segment,
            )?;
            register_segment(
                SegmentRegistryMut {
                    registry: self.segment_registry.get_mut(),
                    dirty_ids: &mut self.dirty_segment_ids,
                },
                segment_catalog_entry(
                    postings_seg_id,
                    SegmentType::Postings,
                    out.postings_record_count,
                    &out.postings_segment,
                    postings_ref,
                )?,
            );

            // DocStats segment.
            let doc_stats_seg_id = self.id_allocator.allocate_segment_id()?;
            let doc_stats_ref = append_segment_at_end(
                &mut self.file.lock(),
                doc_stats_seg_id,
                SegmentType::DocStats,
                out.doc_stats_record_count,
                &out.doc_stats_segment,
            )?;
            register_segment(
                SegmentRegistryMut {
                    registry: self.segment_registry.get_mut(),
                    dirty_ids: &mut self.dirty_segment_ids,
                },
                segment_catalog_entry(
                    doc_stats_seg_id,
                    SegmentType::DocStats,
                    out.doc_stats_record_count,
                    &out.doc_stats_segment,
                    doc_stats_ref,
                )?,
            );

            // Update body.text_index_roots with the new heads.
            let root = text_index_root_mut(body, text_space_id);
            root.term_dictionary_root = Some(term_dict_ref);
            root.postings_root = Some(postings_ref);
            root.doc_stats_root = Some(doc_stats_ref);
            profile.text_indexes_write += t_write.elapsed();
        }
        Ok(())
    }

    pub(crate) fn flush_pending_vector_batches(&mut self) -> Result<()> {
        let keys: Vec<(EmbeddingSpaceId, CodecId)> =
            self.pending_vector_batches.keys().copied().collect();
        for key in keys {
            let batch = self
                .pending_vector_batches
                .remove(&key)
                .expect("BUG: pending batch disappeared between key listing and removal");
            if batch.vectors.is_empty() {
                continue;
            }
            self.flush_pending_batch(batch)?;
        }
        Ok(())
    }

    pub(crate) fn flush_pending_batch(&mut self, mut batch: PendingVectorBatch) -> Result<()> {
        use rayon::prelude::*;
        let item_count = u32::try_from(batch.vectors.len()).map_err(|_| {
            Error::Format(format!(
                "pending vector batch overflow: {} vectors",
                batch.vectors.len()
            ))
        })?;
        let segment_id = batch.segment_id;

        // Streaming encode left only the trailing partial chunk in
        // raw_values; everything before the last `VECTOR_CHUNK_SIZE`
        // boundary is already in `batch.writer`. Encode the trailing
        // tail (parallel rayon, same as the per-chunk drain) and append.
        let dim = batch.dim;
        let tail_count = batch.raw_values.len() / dim;
        if tail_count > 0 {
            let codec = self
                .codec_cache
                .get(&batch.codec_id)
                .ok_or_else(|| {
                    Error::Integrity(format!(
                        "flush_pending_batch: codec {} missing from cache",
                        batch.codec_id.0
                    ))
                })?
                .as_ref();
            let raw_ref: &[f32] = &batch.raw_values;
            let encoded: Vec<Vec<u8>> = (0..tail_count)
                .into_par_iter()
                .map(|i| codec.encode(&raw_ref[i * dim..(i + 1) * dim]))
                .collect::<Result<Vec<_>>>()?;
            let starting_offset = batch.writer.item_count() as usize;
            for (i, enc) in encoded.into_iter().enumerate() {
                let desc = &batch.vectors[starting_offset + i];
                let written_ord = batch.writer.append(desc.vector_id, &enc)?;
                debug_assert_eq!(written_ord, desc.ordinal_in_segment);
            }
        }
        batch.raw_values = Vec::new();
        debug_assert_eq!(batch.writer.item_count() as usize, batch.vectors.len());

        let writer = std::mem::replace(
            &mut batch.writer,
            VectorDataSegmentWriter::new(
                batch.embedding_space_id,
                batch.codec_id,
                batch.base_bytes,
            ),
        );
        let payload = writer.finish()?;
        let segment_ref = segment_ref_at_end(
            &self.file.lock(),
            segment_id,
            SegmentType::VectorData,
            item_count,
            &payload,
        )?;

        let written = append_segment_at_end(
            &mut self.file.lock(),
            segment_id,
            SegmentType::VectorData,
            item_count,
            &payload,
        )?;
        debug_assert_eq!(written, segment_ref);
        sync_file(&self.file.lock(), self.durability)?;

        register_segment(
            SegmentRegistryMut {
                registry: self.segment_registry.get_mut(),
                dirty_ids: &mut self.dirty_segment_ids,
            },
            segment_catalog_entry(
                segment_id,
                SegmentType::VectorData,
                item_count,
                &[],
                segment_ref,
            )?,
        );

        let _ = batch.embedding_space_id;
        let _ = batch.codec_id;
        Ok(())
    }

    pub(crate) fn write_catalog_segments_profiled(
        &mut self,
        snapshot_generation: u64,
        profile: &mut CommitProfile,
    ) -> Result<TocFooterBody> {
        // No commit-time vector tiering: the sign-sketch index is derived
        // in-memory from the QAM phase codes at file-open, so nothing
        // vector-specific is built or persisted here.
        let mut body = TocFooterBody {
            snapshot_generation,
            id_allocator: self.id_allocator.clone(),
            ..self.footer.body.clone()
        };

        let t = std::time::Instant::now();
        self.flush_pending_text_indexes_profiled(&mut body, profile)?;
        profile.text_indexes = t.elapsed();

        let t = std::time::Instant::now();
        flush_catalog_table(
            &mut self.file.lock(),
            &mut self.id_allocator,
            SegmentRegistryMut {
                registry: self.segment_registry.get_mut(),
                dirty_ids: &mut self.dirty_segment_ids,
            },
            &mut body.collection_catalog_root,
            &self.dirty_collection_ids,
            &self.catalog.collections,
            CatalogTableKind::Collection,
            SegmentType::CollectionCatalog,
            |collection| collection.collection_id,
        )?;
        flush_frame_catalog_columnar(
            &mut self.file.lock(),
            &mut self.id_allocator,
            SegmentRegistryMut {
                registry: self.segment_registry.get_mut(),
                dirty_ids: &mut self.dirty_segment_ids,
            },
            &mut body.frame_catalog_root,
            &self.dirty_frame_ids,
            &self.catalog.frames,
        )?;
        flush_catalog_table(
            &mut self.file.lock(),
            &mut self.id_allocator,
            SegmentRegistryMut {
                registry: self.segment_registry.get_mut(),
                dirty_ids: &mut self.dirty_segment_ids,
            },
            &mut body.text_space_catalog_root,
            &self.dirty_text_space_ids,
            &self.catalog.text_spaces,
            CatalogTableKind::TextSpace,
            SegmentType::TextSpaceCatalog,
            |s| s.text_space_id,
        )?;
        flush_catalog_table(
            &mut self.file.lock(),
            &mut self.id_allocator,
            SegmentRegistryMut {
                registry: self.segment_registry.get_mut(),
                dirty_ids: &mut self.dirty_segment_ids,
            },
            &mut body.analyzer_catalog_root,
            &self.dirty_analyzer_ids,
            &self.catalog.analyzers,
            CatalogTableKind::Analyzer,
            SegmentType::AnalyzerCatalog,
            |a| a.analyzer_id,
        )?;
        flush_catalog_table(
            &mut self.file.lock(),
            &mut self.id_allocator,
            SegmentRegistryMut {
                registry: self.segment_registry.get_mut(),
                dirty_ids: &mut self.dirty_segment_ids,
            },
            &mut body.field_schema_catalog_root,
            &self.dirty_field_schema_ids,
            &self.catalog.field_schemas,
            CatalogTableKind::FieldSchema,
            SegmentType::FieldSchemaCatalog,
            |f| f.field_schema_id,
        )?;
        flush_catalog_table(
            &mut self.file.lock(),
            &mut self.id_allocator,
            SegmentRegistryMut {
                registry: self.segment_registry.get_mut(),
                dirty_ids: &mut self.dirty_segment_ids,
            },
            &mut body.retrieval_profile_catalog_root,
            &self.dirty_retrieval_profile_ids,
            &self.catalog.retrieval_profiles,
            CatalogTableKind::RetrievalProfile,
            SegmentType::RetrievalProfileCatalog,
            |p| p.profile_id,
        )?;
        flush_catalog_table(
            &mut self.file.lock(),
            &mut self.id_allocator,
            SegmentRegistryMut {
                registry: self.segment_registry.get_mut(),
                dirty_ids: &mut self.dirty_segment_ids,
            },
            &mut body.embedding_space_catalog_root,
            &self.dirty_embedding_space_ids,
            &self.catalog.embedding_spaces,
            CatalogTableKind::EmbeddingSpace,
            SegmentType::EmbeddingSpaceCatalog,
            |space| space.embedding_space_id,
        )?;
        flush_catalog_table(
            &mut self.file.lock(),
            &mut self.id_allocator,
            SegmentRegistryMut {
                registry: self.segment_registry.get_mut(),
                dirty_ids: &mut self.dirty_segment_ids,
            },
            &mut body.codec_catalog_root,
            &self.dirty_codec_ids,
            &self.catalog.codecs,
            CatalogTableKind::Codec,
            SegmentType::CodecCatalog,
            |codec| codec.codec_id,
        )?;
        flush_catalog_table(
            &mut self.file.lock(),
            &mut self.id_allocator,
            SegmentRegistryMut {
                registry: self.segment_registry.get_mut(),
                dirty_ids: &mut self.dirty_segment_ids,
            },
            &mut body.vector_catalog_root,
            &self.dirty_vector_ids,
            &self.catalog.vectors,
            CatalogTableKind::Vector,
            SegmentType::VectorCatalog,
            |vector| vector.vector_id,
        )?;
        flush_catalog_table(
            &mut self.file.lock(),
            &mut self.id_allocator,
            SegmentRegistryMut {
                registry: self.segment_registry.get_mut(),
                dirty_ids: &mut self.dirty_segment_ids,
            },
            &mut body.fusion_profile_catalog_root,
            &self.dirty_fusion_profile_ids,
            &self.catalog.fusion_profiles,
            CatalogTableKind::FusionProfile,
            SegmentType::FusionProfileCatalog,
            |p| p.fusion_profile_id,
        )?;
        profile.catalog_tables = t.elapsed();

        // Rebuild collection filters for collections whose membership
        // changed (frames or vectors). Updates the on-disk root chain and
        // the in-memory cache.
        let t = std::time::Instant::now();
        let dirty_filters = filter_io::dirty_filter_collections(
            &self.catalog,
            &self.dirty_frame_ids,
            &self.dirty_vector_ids,
        );
        for cid in dirty_filters {
            filter_io::write_collection_filter_segment(
                &mut self.file.lock(),
                &mut self.id_allocator,
                SegmentRegistryMut {
                    registry: self.segment_registry.get_mut(),
                    dirty_ids: &mut self.dirty_segment_ids,
                },
                &mut body,
                &self.catalog,
                cid,
            )?;
            // Refresh the in-memory cache from the freshly-built snapshot
            // so subsequent helpers don't have to re-read.
            let frame_ids: Vec<FrameId> = self
                .catalog
                .frames
                .iter()
                .filter(|f| f.collection_id == cid && f.status == FrameStatus::Active)
                .map(|f| f.frame_id)
                .collect();
            let vector_ids: Vec<VectorId> = self
                .catalog
                .vectors
                .iter()
                .filter(|v| v.collection_id == cid && v.status == VectorStatus::Active)
                .map(|v| v.vector_id)
                .collect();
            let mut frame_ids = frame_ids;
            frame_ids.sort_by_key(|f| f.0);
            let mut vector_ids = vector_ids;
            vector_ids.sort_by_key(|v| v.0);
            self.collection_filter_cache
                .insert(cid, (frame_ids, vector_ids));
        }
        profile.collection_filters = t.elapsed();

        // Rebuild the global time index whenever frames changed. v1 emits
        // a single global segment per commit; partitioned variants are v2.
        let t = std::time::Instant::now();
        if !self.dirty_frame_ids.is_empty() {
            time_index_io::write_time_index_segment(
                &mut self.file.lock(),
                &mut self.id_allocator,
                SegmentRegistryMut {
                    registry: self.segment_registry.get_mut(),
                    dirty_ids: &mut self.dirty_segment_ids,
                },
                &mut body,
                &self.catalog,
            )?;
            // Rebuild the in-memory cache from the same source the segment
            // saw, so reads after commit don't have to re-decode.
            let mut entries: Vec<crate::format::time_index_segment::TimeIndexEntry> = self
                .catalog
                .frames
                .iter()
                .filter(|f| f.status == FrameStatus::Active)
                .map(|f| crate::format::time_index_segment::TimeIndexEntry {
                    frame_id: f.frame_id,
                    collection_id: f.collection_id,
                    created_at: f.created_at,
                    updated_at: f.updated_at,
                })
                .collect();
            entries.sort_by(|a, b| {
                a.created_at
                    .cmp(&b.created_at)
                    .then(a.frame_id.0.cmp(&b.frame_id.0))
            });
            self.time_index_cache = entries;
        }
        profile.time_index = t.elapsed();

        // The segment-registry spine is rooted directly by the TOC and is not
        // self-listed (spec §9.4), so it cannot go through the registered-
        // segment helper above.
        let t = std::time::Instant::now();
        if !self.dirty_segment_ids.is_empty() {
            let segments = dirty_records(
                &self.segment_registry.get_mut().catalog,
                &self.dirty_segment_ids,
                |segment| segment.segment_id,
            );
            let segment_payload = encode_catalog_delta(
                CatalogTableKind::Segment,
                body.segment_catalog_root,
                &segments,
            )?;
            let root = append_segment_at_end(
                &mut self.file.lock(),
                self.id_allocator.allocate_segment_id()?,
                SegmentType::SegmentRegistry,
                segments.len() as u32,
                &segment_payload,
            )?;
            body.segment_catalog_root = Some(root);
        }
        profile.segment_registry = t.elapsed();
        body.id_allocator = self.id_allocator.clone();
        Ok(body)
    }
}
