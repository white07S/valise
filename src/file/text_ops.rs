//! `ValiseFile` text indexing + query.

use super::*;

impl ValiseFile {
    /// Index a frame's payload bytes into the given text space. The payload
    /// is treated as UTF-8; tokenization is driven by the text space's
    /// analyzer. The pending state lands on disk at the next `commit()` call.
    /// Reading the indexed terms / running queries before commit is rejected.
    ///
    /// v1 simplifications: source text is the frame's `payload` bytes
    /// (`search_text_ref` resolution is v2); single-field schema is assumed
    /// (multi-field is v2); a crash before commit drops the pending buffer
    /// and the user must re-call `index_frame_text` (spec §8.2 — pending
    /// state is in-memory only and nothing persists until commit).
    pub fn index_frame_text(
        &mut self,
        frame_id: FrameId,
        text_space_id: TextSpaceId,
    ) -> Result<()> {
        self.ensure_write()?;
        let prof = self.ingest_profile_enabled;
        let _call_start = std::time::Instant::now();

        let _t = std::time::Instant::now();
        // catalog.frames is kept sorted by frame_id by every upsert path,
        // so a binary search is correct and O(log N) — no side-table.
        let frame_idx = self
            .catalog
            .frames
            .binary_search_by_key(&frame_id.0, |f| f.frame_id.0)
            .map_err(|_| {
                Error::Format(format!("index_frame_text: unknown frame_id {}", frame_id.0))
            })?;
        let frame = &self.catalog.frames[frame_idx];
        if frame.status != FrameStatus::Active {
            return Err(Error::Format(format!(
                "index_frame_text: frame {} is not Active",
                frame_id.0
            )));
        }
        let collection_id = frame.collection_id;
        if prof {
            let p = &self.ingest_profile;
            p.idx_frame_lookup_ns.fetch_add(
                _t.elapsed().as_nanos() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
        }

        let _t = std::time::Instant::now();
        let analyzer = self
            .analyzer_cache
            .get(&text_space_id)
            .ok_or_else(|| {
                Error::Format(format!(
                    "index_frame_text: unknown text_space_id {}",
                    text_space_id.0
                ))
            })?
            .clone();
        if prof {
            let p = &self.ingest_profile;
            p.idx_text_space_lookup_ns.fetch_add(
                _t.elapsed().as_nanos() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
        }

        // Split the legacy lumped `idx_analyzer_apply_ns` into three
        // sub-timers (read / utf8 / analyze) so the profile shows where
        // the per-doc 55 µs actually goes. The legacy field is still
        // populated as the sum so older consumers don't break.
        let analyzer_apply_t = std::time::Instant::now();
        let t_read = std::time::Instant::now();
        let payload_bytes = self.read_payload(frame_id)?;
        let read_payload_ns = t_read.elapsed().as_nanos() as u64;

        let t_utf8 = std::time::Instant::now();
        let text = std::str::from_utf8(&payload_bytes).map_err(|err| {
            Error::Format(format!(
                "index_frame_text: frame {} payload is not UTF-8 ({err})",
                frame_id.0
            ))
        })?;
        let utf8_ns = t_utf8.elapsed().as_nanos() as u64;

        let t_analyze = std::time::Instant::now();
        let mut breakdown = crate::text::analyzer::AnalyzeBreakdown::default();
        let tokens = if prof {
            analyzer.analyze_with_breakdown(text, &mut breakdown)
        } else {
            analyzer.analyze(text)
        };
        let analyze_total_ns = t_analyze.elapsed().as_nanos() as u64;
        if prof {
            let p = &self.ingest_profile;
            use std::sync::atomic::Ordering::Relaxed;
            p.idx_read_payload_ns.fetch_add(read_payload_ns, Relaxed);
            p.idx_utf8_validate_ns.fetch_add(utf8_ns, Relaxed);
            p.idx_analyze_total_ns.fetch_add(analyze_total_ns, Relaxed);
            p.idx_analyze_normalize_ns
                .fetch_add(breakdown.normalize_ns, Relaxed);
            p.idx_analyze_tokenize_ns
                .fetch_add(breakdown.tokenize_ns, Relaxed);
            p.idx_analyze_token_loop_ns
                .fetch_add(breakdown.token_loop_ns, Relaxed);
            p.idx_raw_tokens_total
                .fetch_add(breakdown.n_raw_tokens as u64, Relaxed);
            p.idx_output_tokens_total
                .fetch_add(breakdown.n_output_tokens as u64, Relaxed);
            p.idx_analyzer_apply_ns
                .fetch_add(analyzer_apply_t.elapsed().as_nanos() as u64, Relaxed);
        }

        let _t = std::time::Instant::now();
        let pending = PendingFrameIndex::from_tokens(collection_id, tokens);
        if prof {
            let p = &self.ingest_profile;
            p.idx_pending_build_ns.fetch_add(
                _t.elapsed().as_nanos() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
        }

        let _t = std::time::Instant::now();
        self.pending_text_indexes
            .entry(text_space_id)
            .or_default()
            .insert(frame_id, pending);
        if prof {
            let p = &self.ingest_profile;
            p.idx_pending_insert_ns.fetch_add(
                _t.elapsed().as_nanos() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            p.idx_total_ns.fetch_add(
                _call_start.elapsed().as_nanos() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            p.idx_calls
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        self.dirty = true;
        Ok(())
    }

    /// Batch-ingest a slice of `PutFrame`s into a single text space,
    /// parallelizing the analyzer pipeline via rayon.
    ///
    /// The analyzer is the dominant per-doc cost in the BEIR pilot
    /// (~50 µs/doc, 80–90% of total ingest after the Durability fix).
    /// `Analyzer::analyze` is pure (no shared state), so this method
    /// splits the loop into two phases:
    ///
    /// - **Phase 1 — parallel**: clone the cached analyzer once, then
    ///   `par_iter` over `inputs` to UTF-8-validate each payload, run
    ///   the analyzer, and produce a `PendingFrameIndex` per input.
    ///   Pure CPU; scales linearly with rayon's thread pool.
    /// - **Phase 2 — serial**: walk the precomputed `PendingFrameIndex`
    ///   list in input order, calling `put_frame` (single-writer
    ///   `&mut self` work: id allocation + payload segment write +
    ///   catalog upsert) and stashing the pending entry into the
    ///   text-space's pending map. No re-read of the payload — the
    ///   tokens already exist.
    ///
    /// Returns the assigned `FrameId`s in `inputs` order. Frame IDs
    /// are still monotonic; the parallel section never touches the ID
    /// allocator.
    ///
    /// IngestProfile is populated when `VALISE_INGEST_PROFILE=1`. Phase 1
    /// contributes the analyzer sub-counters (`idx_utf8_validate_ns`,
    /// `idx_analyze_*_ns`, `idx_pending_build_ns`) as atomic
    /// fetch-adds from rayon workers (Relaxed ordering, summed
    /// correctly across threads). `idx_read_payload_ns` stays at zero
    /// — the read is skipped entirely. Phase 2 contributes `put_frame_*`
    /// and `idx_pending_insert_ns`.
    pub fn put_text_frames_batch(
        &mut self,
        text_space_id: TextSpaceId,
        inputs: &[PutFrame<'_>],
    ) -> Result<Vec<FrameId>> {
        use rayon::prelude::*;
        use std::sync::atomic::Ordering::Relaxed;

        self.ensure_write()?;
        self.ensure_text_enabled("put_text_frames_batch")?;
        if inputs.is_empty() {
            return Ok(Vec::new());
        }

        // Clone the registered analyzer for parallel use. Clone is
        // cheap: `Analyzer` is a handful of bools + enum tags + a
        // static `HashSet` ref + the Snowball algorithm selector
        // (Copy). The Stemmer state itself is rebuilt once per
        // `analyze()` call inside `run_token_loop`.
        let analyzer = self
            .analyzer_cache
            .get(&text_space_id)
            .ok_or_else(|| {
                Error::Format(format!(
                    "put_text_frames_batch: unknown text_space_id {}",
                    text_space_id.0
                ))
            })?
            .clone();

        let prof = self.ingest_profile_enabled;

        // ---- Phase 1: parallel analyzer + PendingFrameIndex build ----
        //
        // Opt #6 — each rayon worker holds a long-lived `StemCache`
        // via `map_init`. The cache memoizes the Porter2 stem
        // (post-fold byte slice → stem bytes) across the worker's
        // docs. English BEIR corpora hit a Zipf-distributed vocabulary
        // of ~10k unique post-fold forms; with an 8192-entry
        // direct-mapped cache the hit rate exceeds 90% after warmup.
        //
        // Each worker contributes its per-call timings via atomic
        // fetch_add into `self.ingest_profile`. The `&IngestProfile`
        // borrow is `Send + Sync` because every field is `AtomicU64`.
        let prof_ref = &self.ingest_profile;
        let pending: Vec<PendingFrameIndex> = inputs
            .par_iter()
            .map_init(
                crate::text::stem_cache::StemCache::new,
                |stem_cache, input| {
                    let call_start = std::time::Instant::now();
                    let t_utf8 = std::time::Instant::now();
                    let text = std::str::from_utf8(input.payload).map_err(|err| {
                        Error::Format(format!(
                            "put_text_frames_batch: payload is not UTF-8 ({err})"
                        ))
                    })?;
                    let utf8_ns = t_utf8.elapsed().as_nanos() as u64;

                    let t_analyze = std::time::Instant::now();
                    let mut breakdown = crate::text::analyzer::AnalyzeBreakdown::default();
                    // Snapshot cache counters before/after to attribute
                    // per-doc cache activity to the profile atomics.
                    let pre_lookups = stem_cache.lookups;
                    let pre_hits = stem_cache.hits;
                    let pre_bypassed = stem_cache.bypassed;
                    let tokens = analyzer.analyze_with_cache(text, stem_cache, &mut breakdown);
                    let analyze_total_ns = t_analyze.elapsed().as_nanos() as u64;
                    let delta_lookups = stem_cache.lookups - pre_lookups;
                    let delta_hits = stem_cache.hits - pre_hits;
                    let delta_bypassed = stem_cache.bypassed - pre_bypassed;

                    let t_pending = std::time::Instant::now();
                    let pending = PendingFrameIndex::from_tokens(input.collection_id, tokens);
                    let pending_build_ns = t_pending.elapsed().as_nanos() as u64;

                    if prof {
                        prof_ref.idx_utf8_validate_ns.fetch_add(utf8_ns, Relaxed);
                        prof_ref
                            .idx_analyze_total_ns
                            .fetch_add(analyze_total_ns, Relaxed);
                        prof_ref
                            .idx_analyze_normalize_ns
                            .fetch_add(breakdown.normalize_ns, Relaxed);
                        prof_ref
                            .idx_analyze_tokenize_ns
                            .fetch_add(breakdown.tokenize_ns, Relaxed);
                        prof_ref
                            .idx_analyze_token_loop_ns
                            .fetch_add(breakdown.token_loop_ns, Relaxed);
                        prof_ref
                            .idx_raw_tokens_total
                            .fetch_add(breakdown.n_raw_tokens as u64, Relaxed);
                        prof_ref
                            .idx_output_tokens_total
                            .fetch_add(breakdown.n_output_tokens as u64, Relaxed);
                        prof_ref
                            .idx_analyzer_apply_ns
                            .fetch_add(utf8_ns + analyze_total_ns, Relaxed);
                        prof_ref
                            .idx_pending_build_ns
                            .fetch_add(pending_build_ns, Relaxed);
                        prof_ref
                            .idx_stem_cache_lookups
                            .fetch_add(delta_lookups, Relaxed);
                        prof_ref.idx_stem_cache_hits.fetch_add(delta_hits, Relaxed);
                        prof_ref
                            .idx_stem_cache_bypassed
                            .fetch_add(delta_bypassed, Relaxed);
                        prof_ref
                            .idx_total_ns
                            .fetch_add(call_start.elapsed().as_nanos() as u64, Relaxed);
                        prof_ref.idx_calls.fetch_add(1, Relaxed);
                    }
                    Ok(pending)
                },
            )
            .collect::<Result<Vec<_>>>()?;

        // ---- Phase 2: serial put_frame + stash precomputed pending ----
        let mut frame_ids: Vec<FrameId> = Vec::with_capacity(inputs.len());
        for (input, pending_entry) in inputs.iter().zip(pending) {
            let fid = self.put_frame(input.clone())?;
            let t_insert = std::time::Instant::now();
            self.pending_text_indexes
                .entry(text_space_id)
                .or_default()
                .insert(fid, pending_entry);
            if prof {
                self.ingest_profile
                    .idx_pending_insert_ns
                    .fetch_add(t_insert.elapsed().as_nanos() as u64, Relaxed);
            }
            frame_ids.push(fid);
        }
        self.dirty = true;
        Ok(frame_ids)
    }

    /// Run a text query over the canonical primitives of `query.text_space_id`.
    /// The query string is analyzed with the text space's analyzer; the
    /// resulting tokens drive the chosen retrieval algorithm. Pending
    /// (uncommitted) indexes for the text_space cause the call to error —
    /// commit first.
    pub fn query_text(&self, query: TextQuery) -> Result<Vec<crate::retrieval::Hit>> {
        self.ensure_text_enabled("query_text")?;
        if self
            .pending_text_indexes
            .get(&query.text_space_id)
            .is_some_and(|m| !m.is_empty())
        {
            return Err(Error::Format(format!(
                "query_text: text_space {} has uncommitted indexes; call commit() first",
                query.text_space_id.0
            )));
        }
        let state = self
            .text_space_states
            .get(&query.text_space_id)
            .ok_or_else(|| {
                Error::Format(format!(
                    "query_text: unknown text_space_id {}",
                    query.text_space_id.0
                ))
            })?;
        let analyzer = self
            .analyzer_cache
            .get(&query.text_space_id)
            .ok_or_else(|| {
                Error::Integrity(format!(
                    "query_text: analyzer cache missing for text_space {}",
                    query.text_space_id.0
                ))
            })?;
        let tokens = analyzer.analyze(&query.query);

        let top_k = query.top_k;
        let channel_k = query.channel_k;
        match query.algorithm {
            QueryAlgorithm::Profile(profile_id) => {
                let profile = self
                    .catalog
                    .retrieval_profiles
                    .iter()
                    .find(|p| p.profile_id == profile_id)
                    .ok_or_else(|| {
                        Error::Format(format!("query_text: unknown profile_id {}", profile_id.0))
                    })?;
                self.dispatch_profile_query(state, profile, &tokens, top_k, channel_k)
            }
            QueryAlgorithm::Bm25 { k1, b } => crate::retrieval::bm25::bm25_query_topk(
                state,
                &self.catalog.frame_stubs,
                self.tombstoned_frame_count > 0,
                k1,
                b,
                crate::format::catalog::IdfVariant::RobertsonSparckJones,
                &tokens,
                top_k,
                channel_k,
            ),
            QueryAlgorithm::CountCosine => crate::retrieval::tfidf::count_cosine_query(
                state,
                &self.catalog.frame_stubs,
                &tokens,
                top_k,
                channel_k,
            ),
            QueryAlgorithm::TfidfCosine { tf_mode } => crate::retrieval::tfidf::tfidf_cosine_query(
                state,
                &self.catalog.frame_stubs,
                tf_mode,
                &tokens,
                top_k,
                channel_k,
            ),
            QueryAlgorithm::CountCosineApprox => {
                crate::retrieval::tfidf::count_cosine_approx_query(
                    state,
                    &self.catalog.frame_stubs,
                    &tokens,
                    top_k,
                    channel_k,
                )
            }
            QueryAlgorithm::TfidfCosineApprox { tf_mode } => {
                crate::retrieval::tfidf::tfidf_cosine_approx_query(
                    state,
                    &self.catalog.frame_stubs,
                    tf_mode,
                    &tokens,
                    top_k,
                    channel_k,
                )
            }
            QueryAlgorithm::Dice => crate::retrieval::jaccard::dice_query(
                state,
                &self.catalog.frame_stubs,
                &tokens,
                top_k,
            ),
            QueryAlgorithm::Overlap => crate::retrieval::jaccard::overlap_query(
                state,
                &self.catalog.frame_stubs,
                &tokens,
                top_k,
            ),
            QueryAlgorithm::Containment => crate::retrieval::jaccard::containment_query(
                state,
                &self.catalog.frame_stubs,
                &tokens,
                top_k,
            ),
        }
    }

    pub(crate) fn dispatch_profile_query(
        &self,
        state: &TextSpaceState,
        profile: &RetrievalProfileDesc,
        tokens: &[Vec<u8>],
        top_k: Option<usize>,
        channel_k: Option<usize>,
    ) -> Result<Vec<crate::retrieval::Hit>> {
        use crate::format::catalog::{RetrievalProfileParams, RetrievalProfileType};
        match (&profile.profile_type, &profile.params) {
            (RetrievalProfileType::Bm25, RetrievalProfileParams::Bm25 { k1, b, idf_variant }) => {
                crate::retrieval::bm25::bm25_query_topk(
                    state,
                    &self.catalog.frame_stubs,
                    self.tombstoned_frame_count > 0,
                    *k1,
                    *b,
                    *idf_variant,
                    tokens,
                    top_k,
                    channel_k,
                )
            }
            (
                RetrievalProfileType::Tfidf,
                RetrievalProfileParams::Tfidf {
                    tf_mode,
                    idf_mode,
                    norm_mode,
                },
            ) => crate::retrieval::tfidf::tfidf_query(
                state,
                &self.catalog.frame_stubs,
                *tf_mode,
                *idf_mode,
                *norm_mode,
                tokens,
                top_k,
                channel_k,
            ),
            (
                RetrievalProfileType::JaccardExact,
                RetrievalProfileParams::JaccardExact { token_source },
            ) => crate::retrieval::jaccard::jaccard_query(
                state,
                &self.catalog.frame_stubs,
                *token_source,
                tokens,
                top_k,
            ),
            (RetrievalProfileType::JaccardWeighted, _) => Err(Error::Unsupported(
                "JaccardWeighted is reserved for v2".into(),
            )),
            (kind, _) => Err(Error::Format(format!(
                "query_text: profile_type/params mismatch for {kind:?}"
            ))),
        }
    }
}
