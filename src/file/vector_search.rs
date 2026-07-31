// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Vector search engine: sign-sketch candidate scan → family-specific
//! rerank, plus the open-time vector base-pointer / sketch / norm caches
//! the hot path reads.
//!
//! Layered as: [`ValiseFile::vector_search`](crate::ValiseFile::vector_search) resolves the space + codec and
//! dispatches — UPQ spaces route to `upq_search.rs`, QAM spaces to
//! `sketch_then_rerank_impl` (the (5,6) config additionally engages the
//! integer SDOT sliding kernel), anything without a sketch index falls
//! back to `brute_force_vector`. All scores follow the engine convention
//! (smaller = more similar); see `docs/VECTOR_SEARCH.md`.

use super::*;

impl ValiseFile {
    pub fn vector_search(&self, query: VectorSearchQuery) -> Result<Vec<VectorHit>> {
        self.ensure_segment_registry_loaded()?;
        let space = {
            let idx = self
                .catalog
                .embedding_spaces
                .binary_search_by_key(&query.embedding_space_id.0, |s| s.embedding_space_id.0)
                .map_err(|_| {
                    Error::Format(format!(
                        "vector_search: unknown embedding_space_id {}",
                        query.embedding_space_id.0
                    ))
                })?;
            self.catalog.embedding_spaces[idx].clone()
        };
        if (query.query.len() as u32) != space.dimension {
            return Err(Error::Format(format!(
                "vector_search: query has {} dims but embedding space {} expects {}",
                query.query.len(),
                space.embedding_space_id.0,
                space.dimension
            )));
        }
        if space.dtype.is_f8() {
            return Err(Error::Unsupported(
                "vector_search: f8 distance kernels land in a follow-up".into(),
            ));
        }

        // Dispatch: QAM(5,6) spaces with a built sign-sketch index use
        // the sketch-scan → sliding-rerank path; non-QAM (or empty)
        // spaces fall back to a full brute-force scan through the
        // primary codec. The sketch index is derived from the stored
        // QAM phase codes at file-open (see `build_vector_base_ptrs`).
        self.ensure_vector_base_loaded()?;
        let has_sketch = self
            .vector_base
            .read()
            .sketches_by_space
            .contains_key(&space.embedding_space_id);
        if has_sketch {
            // UPQ spaces take the UPQ decoded-i8 rerank path; QAM
            // spaces the sliding/general QAM path.
            let is_upq = space
                .primary_codec_id
                .and_then(|id| self.codec_cache.get(&id))
                .is_some_and(|c| {
                    c.as_any()
                        .downcast_ref::<crate::codec::upq::UpqCodec>()
                        .is_some()
                });
            if is_upq {
                return self.upq_sketch_then_rerank(&space, &query);
            }
            return self.sketch_then_rerank(&space, &query);
        }
        self.brute_force_vector(&space, &query)
    }

    /// Sketch-scan → QAM-sliding rerank. Step 1: Hamming-scan the query's
    /// sign sketch against every active vector's sketch and counting-sort
    /// the closest `channel_k` as candidates. Step 2: rerank those
    /// candidates through the QAM-sliding kernel to recover the
    /// high-precision ordering.
    fn sketch_then_rerank(
        &self,
        space: &EmbeddingSpaceDesc,
        query: &VectorSearchQuery,
    ) -> Result<Vec<VectorHit>> {
        self.sketch_then_rerank_impl(space, query, None)
    }

    /// Public traced variant. Validates the embedding space the same
    /// way [`Self::vector_search`] does, requires a built sign-sketch
    /// index for the space, and dispatches through
    /// `sketch_then_rerank_impl`. Returns the hits plus a
    /// [`VoteSearchTrace`] with per-stage durations. Intended for
    /// benches that want to compute p50/p99 per stage.
    pub fn vector_search_traced(
        &self,
        query: VectorSearchQuery,
    ) -> Result<(Vec<VectorHit>, VoteSearchTrace)> {
        self.ensure_segment_registry_loaded()?;
        let space = {
            let idx = self
                .catalog
                .embedding_spaces
                .binary_search_by_key(&query.embedding_space_id.0, |s| s.embedding_space_id.0)
                .map_err(|_| {
                    Error::Format(format!(
                        "vector_search_traced: unknown embedding_space_id {}",
                        query.embedding_space_id.0
                    ))
                })?;
            self.catalog.embedding_spaces[idx].clone()
        };
        if (query.query.len() as u32) != space.dimension {
            return Err(Error::Format(format!(
                "vector_search_traced: query has {} dims but embedding space {} expects {}",
                query.query.len(),
                space.embedding_space_id.0,
                space.dimension
            )));
        }
        self.ensure_vector_base_loaded()?;
        if !self
            .vector_base
            .read()
            .sketches_by_space
            .contains_key(&space.embedding_space_id)
        {
            return Err(Error::Unsupported(format!(
                "vector_search_traced: embedding space {} has no sign-sketch index (not QAM 5,6, or empty)",
                space.embedding_space_id.0
            )));
        }
        let mut trace = VoteSearchTrace::default();
        let hits = self.sketch_then_rerank_impl(&space, &query, Some(&mut trace))?;
        Ok((hits, trace))
    }

    fn sketch_then_rerank_impl(
        &self,
        space: &EmbeddingSpaceDesc,
        query: &VectorSearchQuery,
        mut trace: Option<&mut VoteSearchTrace>,
    ) -> Result<Vec<VectorHit>> {
        // VALISE_QUERY_PROFILE per-stage instrumentation. `enabled()` is a
        // cached-bool read; with the flag unset every `lap` below is a
        // single branch (no clock reads) and nothing is recorded.
        let qprof_on = query_profile::enabled();
        let mut qprof = query_profile::QueryProfile::default();
        let mut qprof_mark = qprof_on.then(std::time::Instant::now);
        let t_preflight = std::time::Instant::now();
        self.ensure_vector_base_loaded()?;
        let pending_in_space = self
            .pending_vector_batches
            .keys()
            .any(|(esp, _)| *esp == query.embedding_space_id);
        if pending_in_space {
            return Err(Error::Format(format!(
                "vector_search: embedding_space {} has uncommitted vectors; call commit() first",
                query.embedding_space_id.0
            )));
        }
        let primary_id = space.primary_codec_id.ok_or_else(|| {
            Error::Integrity(format!(
                "vector_search: space {} has no primary_codec_id",
                space.embedding_space_id.0
            ))
        })?;
        if let Some(t) = &mut trace {
            t.preflight = t_preflight.elapsed();
        }

        let primary_box = self.codec_cache.get(&primary_id).ok_or_else(|| {
            Error::Integrity(format!(
                "sketch_then_rerank: primary codec {} missing from cache",
                primary_id.0
            ))
        })?;
        let primary_codec: &dyn VectorCodec = primary_box.as_ref();
        let metric = space.metric;
        let base_bytes_per_vector = primary_codec.base_bytes_per_vector();
        let primary_qam = primary_codec
            .as_any()
            .downcast_ref::<crate::codec::qam_lloyd_max::QamLloydMaxCodec>();

        // Default candidate budget when `channel_k = None`: a fixed,
        // corpus-size-INDEPENDENT value (DEFAULT_SKETCH_CANDIDATE_BUDGET). The
        // sketch's coverage of the true top-k saturates by a couple thousand
        // candidates at d≈768, so scaling the budget with N (the old `N/4`
        // rule) only multiplied rerank cost for no recall gain — and at 25 %
        // of the corpus it had effectively stopped pruning. Clamped to the
        // active count for small corpora. See docs/VECTOR_SEARCH.md.
        let active_n = self.vector_by_id_cache.len();
        let candidates_budget = query
            .channel_k
            .unwrap_or_else(|| {
                query
                    .k
                    .saturating_mul(4)
                    .max(DEFAULT_SKETCH_CANDIDATE_BUDGET)
            })
            .min(active_n.max(1));
        // Any QAM codec (any amp/phase bits) — its rotation produces the query
        // sign sketch and supplies `dim`. The (5,6) sliding fast path is
        // selected separately below (`qam_concrete`); other configs fall back
        // to the general asymmetric rerank. Dispatch guarantees a sketch index
        // exists for this space, which is built only for QAM spaces.
        let qam_codec = primary_qam.ok_or_else(|| {
            Error::Integrity(format!(
                "vector_search: space {} primary codec is not QAM",
                space.embedding_space_id.0
            ))
        })?;

        // Adaptive admission control for the heavy parallel region (sketch
        // scan + rerank fan-out). If the rayon pool is saturated we get
        // `None` and run serially (one core) instead of oversubscribing.
        let _fanout_permit = crate::file::query_admission::global().try_acquire();
        let parallel = _fanout_permit.is_some();

        // Candidate generation: sign-sketch Hamming scan over all active
        // vectors → counting-sort the closest `candidates_budget`. The
        // query sketch is sign(rotate(query)); the DB sketches were
        // derived from the stored QAM phase codes at file-open.
        let t_scan = std::time::Instant::now();
        let q_rot = qam_codec.prepare_query_rotated(&query.query)?.0;
        let q_sketch = crate::retrieval::sketch::pack_query_sketch(&q_rot);
        qprof.prep = query_profile::lap(&mut qprof_mark);
        let candidates: Vec<(VectorId, i32)> = {
            let vb = self.vector_base.read();
            let (words, sketches) = vb
                .sketches_by_space
                .get(&space.embedding_space_id)
                .ok_or_else(|| {
                    Error::Integrity(format!(
                        "vector_search: no sketch index for space {}",
                        space.embedding_space_id.0
                    ))
                })?;
            let rows: &[(u32, usize)] = vb
                .candidates_by_space
                .get(&space.embedding_space_id)
                .map(|v| v.as_slice())
                .unwrap_or(&[]);
            qprof.candidates_scanned = rows.len();
            let mut hbuf: Vec<u16> = Vec::new();
            crate::retrieval::sketch::scan_candidates(
                sketches,
                *words,
                rows,
                &q_sketch,
                candidates_budget,
                qam_codec.dim,
                parallel,
                &mut hbuf,
            )
        };
        qprof.stage1_sketch = query_profile::lap(&mut qprof_mark);
        if let Some(t) = &mut trace {
            t.vote.accumulate = t_scan.elapsed();
        }
        let _ = active_n;

        // Rerank the candidates through the primary codec. Pre-build
        // the prepared query exactly once, mirror the brute-force
        // pattern.
        let t_rerank_int = std::time::Instant::now();

        // Fast path: when primary is QamLloydMaxCodec (5+6), bypass the
        // trait `asymmetric_distance_prepared` route and call the
        // sliding engine directly with a concrete prep type. The trait
        // path was producing different scores than the standalone
        // bench in some condition (still being root-caused) — this
        // direct concrete-engine call exactly mirrors what the
        // adaptive-tune bench does and reaches the same recall.
        let qam_concrete = primary_qam.filter(|c| c.amp_bits == 5 && c.phase_bits == 6);
        let prepared_ctx_box;
        let qam_prep_concrete;
        if let Some(qam) = qam_concrete {
            // Build the sliding engine + concrete prep up-front.
            let engine = qam.sliding_engine.get_or_init(|| {
                crate::codec::qam_sliding::QamSlidingEngine::from_codec(qam)
                    .expect("BUG: from_codec failed for 5+6 QAM")
            });
            qam_prep_concrete = Some((engine, engine.prepare_query(&query.query)?));
            prepared_ctx_box = None;
        } else {
            qam_prep_concrete = None;
            prepared_ctx_box = Some(primary_codec.prepare_query(&query.query)?);
        };
        let prepared_ctx_ref: Option<&(dyn std::any::Any + Send + Sync)> = prepared_ctx_box
            .as_ref()
            .map(|b| &**b as &(dyn std::any::Any + Send + Sync));

        let filter = query.collection_filter.clone();
        let filter_predicate = |cid: CollectionId| match &filter {
            Some(set) => set.contains(&cid),
            None => true,
        };

        let vb_guard = self.vector_base.read();
        let ptrs: &[usize] = &vb_guard.ptrs;
        // QAM-only: per-row y_hat_norm cache prebuilt in
        // `build_vector_base_ptrs`. Saves ~384 mul + 1 sqrt per scored
        // candidate at dim 768 vs the slow path's per-call
        // `engine.y_hat_norm(base)`. `None` for non-QAM spaces and for
        // QAM spaces that didn't fill the cache (shouldn't happen but
        // we fall back to the slow path defensively).
        let qam_norms: Option<&Vec<f32>> = qam_prep_concrete
            .as_ref()
            .and_then(|_| vb_guard.qam_norms_by_space.get(&space.embedding_space_id));

        // Step 1: resolve candidates serially — HashMap lookups, filter
        // checks, ptr + cached-norm lookups. Output is a compact slice
        // we can stream through under par_iter without touching shared
        // state. Mirrors the pattern in `brute_force_vector`.
        #[derive(Clone, Copy)]
        struct RerankCand {
            vid: u32,
            ptr: usize,
            /// Cached **inverse** QAM `‖ŷ‖` (i.e. `1.0 / y_hat_norm`)
            /// precomputed at file-open. 0.0 sentinel means no cache
            /// entry for this vid — fall back to the slow
            /// `engine.asymmetric_distance` (which recomputes the
            /// norm and inverts it per call).
            inv_norm: f32,
        }
        // Fast path (no collection filter): the sketch index is per-space, so
        // every candidate vid the scan emits is already in this space — no
        // `embedding_space_id` check needed.
        // Tombstoned vectors have `ptrs[vid] == 0` by construction of
        // `build_vector_base_ptrs` (only Active vectors get a non-zero
        // entry), so the `ptr == 0` skip subsumes the `status != Active`
        // check too. Net effect: drop the HashMap lookup that was
        // running serially for every one of the `channel_k` candidates
        // — ~500 µs at N=50k channel_k=12500, ~2 ms at N=200k
        // channel_k=50000.
        // Optional sub-stage timing inside rerank_int. Gated on env var
        // so it costs ~5 ns when disabled (Instant::now reads bypassed).
        let timing_on = std::env::var("VALISE_RERANK_TIMING").is_ok();
        let mut t_build = std::time::Duration::ZERO;
        let mut t_sort = std::time::Duration::ZERO;
        let mut t_score = std::time::Duration::ZERO;
        let mut t_topk = std::time::Duration::ZERO;
        let mut t_resolve = std::time::Duration::ZERO;
        let t_step = if timing_on {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let has_filter = query.collection_filter.is_some();
        let mut cand_slice: Vec<RerankCand> = Vec::with_capacity(candidates.len());
        if has_filter {
            for (vid, _vote_score) in &candidates {
                let Some(desc) = self.vector_by_id_cache.get(vid) else {
                    continue; // tombstoned post-build
                };
                if !filter_predicate(desc.collection_id) {
                    continue;
                }
                let idx = vid.0 as usize;
                let ptr = ptrs.get(idx).copied().unwrap_or(0);
                if ptr == 0 {
                    continue;
                }
                let inv_norm = qam_norms.and_then(|c| c.get(idx).copied()).unwrap_or(0.0);
                cand_slice.push(RerankCand {
                    vid: vid.0 as u32,
                    ptr,
                    inv_norm,
                });
            }
        } else {
            for (vid, _vote_score) in &candidates {
                let idx = vid.0 as usize;
                let ptr = ptrs.get(idx).copied().unwrap_or(0);
                if ptr == 0 {
                    continue;
                }
                let inv_norm = qam_norms.and_then(|c| c.get(idx).copied()).unwrap_or(0.0);
                cand_slice.push(RerankCand {
                    vid: vid.0 as u32,
                    ptr,
                    inv_norm,
                });
            }
        }
        if let Some(t0) = t_step {
            t_build = t0.elapsed();
        }
        let t_step = if timing_on {
            Some(std::time::Instant::now())
        } else {
            None
        };
        // Sort by ptr so the score loop visits sequential mmap pages.
        // Vote candidates arrive in vote-score order, which is random
        // in memory; this turns ~`n_cand` random reads into a sequential
        // scan and lets the prefetcher do its job.
        // `par_sort` over rayon: 4× cores → drops the serial-sort tax
        // from ~540 µs to ~140 µs at channel_k=50000. Under admission
        // pressure (`!parallel`) we sort serially to stay off the pool.
        if parallel {
            use rayon::slice::ParallelSliceMut;
            cand_slice.par_sort_unstable_by_key(|c| c.ptr);
        } else {
            cand_slice.sort_unstable_by_key(|c| c.ptr);
        }
        if let Some(t0) = t_step {
            t_sort = t0.elapsed();
        }

        // Step 2: parallel score. `ScoreItem` is POD so par_iter+collect
        // stays allocation-light.
        #[derive(Clone, Copy, PartialEq)]
        struct ScoreItem {
            score: f32,
            vid: u32,
            ptr: usize,
        }
        impl Eq for ScoreItem {}
        impl Ord for ScoreItem {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                self.score
                    .total_cmp(&other.score)
                    .then_with(|| self.vid.cmp(&other.vid))
            }
        }
        impl PartialOrd for ScoreItem {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }
        // SAFETY: `c.ptr` was set by `build_vector_base_ptrs` from a
        // valid mmap slice of length `base_bytes_per_vector`; the mmap
        // is owned by `self` and lives as long as `vb_guard`, which
        // outlives this scope.
        let t_step_inner = if timing_on {
            Some(std::time::Instant::now())
        } else {
            None
        };
        // One score per candidate; identical math on both the parallel
        // and serial paths. The serial path (admission fallback) keeps
        // this query off the shared rayon pool entirely.
        let score_one = |c: &RerankCand| -> ScoreItem {
            let base: &[u8] =
                unsafe { std::slice::from_raw_parts(c.ptr as *const u8, base_bytes_per_vector) };
            let score = if let Some((engine, prep)) = qam_prep_concrete.as_ref() {
                if c.inv_norm > 0.0 {
                    engine
                        .asymmetric_distance_with_inv_norm(prep, base, c.inv_norm)
                        .unwrap_or(f32::INFINITY)
                } else {
                    engine
                        .asymmetric_distance(prep, base, metric)
                        .unwrap_or(f32::INFINITY)
                }
            } else {
                primary_codec
                    .asymmetric_distance_prepared(
                        prepared_ctx_ref.expect("non-QAM path must have prepared_ctx"),
                        base,
                        metric,
                    )
                    .unwrap_or(f32::INFINITY)
            };
            ScoreItem {
                score,
                vid: c.vid,
                ptr: c.ptr,
            }
        };
        let scored: Vec<ScoreItem> = if parallel {
            use rayon::prelude::*;
            cand_slice.par_iter().map(score_one).collect()
        } else {
            cand_slice.iter().map(score_one).collect()
        };
        if let Some(t0) = t_step_inner {
            t_score = t0.elapsed();
        }
        let t_step = if timing_on {
            Some(std::time::Instant::now())
        } else {
            None
        };

        // Step 3: bounded top-k. Oversample to 3*k when Full is requested
        // so the f32 rerank can actually move items in/out of the final
        // top-k. Plain `Lossy` keeps just `k`.
        let pool_size = if query.fidelity == VectorFidelity::Full {
            (3 * query.k).min(scored.len())
        } else {
            query.k.min(scored.len())
        };
        let mut top: Vec<ScoreItem> = if scored.len() > pool_size && pool_size > 0 {
            let mut s = scored;
            s.select_nth_unstable_by(pool_size - 1, |a, b| a.cmp(b));
            s.truncate(pool_size);
            s.sort();
            s
        } else {
            let mut s = scored;
            s.sort();
            s
        };
        if let Some(t0) = t_step {
            t_topk = t0.elapsed();
        }
        let t_step = if timing_on {
            Some(std::time::Instant::now())
        } else {
            None
        };

        // Step 4: resolve frame_id / collection_id only on the final
        // winners. The Full rerank may still change ordering, so defer
        // descriptor lookup until after that pass.
        if let Some(t0) = t_step {
            t_resolve = t0.elapsed();
        }
        if timing_on {
            tracing::debug!(
                "[rerank_int] cands={} build={:.0}µs sort={:.0}µs score={:.0}µs topk={:.0}µs resolve={:.0}µs",
                cand_slice.len(),
                t_build.as_secs_f64() * 1e6,
                t_sort.as_secs_f64() * 1e6,
                t_score.as_secs_f64() * 1e6,
                t_topk.as_secs_f64() * 1e6,
                t_resolve.as_secs_f64() * 1e6,
            );
        }
        if let Some(t) = &mut trace {
            t.rerank_int = t_rerank_int.elapsed();
        }
        qprof.stage2_rerank = query_profile::lap(&mut qprof_mark);
        qprof.candidates_reranked = cand_slice.len();
        qprof.survivors = top.len();

        let make_hits = |items: &[ScoreItem]| -> Vec<VectorHit> {
            items
                .iter()
                .filter_map(|si| {
                    self.vector_by_id_cache
                        .get(&VectorId(si.vid as u64))
                        .map(|desc| VectorHit {
                            vector_id: desc.vector_id,
                            frame_id: desc.owner_frame_id,
                            collection_id: desc.collection_id,
                            score: si.score,
                        })
                })
                .collect()
        };

        if query.fidelity == VectorFidelity::Full {
            let t_rerank_full = std::time::Instant::now();
            let force_decode_rerank = std::env::var_os("VALISE_QAM_FULL_RERANK_DECODE").is_some();
            if let Some(qam) = primary_qam.filter(|_| !force_decode_rerank) {
                let (q_rot, q_norm) = qam.prepare_query_rotated(&query.query)?;
                let num_pairs = qam.num_pairs;
                // f32 rerank of the `pool_size` (3*k) survivors. Each
                // `(q_rot, base)` score is independent and read-only on
                // `qam`/`q_rot`. Scores are bit-identical on both paths;
                // the sort below is the sole ordering authority, so the
                // parallel/serial choice never changes results or recall.
                // SAFETY (both paths): `item.ptr` came from
                // `build_vector_base_ptrs` (a valid `base_bytes_per_vector`
                // mmap slice still pinned by `vb_guard`); each item reads a
                // distinct, read-only slice.
                if parallel {
                    // Fan out with per-worker scratch (allocated once per
                    // worker via `for_each_init`, not per item). Turns the
                    // serial ~300µs pass at pool_size=300 into ~30µs.
                    use rayon::prelude::*;
                    top.par_iter_mut().for_each_init(
                        || (vec![0u32; num_pairs], vec![0u32; num_pairs]),
                        |(amp_scratch, phase_scratch), item| {
                            let base: &[u8] = unsafe {
                                std::slice::from_raw_parts(
                                    item.ptr as *const u8,
                                    base_bytes_per_vector,
                                )
                            };
                            item.score = qam
                                .asymmetric_distance_with_rotated_scratch(
                                    &q_rot,
                                    q_norm,
                                    base,
                                    metric,
                                    amp_scratch,
                                    phase_scratch,
                                )
                                .unwrap_or(f32::INFINITY);
                        },
                    );
                } else {
                    // Admission fallback: serial, single reused scratch,
                    // off the shared rayon pool.
                    let mut amp_scratch = vec![0u32; num_pairs];
                    let mut phase_scratch = vec![0u32; num_pairs];
                    for item in &mut top {
                        let base: &[u8] = unsafe {
                            std::slice::from_raw_parts(item.ptr as *const u8, base_bytes_per_vector)
                        };
                        item.score = qam
                            .asymmetric_distance_with_rotated_scratch(
                                &q_rot,
                                q_norm,
                                base,
                                metric,
                                &mut amp_scratch,
                                &mut phase_scratch,
                            )
                            .unwrap_or(f32::INFINITY);
                    }
                }
                top.sort();
                top.truncate(query.k);
            } else {
                // Profiling escape hatch and generic-codec fallback: preserve
                // the previous decode + inverse-rotate behavior exactly.
                let mut hits = make_hits(&top);
                let registry_guard = self.segment_registry.read();
                let segment_by_id_ref: &HashMap<SegmentId, SegmentRef> = &registry_guard.by_id;
                let mmap_ref = self.file_mmap.as_ref().ok_or_else(|| {
                    Error::Integrity("sketch_then_rerank: file mmap is not initialized".into())
                })?;
                for hit in &mut hits {
                    let desc = self
                        .vector_by_id_cache
                        .get(&hit.vector_id)
                        .expect("BUG: hit must be in active set");
                    let seg_ref = segment_by_id_ref
                        .get(&desc.data_segment_id)
                        .copied()
                        .ok_or_else(|| {
                            Error::Integrity(format!(
                                "sketch_then_rerank rerank: segment {} not in registry",
                                desc.data_segment_id.0
                            ))
                        })?;
                    let payload = mmap_segment_payload(mmap_ref, seg_ref, SegmentType::VectorData)?;
                    let reader = VectorDataSegmentReader::open(payload, base_bytes_per_vector)?;
                    let base = reader.base(desc.ordinal_in_segment)?;
                    let decoded = primary_codec.decode_lossy(base)?;
                    hit.score = full_distance(&query.query, &decoded, metric);
                }
                hits.sort_by(|a, b| a.score.total_cmp(&b.score));
                hits.truncate(query.k);
                if let Some(t) = &mut trace {
                    t.rerank_full = t_rerank_full.elapsed();
                }
                qprof.stage3_exact = query_profile::lap(&mut qprof_mark);
                if qprof_on {
                    query_profile::record_query(qprof);
                }
                return Ok(hits);
            }
            if let Some(t) = &mut trace {
                t.rerank_full = t_rerank_full.elapsed();
            }
            qprof.stage3_exact = query_profile::lap(&mut qprof_mark);
        }
        if qprof_on {
            query_profile::record_query(qprof);
        }
        Ok(make_hits(&top))
    }

    /// Brute-force search over the active vectors of a non-f8
    /// embedding space, scoring through the primary codec. Used by
    /// [`Self::vector_search`] when the space has no sign-sketch index —
    /// i.e. any codec other than QAM(5,6) (the QAM(5,6) path takes the
    /// sketch-then-rerank route instead).
    fn brute_force_vector(
        &self,
        space: &EmbeddingSpaceDesc,
        query: &VectorSearchQuery,
    ) -> Result<Vec<VectorHit>> {
        self.ensure_vector_base_loaded()?;
        let pending_in_space = self
            .pending_vector_batches
            .keys()
            .any(|(esp, _)| *esp == query.embedding_space_id);
        if pending_in_space {
            return Err(Error::Format(format!(
                "vector_search: embedding_space {} has uncommitted vectors; call commit() first",
                query.embedding_space_id.0
            )));
        }
        let codec_id = space.primary_codec_id.ok_or_else(|| {
            Error::Integrity(format!(
                "brute_force_vector: embedding_space {} has no primary_codec_id",
                space.embedding_space_id.0
            ))
        })?;
        let codec_box = self.codec_cache.get(&codec_id).ok_or_else(|| {
            Error::Integrity(format!(
                "brute_force_vector: codec {} missing from cache",
                codec_id.0
            ))
        })?;
        let codec_ref: &dyn VectorCodec = codec_box.as_ref();
        let metric = space.metric;
        let base_bytes_per_vector = codec_ref.base_bytes_per_vector();

        // Hot path: specialize on the concrete codec type. The trait
        // dispatch + per-call downcast costs ~50% of brute-force
        // latency at typical dim — peeling that overhead out front
        // (one downcast per query, not per vector) brings us in line
        // with the standalone-bench numbers.
        let codec_any = codec_ref.as_any();
        enum FastPath<'a> {
            Qam {
                engine: &'a crate::codec::qam_sliding::QamSlidingEngine,
                prep: crate::codec::qam_sliding::QamSlidingPreparedQuery,
            },
            Generic {
                ctx: Box<dyn std::any::Any + Send + Sync>,
            },
        }
        let fast = if let Some(qam) =
            codec_any.downcast_ref::<crate::codec::qam_lloyd_max::QamLloydMaxCodec>()
        {
            if qam.amp_bits == 5 && qam.phase_bits == 6 {
                let engine = qam.sliding_engine.get_or_init(|| {
                    crate::codec::qam_sliding::QamSlidingEngine::from_codec(qam)
                        .expect("BUG: QamSlidingEngine::from_codec failed for 5+6 codec")
                });
                let prep = engine.prepare_query(&query.query)?;
                FastPath::Qam { engine, prep }
            } else {
                FastPath::Generic {
                    ctx: codec_ref.prepare_query(&query.query)?,
                }
            }
        } else {
            FastPath::Generic {
                ctx: codec_ref.prepare_query(&query.query)?,
            }
        };

        let filter = query.collection_filter.clone();
        let filter_predicate = move |cid: CollectionId| match &filter {
            Some(set) => set.contains(&cid),
            None => true,
        };

        // Top-k via a max-heap keyed on (score_desc, vector_id) so the
        // smallest distance stays at the root. We keep the heap bounded
        // at `query.k`, popping the worst when full.
        use std::collections::BinaryHeap;
        #[derive(Clone, Copy, PartialEq)]
        struct HeapItem {
            score: f32,
            vid: VectorId,
            frame: FrameId,
            collection: CollectionId,
        }
        impl Eq for HeapItem {}
        impl PartialOrd for HeapItem {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }
        impl Ord for HeapItem {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                // Larger score = farther = "worse"; we want max-heap
                // by score so root pops the worst kept hit.
                self.score
                    .total_cmp(&other.score)
                    .then_with(|| self.vid.0.cmp(&other.vid.0))
            }
        }

        let vb_guard = self.vector_base.read();

        // Build a flat (idx, ptr) candidate slice once. We keep the heap's
        // payload as just `idx` during the parallel scan — frame_id /
        // collection_id are looked up only for the final k winners, so
        // we don't pay a desc lookup per candidate. This mirrors the
        // pattern used by the standalone bench: a contiguous slice
        // iterated under `par_iter`, no HashMap walk in the hot loop.
        let space_id = space.embedding_space_id;
        let has_filter = query.collection_filter.is_some();
        let _t_cand = std::time::Instant::now();
        // Pull the pre-built per-space candidate list. It's already in
        // ptr-sorted order so par_iter chunks visit sequential memory.
        // If a collection filter is set we walk it once here and produce
        // a filtered slice; the unfiltered case borrows the cache directly.
        let cached: &[(u32, usize)] = vb_guard
            .candidates_by_space
            .get(&space_id)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        let candidates_owned: Option<Vec<(u32, usize)>> = if has_filter {
            let filtered: Vec<(u32, usize)> = cached
                .iter()
                .copied()
                .filter(|&(vid, _)| {
                    self.vector_by_id_cache
                        .get(&VectorId(vid as u64))
                        .map(|d| filter_predicate(d.collection_id))
                        .unwrap_or(false)
                })
                .collect();
            Some(filtered)
        } else {
            None
        };
        let candidates: &[(u32, usize)] = candidates_owned.as_deref().unwrap_or(cached);
        if std::env::var("VALISE_BF_TIMING").is_ok() {
            tracing::debug!(
                "[bf] candidates_collect={:?} n={}",
                _t_cand.elapsed(),
                candidates.len()
            );
        }

        #[derive(Clone, Copy, PartialEq)]
        struct ScoreItem {
            score: f32,
            vid: u32,
        }
        impl Eq for ScoreItem {}
        impl PartialOrd for ScoreItem {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }
        impl Ord for ScoreItem {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                self.score
                    .total_cmp(&other.score)
                    .then_with(|| self.vid.cmp(&other.vid))
            }
        }

        let k = query.k.max(1);
        // SAFETY of `unsafe slice::from_raw_parts`: ptr was produced by
        // build_vector_base_ptrs on a valid mmap slice of length
        // base_bytes_per_vector; the mmap is owned by self and lives as
        // long as vb_guard which outlives this scope.
        use rayon::prelude::*;
        let _t_score = std::time::Instant::now();
        let cand_slice: &[(u32, usize)] = candidates;
        let n_cand = cand_slice.len();
        let mut scored: Vec<ScoreItem> = match &fast {
            FastPath::Qam { engine, prep } => (0..n_cand)
                .into_par_iter()
                .map(|i| {
                    let (vid, ptr) = cand_slice[i];
                    let base: &[u8] = unsafe {
                        std::slice::from_raw_parts(ptr as *const u8, base_bytes_per_vector)
                    };
                    let score = engine
                        .asymmetric_distance(prep, base, metric)
                        .unwrap_or(f32::INFINITY);
                    ScoreItem { score, vid }
                })
                .collect(),
            FastPath::Generic { ctx } => {
                let prepared_ctx_ref: &(dyn std::any::Any + Send + Sync) = ctx.as_ref();
                (0..n_cand)
                    .into_par_iter()
                    .map(|i| {
                        let (vid, ptr) = cand_slice[i];
                        let base: &[u8] = unsafe {
                            std::slice::from_raw_parts(ptr as *const u8, base_bytes_per_vector)
                        };
                        let score = codec_ref
                            .asymmetric_distance_prepared(prepared_ctx_ref, base, metric)
                            .unwrap_or(f32::INFINITY);
                        ScoreItem { score, vid }
                    })
                    .collect()
            }
        };
        if std::env::var("VALISE_BF_TIMING").is_ok() {
            tracing::debug!("[bf] par_score={:?}", _t_score.elapsed());
        }
        let _t_topk = std::time::Instant::now();
        // Oversample for the optional f32 rerank: keep `3 * k` candidates from
        // the i8 scan when `VectorFidelity::Full` is requested, so the
        // subsequent decode-and-recompute step can move items in/out of the
        // final top-k. Without oversampling, f32 rerank would only re-score
        // the i8 top-k and recall would be capped at the i8 ceiling.
        let pool_size = if query.fidelity == VectorFidelity::Full {
            (3 * k).min(scored.len())
        } else {
            k.min(scored.len())
        };
        let top_idx: Vec<ScoreItem> = if scored.len() > pool_size {
            let (left, _, _) = scored.select_nth_unstable_by(pool_size, |a, b| {
                a.score.total_cmp(&b.score).then_with(|| a.vid.cmp(&b.vid))
            });
            let mut owned = left.to_vec();
            owned.sort_by(|a, b| a.score.total_cmp(&b.score).then_with(|| a.vid.cmp(&b.vid)));
            owned
        } else {
            scored.sort_by(|a, b| a.score.total_cmp(&b.score).then_with(|| a.vid.cmp(&b.vid)));
            scored
        };

        if std::env::var("VALISE_BF_TIMING").is_ok() {
            tracing::debug!("[bf] topk={:?}", _t_topk.elapsed());
        }
        // Resolve the top-k descriptors only at the end (≤ k = 10 lookups).
        let heap: BinaryHeap<HeapItem> = top_idx
            .into_iter()
            .filter_map(|si| {
                self.vector_by_id_cache
                    .get(&VectorId(si.vid as u64))
                    .map(|desc| HeapItem {
                        score: si.score,
                        vid: desc.vector_id,
                        frame: desc.owner_frame_id,
                        collection: desc.collection_id,
                    })
            })
            .collect();

        let mut hits: Vec<VectorHit> = heap
            .into_iter()
            .map(|i| VectorHit {
                vector_id: i.vid,
                frame_id: i.frame,
                collection_id: i.collection,
                score: i.score,
            })
            .collect();
        hits.sort_by(|a, b| a.score.total_cmp(&b.score));

        // Optional Full rerank. Brute force already scored through the
        // codec's asymmetric path; `Full` re-runs the metric on the
        // dequantized vectors. For large k this is dominated by the
        // decode cost, but only the top-k are touched so it stays
        // bounded.
        if query.fidelity == VectorFidelity::Full {
            let registry_guard = self.segment_registry.read();
            let segment_by_id_ref: &HashMap<SegmentId, SegmentRef> = &registry_guard.by_id;
            let mmap_ref = self.file_mmap.as_ref().ok_or_else(|| {
                Error::Integrity("vector_search: file mmap is not initialized".into())
            })?;
            for hit in &mut hits {
                let desc = self
                    .vector_by_id_cache
                    .get(&hit.vector_id)
                    .expect("BUG: hit must be in active set");
                let seg_ref = segment_by_id_ref
                    .get(&desc.data_segment_id)
                    .copied()
                    .ok_or_else(|| {
                        Error::Integrity(format!(
                            "vector_search rerank: segment {} not in registry",
                            desc.data_segment_id.0
                        ))
                    })?;
                let payload = mmap_segment_payload(mmap_ref, seg_ref, SegmentType::VectorData)?;
                let reader = VectorDataSegmentReader::open(payload, base_bytes_per_vector)?;
                let base = reader.base(desc.ordinal_in_segment)?;
                let decoded = codec_ref.decode_lossy(base)?;
                hit.score = full_distance(&query.query, &decoded, metric);
            }
            hits.sort_by(|a, b| a.score.total_cmp(&b.score));
            hits.truncate(k);
        }
        Ok(hits)
    }
}

/// Bundled state for the naked-pointer vector base cache. Replaces the
/// trio of `vector_base_ptrs`/`_loaded`/`_stride` fields the pre-Stage-1
/// implementation kept on `ValiseFile`. `loaded` is `false` when the cache
/// hasn't been built yet (lazy-deferred at open mirrors the ghost
/// segment registry); read-side `vector_base_ptrs_for_search` flips it
/// to `true` on first ANN call. Refreshed in lockstep with the mmap
/// remap inside `commit()`.
#[derive(Clone, Default)]
pub(crate) struct VectorBasePtrs {
    pub(crate) ptrs: Vec<usize>,
    pub(crate) stride: usize,
    pub(crate) loaded: bool,
    /// Per-embedding-space pre-filtered candidate lists for the brute-
    /// force search hot path. Built once at the same time as `ptrs` (in
    /// `build_vector_base_ptrs`) and reused across queries — avoids the
    /// per-query HashMap walk that otherwise dominates small-corpus
    /// brute-force latency.
    pub(crate) candidates_by_space:
        std::collections::HashMap<crate::format::EmbeddingSpaceId, Vec<(u32, usize)>>,
    /// Per-embedding-space precomputed **inverse** `‖ŷ‖`
    /// (`1.0 / y_hat_norm`) indexed by `vector_id.0`. Populated only
    /// when the space's primary codec is the QAM-sliding engine.
    /// Caching the inverse (not the raw norm) saves one fdiv per
    /// scored candidate inside `sketch_then_rerank_impl`; the fdiv is
    /// paid once per vector at file-open / commit instead.
    pub(crate) qam_norms_by_space:
        std::collections::HashMap<crate::format::EmbeddingSpaceId, Vec<f32>>,
    /// Per-embedding-space dense sign-sketch index `(words, sketches)`,
    /// row-aligned with `candidates_by_space`. Derived from QAM phase
    /// codes at file-open (0 extra storage); drives sketch candidate
    /// generation. Only QAM-sliding (5,6) spaces.
    pub(crate) sketches_by_space:
        std::collections::HashMap<crate::format::EmbeddingSpaceId, (usize, Vec<u64>)>,
    /// UPQ spaces only: vid-indexed decoded-i8 cache rows (`dim` bytes
    /// per vid) for the stage-2 contiguous-dot rerank kernel, and the
    /// matching per-vid dequant factors. Built at file-open from the
    /// packed codes — nothing persisted. See `src/file/upq_search.rs`.
    pub(crate) upq_i8_by_space: std::collections::HashMap<crate::format::EmbeddingSpaceId, Vec<i8>>,
    pub(crate) upq_dequant_by_space:
        std::collections::HashMap<crate::format::EmbeddingSpaceId, Vec<f32>>,
}

impl VectorBasePtrs {
    /// Drop every cached address and mark the cache unloaded.
    ///
    /// `ptrs` holds **absolute addresses into the file mmap**, so any
    /// remap invalidates all of them. Every commit remaps, which means
    /// every commit must either rebuild this cache or call this. Leaving
    /// stale addresses behind is a use-after-free: the next vector search
    /// dereferences them and segfaults.
    ///
    /// Clearing rather than only clearing `loaded` is deliberate — a
    /// reader that forgets the `loaded` check then sees an empty cache
    /// (wrong, recoverable) instead of a dangling pointer (undefined).
    pub(crate) fn invalidate(&mut self) {
        self.ptrs.clear();
        self.candidates_by_space.clear();
        self.qam_norms_by_space.clear();
        self.sketches_by_space.clear();
        self.upq_i8_by_space.clear();
        self.upq_dequant_by_space.clear();
        self.stride = 0;
        self.loaded = false;
    }
}

/// Build the naked-pointer vector base cache: `ptrs[vid.0]` = absolute
/// mmap address of the vector's encoded base record. Built at open +
/// every commit (after `mmap_remap`). Tombstoned vectors and the slot-0
/// sentinel remain 0.
///
/// Returns `(ptrs, stride)`. `stride` is the codec's
/// `base_bytes_per_vector`; when multiple embedding spaces share the
/// same codec stride the value is stable across the cache, otherwise
/// callers should look up the per-space codec's stride at hot-path
/// entry instead of trusting this returned value.
///
/// SAFETY contract: callers must invalidate (clear) this cache before
/// remapping `file_mmap`. The current call sites do this implicitly by
/// rebuilding immediately after the remap.
type VectorBasePtrsResult = (
    Vec<usize>,
    usize,
    std::collections::HashMap<crate::format::EmbeddingSpaceId, Vec<(u32, usize)>>,
    std::collections::HashMap<crate::format::EmbeddingSpaceId, Vec<f32>>,
    // Per-space dense sign-sketch array `(words, sketches)`, row-aligned
    // with `candidates_by_space[space]`. QAM and UPQ spaces.
    std::collections::HashMap<crate::format::EmbeddingSpaceId, (usize, Vec<u64>)>,
    // UPQ only: vid-indexed decoded-i8 cache rows (dim bytes per vid)
    // and the matching per-vid dequant factors. Mirrors the QAM
    // y-hat-norm cache convention (dense by vid, 0/empty = absent).
    std::collections::HashMap<crate::format::EmbeddingSpaceId, Vec<i8>>,
    std::collections::HashMap<crate::format::EmbeddingSpaceId, Vec<f32>>,
);

pub(super) fn build_vector_base_ptrs(
    catalog: &CatalogSnapshot,
    segment_by_id: &HashMap<SegmentId, SegmentRef>,
    file_mmap: &memmap2::Mmap,
    codec_cache: &HashMap<CodecId, Box<dyn VectorCodec>>,
    verified: Option<&parking_lot::RwLock<std::collections::HashSet<crate::format::SegmentId>>>,
) -> Result<VectorBasePtrsResult> {
    // Structured open-time profiling (VALISE_QUERY_PROFILE or
    // VALISE_OPEN_PROFILE): total build, per-family cache build, and
    // sketch derivation, drained via `last_vector_open_profile()`.
    // This function runs at most once per open/commit, so the extra
    // env lookup in `open_enabled` is off the query hot path.
    let oprof_on = query_profile::open_enabled();
    let mut oprof = query_profile::VectorOpenProfile::default();
    let oprof_total = oprof_on.then(std::time::Instant::now);
    if catalog.vectors.is_empty() {
        return Ok((
            Vec::new(),
            0,
            std::collections::HashMap::new(),
            std::collections::HashMap::new(),
            std::collections::HashMap::new(),
            std::collections::HashMap::new(),
            std::collections::HashMap::new(),
        ));
    }
    let mut max_vid: u64 = 0;
    for v in &catalog.vectors {
        if v.vector_id.0 > max_vid {
            max_vid = v.vector_id.0;
        }
    }
    let mut ptrs = vec![0usize; (max_vid as usize) + 1];

    // Group active vectors by data segment so we open each segment once.
    let mut by_seg: HashMap<SegmentId, Vec<&VectorDesc>> = HashMap::new();
    for v in &catalog.vectors {
        if v.status != VectorStatus::Active {
            continue;
        }
        by_seg.entry(v.data_segment_id).or_default().push(v);
    }

    // Per-space QAM y_hat_norm cache, indexed by vector_id.0. Only
    // populated for spaces whose primary codec is the QAM-sliding
    // engine. Allocated lazily on first such space.
    let mut qam_norms_by_space: std::collections::HashMap<
        crate::format::EmbeddingSpaceId,
        Vec<f32>,
    > = std::collections::HashMap::new();
    // UPQ: vid-indexed decoded-i8 rows + dequant factors, built once at
    // open so stage-2 scoring is a contiguous SDOT-shape dot instead of
    // a per-pair unpack (see docs in src/codec/upq.rs).
    let mut upq_i8_by_space: std::collections::HashMap<crate::format::EmbeddingSpaceId, Vec<i8>> =
        std::collections::HashMap::new();
    let mut upq_dequant_by_space: std::collections::HashMap<
        crate::format::EmbeddingSpaceId,
        Vec<f32>,
    > = std::collections::HashMap::new();

    let mut stride: usize = 0;
    let oprof_cache = oprof_on.then(std::time::Instant::now);
    for (seg_id, descs) in by_seg {
        let seg_ref = segment_by_id.get(&seg_id).copied().ok_or_else(|| {
            Error::Integrity(format!(
                "build_vector_base_ptrs: segment {} not in registry",
                seg_id.0
            ))
        })?;
        let payload = mmap_segment_payload(file_mmap, seg_ref, SegmentType::VectorData)?;
        if let Some(v) = verified {
            crate::file::segment_io::verify_payload_wire_bytes(
                seg_ref.segment_id,
                &seg_ref.checksum,
                payload,
                v,
            )?;
        }
        // Look up the codec by traversing one descriptor → embedding
        // space → codec_id. v1 file scenario: one codec, but the lookup
        // is per-segment so multiple codecs would also work.
        let space_id = descs[0].embedding_space_id;
        let space = catalog
            .embedding_spaces
            .iter()
            .find(|s| s.embedding_space_id == space_id)
            .ok_or_else(|| {
                Error::Integrity(format!(
                    "build_vector_base_ptrs: embedding_space {} missing",
                    space_id.0
                ))
            })?;
        // Vector base pointers are anchored to the primary (QAM) codec
        // stride. f8 spaces don't reach this path yet — their raw-base-ptrs
        // are a follow-up.
        let primary_codec_id = space.primary_codec_id.ok_or_else(|| {
            Error::Integrity(format!(
                "build_vector_base_ptrs: embedding_space {} has no primary_codec_id",
                space.embedding_space_id.0
            ))
        })?;
        let codec = codec_cache.get(&primary_codec_id).ok_or_else(|| {
            Error::Integrity(format!(
                "build_vector_base_ptrs: codec {} missing from cache",
                primary_codec_id.0
            ))
        })?;
        let seg_stride = codec.base_bytes_per_vector();
        if stride == 0 {
            stride = seg_stride;
        } else if stride != seg_stride {
            // Mixed strides: callers must resolve per-vector stride at
            // hot path; the cache layout is still correct.
            stride = 0;
        }
        let reader = VectorDataSegmentReader::open(payload, seg_stride)?;

        // If this segment's primary codec is QAM-sliding (5+6), prime
        // the per-row y_hat_norm cache for the whole segment now. This
        // is the same scan we'd otherwise pay per-query inside
        // `engine.asymmetric_distance`.
        let qam_concrete = codec
            .as_ref()
            .as_any()
            .downcast_ref::<crate::codec::qam_lloyd_max::QamLloydMaxCodec>()
            .filter(|c| c.amp_bits == 5 && c.phase_bits == 6);
        let qam_engine = qam_concrete.and_then(|qam| {
            qam.sliding_engine.get_or_init(|| {
                crate::codec::qam_sliding::QamSlidingEngine::from_codec(qam)
                    .expect("BUG: from_codec failed for 5+6 QAM")
            });
            qam.sliding_engine.get()
        });

        let upq_concrete = codec
            .as_ref()
            .as_any()
            .downcast_ref::<crate::codec::upq::UpqCodec>();
        let mut upq_scratch: Vec<f32> = upq_concrete
            .map(|u| vec![0.0_f32; u.dim])
            .unwrap_or_default();

        for desc in descs {
            let base = reader.base(desc.ordinal_in_segment)?;
            ptrs[desc.vector_id.0 as usize] = base.as_ptr() as usize;
            if let Some(upq) = upq_concrete {
                let dim = upq.dim;
                let cache = upq_i8_by_space
                    .entry(space_id)
                    .or_insert_with(|| vec![0i8; ((max_vid as usize) + 1) * dim]);
                let dequant = upq_dequant_by_space
                    .entry(space_id)
                    .or_insert_with(|| vec![0f32; (max_vid as usize) + 1]);
                let vid = desc.vector_id.0 as usize;
                let row = &mut cache[vid * dim..(vid + 1) * dim];
                dequant[vid] = upq.i8_row_from_base(base, &mut upq_scratch, row)?;
            }
            if let Some(engine) = qam_engine {
                // Cache the inverse so per-candidate scoring is a single
                // fmul instead of `1.0 / norm` per call. Pay the one
                // fdiv per vector at file-open time; reuse it across
                // every future query.
                let norm = engine.y_hat_norm(base);
                let inv_norm = if norm > 1e-12 { 1.0 / norm } else { 0.0 };
                let cache = qam_norms_by_space
                    .entry(space_id)
                    .or_insert_with(|| vec![0f32; (max_vid as usize) + 1]);
                cache[desc.vector_id.0 as usize] = inv_norm;
            }
        }
    }
    if let Some(t) = oprof_cache {
        oprof.cache_build = t.elapsed();
    }
    // Pre-build per-space candidate lists in vid-sorted order. The cache
    // is invalidated together with `ptrs` on every commit (mmap_remap).
    let mut candidates_by_space: std::collections::HashMap<
        crate::format::EmbeddingSpaceId,
        Vec<(u32, usize)>,
    > = std::collections::HashMap::new();
    for v in &catalog.vectors {
        if v.status != VectorStatus::Active {
            continue;
        }
        let idx = v.vector_id.0 as usize;
        let ptr = ptrs.get(idx).copied().unwrap_or(0);
        if ptr == 0 {
            continue;
        }
        candidates_by_space
            .entry(v.embedding_space_id)
            .or_default()
            .push((v.vector_id.0 as u32, ptr));
    }
    // Sort each space's list by ptr to keep the par_iter scan addresses
    // monotonically increasing (prefetcher-friendly). Ingest order is
    // monotonic too, so this is usually a no-op.
    for list in candidates_by_space.values_mut() {
        list.sort_unstable_by_key(|&(_, p)| p);
    }

    // Derive the per-space dense sign sketch (row-aligned with
    // candidates_by_space) for EVERY QAM space, at any `(amp_bits,
    // phase_bits)`. Free: the sketch is a function of the stored phase codes,
    // computed from the mmap'd base bytes via the codec's general
    // `sign_sketch` — nothing extra is persisted. (5,6) is bit-identical to
    // the sliding engine's sketch; other configs take the general rerank path.
    let mut sketches_by_space: std::collections::HashMap<
        crate::format::EmbeddingSpaceId,
        (usize, Vec<u64>),
    > = std::collections::HashMap::new();
    let oprof_sketch = oprof_on.then(std::time::Instant::now);
    for (sid, rows) in &candidates_by_space {
        let Some(space) = catalog
            .embedding_spaces
            .iter()
            .find(|s| s.embedding_space_id == *sid)
        else {
            continue;
        };
        let Some(cid) = space.primary_codec_id else {
            continue;
        };
        let Some(codec) = codec_cache.get(&cid) else {
            continue;
        };
        // Family-generic via `VectorCodec::sign_sketch`. `div_ceil`, not
        // truncating `/64`: `sign_sketch` returns `dim.div_ceil(64)` words,
        // so a dim that isn't a multiple of 64 would otherwise size the row
        // one word short and panic in `copy_from_slice`. Trailing padding
        // bits are zero in both DB and query sketches, so they contribute 0
        // to the Hamming distance.
        let words = (codec.dimension() as usize).div_ceil(64);
        let bb = codec.base_bytes_per_vector();
        let mut sk = vec![0u64; rows.len() * words];
        for (row, &(_vid, ptr)) in rows.iter().enumerate() {
            // Safety: ptr is a live base-bytes pointer into the mmap,
            // produced just above; bb is the codec's base stride.
            let base = unsafe { std::slice::from_raw_parts(ptr as *const u8, bb) };
            sk[row * words..(row + 1) * words].copy_from_slice(&codec.sign_sketch(base)?);
        }
        sketches_by_space.insert(*sid, (words, sk));
    }
    if let Some(t) = oprof_sketch {
        oprof.sketch_derive = t.elapsed();
    }
    if let Some(t) = oprof_total {
        oprof.total = t.elapsed();
        query_profile::record_open(oprof);
    }

    Ok((
        ptrs,
        stride,
        candidates_by_space,
        qam_norms_by_space,
        sketches_by_space,
        upq_i8_by_space,
        upq_dequant_by_space,
    ))
}
