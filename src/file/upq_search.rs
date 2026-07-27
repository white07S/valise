//! UPQ vector-search path: sign-sketch scan → decoded-i8 contiguous
//! rerank → optional f32 stage over the survivors.
//!
//! Mirrors the QAM `sketch_then_rerank_impl` skeleton with the UPQ
//! stage-2 kernel measured in `experiments/PARETO_RESEARCH_2026-06.md`
//! Part 7: candidates are scored against the vid-indexed decoded-i8
//! cache built at file-open (`VectorBasePtrs::upq_i8_by_space`) via a
//! prefetched contiguous i8 dot (`codec::upq::score_i8_cache`'s inner
//! shape), then `Full` fidelity re-scores the `3·k` survivors exactly
//! from the packed codes (`UpqCodec::decode_dot_from_base`).
//!
//! Scores follow the engine convention (smaller = more similar):
//! stage-2 reports negated dequantized i8 dots (per-query arbitrary
//! scale, rank-faithful); `Full` reports the true cosine distance.

use std::collections::HashSet;

use super::{
    DEFAULT_SKETCH_CANDIDATE_BUDGET, ValiseFile, VectorFidelity, VectorHit, VectorSearchQuery,
    query_profile,
};
use crate::codec::VectorCodec as _;
use crate::codec::upq::{UpqCodec, dot_i8};
use crate::error::{Error, Result};
use crate::format::catalog::EmbeddingSpaceDesc;
use crate::format::{CollectionId, VectorId};

/// One resolved rerank candidate: vid + its packed-row pointer.
struct UpqCand {
    vid: u32,
    ptr: usize,
}

struct Scored {
    score: f32,
    vid: u32,
    ptr: usize,
}

impl ValiseFile {
    pub(super) fn upq_sketch_then_rerank(
        &self,
        space: &EmbeddingSpaceDesc,
        query: &VectorSearchQuery,
    ) -> Result<Vec<VectorHit>> {
        // VALISE_QUERY_PROFILE per-stage instrumentation (cached-bool
        // gate; every `lap` is a single branch when disabled). Stage
        // boundaries mirror the QAM path's in `vector_search.rs`.
        let qprof_on = query_profile::enabled();
        let mut qprof = query_profile::QueryProfile::default();
        let mut qprof_mark = qprof_on.then(std::time::Instant::now);
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
        let primary_box = self.codec_cache.get(&primary_id).ok_or_else(|| {
            Error::Integrity(format!(
                "upq_sketch_then_rerank: primary codec {} missing from cache",
                primary_id.0
            ))
        })?;
        let upq = primary_box
            .as_ref()
            .as_any()
            .downcast_ref::<UpqCodec>()
            .ok_or_else(|| {
                Error::Integrity(format!(
                    "upq_sketch_then_rerank: space {} primary codec is not UPQ",
                    space.embedding_space_id.0
                ))
            })?;

        // Same fixed, corpus-size-independent default budget as the QAM
        // path (see docs/VECTOR_SEARCH.md §4).
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

        let _fanout_permit = crate::file::query_admission::global().try_acquire();
        let parallel = _fanout_permit.is_some();

        // Stage 1 — sign-sketch Hamming scan (identical machinery to
        // the QAM path; the sketch was derived from UPQ codes at open).
        let (q_rot, q_norm) = upq.rotate_query(&query.query)?;
        let q_sketch = crate::retrieval::sketch::pack_query_sketch(&q_rot);
        let dim = upq.dim;
        let vb = self.vector_base.read();
        qprof.prep = query_profile::lap(&mut qprof_mark);
        let candidates: Vec<(VectorId, i32)> = {
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
                dim,
                parallel,
                &mut hbuf,
            )
        };
        qprof.stage1_sketch = query_profile::lap(&mut qprof_mark);

        // Stage 2 — contiguous i8 rerank against the decoded cache.
        let i8_cache = vb
            .upq_i8_by_space
            .get(&space.embedding_space_id)
            .ok_or_else(|| {
                Error::Integrity(format!(
                    "vector_search: no UPQ i8 cache for space {}",
                    space.embedding_space_id.0
                ))
            })?;
        let dequant = vb
            .upq_dequant_by_space
            .get(&space.embedding_space_id)
            .ok_or_else(|| {
                Error::Integrity(format!(
                    "vector_search: no UPQ dequant cache for space {}",
                    space.embedding_space_id.0
                ))
            })?;
        let mut q_i8 = vec![0i8; dim];
        UpqCodec::prepare_query_i8(&q_rot, &mut q_i8);

        let filter: Option<&HashSet<CollectionId>> = query.collection_filter.as_ref();
        let ptrs: &[usize] = &vb.ptrs;
        let mut cands: Vec<UpqCand> = Vec::with_capacity(candidates.len());
        for (vid, _ham) in &candidates {
            let v = vid.0 as usize;
            if let Some(set) = filter {
                let Some(desc) = self.vector_by_id_cache.get(vid) else {
                    continue;
                };
                if !set.contains(&desc.collection_id) {
                    continue;
                }
            }
            let ptr = ptrs.get(v).copied().unwrap_or(0);
            if ptr == 0 {
                continue; // tombstoned between catalog snapshots; skip
            }
            cands.push(UpqCand {
                vid: vid.0 as u32,
                ptr,
            });
        }
        // Hamming-select order is scattered; the i8 cache is contiguous
        // rows at `vid * dim`, so a vid-sort turns stage 2 into a
        // near-sequential scan (same trick as the QAM path's ptr-sort —
        // the final score sort below is the ordering authority).
        if parallel {
            use rayon::slice::ParallelSliceMut;
            cands.par_sort_unstable_by_key(|c| c.vid);
        } else {
            cands.sort_unstable_by_key(|c| c.vid);
        }

        let score_one = |c: &UpqCand| -> Scored {
            let v = c.vid as usize;
            let row = &i8_cache[v * dim..(v + 1) * dim];
            // Negate: dequantized i8 dot approximates cosine up to a
            // positive per-query constant; engine scores are
            // smaller-is-more-similar.
            let score = -(dot_i8(&q_i8, row) as f32) * dequant[v];
            Scored {
                score,
                vid: c.vid,
                ptr: c.ptr,
            }
        };
        let mut scored: Vec<Scored> = if parallel {
            use rayon::prelude::*;
            cands.par_iter().map(score_one).collect()
        } else {
            cands.iter().map(score_one).collect()
        };

        // Bounded top pool: 3·k for Full so the exact pass can move
        // items in/out of the final top-k; plain k for Lossy.
        let pool_size = if query.fidelity == VectorFidelity::Full {
            (3 * query.k).min(scored.len())
        } else {
            query.k.min(scored.len())
        };
        if scored.len() > pool_size && pool_size > 0 {
            scored.select_nth_unstable_by(pool_size - 1, |a, b| {
                a.score.total_cmp(&b.score).then_with(|| a.vid.cmp(&b.vid))
            });
            scored.truncate(pool_size);
        }
        scored.sort_by(|a, b| a.score.total_cmp(&b.score).then_with(|| a.vid.cmp(&b.vid)));
        qprof.stage2_rerank = query_profile::lap(&mut qprof_mark);
        qprof.candidates_reranked = cands.len();
        qprof.survivors = scored.len();

        // Stage 3 — exact rescoring of the survivors off the packed
        // codes (cosine distance on the σ-scaled reconstruction).
        // Parallel when the fanout permit was granted: the pool is
        // 3·k survivors × num_pairs decode-LUT gathers each, which is
        // the dominant serial cost at large k (mirrors the QAM path's
        // parallel exact pass).
        if query.fidelity == VectorFidelity::Full {
            let base_bytes = upq.base_bytes_per_vector();
            let rescore_one = |item: &mut Scored| -> Result<()> {
                // Safety: ptr is a live base-row pointer into the mmap
                // captured by `vb`, which outlives this scope.
                let base: &[u8] =
                    unsafe { std::slice::from_raw_parts(item.ptr as *const u8, base_bytes) };
                item.score = upq.decode_dot_from_base(base, &q_rot, q_norm)?;
                Ok(())
            };
            if parallel {
                use rayon::prelude::*;
                scored.par_iter_mut().try_for_each(rescore_one)?;
            } else {
                scored.iter_mut().try_for_each(rescore_one)?;
            }
            scored.sort_by(|a, b| a.score.total_cmp(&b.score).then_with(|| a.vid.cmp(&b.vid)));
            scored.truncate(query.k);
            qprof.stage3_exact = query_profile::lap(&mut qprof_mark);
        }
        if qprof_on {
            query_profile::record_query(qprof);
        }

        Ok(scored
            .iter()
            .take(query.k)
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
            .collect())
    }
}
