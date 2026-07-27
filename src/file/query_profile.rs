// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Env-gated per-query vector-search profiling + open-time vector-base
//! build profiling.
//!
//! Activation is opt-in via the `VALISE_QUERY_PROFILE` env var, captured
//! **once per process** into an `OnceLock<bool>` — the disabled hot-path
//! cost is a single branch on a cached bool, no `Instant::now()` calls
//! and no env lookups per query. This mirrors the repo's profiling
//! idiom (`VALISE_INGEST_PROFILE` → [`super::profile::IngestProfile`],
//! `VALISE_OPEN_PROFILE` → open-phase timings) but reports **structured
//! values** through a thread-local slot instead of stderr lines, so a
//! bench harness reads numbers instead of parsing logs.
//!
//! When enabled, each `vector_search` call that goes through a
//! sign-sketch path (QAM `sketch_then_rerank_impl` in
//! `vector_search.rs`, UPQ `upq_sketch_then_rerank` in `upq_search.rs`)
//! records a [`QueryProfile`] for the calling thread; the caller drains
//! it with [`last_query_profile`]. The open-time vector-base build
//! (`build_vector_base_ptrs` — pointer walk + per-family caches +
//! sign-sketch derivation) records a [`VectorOpenProfile`] the same way
//! ([`last_vector_open_profile`]); the open-side recording also honors
//! `VALISE_OPEN_PROFILE` so it composes with the existing open profiling.

use std::cell::Cell;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

/// Per-stage wall-clock + candidate-count breakdown of one
/// `vector_search` call through a sign-sketch path. Stage boundaries
/// match the existing delimitation in `vector_search.rs` /
/// `upq_search.rs`:
///
/// - `prep`: space/codec resolution, admission, query rotation and
///   sign-sketch packing — everything before the Hamming scan.
/// - `stage1_sketch`: the sign-sketch Hamming scan + counting-sort
///   candidate selection over all active rows of the space.
/// - `stage2_rerank`: candidate resolution + integer rerank (QAM
///   sliding i8 / UPQ decoded-i8 cache dot) + bounded top-pool
///   selection.
/// - `stage3_exact`: the exact rescoring pass over the survivors.
///   `Duration::ZERO` unless the query asked for
///   [`super::VectorFidelity::Full`].
///
/// `prep + stage1 + stage2 + stage3` ≈ the whole call (the residue is
/// dispatch + final top-k descriptor resolution, both `O(k)`).
#[derive(Clone, Copy, Debug, Default)]
pub struct QueryProfile {
    pub prep: Duration,
    pub stage1_sketch: Duration,
    pub stage2_rerank: Duration,
    pub stage3_exact: Duration,
    /// Rows the sketch scan visited: the active row count of the
    /// queried embedding space.
    pub candidates_scanned: usize,
    /// Candidates that actually reached the stage-2 rerank kernel
    /// (post filter/tombstone resolution — the effective `channel_k`).
    pub candidates_reranked: usize,
    /// Size of the top pool entering stage 3 (`3·k` under `Full`,
    /// `k` under `Lossy` — where stage 3 then doesn't run).
    pub survivors: usize,
}

/// Wall-clock breakdown of one `build_vector_base_ptrs` run — the
/// open-time (or post-commit) derivation of everything `vector_search`
/// reads from memory instead of disk.
#[derive(Clone, Copy, Debug, Default)]
pub struct VectorOpenProfile {
    /// The whole build: pointer walk + caches + sketch derivation.
    pub total: Duration,
    /// Segment walk: base-pointer table plus the per-family caches
    /// (QAM inverse-norms / UPQ decoded-i8 rows + dequant factors).
    pub cache_build: Duration,
    /// Sign-sketch derivation from the stored codes (all spaces).
    pub sketch_derive: Duration,
}

/// `VALISE_QUERY_PROFILE` presence, resolved once per process.
pub(crate) fn enabled() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| std::env::var_os("VALISE_QUERY_PROFILE").is_some())
}

/// Open-side gate: the structured open profile is recorded when either
/// profiling env var is present. Called only from the (rare) vector-base
/// build, so the extra env lookup is not on a hot path.
pub(crate) fn open_enabled() -> bool {
    enabled() || std::env::var_os("VALISE_OPEN_PROFILE").is_some()
}

thread_local! {
    static LAST_QUERY: Cell<Option<QueryProfile>> = const { Cell::new(None) };
    static LAST_OPEN: Cell<Option<VectorOpenProfile>> = const { Cell::new(None) };
}

pub(crate) fn record_query(profile: QueryProfile) {
    LAST_QUERY.with(|slot| slot.set(Some(profile)));
}

pub(crate) fn record_open(profile: VectorOpenProfile) {
    LAST_OPEN.with(|slot| slot.set(Some(profile)));
}

/// Take (and clear) the profile recorded by the calling thread's most
/// recent sketch-path `vector_search` call. `None` when
/// `VALISE_QUERY_PROFILE` is unset, when the last call on this thread took
/// the brute-force path, or when it was already taken.
pub fn last_query_profile() -> Option<QueryProfile> {
    LAST_QUERY.with(Cell::take)
}

/// Take (and clear) the profile recorded by the calling thread's most
/// recent vector-base build (lazy first-search build or post-commit
/// rebuild). `None` unless `VALISE_QUERY_PROFILE` or `VALISE_OPEN_PROFILE`
/// was set when the build ran on this thread.
pub fn last_vector_open_profile() -> Option<VectorOpenProfile> {
    LAST_OPEN.with(Cell::take)
}

/// Boundary lap timer: returns the elapsed time since `*mark` and
/// advances the mark to now. A `None` mark (profiling disabled) costs
/// one branch and returns `Duration::ZERO` — no clock reads.
pub(crate) fn lap(mark: &mut Option<Instant>) -> Duration {
    match mark {
        Some(m) => {
            let now = Instant::now();
            let elapsed = now.duration_since(*m);
            *mark = Some(now);
            elapsed
        }
        None => Duration::ZERO,
    }
}
