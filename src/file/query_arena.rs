//! Thread-local scratch arena for text query scoring.
//!
//! On wide indexes (Quora at 522k frames, MS-MARCO at 8.8M), the
//! per-query `dots`/`scores`/`touched` allocations of the cosine /
//! TF-IDF scorers dominate the query cost: each one is a fresh
//! `Vec<f32>` of `n_frames` entries that the OS must back with new
//! 4 KB pages, with the kernel zero-filling them for security. That's
//! megabytes of memset and thousands of page faults per query before
//! a single posting is read.
//!
//! `QueryArena` reuses two long-lived buffers — `scores` and
//! `doc_generations`, both indexed by `FrameId.0` — and tags each
//! entry with the query's monotonically-increasing `generation`. An
//! entry is "live" for the current query iff its generation stamp
//! matches `arena.generation`; anything else is implicitly zero.
//! No memset, no page faults, no allocator traffic on the hot path.
//!
//! The arena is thread-local so concurrent queries on the same
//! `ValiseFile` don't fight over it.

use std::cell::RefCell;

/// Reusable per-query scratch buffers keyed by frame_id.
///
/// Invariant: `scores.len() == doc_generations.len() >= n_frames`
/// where `n_frames` is the largest value ever passed to
/// [`begin_query`]. Buffers grow monotonically; they never shrink.
///
/// [`begin_query`]: Self::begin_query
pub(crate) struct QueryArena {
    /// Per-doc score accumulator. `scores[idx]` is meaningful only
    /// when `doc_generations[idx] == generation`.
    pub(crate) scores: Vec<f32>,
    /// Per-doc generation stamp. Matches `generation` iff the doc has
    /// already been touched in the current query.
    pub(crate) doc_generations: Vec<u32>,
    /// Monotonic generation. Bumped at the start of every query.
    pub(crate) generation: u32,
    /// Frame IDs touched by the current query (cleared on each
    /// `begin_query`).
    pub(crate) touched: Vec<u32>,
}

impl QueryArena {
    pub(crate) const fn new() -> Self {
        Self {
            scores: Vec::new(),
            doc_generations: Vec::new(),
            generation: 0,
            touched: Vec::new(),
        }
    }

    /// Open a fresh query against an index with `n_frames` docs.
    /// Resizes buffers on first growth, bumps the generation
    /// counter — no memset of the live region.
    pub(crate) fn begin_query(&mut self, n_frames: usize) {
        if self.scores.len() < n_frames {
            self.scores.resize(n_frames, 0.0);
            self.doc_generations.resize(n_frames, 0);
        }
        self.touched.clear();
        if self.generation == u32::MAX {
            // Wrap. The only memset in the arena's lifetime, and it
            // happens at most once every 4 billion queries.
            for g in &mut self.doc_generations {
                *g = 0;
            }
            self.generation = 1;
        } else {
            self.generation += 1;
        }
    }

    /// Add `delta` to `scores[idx]`. If `idx` hasn't been touched
    /// yet in this query, initialize to `delta` and record it in
    /// `touched`. The generation check replaces the unpredictable
    /// `if prev == 0.0` branch that the old scorers used.
    #[inline]
    pub(crate) fn accumulate(&mut self, idx: usize, delta: f32) {
        if self.doc_generations[idx] != self.generation {
            self.doc_generations[idx] = self.generation;
            self.scores[idx] = delta;
            self.touched.push(idx as u32);
        } else {
            self.scores[idx] += delta;
        }
    }
}

thread_local! {
    static QUERY_ARENA: RefCell<QueryArena> = const { RefCell::new(QueryArena::new()) };
}

/// Run `f` with mutable access to the thread-local arena.
pub(crate) fn with_arena<R>(f: impl FnOnce(&mut QueryArena) -> R) -> R {
    QUERY_ARENA.with(|cell| f(&mut cell.borrow_mut()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generation_bumps_invalidate_prior_writes() {
        let mut a = QueryArena::new();
        a.begin_query(8);
        a.accumulate(3, 1.5);
        a.accumulate(3, 0.5);
        assert_eq!(a.scores[3], 2.0);
        assert_eq!(a.touched, vec![3]);

        a.begin_query(8);
        // Without any memset, slot 3 still holds 2.0 — but the
        // generation has moved, so the touched list is empty and
        // any subsequent accumulate would re-initialize the slot.
        assert!(a.touched.is_empty());
        assert_ne!(a.doc_generations[3], a.generation);
    }

    #[test]
    fn buffers_grow_to_largest_request() {
        let mut a = QueryArena::new();
        a.begin_query(4);
        assert!(a.scores.len() >= 4);
        a.begin_query(16);
        assert!(a.scores.len() >= 16);
        a.begin_query(2);
        // Shrink-resistant: prior capacity is retained.
        assert!(a.scores.len() >= 16);
    }

    #[test]
    fn generation_wrap_resets_stamps() {
        let mut a = QueryArena::new();
        a.begin_query(4);
        // Manually fast-forward to u32::MAX so the next call wraps.
        a.generation = u32::MAX;
        a.doc_generations.fill(u32::MAX);
        a.scores.fill(99.0);
        a.begin_query(4);
        assert_eq!(a.generation, 1);
        assert!(a.doc_generations.iter().all(|&g| g == 0));
    }
}
