// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! 4-way set-associative AoS memoization cache for the Porter2 stemmer.
//!
//! For English text with the BEIR-standard pipeline (lowercase +
//! Lucene-33 stopwords + Porter2 stem) the stemmer is the single
//! largest per-token cost (~400 ns/call: Snowball state machine + the
//! `Cow::Owned(String)` allocation on transforming output). This
//! cache memoizes (post-fold-byte-slice → stem-byte-slice) per rayon
//! worker, eliminating the stemmer call on hits.
//!
//! ## Layout
//!
//! - **Array of Structures.** Each `CacheSlot` packs `key` + `val`
//!   into exactly one 64 B cache line via `#[repr(C, align(64))]`. A
//!   single L2→L1 line fetch on hit covers both the key comparison
//!   and the value read.
//! - **4-way set-associative.** Hash maps to a set of 4 contiguous
//!   slots; all 4 are probed. Tested wider associativity and tree
//!   pseudo-LRU eviction: neither raised the BEIR-pilot hit rate
//!   above the ~42% structural ceiling on this workload (per-worker
//!   vocab dominates capacity), so this revision keeps the cheaper
//!   variant. Round-robin eviction via a global lookup counter.
//! - **8192 slots × 64 B = 512 KB per worker.** Fits in M-series L2.
//!   At 8 workers the global footprint is 4 MB — far below the 8–16
//!   MB shared L2.
//!
//! Cache hit produces ~30 ns of work (hash + line load + memcmp +
//! memcpy) replacing ~400 ns of stem work — a ~13× speedup on the
//! cache-hit path.

use std::hash::Hasher;

use rust_stemmers::Stemmer;
use rustc_hash::FxHasher;

use crate::text::inline_token::InlineToken;

/// Total slot count. Power of two so we mask instead of mod.
pub(crate) const CACHE_SIZE: usize = 8192;
/// Ways per set. 4 is the canonical sweet spot for hardware caches.
pub(crate) const WAYS: usize = 4;
/// Number of sets. Each set is `WAYS` consecutive slots.
pub(crate) const N_SETS: usize = CACHE_SIZE / WAYS;
const SET_MASK: usize = N_SETS - 1;

/// One slot = one cache line on x86 / Apple Silicon.
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug)]
struct CacheSlot {
    key: InlineToken,
    val: InlineToken,
}

const _: () = assert!(std::mem::size_of::<CacheSlot>() == 64);

impl CacheSlot {
    const fn empty() -> Self {
        Self {
            key: InlineToken::empty(),
            val: InlineToken::empty(),
        }
    }
}

pub(crate) struct StemCache {
    slots: Box<[CacheSlot; CACHE_SIZE]>,
    pub(crate) lookups: u64,
    pub(crate) hits: u64,
    pub(crate) bypassed: u64,
}

impl StemCache {
    pub(crate) fn new() -> Self {
        let v = vec![CacheSlot::empty(); CACHE_SIZE];
        let slots: Box<[CacheSlot; CACHE_SIZE]> = v
            .into_boxed_slice()
            .try_into()
            .expect("vec length == CACHE_SIZE");
        Self {
            slots,
            lookups: 0,
            hits: 0,
            bypassed: 0,
        }
    }

    /// Stem `folded` and write the result into `out_buf`.
    ///
    /// `folded.len() > INLINE_TOKEN_CAP` (31 B) bypasses the cache;
    /// stem outputs > 31 B are returned correctly but not memoized.
    #[inline]
    pub(crate) fn stem_into(
        &mut self,
        folded: &[u8],
        stemmer: Option<&Stemmer>,
        out_buf: &mut Vec<u8>,
    ) {
        out_buf.clear();
        self.lookups = self.lookups.saturating_add(1);

        let Some(stem) = stemmer else {
            out_buf.extend_from_slice(folded);
            return;
        };

        let Some(key_token) = InlineToken::new(folded) else {
            self.bypassed = self.bypassed.saturating_add(1);
            self.stem_direct(folded, stem, out_buf);
            return;
        };

        let mut hasher = FxHasher::default();
        hasher.write(folded);
        let set_idx = (hasher.finish() as usize) & SET_MASK;
        let base = set_idx * WAYS;

        // Probe all WAYS ways. First slot pulls a 64 B line carrying
        // both key and val; the next 3 slots occupy adjacent lines.
        for way in 0..WAYS {
            let slot = &self.slots[base + way];
            if slot.key == key_token {
                self.hits = self.hits.saturating_add(1);
                out_buf.extend_from_slice(slot.val.as_slice());
                return;
            }
        }

        // Miss. Stem directly + write back into round-robin victim.
        self.stem_direct(folded, stem, out_buf);
        let victim_way = (self.lookups as usize) & (WAYS - 1);
        let victim = base + victim_way;
        if let Some(stem_token) = InlineToken::new(out_buf) {
            self.slots[victim].key = key_token;
            self.slots[victim].val = stem_token;
        } else {
            self.bypassed = self.bypassed.saturating_add(1);
        }
    }

    #[inline]
    fn stem_direct(&self, folded: &[u8], stemmer: &Stemmer, out_buf: &mut Vec<u8>) {
        // SAFETY: the analyzer fold pipeline guarantees `folded` is
        // valid UTF-8 — `case_fold` + `accent_fold` output is UTF-8 by
        // construction, and the ASCII fast-path emits only
        // ASCII-alphanumeric byte sequences which are trivially valid.
        let s = unsafe { std::str::from_utf8_unchecked(folded) };
        match stemmer.stem(s) {
            std::borrow::Cow::Owned(string) => out_buf.extend_from_slice(string.as_bytes()),
            std::borrow::Cow::Borrowed(b) => out_buf.extend_from_slice(b.as_bytes()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text::inline_token::INLINE_TOKEN_CAP;
    use rust_stemmers::Algorithm;

    #[test]
    fn slot_is_one_cache_line() {
        assert_eq!(std::mem::size_of::<CacheSlot>(), 64);
        assert_eq!(std::mem::align_of::<CacheSlot>(), 64);
    }

    #[test]
    fn empty_cache_miss_then_hit() {
        let mut cache = StemCache::new();
        let stem = Stemmer::create(Algorithm::English);
        let mut buf = Vec::new();
        cache.stem_into(b"running", Some(&stem), &mut buf);
        assert_eq!(&*buf, b"run");
        assert_eq!(cache.hits, 0);
        cache.stem_into(b"running", Some(&stem), &mut buf);
        assert_eq!(&*buf, b"run");
        assert_eq!(cache.hits, 1);
    }

    #[test]
    fn distinct_tokens_distinguished() {
        let mut cache = StemCache::new();
        let stem = Stemmer::create(Algorithm::English);
        let mut buf = Vec::new();
        cache.stem_into(b"running", Some(&stem), &mut buf);
        assert_eq!(&*buf, b"run");
        cache.stem_into(b"runner", Some(&stem), &mut buf);
        // Porter2 keeps "runner" as the agent noun.
        assert_eq!(&*buf, b"runner");
    }

    #[test]
    fn overlong_input_bypasses() {
        let mut cache = StemCache::new();
        let stem = Stemmer::create(Algorithm::English);
        let long = b"abcdefghijklmnopqrstuvwxyz0123456789";
        assert!(long.len() > INLINE_TOKEN_CAP);
        let mut buf = Vec::new();
        cache.stem_into(long, Some(&stem), &mut buf);
        assert_eq!(cache.bypassed, 1);
    }

    #[test]
    fn no_stemmer_is_passthrough() {
        let mut cache = StemCache::new();
        let mut buf = Vec::new();
        cache.stem_into(b"running", None, &mut buf);
        assert_eq!(&*buf, b"running");
    }
}
