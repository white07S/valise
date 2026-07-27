//! Adaptive admission control for query-time rayon fan-out.
//!
//! A single vote-then-rerank query already fans out across every core
//! (the rerank `par_iter` over `channel_k` candidates, plus the vote
//! accumulate's per-shard `par_iter_mut`). When many reader threads each
//! launch that fan-out into the one process-global rayon pool, the pool
//! is oversubscribed by `readers ×` the per-query width — work-stealing
//! jitter across all those independent join scopes blows up tail latency.
//!
//! Rather than *queue* excess queries (which just moves the wait into the
//! client's latency), this limiter lets a query **fall back to a serial
//! execution path** when the pool is already saturated. The policy:
//!
//! - [`try_acquire`](FanoutLimiter::try_acquire) succeeds while fewer
//!   than `permits` queries hold a permit → that query fans out across
//!   rayon (fast at low concurrency).
//! - When permits are exhausted it returns `None` immediately → that
//!   query runs serially on its own thread (one core, zero pool
//!   contention). Under heavy concurrency the cores stay busy via
//!   inter-query parallelism instead of intra-query fan-out.
//!
//! The cap is process-global because the rayon pool is process-global.
//! Override with `VALISE_QUERY_FANOUT` (`0` = unlimited → always parallel,
//! the pre-admission behavior); the default is tuned by the concurrency
//! sweep in `bench/examples/vote_trace.rs`.

use std::sync::OnceLock;

use parking_lot::Mutex;

/// Default number of queries allowed to fan out across rayon at once.
/// One query reaches ~6 effective cores on the rerank, so a couple of
/// concurrent fan-outs already fill a 12-P-core machine; further queries
/// do better running serially than fighting for pool workers.
const DEFAULT_PERMITS: usize = 2;

/// Sentinel meaning "no cap" — `try_acquire` always succeeds and never
/// touches the mutex (so unlimited mode adds zero hot-path overhead).
const UNLIMITED: usize = usize::MAX;

pub(crate) struct FanoutLimiter {
    /// `usize::MAX` ([`UNLIMITED`]) short-circuits the mutex entirely.
    permits: usize,
    /// Permits currently held. Only touched when `permits != UNLIMITED`.
    in_use: Mutex<usize>,
}

impl FanoutLimiter {
    fn new(permits: usize) -> Self {
        Self {
            permits,
            in_use: Mutex::new(0),
        }
    }

    /// Non-blocking: take a permit if one is free, else return `None`.
    /// `None` is the signal to run the query serially. Never blocks, so
    /// no query ever waits in a queue.
    pub(crate) fn try_acquire(&self) -> Option<FanoutPermit<'_>> {
        if self.permits == UNLIMITED {
            // Unlimited → always parallel, no synchronization.
            return Some(FanoutPermit { limiter: None });
        }
        let mut in_use = self.in_use.lock();
        if *in_use >= self.permits {
            return None;
        }
        *in_use += 1;
        Some(FanoutPermit {
            limiter: Some(self),
        })
    }
}

/// RAII permit. Releasing on drop returns the permit even if the scored
/// query bails out with `?`. A `None` limiter is the unlimited no-op.
pub(crate) struct FanoutPermit<'a> {
    limiter: Option<&'a FanoutLimiter>,
}

impl Drop for FanoutPermit<'_> {
    fn drop(&mut self) {
        if let Some(l) = self.limiter {
            *l.in_use.lock() -= 1;
        }
    }
}

/// The process-global limiter, initialized once from `VALISE_QUERY_FANOUT`
/// (or [`DEFAULT_PERMITS`]). `VALISE_QUERY_FANOUT=0` → unlimited.
pub(crate) fn global() -> &'static FanoutLimiter {
    static LIMITER: OnceLock<FanoutLimiter> = OnceLock::new();
    LIMITER.get_or_init(|| {
        let permits = match std::env::var("VALISE_QUERY_FANOUT")
            .ok()
            .and_then(|s| s.parse().ok())
        {
            Some(0) => UNLIMITED,
            Some(k) => k,
            None => DEFAULT_PERMITS,
        };
        FanoutLimiter::new(permits)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn admits_up_to_permits_then_falls_back() {
        let limiter = FanoutLimiter::new(2);
        let p1 = limiter.try_acquire();
        let p2 = limiter.try_acquire();
        assert!(p1.is_some() && p2.is_some());
        // Third query is denied a permit → caller runs serial.
        assert!(limiter.try_acquire().is_none());
        // Dropping one frees a slot.
        drop(p1);
        assert!(limiter.try_acquire().is_some());
    }

    #[test]
    fn unlimited_always_admits_without_locking() {
        let limiter = FanoutLimiter::new(UNLIMITED);
        let _a = limiter.try_acquire().expect("unlimited admits");
        let _b = limiter.try_acquire().expect("unlimited admits");
        let _c = limiter.try_acquire().expect("unlimited admits");
    }
}
