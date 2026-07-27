//! Concurrency primitives for Valise.
//!
//! See `docs/CONCURRENCY.md` for the design, stage roadmap, and the
//! running decision log.
//!
//! ## Stage status
//!
//! Stages 1–5++ have landed. `Database`, `Snapshot`, `ReadConnection`, and
//! `WriteConnection` are live and re-exported from the crate root (see
//! `lib.rs`). The current write path is the Stage 5++ avalanche commit with
//! a `GroupFsync` group-commit barrier; there is no embedded WAL (the v2
//! format removed it — see `MIGRATION.md`).
//!
//! ## Module layout
//!
//! - [`snapshot`] — `Snapshot`, the immutable per-publication reader pin.
//! - [`database`] — `Database`, the per-`(dev, ino)` shared state.
//! - [`connection`] — `ReadConnection` and `WriteConnection` handles.
//! - `coordination` — typed view of the in-header coordination region.
//! - `locks` — OFD / `F_SETLK` byte-lock abstraction.
//! - `writer_pipeline` — group-commit leader/followers state machine.
//!
//! ## Debug tracing
//!
//! Set `VALISE_CONCURRENCY_DEBUG` in the environment to enable verbose
//! `tracing`-level instrumentation inside this module. Off by default; the
//! lookup is cached after the first call.

use std::sync::OnceLock;

pub mod connection;
pub(crate) mod coordination;
pub mod database;
pub(crate) mod locks;
pub mod snapshot;
pub(crate) mod writer_pipeline;

/// Returns `true` when verbose concurrency tracing is enabled via the
/// `VALISE_CONCURRENCY_DEBUG` environment variable. Cached on first call; later
/// calls are a single atomic load.
#[allow(
    dead_code,
    reason = "reserved for opt-in concurrency tracing; only the test exercises it today"
)]
pub(crate) fn debug_enabled() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| std::env::var_os("VALISE_CONCURRENCY_DEBUG").is_some())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debug_flag_is_cached() {
        // The flag is read once and cached; calling it twice must not panic
        // and must return a consistent value.
        let first = debug_enabled();
        let second = debug_enabled();
        assert_eq!(first, second);
    }
}
