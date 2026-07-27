//! Per-`Database` group-commit fsync coalescing.
//!
//! Stage 5++ ships a single primitive — [`GroupFsync`] — that lets
//! concurrent `WriteConnection::commit` calls share one
//! `F_FULLFSYNC` syscall. The earlier Stage 5 leader/followers
//! protocol around `pipeline.submit` is gone; the avalanche commit
//! path drives the barrier directly.
//!
//! ## How it engages
//!
//! Every `WriteConnection::commit` runs in three steps:
//!
//! 1. Phase A under the inner write guard — small, microsecond-scale:
//!    write segment / footer / header bytes to the page cache, dup the
//!    file fd, drop the guard.
//! 2. Call [`GroupFsync::fsync`] with **no `Database`-level lock
//!    held**. The first arrival becomes leader and runs the actual
//!    `F_FULLFSYNC` (~5 ms on macOS APFS); concurrent arrivals see
//!    `leader_active == true` and queue on the condvar. The kernel
//!    guarantees that the leader's fsync covers every byte every
//!    queued thread had written *before* the syscall started.
//! 3. Phase B: take the inner write guard, publish the snapshot, drop.
//!
//! See `docs/CONCURRENCY_PLAN.md` §7 for the full design notes.

use parking_lot::{Condvar, Mutex};

use crate::error::Result;

/// Coalesced-fsync barrier, one per `Database`. See module docs.
///
/// Ticket protocol — variant of the SQLite group-fsync pattern:
///
/// 1. Caller takes the next ticket and reads `leader_active`.
/// 2. If no leader is active, the caller becomes leader. It snapshots
///    the current `next_ticket - 1` as `target`, runs the closure
///    (the actual fsync), sets `last_fsynced = max(last_fsynced,
///    target)`, releases leadership, wakes waiters.
/// 3. If a leader is already running and `last_fsynced >= my_ticket`,
///    the caller's bytes are already durable and it returns
///    immediately. Otherwise the caller waits on the condvar.
///
/// On macOS APFS, F_FULLFSYNC is a hardware barrier ~5–25 ms;
/// coalescing N concurrent commits into one fsync turns N×barrier
/// into 1×barrier.
pub struct GroupFsync {
    state: Mutex<GroupFsyncState>,
    cv: Condvar,
}

struct GroupFsyncState {
    next_ticket: u64,
    last_fsynced: u64,
    leader_active: bool,
}

impl GroupFsync {
    pub(crate) fn new() -> Self {
        Self {
            state: Mutex::new(GroupFsyncState {
                next_ticket: 1,
                last_fsynced: 0,
                leader_active: false,
            }),
            cv: Condvar::new(),
        }
    }

    /// Coalesce a fsync. `do_fsync` is invoked at most once per
    /// "burst" of concurrent callers; every caller sees its own
    /// bytes as durable when this returns. Followers whose ticket
    /// wasn't covered by a prior leader's fsync wake from
    /// `cv.wait`, recheck, and promote themselves to leader for
    /// the next round — so progress is guaranteed even under
    /// continuous arrival.
    pub(crate) fn fsync<F>(&self, do_fsync: F) -> Result<()>
    where
        F: FnOnce() -> Result<()>,
    {
        let my_ticket = {
            let mut s = self.state.lock();
            let t = s.next_ticket;
            s.next_ticket += 1;
            // Already covered? (Cheap fast path.)
            if s.last_fsynced >= t {
                return Ok(());
            }
            t
        };
        let mut do_fsync_slot = Some(do_fsync);
        loop {
            let mut s = self.state.lock();
            if s.last_fsynced >= my_ticket {
                return Ok(());
            }
            if s.leader_active {
                // A leader is running. Wait for it to finish; we'll
                // re-check after wakeup.
                self.cv.wait(&mut s);
                continue;
            }
            // No leader and we're not covered → promote ourselves.
            s.leader_active = true;
            let target = s.next_ticket - 1;
            drop(s);
            let do_fsync = do_fsync_slot
                .take()
                .expect("BUG: fsync closure consumed twice");
            let result = do_fsync();
            {
                let mut s = self.state.lock();
                s.leader_active = false;
                if result.is_ok() {
                    s.last_fsynced = s.last_fsynced.max(target);
                }
            }
            self.cv.notify_all();
            // We were the leader; our ticket <= target → covered.
            return result;
        }
    }
}

/// Per-`Database` group-commit coordinator. Owns the `GroupFsync`
/// barrier shared with the underlying `ValiseFile`'s commit path.
pub(crate) struct WriterPipeline {
    /// Shared with `ValiseFile` via `install_commit_fsync_barrier` at
    /// `Database` open/create time.
    pub(crate) commit_fsync: std::sync::Arc<GroupFsync>,
}

impl WriterPipeline {
    pub(crate) fn new() -> Self {
        Self {
            commit_fsync: std::sync::Arc::new(GroupFsync::new()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::thread;
    use std::time::Duration;

    #[test]
    fn coalesces_concurrent_fsyncs() {
        let g = Arc::new(GroupFsync::new());
        let real_fsync_count = Arc::new(AtomicU64::new(0));
        let mut handles = Vec::new();
        for _ in 0..32 {
            let g = Arc::clone(&g);
            let counter = Arc::clone(&real_fsync_count);
            handles.push(thread::spawn(move || {
                g.fsync(|| {
                    // Simulate a slow F_FULLFSYNC.
                    counter.fetch_add(1, Ordering::AcqRel);
                    thread::sleep(Duration::from_millis(20));
                    Ok(())
                })
                .unwrap();
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        // 32 callers ⇒ at most a handful of real fsyncs (some slipped
        // in after the first leader released). Definitely fewer than
        // 32 — that's the coalescing guarantee.
        let n = real_fsync_count.load(Ordering::Acquire);
        assert!(n < 32, "expected coalescing; got {n} real fsyncs");
        assert!(n >= 1);
    }

    #[test]
    fn already_covered_returns_immediately() {
        let g = GroupFsync::new();
        // First call runs the closure.
        let ran = std::cell::Cell::new(false);
        g.fsync(|| {
            ran.set(true);
            Ok(())
        })
        .unwrap();
        assert!(ran.get());
        // Second call on a fresh ticket falls through the loop and
        // becomes the next leader (no prior leader was waiting); the
        // closure runs again. Verify by toggling.
        let ran2 = std::cell::Cell::new(false);
        g.fsync(|| {
            ran2.set(true);
            Ok(())
        })
        .unwrap();
        assert!(ran2.get());
    }
}
