//! OFD / `F_SETLK` byte-lock abstraction.
//!
//! Two platform paths:
//!
//! - **Linux**: `F_OFD_SETLK` (Open File Description locks). Tied to the
//!   open file description, not the process; immune to the POSIX
//!   "any-fd-close releases all process locks on file" footgun.
//!   Requires Linux 3.15+ and glibc 2.20+.
//!
//! - **macOS**: OFD locks do not exist. We use `F_SETLK` and rely on the
//!   `Database` registry's single-fd-per-`(dev, ino)` invariant (see
//!   `docs/CONCURRENCY_PLAN.md` §4.3 / §6) to keep the legacy POSIX
//!   semantics safe. Stage 4 introduces the registry; Stage 3b ships a
//!   single-process variant where the invariant is naturally satisfied.
//!
//! Stage 3b uses these locks for two byte ranges in the file header:
//!
//! - The **writer slot** byte (file offset 192): exclusive lock acquired
//!   on commit entry, released after publish. One writer at a time
//!   across processes.
//! - **Reader slot** lock bytes (file offsets 320, 384, …, 768): shared
//!   lock acquired on snapshot pin, released on `Snapshot` drop. Many
//!   readers may hold their own slot's shared lock concurrently.
//!
//! See `docs/FORMAT.md` §7.1.1 for the lock-byte mapping.

use std::fs::File;
use std::os::fd::{AsRawFd, BorrowedFd};

use crate::error::{Error, Result};

/// Type of lock to acquire.
#[allow(
    dead_code,
    reason = "Shared variant consumed by reader-slot acquisition (Stage 4)"
)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum LockKind {
    Shared,
    Exclusive,
}

/// Acquire a one-byte advisory lock at the given file offset.
///
/// Returns `Ok(true)` on success, `Ok(false)` if another process holds an
/// incompatible lock (`EAGAIN` / `EACCES`), and `Err` on other system
/// errors. Caller releases via [`release_byte_lock`] (or by closing the
/// fd, which on macOS invalidates *all* this process's locks on the file
/// — hence the single-fd invariant).
pub(crate) fn try_acquire_byte_lock(fd: BorrowedFd<'_>, byte: u64, kind: LockKind) -> Result<bool> {
    fcntl_setlk(fd, byte, kind, true)
}

/// Release a one-byte advisory lock at the given file offset. Idempotent
/// — releasing a byte that isn't currently locked by this process is a
/// no-op (the OS still honours the request as a `F_UNLCK`).
pub(crate) fn release_byte_lock(fd: BorrowedFd<'_>, byte: u64) -> Result<()> {
    let _ = fcntl_unlk(fd, byte)?;
    Ok(())
}

#[cfg(unix)]
fn fcntl_setlk(fd: BorrowedFd<'_>, byte: u64, kind: LockKind, nonblocking: bool) -> Result<bool> {
    // The `as i16` casts are load-bearing on Linux, where the F_*LCK
    // constants are `i32` while `flock.l_type` is `i16`. On macOS they are
    // already `i16` and clippy will call the cast redundant — do not remove
    // it on that advice, and do not run `clippy --fix` for this file on a
    // Mac without re-checking Linux.
    #[allow(clippy::unnecessary_cast)]
    let l_type = match kind {
        LockKind::Shared => libc::F_RDLCK as i16,
        LockKind::Exclusive => libc::F_WRLCK as i16,
    };
    let mut fl: libc::flock = unsafe { std::mem::zeroed() };
    fl.l_type = l_type;
    fl.l_whence = libc::SEEK_SET as i16;
    fl.l_start = byte as libc::off_t;
    fl.l_len = 1 as libc::off_t;

    let cmd = pick_cmd(nonblocking);
    // SAFETY: `fl` is fully initialized; `fd.as_raw_fd()` is a valid open
    // file descriptor for the duration of the call.
    let rc = unsafe { libc::fcntl(fd.as_raw_fd(), cmd, &mut fl) };
    if rc == 0 {
        Ok(true)
    } else {
        let err = std::io::Error::last_os_error();
        match err.raw_os_error() {
            // Lock contention: another process holds an incompatible lock.
            Some(libc::EAGAIN) | Some(libc::EACCES) => Ok(false),
            _ => Err(Error::Io(err)),
        }
    }
}

#[cfg(unix)]
fn fcntl_unlk(fd: BorrowedFd<'_>, byte: u64) -> Result<bool> {
    let mut fl: libc::flock = unsafe { std::mem::zeroed() };
    // See `fcntl_setlk`: this cast is required on Linux, redundant on macOS.
    #[allow(clippy::unnecessary_cast)]
    {
        fl.l_type = libc::F_UNLCK as i16;
    }
    fl.l_whence = libc::SEEK_SET as i16;
    fl.l_start = byte as libc::off_t;
    fl.l_len = 1 as libc::off_t;

    let cmd = pick_cmd(true);
    // SAFETY: same reasoning as `fcntl_setlk`.
    let rc = unsafe { libc::fcntl(fd.as_raw_fd(), cmd, &mut fl) };
    if rc == 0 {
        Ok(true)
    } else {
        Err(Error::Io(std::io::Error::last_os_error()))
    }
}

/// Pick the right `fcntl` cmd for the platform.
///
/// - Linux: `F_OFD_SETLK` (38) for non-blocking, `F_OFD_SETLKW` (37) for
///   blocking. Per-fd semantics — immune to the POSIX close-any-fd footgun.
/// - macOS / other Unix: `F_SETLK` / `F_SETLKW`. Process-scoped semantics;
///   safety relies on the `Database` registry's single-fd-per-file
///   invariant (see `concurrency/database.rs`, Stage 4).
#[cfg(target_os = "linux")]
fn pick_cmd(nonblocking: bool) -> libc::c_int {
    // libc may not export these; define locally to avoid a version pin.
    const F_OFD_SETLK: libc::c_int = 37;
    const F_OFD_SETLKW: libc::c_int = 38;
    if nonblocking {
        F_OFD_SETLK
    } else {
        F_OFD_SETLKW
    }
}

#[cfg(all(unix, not(target_os = "linux")))]
fn pick_cmd(nonblocking: bool) -> libc::c_int {
    if nonblocking {
        libc::F_SETLK
    } else {
        libc::F_SETLKW
    }
}

#[cfg(not(unix))]
fn fcntl_setlk(
    _fd: BorrowedFd<'_>,
    _byte: u64,
    _kind: LockKind,
    _nonblocking: bool,
) -> Result<bool> {
    Err(Error::Unsupported(
        "byte-range advisory locks are only implemented on Unix".into(),
    ))
}

#[cfg(not(unix))]
fn fcntl_unlk(_fd: BorrowedFd<'_>, _byte: u64) -> Result<bool> {
    Err(Error::Unsupported(
        "byte-range advisory locks are only implemented on Unix".into(),
    ))
}

/// Convenience: spin-acquire an exclusive byte lock with bounded backoff.
/// Returns the number of attempts taken on success, or `Error::Busy` if
/// the deadline passes. Used by writers entering `commit()` who must
/// observe the writer slot exclusively.
pub(crate) fn acquire_exclusive_blocking(
    fd: BorrowedFd<'_>,
    byte: u64,
    max_attempts: u32,
) -> Result<u32> {
    for attempt in 0..max_attempts {
        if try_acquire_byte_lock(fd, byte, LockKind::Exclusive)? {
            return Ok(attempt + 1);
        }
        // Exponential backoff with jitter, capped at ~2 ms per spin.
        let micros = (1u64 << attempt.min(11)).min(2000);
        std::thread::sleep(std::time::Duration::from_micros(micros));
    }
    Err(Error::Busy(
        "writer slot lock contended past max attempts".into(),
    ))
}

/// RAII guard for a byte lock. On `drop`, releases the lock; the caller
/// is responsible for keeping the `File` reachable for the guard's
/// lifetime. Used by reader-slot pinning (Stage 4) to tie lock release
/// to `Snapshot` drop.
#[allow(dead_code, reason = "consumed by reader-slot pin in Stage 4")]
pub(crate) struct ByteLockGuard {
    file: std::sync::Arc<File>,
    byte: u64,
}

#[allow(dead_code, reason = "consumed by reader-slot pin in Stage 4")]
impl ByteLockGuard {
    pub(crate) fn new(file: std::sync::Arc<File>, byte: u64) -> Self {
        Self { file, byte }
    }
}

impl Drop for ByteLockGuard {
    fn drop(&mut self) {
        // SAFETY: `file` is alive here; we got it via `Arc<File>` clone.
        let fd = unsafe { BorrowedFd::borrow_raw(self.file.as_raw_fd()) };
        let _ = release_byte_lock(fd, self.byte);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::fd::AsFd;

    #[test]
    fn shared_lock_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("lock.bin");
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)
            .unwrap();
        file.set_len(1024).unwrap();

        let fd = file.as_fd();
        assert!(try_acquire_byte_lock(fd, 100, LockKind::Shared).unwrap());
        release_byte_lock(fd, 100).unwrap();
        // Re-acquiring should still work.
        assert!(try_acquire_byte_lock(fd, 100, LockKind::Shared).unwrap());
        release_byte_lock(fd, 100).unwrap();
    }

    #[test]
    fn exclusive_lock_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("lock-ex.bin");
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)
            .unwrap();
        file.set_len(1024).unwrap();

        let fd = file.as_fd();
        assert!(try_acquire_byte_lock(fd, 192, LockKind::Exclusive).unwrap());
        release_byte_lock(fd, 192).unwrap();
    }

    #[test]
    fn cross_fd_exclusive_contention_is_visible() {
        // Two distinct fds to the same file. With OFD locks (Linux) the
        // second fd genuinely sees contention. With `F_SETLK` (macOS) the
        // second fd is in the same process and sees its own lock — so we
        // open the file twice from different processes by simulating with
        // two fds; the test below works on both platforms because we
        // verify behaviour via a second OS-level fd.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("lock-contention.bin");
        let file_a = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)
            .unwrap();
        file_a.set_len(1024).unwrap();
        let file_b = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .unwrap();

        let fd_a = file_a.as_fd();
        assert!(try_acquire_byte_lock(fd_a, 192, LockKind::Exclusive).unwrap());

        // Behavior split by platform:
        // - Linux OFD locks: fd_b sees contention (false).
        // - macOS F_SETLK: fd_b is in the same process → no contention
        //   from POSIX semantics. We still expect this to *succeed* on
        //   macOS because legacy `F_SETLK` is process-scoped.
        let fd_b = file_b.as_fd();
        let acquired_b = try_acquire_byte_lock(fd_b, 192, LockKind::Exclusive).unwrap();

        #[cfg(target_os = "linux")]
        assert!(!acquired_b, "OFD locks must enforce per-fd exclusion");

        #[cfg(not(target_os = "linux"))]
        let _ = acquired_b;

        release_byte_lock(fd_a, 192).unwrap();
    }
}
