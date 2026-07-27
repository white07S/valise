//! Low-level file I/O helpers.
//!
//! The first implementation keeps most append and recovery orchestration inside
//! `file.rs` while the format codecs stabilize. This module owns reusable
//! durability primitives that do not change on-disk bytes.

use std::{fs::File, path::Path};

#[cfg(target_os = "macos")]
use std::io;
#[cfg(target_os = "macos")]
use std::os::fd::AsRawFd;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Durability {
    /// Flush file contents and metadata using the platform's standard sync.
    SyncAll,

    /// Request the strongest available flush to stable storage.
    #[default]
    FullSync,

    /// Defer durability to explicit `flush()` / `commit()` boundaries.
    /// Per-call `sync_file` becomes a no-op; bytes accumulate in the OS
    /// page cache between barriers. Crash semantics: only the prefix of
    /// writes that completed before the most recent `flush()` (or
    /// `commit()`) is guaranteed recoverable. Useful for bulk ingest;
    /// unsafe to assume per-call durability.
    Buffered,
}

pub fn sync_file(file: &File, durability: Durability) -> crate::Result<()> {
    match durability {
        Durability::SyncAll => {
            file.sync_all()?;
            Ok(())
        }
        Durability::FullSync => sync_file_full(file),
        // Per-call sync is suppressed; the next explicit `flush()` /
        // `commit()` issues a `FullSync` over everything in the page cache.
        Durability::Buffered => Ok(()),
    }
}

#[cfg(target_os = "macos")]
fn sync_file_full(file: &File) -> crate::Result<()> {
    match full_fsync(file) {
        Ok(()) => Ok(()),
        Err(error) if full_fsync_is_unsupported(&error) => {
            file.sync_all()?;
            Ok(())
        }
        Err(error) => Err(error.into()),
    }
}

#[cfg(target_os = "macos")]
fn full_fsync(file: &File) -> io::Result<()> {
    let result = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_FULLFSYNC) };
    if result == -1 {
        Err(io::Error::last_os_error())
    } else {
        Ok(())
    }
}

#[cfg(target_os = "macos")]
fn full_fsync_is_unsupported(error: &io::Error) -> bool {
    let unsupported_codes = [
        libc::EINVAL,
        libc::ENOTTY,
        libc::ENOTSUP,
        libc::EOPNOTSUPP,
        libc::ENOSYS,
    ];

    error
        .raw_os_error()
        .is_some_and(|code| unsupported_codes.contains(&code))
}

#[cfg(not(target_os = "macos"))]
fn sync_file_full(file: &File) -> crate::Result<()> {
    file.sync_all()?;
    Ok(())
}

#[cfg(unix)]
pub fn sync_parent_dir(path: &Path, durability: Durability) -> crate::Result<()> {
    if durability == Durability::Buffered {
        // Per-call directory syncs are skipped under Buffered; an explicit
        // `flush()` reissues the directory sync at the durability barrier.
        return Ok(());
    }
    let parent = parent_dir(path);
    let dir = File::open(parent)?;

    match sync_file(&dir, durability) {
        Ok(()) => Ok(()),
        Err(error) if dir_sync_is_unsupported(&error) => Ok(()),
        Err(error) => Err(error),
    }
}

#[cfg(unix)]
fn parent_dir(path: &Path) -> &Path {
    path.parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
}

#[cfg(unix)]
fn dir_sync_is_unsupported(error: &crate::Error) -> bool {
    let crate::Error::Io(error) = error else {
        return false;
    };

    let unsupported_codes = [libc::EINVAL, libc::ENOTSUP, libc::EOPNOTSUPP, libc::ENOSYS];

    error
        .raw_os_error()
        .is_some_and(|code| unsupported_codes.contains(&code))
}

#[cfg(not(unix))]
pub fn sync_parent_dir(_path: &Path, _durability: Durability) -> crate::Result<()> {
    Ok(())
}
