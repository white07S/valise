// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Bulk ingest: a thin batching wrapper over [`Writer`] that group-commits.
//!
//! Value is fsync amortization (durability defaults to `Buffered`, so the
//! writer only fsyncs at `commit`) plus ergonomics. It is **not** wired to the
//! engine's parallel `put_text_frames_batch` path (that bypasses the
//! upsert/identity machinery) — a perf follow-up.
//!
//! Upsert safety is size-independent: if a key already seen in the current
//! (un-flushed) window is put again, the window is flushed first so the prior
//! version is committed and the re-put is a valid update — otherwise the
//! same-batch "can't update an uncommitted key" rule would surface or not
//! depending on the invisible batch size.

use std::hash::{Hash, Hasher};

use rustc_hash::{FxHashSet, FxHasher};

use crate::db::record::{Key, Record};
use crate::db::writer::Writer;
use crate::error::Result;

/// Batching loader. Auto-commits every `threshold` records; `flush` commits the
/// tail. Borrows the writer for its lifetime.
pub struct Bulk<'w, 's> {
    writer: &'w mut Writer<'s>,
    threshold: usize,
    pending: usize,
    /// 64-bit digests of `(coll, key)` seen in the current window — no per-put
    /// `String`/`Vec` alloc. A hash collision only forces a redundant (still
    /// correct) flush, so the digest is sound for dedup.
    seen: FxHashSet<u64>,
}

impl<'w, 's> Bulk<'w, 's> {
    pub(crate) fn new(writer: &'w mut Writer<'s>, threshold: usize) -> Self {
        Self {
            writer,
            threshold: threshold.max(1),
            pending: 0,
            seen: FxHashSet::default(),
        }
    }

    /// Buffer a put; auto-flushes at the batch threshold (and before a repeated
    /// key, to keep upsert behavior size-independent).
    pub fn put(&mut self, coll: &str, key: impl Into<Key>, rec: Record<'_>) -> Result<()> {
        let key: Key = key.into();
        let mut hasher = FxHasher::default();
        coll.hash(&mut hasher);
        key.hash(&mut hasher);
        let digest = hasher.finish();
        if self.seen.contains(&digest) {
            self.flush()?;
        }
        self.writer.put(coll, key, rec)?;
        self.seen.insert(digest);
        self.pending += 1;
        if self.pending >= self.threshold {
            self.flush()?;
        }
        Ok(())
    }

    /// Group-commit the buffered records.
    pub fn flush(&mut self) -> Result<()> {
        if self.pending == 0 {
            return Ok(());
        }
        self.writer.commit()?;
        self.pending = 0;
        self.seen.clear();
        Ok(())
    }
}
