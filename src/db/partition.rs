//! Partitioning + views — the AI-memory generalization (DB-layer plan §6).
//!
//! A [`Partitioned`] routes records into physical collections named
//! `"{base}:{period}"` that all share one [`Shape`] (hence one set of
//! file-global spaces — no codec duplication). A [`View`] selects a window of
//! those partitions for a fan-out-free [`crate::db::Reader::search_view`] (which
//! lowers to a single multi-collection search). [`Partitioned::forget_before`]
//! is the cheap whole-window soft-forget.
//!
//! Partition names are human-readable, dependency-free civil dates
//! (`memory:2026-05-25`), parsed back by `forget_before`/`view` — no sidecar.
//!
//! Caveat (late arrival): routing uses `record.created_at`, so a record whose
//! timestamp falls in an already-forgotten window will lazily recreate that
//! partition. `forget_before` forgets what exists at call time; it is not a
//! permanently-closed window. A persisted watermark is future work.

use std::sync::Arc;

use crate::db::Handles;
use crate::db::collection::Shape;
use crate::db::record::Record;
use crate::db::schema::Schema;
use crate::db::space::SchemaRegistry;
use crate::error::{Error, Result};
use crate::format::{CollectionId, FrameId, VectorId};

/// How records are routed to physical partitions.
#[derive(Clone)]
pub enum Partition {
    /// One partition per UTC calendar day (`base:YYYY-MM-DD`).
    ByDay,
    /// One partition per UTC calendar month (`base:YYYY-MM`).
    ByMonth,
    /// Caller-supplied router. The closure may capture (e.g. a tenant id) and
    /// read the record's `created_at`. Non-time partitions cannot be
    /// `forget_before`'d (no period to compare).
    Custom(Arc<dyn Fn(&Record) -> String + Send + Sync>),
}

impl Partition {
    fn is_time(&self) -> bool {
        matches!(self, Partition::ByDay | Partition::ByMonth)
    }
}

/// A window of partitions for a view.
#[derive(Clone, Debug)]
pub enum Window {
    /// Partitions intersecting `[now - days, now]`.
    LastDays(u32),
    /// Partitions intersecting `[from, to]` (inclusive unix seconds).
    Range { from: i64, to: i64 },
    /// Explicit period suffixes (e.g. `"2026-05-25"`); resolved to existing
    /// `"{base}:{suffix}"` collections.
    Partitions(Vec<String>),
}

/// A resolved set of physical partition collections plus their shared shape.
#[derive(Clone, Debug)]
pub struct View {
    pub(crate) collections: Vec<String>,
    pub(crate) shape: Shape,
}

impl View {
    /// The physical collection names in this view.
    #[must_use]
    pub fn collections(&self) -> &[String] {
        &self.collections
    }
}

/// A logical partitioned collection. Cheap to clone; self-sufficient (holds the
/// store's shared handles).
#[derive(Clone)]
pub struct Partitioned {
    handles: Handles,
    base: String,
    /// The declared schema, kept so lazily created partition collections can
    /// persist their own schema docs (see `schema_doc.rs`). Their docs bind
    /// the base's `~auto/{base}/{field}` spaces, so reopen restores every
    /// partition's shape (and any still-deferred codec choice) from file.
    schema: Schema,
    shape: Shape,
    by: Partition,
}

impl Partitioned {
    pub(crate) fn new(
        handles: Handles,
        base: &str,
        schema: Schema,
        shape: Shape,
        by: Partition,
    ) -> Self {
        Self {
            handles,
            base: base.to_owned(),
            schema,
            shape,
            by,
        }
    }

    pub(crate) fn base(&self) -> &str {
        &self.base
    }

    pub(crate) fn declared_schema(&self) -> &Schema {
        &self.schema
    }

    pub(crate) fn shape(&self) -> &Shape {
        &self.shape
    }

    /// The physical collection name a record routes to.
    pub(crate) fn route(&self, rec: &Record<'_>, now: i64) -> String {
        match &self.by {
            Partition::ByDay => {
                let (y, m, d) = civil_from_days(rec.created_at().unwrap_or(now).div_euclid(86_400));
                format!("{}:{y:04}-{m:02}-{d:02}", self.base)
            }
            Partition::ByMonth => {
                let (y, m, _) = civil_from_days(rec.created_at().unwrap_or(now).div_euclid(86_400));
                format!("{}:{y:04}-{m:02}", self.base)
            }
            Partition::Custom(f) => format!("{}:{}", self.base, f(rec)),
        }
    }

    /// Resolve the partitions intersecting `window` to a [`View`]. For
    /// `Window::LastDays`, "now" is the system clock — use [`Self::view_as_of`]
    /// for a deterministic / testable reference time.
    pub fn view(&self, window: Window) -> View {
        self.view_as_of(window, now_secs())
    }

    /// Like [`Self::view`] but with an explicit "now" (unix seconds) for the
    /// `LastDays` window — deterministic, mirroring `Search::now`.
    pub fn view_as_of(&self, window: Window, now: i64) -> View {
        let p = self.handles.current();
        let schema = p.schema.lock();
        let collections = match window {
            Window::Partitions(suffixes) => suffixes
                .into_iter()
                .map(|s| format!("{}:{s}", self.base))
                .filter(|n| schema.collections.contains_key(n))
                .collect(),
            Window::LastDays(days) => {
                let from = now - i64::from(days) * 86_400;
                self.in_window(&schema, from, now)
            }
            Window::Range { from, to } => self.in_window(&schema, from, to),
        };
        View {
            collections,
            shape: self.shape.clone(),
        }
    }

    /// A view over every existing partition.
    pub fn all(&self) -> View {
        let p = self.handles.current();
        let schema = p.schema.lock();
        View {
            collections: self.existing(&schema).into_iter().map(|(n, _)| n).collect(),
            shape: self.shape.clone(),
        }
    }

    /// SOFT forget: tombstone every record in partitions whose entire period
    /// precedes `cutoff` (period_end ≤ cutoff). Returns the number of frames
    /// tombstoned. Bytes are reclaimed only at compaction (P4).
    ///
    /// Acquires the single writer for the whole tombstone+commit, so it must
    /// **not** be called while a [`crate::db::Writer`] is held on the same
    /// thread (the writer lock is not reentrant → deadlock).
    pub fn forget_before(&self, cutoff: i64) -> Result<usize> {
        if !self.by.is_time() {
            return Err(Error::Unsupported(
                "forget_before requires a time partition (ByDay/ByMonth)".into(),
            ));
        }

        // Take the writer first so collection ids are stable (excludes a
        // concurrent compaction, which would renumber them).
        let mut writer = self.handles.writer();
        let p = self.handles.current();

        let expired: Vec<(String, CollectionId)> = {
            let schema = p.schema.lock();
            self.existing(&schema)
                .into_iter()
                .filter(|(name, _)| self.period_of(name).is_some_and(|(_, end)| end <= cutoff))
                .collect()
        };
        if expired.is_empty() {
            return Ok(0);
        }

        let mut count = 0usize;
        {
            let mut valise = p.db.inner.write();
            for (_, cid) in &expired {
                let vids: Vec<VectorId> = valise.collection_member_vectors(*cid).to_vec();
                let fids: Vec<FrameId> = valise.collection_member_frames(*cid).to_vec();
                for v in vids {
                    // Tolerate already-tombstoned / unknown members.
                    let _ = valise.delete_vector(v);
                }
                for f in fids {
                    if valise.delete_frame(f).is_ok() {
                        count += 1;
                    }
                }
            }
        }
        writer.commit()?;
        {
            let mut identity = p.identity.lock();
            for (_, cid) in &expired {
                identity.remove_collection(*cid);
            }
        }
        Ok(count)
    }

    /// Existing physical partitions of this base: `(full_name, collection_id)`.
    fn existing(&self, schema: &SchemaRegistry) -> Vec<(String, CollectionId)> {
        let prefix = format!("{}:", self.base);
        schema
            .collections
            .iter()
            .filter(|(name, _)| name.starts_with(&prefix))
            .map(|(name, rec)| (name.clone(), rec.collection_id))
            .collect()
    }

    fn in_window(&self, schema: &SchemaRegistry, from: i64, to: i64) -> Vec<String> {
        self.existing(schema)
            .into_iter()
            .filter(|(name, _)| {
                self.period_of(name)
                    // half-open period [start, end) intersects closed [from, to]
                    .is_some_and(|(start, end)| end > from && start <= to)
            })
            .map(|(name, _)| name)
            .collect()
    }

    /// Parse the `[start, end)` unix-second period from a partition name, or
    /// `None` for a non-time / unparseable suffix.
    fn period_of(&self, full_name: &str) -> Option<(i64, i64)> {
        let suffix = full_name.strip_prefix(&self.base)?.strip_prefix(':')?;
        match self.by {
            Partition::ByDay => {
                let (y, m, d) = parse_ymd(suffix)?;
                let start = days_from_civil(y, m, d) * 86_400;
                Some((start, start + 86_400))
            }
            Partition::ByMonth => {
                let (y, m) = parse_ym(suffix)?;
                let start = days_from_civil(y, m, 1) * 86_400;
                let (ny, nm) = if m == 12 { (y + 1, 1) } else { (y, m + 1) };
                let end = days_from_civil(ny, nm, 1) * 86_400;
                Some((start, end))
            }
            Partition::Custom(_) => None,
        }
    }
}

fn parse_ymd(s: &str) -> Option<(i64, u32, u32)> {
    let mut it = s.split('-');
    let y = it.next()?.parse().ok()?;
    let m = it.next()?.parse().ok()?;
    let d = it.next()?.parse().ok()?;
    if it.next().is_some() {
        return None;
    }
    Some((y, m, d))
}

fn parse_ym(s: &str) -> Option<(i64, u32)> {
    let mut it = s.split('-');
    let y = it.next()?.parse().ok()?;
    let m = it.next()?.parse().ok()?;
    if it.next().is_some() {
        return None;
    }
    Some((y, m))
}

// Howard Hinnant's civil-date algorithms (public domain). Dependency-free
// proleptic-Gregorian conversions between days-since-1970-01-01 and y/m/d.

fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let m = i64::from(m);
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + i64::from(d) - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

fn now_secs() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn civil_roundtrip() {
        for &(y, m, d) in &[
            (1970, 1, 1),
            (2000, 2, 29),
            (2026, 5, 25),
            (1969, 12, 31),
            (2024, 12, 31),
        ] {
            let days = days_from_civil(y, m, d);
            assert_eq!(civil_from_days(days), (y, m, d), "ymd {y}-{m}-{d}");
        }
    }

    #[test]
    fn epoch_is_day_zero() {
        assert_eq!(days_from_civil(1970, 1, 1), 0);
        assert_eq!(civil_from_days(0), (1970, 1, 1));
    }

    #[test]
    fn parse_helpers() {
        assert_eq!(parse_ymd("2026-05-25"), Some((2026, 5, 25)));
        assert_eq!(parse_ymd("2026-05"), None);
        assert_eq!(parse_ym("2026-05"), Some((2026, 5)));
        assert_eq!(parse_ym("2026-05-25"), None);
    }
}
