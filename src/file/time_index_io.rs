// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Time-index build and reload (Phase 3 Group F).
//!
//! v1: one global segment per commit, full snapshot of all active frames.
//! `TimeQuery` filters in-memory by `[from, to]` and optional collection.

use std::fs::File;

use crate::{
    error::Result,
    format::{
        FrameId,
        catalog::{CatalogSnapshot, FrameStatus},
        registry::IdAllocator,
        segment::SegmentType,
        time_index_segment::{
            TimeIndexEntry, TimeIndexSegment, decode_time_index_segment, encode_time_index_segment,
        },
        toc::{TimeIndexRoot, TocFooterBody},
    },
};

use super::segment_io::{
    SegmentRegistryMut, append_registered_segment, read_segment_payload_typed,
};

/// Snapshot of the global time index, sorted ascending by `created_at`.
pub(crate) type TimeIndexCache = Vec<TimeIndexEntry>;

fn build_snapshot(catalog: &CatalogSnapshot) -> TimeIndexSegment {
    let mut entries: Vec<TimeIndexEntry> = catalog
        .frames
        .iter()
        .filter(|f| f.status == FrameStatus::Active)
        .map(|f| TimeIndexEntry {
            frame_id: f.frame_id,
            collection_id: f.collection_id,
            created_at: f.created_at,
            updated_at: f.updated_at,
        })
        .collect();
    // Stable sort: ties on created_at break by frame_id ascending so the
    // on-disk layout is deterministic.
    entries.sort_by(|a, b| {
        a.created_at
            .cmp(&b.created_at)
            .then(a.frame_id.0.cmp(&b.frame_id.0))
    });
    TimeIndexSegment {
        collection_id: None,
        entries,
        previous_root: None,
    }
}

/// Rebuild the global time index from the current catalog and append it to
/// the segment chain. Updates `body.time_index_roots` to a single global
/// entry pointing at the new root (replacing any previous global root).
pub(crate) fn write_time_index_segment(
    file: &mut File,
    id_allocator: &mut IdAllocator,
    segments: SegmentRegistryMut<'_>,
    body: &mut TocFooterBody,
    catalog: &CatalogSnapshot,
) -> Result<()> {
    let mut snapshot = build_snapshot(catalog);
    snapshot.previous_root = body
        .time_index_roots
        .iter()
        .find(|r| r.collection_id.is_none())
        .map(|r| r.root);
    let payload = encode_time_index_segment(&snapshot)?;
    let item_count = u32::try_from(snapshot.entries.len()).unwrap_or(u32::MAX);
    let root = append_registered_segment(
        file,
        id_allocator,
        segments,
        SegmentType::TimeIndex,
        item_count,
        &payload,
    )?;
    if let Some(existing) = body
        .time_index_roots
        .iter_mut()
        .find(|r| r.collection_id.is_none())
    {
        existing.root = root;
    } else {
        body.time_index_roots.push(TimeIndexRoot {
            collection_id: None,
            root,
        });
    }
    Ok(())
}

pub(crate) fn load_time_index_cache(
    file: &mut File,
    body: &TocFooterBody,
) -> Result<TimeIndexCache> {
    let Some(entry) = body
        .time_index_roots
        .iter()
        .find(|r| r.collection_id.is_none())
    else {
        return Ok(TimeIndexCache::new());
    };
    let payload = read_segment_payload_typed(file, entry.root, SegmentType::TimeIndex)?;
    let seg = decode_time_index_segment(&payload)?;
    Ok(seg.entries)
}

/// In-memory time-range query — `from` and `to` are inclusive Unix epoch
/// seconds. The cache is sorted by `created_at`, so binary search bounds
/// the scan.
pub(crate) fn time_range(
    cache: &[TimeIndexEntry],
    from: i64,
    to: i64,
    collection: Option<crate::format::CollectionId>,
) -> Vec<FrameId> {
    if cache.is_empty() || from > to {
        return Vec::new();
    }
    // partition_point returns first index whose entry.created_at >= from.
    let lo = cache.partition_point(|e| e.created_at < from);
    let mut out = Vec::new();
    for entry in &cache[lo..] {
        if entry.created_at > to {
            break;
        }
        if let Some(cid) = collection
            && entry.collection_id != cid
        {
            continue;
        }
        out.push(entry.frame_id);
    }
    out
}
