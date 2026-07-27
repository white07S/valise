//! Collection-filter build and reload (Phase 3 Group E).
//!
//! v1 strategy: per-collection full snapshot at commit. Each segment carries
//! the collection's complete sorted frame_id and vector_id sets. Replay
//! walks the chain newest-first; the latest segment per collection wins.

use std::collections::{HashMap, HashSet};
use std::fs::File;

use crate::{
    error::Result,
    format::{
        CollectionId, FrameId, VectorId,
        catalog::{CatalogSnapshot, FrameStatus, VectorStatus},
        collection_filter::{
            CollectionFilterSegment, decode_collection_filter_segment,
            encode_collection_filter_segment,
        },
        registry::IdAllocator,
        segment::SegmentType,
        toc::{CollectionFilterRoot, TocFooterBody},
    },
};

use super::segment_io::{
    SegmentRegistryMut, append_registered_segment, read_segment_payload_typed,
};

/// Build the membership snapshot for `collection_id` from the current
/// catalog. Excludes tombstoned frames and vectors.
fn build_snapshot(
    catalog: &CatalogSnapshot,
    collection_id: CollectionId,
) -> CollectionFilterSegment {
    let mut frame_ids: Vec<FrameId> = catalog
        .frames
        .iter()
        .filter(|f| f.collection_id == collection_id && f.status == FrameStatus::Active)
        .map(|f| f.frame_id)
        .collect();
    frame_ids.sort_by_key(|f| f.0);

    let mut vector_ids: Vec<VectorId> = catalog
        .vectors
        .iter()
        .filter(|v| v.collection_id == collection_id && v.status == VectorStatus::Active)
        .map(|v| v.vector_id)
        .collect();
    vector_ids.sort_by_key(|v| v.0);

    CollectionFilterSegment {
        collection_id,
        frame_ids,
        vector_ids,
        previous_root: None, // filled in by caller
    }
}

/// Identify which collections need to be re-snapshotted because at least
/// one of their frames or vectors changed since the last commit.
pub(crate) fn dirty_filter_collections(
    catalog: &CatalogSnapshot,
    dirty_frame_ids: &HashSet<FrameId>,
    dirty_vector_ids: &HashSet<VectorId>,
) -> Vec<CollectionId> {
    // Inverted loops: iterate the small dirty sets + binary-search the
    // sorted catalog slices. O(K log N) per side instead of O(N + V)
    // per side. With 20 k frames + 20 k vectors and one dirty frame,
    // 14 lookups vs 40 000 iterations.
    let mut affected: HashSet<CollectionId> = HashSet::new();
    for fid in dirty_frame_ids {
        if let Ok(idx) = catalog
            .frames
            .binary_search_by_key(&fid.0, |f| f.frame_id.0)
        {
            affected.insert(catalog.frames[idx].collection_id);
        }
    }
    for vid in dirty_vector_ids {
        if let Ok(idx) = catalog
            .vectors
            .binary_search_by_key(&vid.0, |v| v.vector_id.0)
        {
            affected.insert(catalog.vectors[idx].collection_id);
        }
    }
    let mut out: Vec<CollectionId> = affected.into_iter().collect();
    out.sort_by_key(|c| c.0);
    out
}

/// Build, write, and register one collection's filter segment. Updates
/// `body.collection_filter_roots` (replaces the prior entry for this
/// collection or appends a new one).
pub(crate) fn write_collection_filter_segment(
    file: &mut File,
    id_allocator: &mut IdAllocator,
    segments: SegmentRegistryMut<'_>,
    body: &mut TocFooterBody,
    catalog: &CatalogSnapshot,
    collection_id: CollectionId,
) -> Result<()> {
    let mut snapshot = build_snapshot(catalog, collection_id);
    let prior = body
        .collection_filter_roots
        .iter()
        .find(|r| r.collection_id == collection_id)
        .map(|r| r.frame_filter_root)
        .unwrap_or(None);
    snapshot.previous_root = prior;
    let payload = encode_collection_filter_segment(&snapshot)?;
    let item_count = (snapshot.frame_ids.len() + snapshot.vector_ids.len()) as u32;
    let root = append_registered_segment(
        file,
        id_allocator,
        segments,
        SegmentType::CollectionFilter,
        item_count,
        &payload,
    )?;
    if let Some(existing) = body
        .collection_filter_roots
        .iter_mut()
        .find(|r| r.collection_id == collection_id)
    {
        existing.frame_filter_root = Some(root);
        existing.vector_filter_root = Some(root);
    } else {
        body.collection_filter_roots.push(CollectionFilterRoot {
            collection_id,
            frame_filter_root: Some(root),
            vector_filter_root: Some(root),
        });
    }
    Ok(())
}

/// Cache of per-collection membership, materialized on demand from
/// `body.collection_filter_roots`. Each entry is the LATEST segment for
/// that collection; older segments in the chain are not visited at v1
/// because the latest carries a full snapshot.
pub(crate) type CollectionFilterCache = HashMap<CollectionId, (Vec<FrameId>, Vec<VectorId>)>;

pub(crate) fn load_collection_filter_cache(
    file: &mut File,
    body: &TocFooterBody,
) -> Result<CollectionFilterCache> {
    let mut out = CollectionFilterCache::new();
    for entry in &body.collection_filter_roots {
        let Some(root) = entry.frame_filter_root else {
            continue;
        };
        let payload = read_segment_payload_typed(file, root, SegmentType::CollectionFilter)?;
        let seg = decode_collection_filter_segment(&payload)?;
        out.insert(seg.collection_id, (seg.frame_ids, seg.vector_ids));
    }
    Ok(out)
}
