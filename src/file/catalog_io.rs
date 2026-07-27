// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use std::{
    collections::{HashMap, HashSet},
    fs::File,
};

use crate::{
    Result,
    error::Error,
    format::{
        FrameId, bincode_config,
        catalog::{
            AnalyzerDesc, CatalogSnapshot, CodecDesc, CollectionDesc, EmbeddingSpaceDesc,
            FieldSchemaDesc, FrameDesc, FrameStatus, FrameStub, RetrievalProfileDesc,
            TextSpaceDesc, VectorDesc,
        },
        catalog_codec::{
            CATALOG_HEADER_LEN, CatalogTableKind, decode_catalog_delta,
            decode_catalog_previous_root, decode_catalog_record_envelope,
        },
        registry::SegmentCatalogEntry,
        segment::{SegmentRef, SegmentType},
        toc::TocFooterBody,
    },
};

use super::segment_io::{read_segment_payload, read_segment_payload_typed};

/// Lazy locator for one frame's heavy `FrameDesc` payload bytes within a
/// committed catalog segment. Built at open by the slim decoder; consumed
/// by `ValiseFile::frame_full(id)` when callers need the full descriptor.
#[derive(Clone, Copy, Debug)]
pub(crate) struct FrameLocator {
    pub(crate) catalog_segment: SegmentRef,
    pub(crate) payload_offset: usize,
    pub(crate) payload_len: usize,
}

pub(super) fn read_catalog_snapshot(
    file: &mut File,
    body: &TocFooterBody,
) -> Result<(CatalogSnapshot, HashMap<FrameId, FrameLocator>)> {
    let prof = std::env::var_os("VALISE_OPEN_PROFILE").is_some();
    let m = std::time::Instant::now();
    let collections = read_collection_catalog(file, body)?;
    if prof {
        crate::prof_eprintln!(
            "    [open] catalog.collections:    {:.3}s",
            m.elapsed().as_secs_f64()
        );
    }
    // Ghost catalog: decode only the slim (frame_id, status) view at open.
    // Heavy `FrameDesc` fields stay on disk; load lazily if a public API
    // explicitly asks for them. The retrieval hot path consumes
    // `frame_stubs` via `build_tombstone_set`, never the full Vec.
    let m = std::time::Instant::now();
    let (frame_stubs, frame_locators, v2_frames) = read_frame_catalog_slim(file, body)?;
    if prof {
        crate::prof_eprintln!(
            "    [open] catalog.frame_stubs:    {:.3}s ({})",
            m.elapsed().as_secs_f64(),
            frame_stubs.len()
        );
    }
    let m = std::time::Instant::now();
    let snapshot = CatalogSnapshot {
        collections,
        // v2 columnar FrameCatalog decodes the full FrameDescs eagerly
        // at open (no slim/locator path — the columnar block can't be
        // randomly seeked into). v1 chains leave this empty and lazy-
        // load via `frame_full`.
        frames: v2_frames,
        frame_stubs,
        text_spaces: read_catalog_chain_dedup(
            file,
            body.text_space_catalog_root,
            CatalogTableKind::TextSpace,
            |t: &crate::format::catalog::TextSpaceDesc| t.text_space_id.0 as u64,
        )?,
        analyzers: read_catalog_chain_dedup(
            file,
            body.analyzer_catalog_root,
            CatalogTableKind::Analyzer,
            |a: &crate::format::catalog::AnalyzerDesc| a.analyzer_id.0 as u64,
        )?,
        field_schemas: read_catalog_chain_dedup(
            file,
            body.field_schema_catalog_root,
            CatalogTableKind::FieldSchema,
            |f: &crate::format::catalog::FieldSchemaDesc| f.field_schema_id.0 as u64,
        )?,
        retrieval_profiles: read_catalog_chain_dedup(
            file,
            body.retrieval_profile_catalog_root,
            CatalogTableKind::RetrievalProfile,
            |p: &crate::format::catalog::RetrievalProfileDesc| p.profile_id.0 as u64,
        )?,
        embedding_spaces: read_catalog_chain_dedup(
            file,
            body.embedding_space_catalog_root,
            CatalogTableKind::EmbeddingSpace,
            |s: &crate::format::catalog::EmbeddingSpaceDesc| s.embedding_space_id.0 as u64,
        )?,
        codecs: read_catalog_chain_dedup(
            file,
            body.codec_catalog_root,
            CatalogTableKind::Codec,
            |c: &crate::format::catalog::CodecDesc| c.codec_id.0 as u64,
        )?,
        vectors: read_catalog_chain_dedup(
            file,
            body.vector_catalog_root,
            CatalogTableKind::Vector,
            |v: &crate::format::catalog::VectorDesc| v.vector_id.0,
        )?,
        fusion_profiles: read_catalog_chain_dedup(
            file,
            body.fusion_profile_catalog_root,
            CatalogTableKind::FusionProfile,
            |p: &crate::format::catalog::FusionProfileDesc| p.fusion_profile_id.0 as u64,
        )?,
    };
    if prof {
        crate::prof_eprintln!(
            "    [open] catalog.misc_tables:    {:.3}s",
            m.elapsed().as_secs_f64()
        );
    }
    Ok((snapshot, frame_locators))
}

/// Walk the chain rooted at `latest_root` newest-first, decoding each
/// segment's records and applying newest-wins dedup keyed by `id_of`.
/// Returns records sorted ascending by id for deterministic output.
fn read_catalog_chain_dedup<T, F>(
    file: &mut File,
    latest_root: Option<SegmentRef>,
    table: CatalogTableKind,
    id_of: F,
) -> Result<Vec<T>>
where
    T: serde::de::DeserializeOwned,
    F: Fn(&T) -> u64,
{
    let prof = std::env::var_os("VALISE_OPEN_PROFILE").is_some();
    let t_total = std::time::Instant::now();
    let roots = catalog_chain_roots(file, latest_root, table)?;

    // Single-segment fast path: skip the HashMap dedup, just sort the
    // already-monotonic list. id_allocator emits ids in ascending order
    // and a single segment carries them in append order.
    if roots.len() == 1 {
        let mut out: Vec<T> = read_catalog_root(file, Some(roots[0]), table)?;
        out.sort_by_key(|r| id_of(r));
        if prof && t_total.elapsed().as_millis() >= 1 {
            crate::prof_eprintln!(
                "      [catalog/{:?}/single] records={} total={:.3}s",
                table,
                out.len(),
                t_total.elapsed().as_secs_f64(),
            );
        }
        return Ok(out);
    }

    // Multi-segment: HashMap dedup (newest wins).
    let mut by_id: HashMap<u64, T> = HashMap::new();
    let mut total_records = 0usize;
    for root in roots {
        let records: Vec<T> = read_catalog_root(file, Some(root), table)?;
        total_records += records.len();
        for record in records {
            by_id.insert(id_of(&record), record);
        }
    }
    let mut out: Vec<(u64, T)> = by_id.into_iter().collect();
    out.sort_by_key(|(id, _)| *id);
    if prof && t_total.elapsed().as_millis() >= 1 {
        crate::prof_eprintln!(
            "      [catalog/{:?}/multi] records={} total={:.3}s",
            table,
            total_records,
            t_total.elapsed().as_secs_f64(),
        );
    }
    Ok(out.into_iter().map(|(_, r)| r).collect())
}

pub(super) fn read_segment_catalog(
    file: &mut File,
    body: &TocFooterBody,
) -> Result<Vec<SegmentCatalogEntry>> {
    let roots = segment_registry_chain_roots(file, body.segment_catalog_root)?;
    let mut entries_by_id = HashMap::new();
    for root in roots {
        let payload = read_segment_payload_typed(file, root, SegmentType::SegmentRegistry)?;
        let records: Vec<SegmentCatalogEntry> =
            decode_catalog_delta(CatalogTableKind::Segment, &payload)?.records;
        for record in records {
            entries_by_id.insert(record.segment_id, record);
        }
    }
    let mut entries: Vec<_> = entries_by_id.into_values().collect();
    entries.sort_by_key(|entry| entry.segment_id.0);
    Ok(entries)
}

fn segment_registry_chain_roots(
    file: &mut File,
    latest_root: Option<SegmentRef>,
) -> Result<Vec<SegmentRef>> {
    let mut roots_newest_first = Vec::new();
    let mut next = latest_root;
    while let Some(root) = next {
        let payload = read_segment_payload_typed(file, root, SegmentType::SegmentRegistry)?;
        let previous_root = decode_catalog_previous_root(CatalogTableKind::Segment, &payload)?;
        roots_newest_first.push(root);
        next = previous_root;
    }
    roots_newest_first.reverse();
    Ok(roots_newest_first)
}

fn read_collection_catalog(file: &mut File, body: &TocFooterBody) -> Result<Vec<CollectionDesc>> {
    let roots = catalog_chain_roots(
        file,
        body.collection_catalog_root,
        CatalogTableKind::Collection,
    )?;
    let mut catalog = CatalogSnapshot::default();
    for root in roots {
        let records: Vec<CollectionDesc> =
            read_catalog_root(file, Some(root), CatalogTableKind::Collection)?;
        for record in records {
            upsert_collection(&mut catalog, record);
        }
    }
    Ok(catalog.collections)
}

/// Ghost-Catalog open path: walk the Frame catalog chain and build only
/// the slim `(frame_id, status)` index — no bincode decode of heavy
/// `FrameDesc` fields. ~16 µs for 200 k frames vs ~52 ms for the full
/// `read_frame_catalog`.
///
/// **How this works without parsing the full payload:**
/// - Each catalog record has a length-prefixed envelope (8 B header +
///   `payload_len` bytes), so `decode_catalog_record_envelope` walks
///   record-by-record by reading only the header.
/// - `frame_id` is the **first** field of `FrameDesc`. Bincode's varint
///   config encodes a `u64` as 1-9 bytes at the start of the payload —
///   we read it via `read_varint_u64`.
/// - `status` is the **last** field of `FrameDesc` (a 2-variant enum).
///   Bincode encodes the variant discriminant as a 1-byte varint, so it
///   sits at `payload[payload_len - 1]`. The middle ~14 fields stay
///   on disk untouched.
///
/// This decode is **field-order-sensitive**: if `FrameDesc`'s field
/// order changes, the slim decoder must move in lockstep. There is a
/// roundtrip test in this module pinning the contract.
pub(super) fn read_frame_catalog_slim(
    file: &mut File,
    body: &TocFooterBody,
) -> Result<(
    Vec<FrameStub>,
    HashMap<FrameId, FrameLocator>,
    Vec<FrameDesc>,
)> {
    let prof = std::env::var_os("VALISE_OPEN_PROFILE").is_some();
    let t_total = std::time::Instant::now();
    // Detect the FrameCatalog wire format on the most recent root —
    // v2 (VLF2 magic, columnar) routes to a separate decoder that
    // populates `Vec<FrameDesc>` eagerly. v1 (VALISEC magic, bincode rows)
    // sticks with the slim+locator ghost-catalog path that's been the
    // open hot path since spec §10.
    if let Some(root) = body.frame_catalog_root {
        let head_payload = read_segment_payload_typed(file, root, SegmentType::FrameCatalog)?;
        if head_payload.starts_with(b"VLF2") {
            let (stubs, frames) = read_frame_catalog_v2_chain(file, root, &head_payload)?;
            if prof {
                crate::prof_eprintln!(
                    "      [slim_frame/v2] records={} total={:.3}s",
                    stubs.len(),
                    t_total.elapsed().as_secs_f64(),
                );
            }
            return Ok((stubs, HashMap::new(), frames));
        }
    }
    let roots = catalog_chain_roots(file, body.frame_catalog_root, CatalogTableKind::Frame)?;

    // Single-segment fast path: when the catalog chain has exactly one
    // root, the on-disk record stream is already monotonic by frame_id
    // (id_allocator hands out IDs in order, append-only). Skip the
    // HashMap dedup and push directly to a presized Vec — saves ~25 ms
    // per 200 k records. Multi-segment falls back to HashMap for
    // newest-wins resolution.
    if roots.len() == 1 {
        let root = roots[0];
        let payload = read_segment_payload_typed(file, root, SegmentType::FrameCatalog)?;
        if payload.len() < CATALOG_HEADER_LEN {
            return Err(Error::Format("frame catalog truncated".into()));
        }
        let count = u32::from_le_bytes(payload[8..12].try_into().unwrap()) as usize;
        let mut stubs: Vec<FrameStub> = Vec::with_capacity(count);
        let mut locators: HashMap<FrameId, FrameLocator> = HashMap::with_capacity(count);
        let mut offset = CATALOG_HEADER_LEN;
        for _ in 0..count {
            let (envelope, consumed) = decode_catalog_record_envelope(&payload[offset..])?;
            if envelope.payload.is_empty() {
                return Err(Error::Format("frame catalog record empty".into()));
            }
            let payload_offset_in_segment = offset + (consumed - envelope.payload.len());
            let payload_len = envelope.payload.len();
            let (fid, _): (FrameId, _) =
                bincode::serde::decode_from_slice(envelope.payload, bincode_config())
                    .map_err(|e| Error::Serde(e.to_string()))?;
            let status_byte = *envelope.payload.last().unwrap();
            let status = match status_byte {
                0 => FrameStatus::Active,
                1 => FrameStatus::Tombstoned,
                other => {
                    return Err(Error::Format(format!(
                        "frame catalog: unknown FrameStatus discriminant {other}"
                    )));
                }
            };
            stubs.push(FrameStub {
                frame_id: fid,
                status,
            });
            locators.insert(
                fid,
                FrameLocator {
                    catalog_segment: root,
                    payload_offset: payload_offset_in_segment,
                    payload_len,
                },
            );
            offset += consumed;
        }
        if prof {
            crate::prof_eprintln!(
                "      [slim_frame/single] records={} total={:.3}s",
                stubs.len(),
                t_total.elapsed().as_secs_f64(),
            );
        }
        return Ok((stubs, locators, Vec::new()));
    }

    // Multi-segment slow path: dedup via HashMap (newest-wins).
    let mut by_id: HashMap<u64, (FrameStatus, FrameLocator)> = HashMap::new();
    for root in roots {
        let payload = read_segment_payload_typed(file, root, SegmentType::FrameCatalog)?;
        if payload.len() < CATALOG_HEADER_LEN {
            return Err(Error::Format("frame catalog truncated".into()));
        }
        let count = u32::from_le_bytes(payload[8..12].try_into().unwrap()) as usize;
        let mut offset = CATALOG_HEADER_LEN;
        for _ in 0..count {
            let (envelope, consumed) = decode_catalog_record_envelope(&payload[offset..])?;
            if envelope.payload.is_empty() {
                return Err(Error::Format("frame catalog record empty".into()));
            }
            let payload_offset_in_segment = offset + (consumed - envelope.payload.len());
            let payload_len = envelope.payload.len();
            let (frame_id, _): (FrameId, _) =
                bincode::serde::decode_from_slice(envelope.payload, bincode_config())
                    .map_err(|e| Error::Serde(e.to_string()))?;
            let frame_id = frame_id.0;
            let status_byte = *envelope.payload.last().unwrap();
            let status = match status_byte {
                0 => FrameStatus::Active,
                1 => FrameStatus::Tombstoned,
                other => {
                    return Err(Error::Format(format!(
                        "frame catalog: unknown FrameStatus discriminant {other}"
                    )));
                }
            };
            by_id.insert(
                frame_id,
                (
                    status,
                    FrameLocator {
                        catalog_segment: root,
                        payload_offset: payload_offset_in_segment,
                        payload_len,
                    },
                ),
            );
            offset += consumed;
        }
    }
    let mut sorted: Vec<(u64, FrameStatus, FrameLocator)> = by_id
        .into_iter()
        .map(|(id, (status, locator))| (id, status, locator))
        .collect();
    sorted.sort_by_key(|&(id, _, _)| id);
    if prof {
        crate::prof_eprintln!(
            "      [slim_frame/multi] records={} total={:.3}s",
            sorted.len(),
            t_total.elapsed().as_secs_f64(),
        );
    }
    let stubs: Vec<FrameStub> = sorted
        .iter()
        .map(|&(id, s, _)| FrameStub {
            frame_id: FrameId(id),
            status: s,
        })
        .collect();
    let locators: HashMap<FrameId, FrameLocator> = sorted
        .into_iter()
        .map(|(id, _, locator)| (FrameId(id), locator))
        .collect();
    Ok((stubs, locators, Vec::new()))
}

/// Walk a v2 (`VLF2`) FrameCatalog chain newest-first, union all
/// FrameDescs with newest-wins on `frame_id`, and return both the
/// dedupe'd full Vec and the slim `(frame_id, status)` stubs derived
/// from it. The columnar wire form decodes every field at once, so
/// there is no separate ghost-catalog path; callers receive both
/// views from one walk.
fn read_frame_catalog_v2_chain(
    file: &mut File,
    head_root: SegmentRef,
    head_payload: &[u8],
) -> Result<(Vec<FrameStub>, Vec<FrameDesc>)> {
    use crate::format::frame_catalog_codec::{
        decode_frame_catalog, decode_frame_catalog_previous_root,
    };
    let mut by_id: HashMap<u64, FrameDesc> = HashMap::new();

    let head_delta = decode_frame_catalog(head_payload)?;
    let mut next = head_delta.previous_root;
    for frame in head_delta.frames {
        by_id.insert(frame.frame_id.0, frame);
    }
    while let Some(root) = next {
        let payload = read_segment_payload_typed(file, root, SegmentType::FrameCatalog)?;
        if !payload.starts_with(b"VLF2") {
            return Err(Error::Format(
                "FrameCatalog chain mixes v2 and v1 wire formats; rebuild required".into(),
            ));
        }
        let prev_for_this = decode_frame_catalog_previous_root(&payload)?;
        let delta = decode_frame_catalog(&payload)?;
        for frame in delta.frames {
            by_id.entry(frame.frame_id.0).or_insert(frame);
        }
        next = prev_for_this;
    }
    let _ = head_root;
    let mut frames: Vec<FrameDesc> = by_id.into_values().collect();
    frames.sort_by_key(|f| f.frame_id.0);
    let stubs: Vec<FrameStub> = frames
        .iter()
        .map(|f| FrameStub {
            frame_id: f.frame_id,
            status: f.status,
        })
        .collect();
    Ok((stubs, frames))
}

fn catalog_chain_roots(
    file: &mut File,
    latest_root: Option<SegmentRef>,
    table: CatalogTableKind,
) -> Result<Vec<SegmentRef>> {
    let mut roots_newest_first = Vec::new();
    let mut next = latest_root;
    while let Some(root) = next {
        let payload = read_segment_payload(file, root)?;
        let previous_root = decode_catalog_previous_root(table, &payload)?;
        roots_newest_first.push(root);
        next = previous_root;
    }
    roots_newest_first.reverse();
    Ok(roots_newest_first)
}

fn read_catalog_root<T: serde::de::DeserializeOwned>(
    file: &mut File,
    root: Option<SegmentRef>,
    table: CatalogTableKind,
) -> Result<Vec<T>> {
    let Some(root) = root else {
        return Ok(Vec::new());
    };
    let payload = read_segment_payload(file, root)?;
    Ok(decode_catalog_delta(table, &payload)?.records)
}

/// Pluck the dirty subset out of a sorted catalog table.
///
/// `records` is kept sorted by `id_of` (every `upsert_*` uses
/// `binary_search_by_key`, so the slice is monotonic by ID). Old shape
/// did `records.iter().filter(...)` which is O(N) per call — *every*
/// commit performed 11+ such full-table scans, dominating phase A
/// (3.5 ms at 20 k vectors). The new shape iterates the (small)
/// `dirty_ids`, sorts them to preserve the deterministic ascending
/// output ordering callers downstream rely on
/// (`read_catalog_chain_dedup`), and binary-searches the sorted
/// `records` for each. O(K log N) per call where K = `dirty_ids.len()`
/// — for a one-frame commit on 20 k vectors that's 13 lookups vs
/// 20 000 iterations.
pub(super) fn dirty_records<'a, T, Id>(
    records: &'a [T],
    dirty_ids: &HashSet<Id>,
    id_of: impl Fn(&T) -> Id,
) -> Vec<T>
where
    T: Clone + 'a,
    Id: Eq + Ord + Copy + std::hash::Hash,
{
    if dirty_ids.is_empty() {
        return Vec::new();
    }
    let mut keys: Vec<Id> = dirty_ids.iter().copied().collect();
    keys.sort_unstable();
    let mut out = Vec::with_capacity(keys.len());
    for k in keys {
        if let Ok(idx) = records.binary_search_by_key(&k, &id_of) {
            out.push(records[idx].clone());
        }
    }
    out
}

// All `upsert_*` functions below use the same shape: catalogs are kept
// sorted by id, so a binary_search_by_key resolves to either Ok(idx)
// (replace in place) or Err(idx) (insert at the sorted position). For
// monotonic id allocation — the common ingest pattern — the Err arm is
// always Err(len), making `Vec::insert(len, …)` an amortized O(1)
// append. No sort, no linear scan, no side-table.

pub(super) fn upsert_collection(catalog: &mut CatalogSnapshot, item: CollectionDesc) {
    match catalog
        .collections
        .binary_search_by_key(&item.collection_id.0, |e| e.collection_id.0)
    {
        Ok(idx) => catalog.collections[idx] = item,
        Err(idx) => catalog.collections.insert(idx, item),
    }
}

pub(super) fn upsert_frame(catalog: &mut CatalogSnapshot, item: FrameDesc) {
    let stub = FrameStub {
        frame_id: item.frame_id,
        status: item.status,
    };
    match catalog
        .frames
        .binary_search_by_key(&item.frame_id.0, |e| e.frame_id.0)
    {
        Ok(idx) => catalog.frames[idx] = item,
        Err(idx) => catalog.frames.insert(idx, item),
    }
    // Mirror into the slim `frame_stubs` index so the retrieval hot path
    // (BM25 / jaccard / tfidf tombstone filter) sees post-write deltas.
    match catalog
        .frame_stubs
        .binary_search_by_key(&stub.frame_id.0, |s| s.frame_id.0)
    {
        Ok(idx) => catalog.frame_stubs[idx] = stub,
        Err(idx) => catalog.frame_stubs.insert(idx, stub),
    }
}

pub(super) fn upsert_embedding_space(catalog: &mut CatalogSnapshot, item: EmbeddingSpaceDesc) {
    match catalog
        .embedding_spaces
        .binary_search_by_key(&item.embedding_space_id.0, |e| e.embedding_space_id.0)
    {
        Ok(idx) => catalog.embedding_spaces[idx] = item,
        Err(idx) => catalog.embedding_spaces.insert(idx, item),
    }
}

pub(super) fn upsert_codec(catalog: &mut CatalogSnapshot, item: CodecDesc) {
    match catalog
        .codecs
        .binary_search_by_key(&item.codec_id.0, |e| e.codec_id.0)
    {
        Ok(idx) => catalog.codecs[idx] = item,
        Err(idx) => catalog.codecs.insert(idx, item),
    }
}

pub(super) fn upsert_vector(catalog: &mut CatalogSnapshot, item: VectorDesc) {
    match catalog
        .vectors
        .binary_search_by_key(&item.vector_id.0, |e| e.vector_id.0)
    {
        Ok(idx) => catalog.vectors[idx] = item,
        Err(idx) => catalog.vectors.insert(idx, item),
    }
}

pub(super) fn upsert_text_space(catalog: &mut CatalogSnapshot, item: TextSpaceDesc) {
    match catalog
        .text_spaces
        .binary_search_by_key(&item.text_space_id.0, |e| e.text_space_id.0)
    {
        Ok(idx) => catalog.text_spaces[idx] = item,
        Err(idx) => catalog.text_spaces.insert(idx, item),
    }
}

pub(super) fn upsert_analyzer(catalog: &mut CatalogSnapshot, item: AnalyzerDesc) {
    match catalog
        .analyzers
        .binary_search_by_key(&item.analyzer_id.0, |e| e.analyzer_id.0)
    {
        Ok(idx) => catalog.analyzers[idx] = item,
        Err(idx) => catalog.analyzers.insert(idx, item),
    }
}

pub(super) fn upsert_field_schema(catalog: &mut CatalogSnapshot, item: FieldSchemaDesc) {
    match catalog
        .field_schemas
        .binary_search_by_key(&item.field_schema_id.0, |e| e.field_schema_id.0)
    {
        Ok(idx) => catalog.field_schemas[idx] = item,
        Err(idx) => catalog.field_schemas.insert(idx, item),
    }
}

pub(super) fn upsert_retrieval_profile(catalog: &mut CatalogSnapshot, item: RetrievalProfileDesc) {
    match catalog
        .retrieval_profiles
        .binary_search_by_key(&item.profile_id.0, |e| e.profile_id.0)
    {
        Ok(idx) => catalog.retrieval_profiles[idx] = item,
        Err(idx) => catalog.retrieval_profiles.insert(idx, item),
    }
}

pub(super) fn upsert_fusion_profile(
    catalog: &mut CatalogSnapshot,
    item: crate::format::catalog::FusionProfileDesc,
) {
    match catalog
        .fusion_profiles
        .binary_search_by_key(&item.fusion_profile_id.0, |e| e.fusion_profile_id.0)
    {
        Ok(idx) => catalog.fusion_profiles[idx] = item,
        Err(idx) => catalog.fusion_profiles.insert(idx, item),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::{
        CollectionId,
        catalog::{FrameDesc, FrameRole, FrameStatus},
        catalog_codec::encode_catalog_delta,
        payload::PayloadRef,
    };

    fn dummy_frame(frame_id: u64, status: FrameStatus) -> FrameDesc {
        FrameDesc {
            frame_id: FrameId(frame_id),
            collection_id: CollectionId(1),
            role: FrameRole::Document,
            created_at: 1_700_000_000,
            updated_at: 1_700_000_000,
            uri_ref: None,
            title_ref: None,
            payload_ref: Some(PayloadRef {
                segment_id: crate::format::SegmentId(7),
                bytes_offset: 0,
                bytes_length: 32,
            }),
            payload_encoding: crate::format::catalog::PayloadEncoding::Raw,
            search_text_ref: None,
            parent_frame_id: None,
            chunk_index: None,
            chunk_count: None,
            metadata_ref: None,
            status,
        }
    }

    #[test]
    fn frame_stub_field_order_contract() {
        // Pin the on-disk Frame catalog field order: encode 100 FrameDescs
        // with varied frame_id and status, then re-extract (frame_id,
        // status) via the **slim** decoder's record-walk logic. This test
        // breaks loudly if FrameDesc's leading-frame_id / trailing-status
        // contract ever drifts (the slim decoder relies on the leading
        // varint frame_id and the trailing 1-byte status discriminant).
        use crate::format::catalog_codec::{CatalogTableKind, decode_catalog_record_envelope};

        let mut frames = Vec::with_capacity(100);
        for i in 0..100u64 {
            let status = if i.is_multiple_of(7) {
                FrameStatus::Tombstoned
            } else {
                FrameStatus::Active
            };
            // Mix in some big frame_ids to force varying varint widths.
            let frame_id = i.wrapping_mul(0x0001_0001_0001_0001);
            frames.push(dummy_frame(frame_id, status));
        }
        let payload = encode_catalog_delta(CatalogTableKind::Frame, None, &frames)
            .expect("encode Frame catalog delta");

        // Walk the records via the same envelope decoder the slim path
        // uses, decode `frame_id` (leading varint via bincode) + `status`
        // (trailing byte), and compare to the source descriptors.
        let count = u32::from_le_bytes(payload[8..12].try_into().unwrap()) as usize;
        assert_eq!(count, frames.len());
        let mut offset = CATALOG_HEADER_LEN;
        for expected in &frames {
            let (env, consumed) =
                decode_catalog_record_envelope(&payload[offset..]).expect("decode envelope");
            // Leading frame_id via bincode (varint).
            let (frame_id, _): (FrameId, _) =
                bincode::serde::decode_from_slice(env.payload, bincode_config())
                    .expect("decode FrameId");
            assert_eq!(frame_id, expected.frame_id, "frame_id mismatch");
            // Trailing status via 1-byte discriminant.
            let status_byte = *env.payload.last().unwrap();
            let status = match status_byte {
                0 => FrameStatus::Active,
                1 => FrameStatus::Tombstoned,
                other => panic!("unknown status discriminant {other}"),
            };
            assert_eq!(status, expected.status, "status mismatch");
            offset += consumed;
        }
    }
}
