// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Columnar `FrameCatalog` segment codec, spec §10 v2 (magic `VLF2`).
//!
//! Naming: this is a `_codec` file (pure encode/decode) because the row
//! type it serializes — `FrameDesc` — lives in `format/catalog.rs`
//! alongside the other catalog descriptors, not in a sibling
//! `frame_catalog.rs`. See the `format/` file vocabulary rule in `SKILL.md`.
//!
//! Replaces the bincode-per-row catalog envelope for `SegmentType::FrameCatalog`.
//! The catalog envelope used by every OTHER catalog table (Collection,
//! TextSpace, Codec, …) stays at v1 — only the Frame table is large
//! enough (one row per put_frame, 522 k rows for BEIR-quora) to be
//! worth a hand-coded columnar layout.
//!
//! Wire format:
//!
//!   MAGIC "VLF2"                        (4)
//!   VERSION u16 = 2                     (2)
//!   FRAME_COUNT u32                      (4)
//!   PREVIOUS_ROOT_PRESENCE u8            (1)
//!   PADDING u8 = 0                       (1)
//!   PREVIOUS_ROOT_BODY                   (0 or 56 bytes: SegmentId u64 |
//!                                          offset u64 | length u64 |
//!                                          checksum 32 B)
//!   -- columns (only when frame_count > 0) --
//!   FRAME_ID_BLOB        (u32 len + bytes)
//!     - first u64 frame_id raw, then uvarint deltas vs prev
//!   COLLECTION_RLE_BLOB  (u32 len + bytes)
//!     - u32 run_count, then (u32 cid, uvarint run_len) × runs
//!   ROLE_RLE_BLOB        (u32 len + bytes)
//!   STATUS_RLE_BLOB      (u32 len + bytes)
//!   PAYLOAD_ENCODING_RLE (u32 len + bytes)
//!   CREATED_BLOB         (u32 len + bytes)
//!     - i64 first raw, then zigzag delta-of-delta uvarints
//!   UPDATED_OFFSET_BLOB  (u32 len + bytes)
//!     - per-row zigzag uvarint of (updated_at − created_at)
//!   PAYLOAD_REF_PRESENCE (u32 len + bytes): packed presence list
//!     - u32 present_count, then `present_count` u32 row indices
//!   PAYLOAD_REF_BLOB     (u32 len + bytes): per-present-row payload refs
//!     - first: u64 segment_id, uvarint bytes_offset, uvarint bytes_length
//!     - subsequent: zigzag varint segment_id delta, zigzag varint
//!         bytes_offset delta, uvarint bytes_length
//!   SPARSE_OPTIONS:                      (7 columns: uri_ref, title_ref,
//!                                          search_text_ref, parent_frame_id,
//!                                          chunk_index, chunk_count,
//!                                          metadata_ref)
//!     for each column:
//!       PRESENCE_LIST (u32 count + count × u32 row indices)
//!       VALUES        (variable, packed per the column's type)
//!
//! Quora-shape ingest (522 k frames, all `payload_ref = Some`, all
//! other Optional fields `None`, single collection, single role, single
//! status, single encoding, monotonic timestamps within a second):
//! frame_id ~1 B/row, RLE 0 B/row × 4, created delta-of-delta ~1 B/row,
//! updated_offset 1 B/row, payload_ref presence (full bitmap is "every
//! row present" so it packs as a single run record below), payload_ref
//! delta 3–4 B/row, 7 × (4-byte "no rows present") ≈ 7–8 B/row total.
//! v1 row-shaped bincode was ~33.7 B/row.

use crate::{
    error::{Error, Result},
    format::{
        CollectionId, FrameId, SegmentId,
        catalog::{FrameDesc, FrameRole, FrameStatus, MetadataRef, PayloadEncoding},
        payload::PayloadRef,
        segment::SegmentRef,
    },
};

const MAGIC: [u8; 4] = *b"VLF2";
const VERSION: u16 = 2;

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct FrameCatalogDelta {
    pub(crate) previous_root: Option<SegmentRef>,
    pub(crate) frames: Vec<FrameDesc>,
}

pub(crate) fn encode_frame_catalog(delta: &FrameCatalogDelta) -> Result<Vec<u8>> {
    let mut buf = Vec::new();
    buf.extend_from_slice(&MAGIC);
    buf.extend_from_slice(&VERSION.to_le_bytes());
    let frame_count = u32::try_from(delta.frames.len())
        .map_err(|_| Error::Format("FrameCatalog frame count overflow".into()))?;
    buf.extend_from_slice(&frame_count.to_le_bytes());
    write_previous_root(&mut buf, delta.previous_root);

    if delta.frames.is_empty() {
        return Ok(buf);
    }

    // --- frame_id column ---
    let mut col = Vec::with_capacity(delta.frames.len() + 8);
    col.extend_from_slice(&delta.frames[0].frame_id.0.to_le_bytes());
    let mut prev = delta.frames[0].frame_id.0;
    for f in &delta.frames[1..] {
        let cur = f.frame_id.0;
        if cur < prev {
            return Err(Error::Format(
                "FrameCatalog frame_ids must be monotone ascending in v2 columnar layout".into(),
            ));
        }
        write_uvarint(&mut col, cur - prev);
        prev = cur;
    }
    write_blob(&mut buf, &col)?;

    // --- collection_id (RLE) ---
    write_blob(
        &mut buf,
        &encode_rle_u32(delta.frames.iter().map(|f| f.collection_id.0))?,
    )?;
    // --- role (RLE, 1-byte discriminant) ---
    write_blob(
        &mut buf,
        &encode_rle_u8(delta.frames.iter().map(|f| role_disc(f.role)))?,
    )?;
    // --- status (RLE, 1-byte discriminant) ---
    write_blob(
        &mut buf,
        &encode_rle_u8(delta.frames.iter().map(|f| status_disc(f.status)))?,
    )?;
    // --- payload_encoding (RLE, 1-byte discriminant) ---
    write_blob(
        &mut buf,
        &encode_rle_u8(
            delta
                .frames
                .iter()
                .map(|f| payload_encoding_disc(f.payload_encoding)),
        )?,
    )?;

    // --- created_at (i64 first + zigzag delta-of-delta uvarints) ---
    let mut col = Vec::with_capacity(delta.frames.len() + 8);
    let first_created = delta.frames[0].created_at;
    col.extend_from_slice(&first_created.to_le_bytes());
    let mut prev_value = first_created;
    let mut prev_delta: i64 = 0;
    for f in &delta.frames[1..] {
        let delta_v = f
            .created_at
            .checked_sub(prev_value)
            .ok_or_else(|| Error::Format("FrameCatalog created_at delta overflow".into()))?;
        let dod = delta_v.checked_sub(prev_delta).ok_or_else(|| {
            Error::Format("FrameCatalog created_at delta-of-delta overflow".into())
        })?;
        write_ivarint(&mut col, dod);
        prev_value = f.created_at;
        prev_delta = delta_v;
    }
    write_blob(&mut buf, &col)?;

    // --- updated_at offset (zigzag uvarint of (updated_at - created_at)) ---
    let mut col = Vec::with_capacity(delta.frames.len());
    for f in &delta.frames {
        let off = f
            .updated_at
            .checked_sub(f.created_at)
            .ok_or_else(|| Error::Format("FrameCatalog updated_at offset overflow".into()))?;
        write_ivarint(&mut col, off);
    }
    write_blob(&mut buf, &col)?;

    // --- payload_ref column ---
    let payload_ref_presence: Vec<u32> = delta
        .frames
        .iter()
        .enumerate()
        .filter_map(|(i, f)| {
            f.payload_ref
                .as_ref()
                .map(|_| u32::try_from(i).expect("FrameCatalog row index fits in u32"))
        })
        .collect();
    write_blob(&mut buf, &encode_presence(&payload_ref_presence)?)?;
    let mut col = Vec::new();
    let mut prev_segment: i128 = 0;
    let mut prev_bytes_offset: i128 = 0;
    for (i, idx) in payload_ref_presence.iter().enumerate() {
        let pref = delta.frames[*idx as usize]
            .payload_ref
            .expect("payload_ref Some by construction");
        if i == 0 {
            col.extend_from_slice(&pref.segment_id.0.to_le_bytes());
            write_uvarint(&mut col, pref.bytes_offset);
            write_uvarint(&mut col, pref.bytes_length);
        } else {
            let seg_delta = (pref.segment_id.0 as i128) - prev_segment;
            let off_delta = (pref.bytes_offset as i128) - prev_bytes_offset;
            write_ivarint(&mut col, i128_to_i64(seg_delta, "segment_id delta")?);
            write_ivarint(&mut col, i128_to_i64(off_delta, "bytes_offset delta")?);
            write_uvarint(&mut col, pref.bytes_length);
        }
        prev_segment = pref.segment_id.0 as i128;
        prev_bytes_offset = pref.bytes_offset as i128;
    }
    write_blob(&mut buf, &col)?;

    // --- 4 sparse MetadataRef columns ---
    encode_optional_metadata_column(&mut buf, &delta.frames, |f| f.uri_ref.as_ref())?;
    encode_optional_metadata_column(&mut buf, &delta.frames, |f| f.title_ref.as_ref())?;
    encode_optional_metadata_column(&mut buf, &delta.frames, |f| f.search_text_ref.as_ref())?;
    encode_optional_metadata_column(&mut buf, &delta.frames, |f| f.metadata_ref.as_ref())?;
    // --- parent_frame_id column (Option<FrameId> = Option<u64>) ---
    encode_optional_u64_column(&mut buf, &delta.frames, |f| f.parent_frame_id.map(|p| p.0))?;
    // --- chunk_index column (Option<u32>) ---
    encode_optional_u32_column(&mut buf, &delta.frames, |f| f.chunk_index)?;
    // --- chunk_count column (Option<u32>) ---
    encode_optional_u32_column(&mut buf, &delta.frames, |f| f.chunk_count)?;

    Ok(buf)
}

pub(crate) fn decode_frame_catalog(bytes: &[u8]) -> Result<FrameCatalogDelta> {
    let (delta, _full) = decode_frame_catalog_full(bytes, false)?;
    Ok(delta)
}

pub(crate) fn decode_frame_catalog_previous_root(bytes: &[u8]) -> Result<Option<SegmentRef>> {
    let mut cur = Cursor::new(bytes);
    let magic = cur.read_array::<4>()?;
    if magic != MAGIC {
        return Err(Error::Format(format!(
            "FrameCatalog magic mismatch: expected VLF2, got {magic:?}"
        )));
    }
    let version = cur.read_u16()?;
    if version != VERSION {
        return Err(Error::Format(format!(
            "FrameCatalog version mismatch: expected {VERSION}, got {version}"
        )));
    }
    let _count = cur.read_u32()?;
    read_previous_root(&mut cur)
}

fn decode_frame_catalog_full(bytes: &[u8], _trace: bool) -> Result<(FrameCatalogDelta, usize)> {
    let mut cur = Cursor::new(bytes);
    let magic = cur.read_array::<4>()?;
    if magic != MAGIC {
        return Err(Error::Format(format!(
            "FrameCatalog magic mismatch: expected VLF2, got {magic:?}"
        )));
    }
    let version = cur.read_u16()?;
    if version != VERSION {
        return Err(Error::Format(format!(
            "FrameCatalog version mismatch: expected {VERSION}, got {version}"
        )));
    }
    let count = cur.read_u32()? as usize;
    let previous_root = read_previous_root(&mut cur)?;
    if count == 0 {
        return Ok((
            FrameCatalogDelta {
                previous_root,
                frames: Vec::new(),
            },
            cur.pos,
        ));
    }

    // frame_id column
    let frame_blob = cur.read_blob()?;
    let mut fc = Cursor::new(frame_blob);
    let first_frame = fc.read_u64()?;
    let mut frame_ids = Vec::with_capacity(count);
    frame_ids.push(first_frame);
    let mut prev = first_frame;
    for _ in 1..count {
        let delta = fc.read_uvarint()?;
        let next = prev
            .checked_add(delta)
            .ok_or_else(|| Error::Format("FrameCatalog frame_id delta overflow".into()))?;
        frame_ids.push(next);
        prev = next;
    }
    // collection_id RLE
    let collection_ids: Vec<u32> = decode_rle_u32(cur.read_blob()?, count)?;
    // role
    let role_discs: Vec<u8> = decode_rle_u8(cur.read_blob()?, count)?;
    let roles: Vec<FrameRole> = role_discs
        .into_iter()
        .map(role_from_disc)
        .collect::<Result<Vec<_>>>()?;
    // status
    let status_discs: Vec<u8> = decode_rle_u8(cur.read_blob()?, count)?;
    let statuses: Vec<FrameStatus> = status_discs
        .into_iter()
        .map(status_from_disc)
        .collect::<Result<Vec<_>>>()?;
    // payload_encoding
    let enc_discs: Vec<u8> = decode_rle_u8(cur.read_blob()?, count)?;
    let encodings: Vec<PayloadEncoding> = enc_discs
        .into_iter()
        .map(payload_encoding_from_disc)
        .collect::<Result<Vec<_>>>()?;
    // created_at
    let created_blob = cur.read_blob()?;
    let mut cc = Cursor::new(created_blob);
    let first_created = cc.read_i64()?;
    let mut created = Vec::with_capacity(count);
    created.push(first_created);
    let mut prev_value = first_created;
    let mut prev_delta: i64 = 0;
    for _ in 1..count {
        let dod = cc.read_ivarint()?;
        let delta_v = prev_delta
            .checked_add(dod)
            .ok_or_else(|| Error::Format("FrameCatalog created delta overflow".into()))?;
        let value = prev_value
            .checked_add(delta_v)
            .ok_or_else(|| Error::Format("FrameCatalog created value overflow".into()))?;
        created.push(value);
        prev_value = value;
        prev_delta = delta_v;
    }
    // updated_at offset
    let updated_blob = cur.read_blob()?;
    let mut uc = Cursor::new(updated_blob);
    let mut updated = Vec::with_capacity(count);
    for c in &created {
        let off = uc.read_ivarint()?;
        let value = c
            .checked_add(off)
            .ok_or_else(|| Error::Format("FrameCatalog updated_at overflow".into()))?;
        updated.push(value);
    }
    // payload_ref
    let presence_indices = decode_presence(cur.read_blob()?)?;
    let payload_blob = cur.read_blob()?;
    let mut pc = Cursor::new(payload_blob);
    let mut payload_refs: Vec<Option<PayloadRef>> = vec![None; count];
    let mut prev_segment: i128 = 0;
    let mut prev_offset: i128 = 0;
    for (i, idx) in presence_indices.iter().enumerate() {
        let seg_id = if i == 0 {
            pc.read_u64()?
        } else {
            let delta = pc.read_ivarint()?;
            (prev_segment + delta as i128) as u64
        };
        let bytes_offset = if i == 0 {
            pc.read_uvarint()?
        } else {
            let delta = pc.read_ivarint()?;
            (prev_offset + delta as i128) as u64
        };
        let bytes_length = pc.read_uvarint()?;
        prev_segment = seg_id as i128;
        prev_offset = bytes_offset as i128;
        let row = *idx as usize;
        if row >= count {
            return Err(Error::Format(format!(
                "FrameCatalog payload_ref presence index {row} out of bounds (count={count})"
            )));
        }
        payload_refs[row] = Some(PayloadRef {
            segment_id: SegmentId(seg_id),
            bytes_offset,
            bytes_length,
        });
    }
    // sparse MetadataRef columns
    let uri_refs = decode_optional_metadata_column(&mut cur, count)?;
    let title_refs = decode_optional_metadata_column(&mut cur, count)?;
    let search_text_refs = decode_optional_metadata_column(&mut cur, count)?;
    let metadata_refs = decode_optional_metadata_column(&mut cur, count)?;
    let parent_frame_ids = decode_optional_u64_column(&mut cur, count)?
        .into_iter()
        .map(|opt| opt.map(FrameId))
        .collect::<Vec<_>>();
    let chunk_indices = decode_optional_u32_column(&mut cur, count)?;
    let chunk_counts = decode_optional_u32_column(&mut cur, count)?;

    let mut frames = Vec::with_capacity(count);
    for i in 0..count {
        frames.push(FrameDesc {
            frame_id: FrameId(frame_ids[i]),
            collection_id: CollectionId(collection_ids[i]),
            role: roles[i],
            created_at: created[i],
            updated_at: updated[i],
            uri_ref: uri_refs[i].clone(),
            title_ref: title_refs[i].clone(),
            payload_ref: payload_refs[i],
            payload_encoding: encodings[i],
            search_text_ref: search_text_refs[i].clone(),
            parent_frame_id: parent_frame_ids[i],
            chunk_index: chunk_indices[i],
            chunk_count: chunk_counts[i],
            metadata_ref: metadata_refs[i].clone(),
            status: statuses[i],
        });
    }

    Ok((
        FrameCatalogDelta {
            previous_root,
            frames,
        },
        cur.pos,
    ))
}

fn write_previous_root(buf: &mut Vec<u8>, maybe_ref: Option<SegmentRef>) {
    if let Some(r) = maybe_ref {
        buf.push(1u8);
        buf.push(0u8); // padding
        buf.extend_from_slice(&r.segment_id.0.to_le_bytes());
        buf.extend_from_slice(&r.offset.to_le_bytes());
        buf.extend_from_slice(&r.length.to_le_bytes());
        buf.extend_from_slice(&r.checksum);
    } else {
        buf.push(0u8);
        buf.push(0u8); // padding
    }
}

fn read_previous_root(cur: &mut Cursor<'_>) -> Result<Option<SegmentRef>> {
    let flag = cur.read_u8()?;
    let _padding = cur.read_u8()?;
    match flag {
        0 => Ok(None),
        1 => {
            let segment_id = SegmentId(cur.read_u64()?);
            let offset = cur.read_u64()?;
            let length = cur.read_u64()?;
            let checksum = cur.read_array::<32>()?;
            Ok(Some(SegmentRef {
                segment_id,
                offset,
                length,
                checksum,
            }))
        }
        other => Err(Error::Format(format!(
            "FrameCatalog previous_root flag must be 0 or 1, got {other}"
        ))),
    }
}

fn encode_rle_u8<I: IntoIterator<Item = u8>>(values: I) -> Result<Vec<u8>> {
    let mut runs: Vec<(u8, u32)> = Vec::new();
    for v in values {
        match runs.last_mut() {
            Some((cur_v, len)) if *cur_v == v => {
                *len = len
                    .checked_add(1)
                    .ok_or_else(|| Error::Format("FrameCatalog RLE u8 run too long".into()))?;
            }
            _ => runs.push((v, 1)),
        }
    }
    let mut out = Vec::with_capacity(4 + runs.len() * 2);
    let run_count = u32::try_from(runs.len())
        .map_err(|_| Error::Format("FrameCatalog RLE u8 run count overflow".into()))?;
    out.extend_from_slice(&run_count.to_le_bytes());
    for (v, len) in runs {
        out.push(v);
        write_uvarint(&mut out, u64::from(len));
    }
    Ok(out)
}

fn encode_rle_u32<I: IntoIterator<Item = u32>>(values: I) -> Result<Vec<u8>> {
    let mut runs: Vec<(u32, u32)> = Vec::new();
    for v in values {
        match runs.last_mut() {
            Some((cur_v, len)) if *cur_v == v => {
                *len = len
                    .checked_add(1)
                    .ok_or_else(|| Error::Format("FrameCatalog RLE u32 run too long".into()))?;
            }
            _ => runs.push((v, 1)),
        }
    }
    let mut out = Vec::with_capacity(4 + runs.len() * 8);
    let run_count = u32::try_from(runs.len())
        .map_err(|_| Error::Format("FrameCatalog RLE u32 run count overflow".into()))?;
    out.extend_from_slice(&run_count.to_le_bytes());
    for (v, len) in runs {
        out.extend_from_slice(&v.to_le_bytes());
        write_uvarint(&mut out, u64::from(len));
    }
    Ok(out)
}

fn decode_rle_u8(bytes: &[u8], expected_count: usize) -> Result<Vec<u8>> {
    let mut cur = Cursor::new(bytes);
    let run_count = cur.read_u32()? as usize;
    let mut out = Vec::with_capacity(expected_count);
    let mut total: u64 = 0;
    for _ in 0..run_count {
        let v = cur.read_u8()?;
        let len = cur.read_uvarint()?;
        total += len;
        for _ in 0..len {
            out.push(v);
        }
    }
    if total != expected_count as u64 {
        return Err(Error::Format(format!(
            "FrameCatalog RLE u8 covers {total} rows but count={expected_count}"
        )));
    }
    if !cur.is_empty() {
        return Err(Error::Format(format!(
            "FrameCatalog RLE u8 trailing: {} bytes",
            cur.remaining()
        )));
    }
    Ok(out)
}

fn decode_rle_u32(bytes: &[u8], expected_count: usize) -> Result<Vec<u32>> {
    let mut cur = Cursor::new(bytes);
    let run_count = cur.read_u32()? as usize;
    let mut out = Vec::with_capacity(expected_count);
    let mut total: u64 = 0;
    for _ in 0..run_count {
        let v = cur.read_u32()?;
        let len = cur.read_uvarint()?;
        total += len;
        for _ in 0..len {
            out.push(v);
        }
    }
    if total != expected_count as u64 {
        return Err(Error::Format(format!(
            "FrameCatalog RLE u32 covers {total} rows but count={expected_count}"
        )));
    }
    if !cur.is_empty() {
        return Err(Error::Format(format!(
            "FrameCatalog RLE u32 trailing: {} bytes",
            cur.remaining()
        )));
    }
    Ok(out)
}

fn encode_presence(indices: &[u32]) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(4 + indices.len() * 4);
    let count = u32::try_from(indices.len())
        .map_err(|_| Error::Format("FrameCatalog presence count overflow".into()))?;
    out.extend_from_slice(&count.to_le_bytes());
    // Indices are monotonic by construction (filter_map over the
    // catalog's row order). Delta-encode them as uvarints to keep
    // dense lists tight.
    let mut prev: u32 = 0;
    for (i, &idx) in indices.iter().enumerate() {
        if i == 0 {
            write_uvarint(&mut out, u64::from(idx));
        } else if idx < prev {
            return Err(Error::Format(
                "FrameCatalog presence indices must be ascending".into(),
            ));
        } else {
            write_uvarint(&mut out, u64::from(idx - prev));
        }
        prev = idx;
    }
    Ok(out)
}

fn decode_presence(bytes: &[u8]) -> Result<Vec<u32>> {
    let mut cur = Cursor::new(bytes);
    let count = cur.read_u32()? as usize;
    let mut out = Vec::with_capacity(count);
    let mut prev: u32 = 0;
    for i in 0..count {
        let delta = u32::try_from(cur.read_uvarint()?)
            .map_err(|_| Error::Format("FrameCatalog presence index overflow".into()))?;
        let idx = if i == 0 {
            delta
        } else {
            prev.checked_add(delta)
                .ok_or_else(|| Error::Format("FrameCatalog presence index overflow".into()))?
        };
        out.push(idx);
        prev = idx;
    }
    if !cur.is_empty() {
        return Err(Error::Format(format!(
            "FrameCatalog presence trailing: {} bytes",
            cur.remaining()
        )));
    }
    Ok(out)
}

fn encode_optional_metadata_column<F: Fn(&FrameDesc) -> Option<&MetadataRef>>(
    buf: &mut Vec<u8>,
    frames: &[FrameDesc],
    get: F,
) -> Result<()> {
    let presence: Vec<u32> = frames
        .iter()
        .enumerate()
        .filter_map(|(i, f)| get(f).map(|_| u32::try_from(i).unwrap()))
        .collect();
    write_blob(buf, &encode_presence(&presence)?)?;
    let mut values = Vec::new();
    for idx in &presence {
        let m = get(&frames[*idx as usize]).expect("Some by presence");
        values.extend_from_slice(&m.segment_id.0.to_le_bytes());
        write_uvarint(&mut values, m.bytes_offset);
        write_uvarint(&mut values, m.bytes_length);
        values.extend_from_slice(&m.checksum);
    }
    write_blob(buf, &values)?;
    Ok(())
}

fn decode_optional_metadata_column(
    cur: &mut Cursor<'_>,
    count: usize,
) -> Result<Vec<Option<MetadataRef>>> {
    let presence = decode_presence(cur.read_blob()?)?;
    let values_blob = cur.read_blob()?;
    let mut vc = Cursor::new(values_blob);
    let mut out: Vec<Option<MetadataRef>> = vec![None; count];
    for idx in presence {
        let segment_id = SegmentId(vc.read_u64()?);
        let bytes_offset = vc.read_uvarint()?;
        let bytes_length = vc.read_uvarint()?;
        let checksum = vc.read_array::<32>()?;
        let row = idx as usize;
        if row >= count {
            return Err(Error::Format(format!(
                "FrameCatalog metadata presence index {row} out of bounds (count={count})"
            )));
        }
        out[row] = Some(MetadataRef {
            segment_id,
            bytes_offset,
            bytes_length,
            checksum,
        });
    }
    if !vc.is_empty() {
        return Err(Error::Format(format!(
            "FrameCatalog metadata values trailing: {} bytes",
            vc.remaining()
        )));
    }
    Ok(out)
}

fn encode_optional_u64_column<F: Fn(&FrameDesc) -> Option<u64>>(
    buf: &mut Vec<u8>,
    frames: &[FrameDesc],
    get: F,
) -> Result<()> {
    let presence: Vec<u32> = frames
        .iter()
        .enumerate()
        .filter_map(|(i, f)| get(f).map(|_| u32::try_from(i).unwrap()))
        .collect();
    write_blob(buf, &encode_presence(&presence)?)?;
    let mut values = Vec::new();
    for idx in &presence {
        let v = get(&frames[*idx as usize]).expect("Some by presence");
        write_uvarint(&mut values, v);
    }
    write_blob(buf, &values)?;
    Ok(())
}

fn decode_optional_u64_column(cur: &mut Cursor<'_>, count: usize) -> Result<Vec<Option<u64>>> {
    let presence = decode_presence(cur.read_blob()?)?;
    let values_blob = cur.read_blob()?;
    let mut vc = Cursor::new(values_blob);
    let mut out: Vec<Option<u64>> = vec![None; count];
    for idx in presence {
        let v = vc.read_uvarint()?;
        out[idx as usize] = Some(v);
    }
    if !vc.is_empty() {
        return Err(Error::Format(format!(
            "FrameCatalog optional u64 values trailing: {} bytes",
            vc.remaining()
        )));
    }
    Ok(out)
}

fn encode_optional_u32_column<F: Fn(&FrameDesc) -> Option<u32>>(
    buf: &mut Vec<u8>,
    frames: &[FrameDesc],
    get: F,
) -> Result<()> {
    let presence: Vec<u32> = frames
        .iter()
        .enumerate()
        .filter_map(|(i, f)| get(f).map(|_| u32::try_from(i).unwrap()))
        .collect();
    write_blob(buf, &encode_presence(&presence)?)?;
    let mut values = Vec::new();
    for idx in &presence {
        let v = get(&frames[*idx as usize]).expect("Some by presence");
        write_uvarint(&mut values, u64::from(v));
    }
    write_blob(buf, &values)?;
    Ok(())
}

fn decode_optional_u32_column(cur: &mut Cursor<'_>, count: usize) -> Result<Vec<Option<u32>>> {
    let presence = decode_presence(cur.read_blob()?)?;
    let values_blob = cur.read_blob()?;
    let mut vc = Cursor::new(values_blob);
    let mut out: Vec<Option<u32>> = vec![None; count];
    for idx in presence {
        let v = u32::try_from(vc.read_uvarint()?)
            .map_err(|_| Error::Format("FrameCatalog optional u32 overflow".into()))?;
        out[idx as usize] = Some(v);
    }
    if !vc.is_empty() {
        return Err(Error::Format(format!(
            "FrameCatalog optional u32 values trailing: {} bytes",
            vc.remaining()
        )));
    }
    Ok(out)
}

fn write_blob(buf: &mut Vec<u8>, blob: &[u8]) -> Result<()> {
    let len = u32::try_from(blob.len())
        .map_err(|_| Error::Format("FrameCatalog blob too large".into()))?;
    buf.extend_from_slice(&len.to_le_bytes());
    buf.extend_from_slice(blob);
    Ok(())
}

fn write_uvarint(buf: &mut Vec<u8>, mut value: u64) {
    while value >= 0x80 {
        buf.push(((value as u8) & 0x7F) | 0x80);
        value >>= 7;
    }
    buf.push(value as u8);
}

fn write_ivarint(buf: &mut Vec<u8>, value: i64) {
    let zz = ((value << 1) ^ (value >> 63)) as u64;
    write_uvarint(buf, zz);
}

fn i128_to_i64(v: i128, label: &str) -> Result<i64> {
    i64::try_from(v).map_err(|_| Error::Format(format!("FrameCatalog {label} out of i64 range")))
}

fn role_disc(r: FrameRole) -> u8 {
    match r {
        FrameRole::Document => 0,
        FrameRole::Chunk => 1,
        FrameRole::Tombstone => 2,
    }
}
fn role_from_disc(d: u8) -> Result<FrameRole> {
    Ok(match d {
        0 => FrameRole::Document,
        1 => FrameRole::Chunk,
        2 => FrameRole::Tombstone,
        other => {
            return Err(Error::Format(format!(
                "FrameCatalog unknown role disc {other}"
            )));
        }
    })
}
fn status_disc(s: FrameStatus) -> u8 {
    match s {
        FrameStatus::Active => 0,
        FrameStatus::Tombstoned => 1,
    }
}
fn status_from_disc(d: u8) -> Result<FrameStatus> {
    Ok(match d {
        0 => FrameStatus::Active,
        1 => FrameStatus::Tombstoned,
        other => {
            return Err(Error::Format(format!(
                "FrameCatalog unknown status disc {other}"
            )));
        }
    })
}
fn payload_encoding_disc(e: PayloadEncoding) -> u8 {
    match e {
        PayloadEncoding::Raw => 0,
        PayloadEncoding::Zstd => 1,
        PayloadEncoding::Lz4 => 2,
    }
}
fn payload_encoding_from_disc(d: u8) -> Result<PayloadEncoding> {
    Ok(match d {
        0 => PayloadEncoding::Raw,
        1 => PayloadEncoding::Zstd,
        2 => PayloadEncoding::Lz4,
        other => {
            return Err(Error::Format(format!(
                "FrameCatalog unknown payload_encoding disc {other}"
            )));
        }
    })
}

struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }
    fn remaining(&self) -> usize {
        self.bytes.len() - self.pos
    }
    fn is_empty(&self) -> bool {
        self.pos == self.bytes.len()
    }
    fn read_array<const N: usize>(&mut self) -> Result<[u8; N]> {
        if self.pos + N > self.bytes.len() {
            return Err(Error::Format(format!(
                "FrameCatalog truncated: need {N} bytes at offset {}",
                self.pos
            )));
        }
        let mut out = [0u8; N];
        out.copy_from_slice(&self.bytes[self.pos..self.pos + N]);
        self.pos += N;
        Ok(out)
    }
    fn read_u8(&mut self) -> Result<u8> {
        Ok(self.read_array::<1>()?[0])
    }
    fn read_u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.read_array::<2>()?))
    }
    fn read_u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.read_array::<4>()?))
    }
    fn read_u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.read_array::<8>()?))
    }
    fn read_i64(&mut self) -> Result<i64> {
        Ok(i64::from_le_bytes(self.read_array::<8>()?))
    }
    fn read_blob(&mut self) -> Result<&'a [u8]> {
        let len = self.read_u32()? as usize;
        if self.pos + len > self.bytes.len() {
            return Err(Error::Format(format!(
                "FrameCatalog blob truncated: declared {len} but only {} bytes left",
                self.bytes.len() - self.pos
            )));
        }
        let out = &self.bytes[self.pos..self.pos + len];
        self.pos += len;
        Ok(out)
    }
    fn read_uvarint(&mut self) -> Result<u64> {
        let mut result: u64 = 0;
        let mut shift: u32 = 0;
        loop {
            let b = self.read_u8()?;
            let chunk = u64::from(b & 0x7F);
            result |= chunk
                .checked_shl(shift)
                .ok_or_else(|| Error::Format("FrameCatalog varint overflow".into()))?;
            if b & 0x80 == 0 {
                return Ok(result);
            }
            shift += 7;
            if shift >= 64 {
                return Err(Error::Format("FrameCatalog varint too long".into()));
            }
        }
    }
    fn read_ivarint(&mut self) -> Result<i64> {
        let zz = self.read_uvarint()?;
        Ok(((zz >> 1) as i64) ^ -((zz & 1) as i64))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::catalog::FrameStub;

    fn quora_frame(frame_id: u64, now: i64, seg_id: u64, offset: u64, len: u64) -> FrameDesc {
        FrameDesc {
            frame_id: FrameId(frame_id),
            collection_id: CollectionId(1),
            role: FrameRole::Document,
            created_at: now,
            updated_at: now,
            uri_ref: None,
            title_ref: None,
            payload_ref: Some(PayloadRef {
                segment_id: SegmentId(seg_id),
                bytes_offset: offset,
                bytes_length: len,
            }),
            payload_encoding: PayloadEncoding::Raw,
            search_text_ref: None,
            parent_frame_id: None,
            chunk_index: None,
            chunk_count: None,
            metadata_ref: None,
            status: FrameStatus::Active,
        }
    }

    #[test]
    fn roundtrip_empty() {
        let delta = FrameCatalogDelta {
            previous_root: None,
            frames: Vec::new(),
        };
        let bytes = encode_frame_catalog(&delta).expect("encode");
        let decoded = decode_frame_catalog(&bytes).expect("decode");
        assert_eq!(decoded, delta);
        // previous_root extraction must also work on empty
        let prev = decode_frame_catalog_previous_root(&bytes).expect("prev");
        assert_eq!(prev, None);
    }

    #[test]
    fn roundtrip_with_previous_root_and_frames() {
        let prev = SegmentRef {
            segment_id: SegmentId(42),
            offset: 4096,
            length: 1024,
            checksum: [0xAB; 32],
        };
        let delta = FrameCatalogDelta {
            previous_root: Some(prev),
            frames: vec![
                quora_frame(1, 1_700_000_000, 100, 0, 50),
                FrameDesc {
                    uri_ref: Some(MetadataRef {
                        segment_id: SegmentId(99),
                        bytes_offset: 17,
                        bytes_length: 24,
                        checksum: [0x11; 32],
                    }),
                    parent_frame_id: Some(FrameId(1)),
                    chunk_index: Some(2),
                    chunk_count: Some(5),
                    role: FrameRole::Chunk,
                    status: FrameStatus::Tombstoned,
                    ..quora_frame(7, 1_700_000_001, 101, 50, 75)
                },
            ],
        };
        let bytes = encode_frame_catalog(&delta).expect("encode");
        let decoded = decode_frame_catalog(&bytes).expect("decode");
        assert_eq!(decoded, delta);
        let prev_extracted = decode_frame_catalog_previous_root(&bytes).expect("prev");
        assert_eq!(prev_extracted, Some(prev));
        // Derive `FrameStub`s the same way `read_frame_catalog_v2_chain`
        // does: walk the full decode and project (frame_id, status).
        // No separate slim decoder exists for the columnar v2 wire — the
        // block isn't randomly accessible without decoding every column.
        let stubs: Vec<FrameStub> = decoded
            .frames
            .iter()
            .map(|f| FrameStub {
                frame_id: f.frame_id,
                status: f.status,
            })
            .collect();
        assert_eq!(stubs.len(), 2);
        assert_eq!(stubs[0].status, FrameStatus::Active);
        assert_eq!(stubs[1].status, FrameStatus::Tombstoned);
    }

    /// Synthetic Quora ingest: 522 k sequential frames, one collection,
    /// all `Active`, all `Document`, monotonic timestamps within a
    /// second, all 7 sparse Option fields `None`. v2 columnar must
    /// produce ≤ 10 B/row — a >3× win over the 41.7 B/row v1 bincode
    /// layout. Most of that budget goes into the payload_ref column
    /// (3 varints/row ≈ 4 B with the segment_id/offset deltas).
    #[test]
    fn columnar_compresses_quora_shape() {
        let now: i64 = 1_700_000_000;
        // Spread payloads across 8 segments to mirror the production
        // ~4 MiB batched commit path. bytes_offset deltas climb by
        // ~64 B (= avg Quora doc size) per row inside each segment.
        let frames: Vec<FrameDesc> = (1..=522_931u64)
            .map(|i| {
                let seg = 100 + (i - 1) / 65_366;
                let off = ((i - 1) % 65_366) * 64;
                quora_frame(i, now, seg, off, 50)
            })
            .collect();
        let delta = FrameCatalogDelta {
            previous_root: None,
            frames,
        };
        let bytes = encode_frame_catalog(&delta).expect("encode");
        let per_row = bytes.len() as f64 / 522_931.0;
        assert!(
            per_row < 10.0,
            "Quora-shaped FrameCatalog v2 encode is {per_row:.2} B/row, expected < 10",
        );
        let decoded = decode_frame_catalog(&bytes).expect("decode");
        assert_eq!(decoded.frames.len(), 522_931);
        assert_eq!(decoded.frames[0].frame_id.0, 1);
        assert_eq!(decoded.frames[522_930].frame_id.0, 522_931);
    }

    #[test]
    fn rejects_magic() {
        let mut bytes = encode_frame_catalog(&FrameCatalogDelta {
            previous_root: None,
            frames: vec![quora_frame(1, 0, 1, 0, 1)],
        })
        .unwrap();
        bytes[0] = b'X';
        let err = decode_frame_catalog(&bytes).expect_err("magic");
        assert!(err.to_string().contains("magic"));
    }

    #[test]
    fn rejects_version() {
        let mut bytes = encode_frame_catalog(&FrameCatalogDelta {
            previous_root: None,
            frames: vec![quora_frame(1, 0, 1, 0, 1)],
        })
        .unwrap();
        bytes[4..6].copy_from_slice(&99u16.to_le_bytes());
        let err = decode_frame_catalog(&bytes).expect_err("version");
        assert!(err.to_string().contains("version"));
    }
}
