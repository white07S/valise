//! `TimeIndexSegment` payload codec, spec §17 (magic `NXTI`).
//!
//! Naming: `_segment` (not `_codec`) because this file owns the
//! `TimeIndexSegment` payload *type* plus its wire codec. The per-entry
//! record `TimeIndexEntry` lives in the sibling `time_index.rs`. See the
//! `format/` file vocabulary rule in `SKILL.md`.
//!
//! v2 wire form (columnar):
//!
//!   MAGIC (4)
//!   VERSION (2) = 2
//!   PARTITION_FLAG (1) + PADDING_OR_CID (4)
//!   COUNT (u32, little-endian)
//!   PREVIOUS_ROOT (variable, optional SegmentRef)
//!   -- column blobs (only when COUNT > 0) --
//!   FRAME_ID_BYTES (u32 len + bytes)     unsigned-LEB128 deltas
//!   CREATED_BYTES  (u32 len + bytes)     zigzag-LEB128 delta-of-deltas, prefixed by raw i64 base
//!   UPDATED_BYTES  (u32 len + bytes)     zigzag-LEB128 of (updated_at - created_at) per entry
//!   COLLECTION_RUNS (u32 run_count + (collection_id u32, run_len varint) × runs)
//!
//! Rationale: in append-only ingest each `frame_id` is +1 from the
//! previous (delta = 1 → 1 byte LEB128). For ingest bursts within the
//! same second, `created_at` deltas are 0 → delta-of-delta is 0 → 1
//! byte LEB128. `updated_at == created_at` is the common case → 1 byte
//! zigzag-LEB128 zero. `collection_id` is typically one-hot per build,
//! so RLE collapses 522 k 4-byte fields to a single run record.
//!
//! v1 was a row-by-row dump of `(u64, u32, i64, i64) = 28 B/entry`. v2
//! is rejected by v1 readers (version mismatch) and v1 is rejected by
//! v2 readers — there is no on-disk compatibility, consistent with the
//! v0.1-alpha policy that lets us iterate the format freely.

use crate::{
    error::{Error, Result},
    format::{CollectionId, FrameId, segment::SegmentRef},
};

const MAGIC: [u8; 4] = *b"VLTI";
const VERSION: u16 = 2;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct TimeIndexSegment {
    /// Optional partition key. v1 always emits `None` (global index);
    /// readers must accept `Some(collection_id)` for v2 forward-compat.
    pub(crate) collection_id: Option<CollectionId>,
    pub(crate) entries: Vec<TimeIndexEntry>,
    pub(crate) previous_root: Option<SegmentRef>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct TimeIndexEntry {
    pub(crate) frame_id: FrameId,
    pub(crate) collection_id: CollectionId,
    pub(crate) created_at: i64,
    pub(crate) updated_at: i64,
}

pub(crate) fn encode_time_index_segment(seg: &TimeIndexSegment) -> Result<Vec<u8>> {
    if !seg
        .entries
        .windows(2)
        .all(|w| w[0].created_at <= w[1].created_at)
    {
        return Err(Error::Format(
            "TimeIndexSegment entries must be sorted ascending by created_at".into(),
        ));
    }
    let mut buf = Vec::new();
    buf.extend_from_slice(&MAGIC);
    buf.extend_from_slice(&VERSION.to_le_bytes());
    match seg.collection_id {
        None => {
            buf.push(0u8);
            buf.extend_from_slice(&[0u8; 4]);
        }
        Some(cid) => {
            buf.push(1u8);
            buf.extend_from_slice(&cid.0.to_le_bytes());
        }
    }
    let count = u32::try_from(seg.entries.len())
        .map_err(|_| Error::Format("TimeIndexSegment entry count overflow".into()))?;
    buf.extend_from_slice(&count.to_le_bytes());
    write_optional_segment_ref(&mut buf, seg.previous_root);

    if seg.entries.is_empty() {
        return Ok(buf);
    }

    // --- frame_id column (deltas, unsigned LEB128) ---
    // The first frame_id is stored raw (8 bytes prepended inside the
    // blob); subsequent entries store (cur - prev) as a varint.
    let mut frame_bytes: Vec<u8> = Vec::with_capacity(seg.entries.len() + 8);
    frame_bytes.extend_from_slice(&seg.entries[0].frame_id.0.to_le_bytes());
    for w in seg.entries.windows(2) {
        let prev = w[0].frame_id.0;
        let cur = w[1].frame_id.0;
        if cur < prev {
            return Err(Error::Format(
                "TimeIndexSegment frame_ids must be monotone within the same created_at".into(),
            ));
        }
        write_uvarint(&mut frame_bytes, cur - prev);
    }
    write_blob(&mut buf, &frame_bytes)?;

    // --- created_at column (delta-of-delta, zigzag LEB128) ---
    // Layout: i64 first_created_at, varint first_delta (signed
    // 0-against), then zigzag deltas-of-deltas.
    let mut created_bytes: Vec<u8> = Vec::with_capacity(seg.entries.len() + 16);
    let first_created = seg.entries[0].created_at;
    created_bytes.extend_from_slice(&first_created.to_le_bytes());
    let mut prev_delta: i64 = 0;
    let mut prev_value: i64 = first_created;
    for entry in &seg.entries[1..] {
        let delta = entry
            .created_at
            .checked_sub(prev_value)
            .ok_or_else(|| Error::Format("TimeIndexSegment created_at delta overflow".into()))?;
        let dod = delta.checked_sub(prev_delta).ok_or_else(|| {
            Error::Format("TimeIndexSegment created_at delta-of-delta overflow".into())
        })?;
        write_ivarint(&mut created_bytes, dod);
        prev_value = entry.created_at;
        prev_delta = delta;
    }
    write_blob(&mut buf, &created_bytes)?;

    // --- updated_at column (zigzag LEB128 of (updated_at - created_at)) ---
    let mut updated_bytes: Vec<u8> = Vec::with_capacity(seg.entries.len());
    for entry in &seg.entries {
        let offset = entry
            .updated_at
            .checked_sub(entry.created_at)
            .ok_or_else(|| Error::Format("TimeIndexSegment updated/created overflow".into()))?;
        write_ivarint(&mut updated_bytes, offset);
    }
    write_blob(&mut buf, &updated_bytes)?;

    // --- collection_id column (RLE) ---
    let mut runs: Vec<(u32, u32)> = Vec::new();
    let mut iter = seg.entries.iter();
    if let Some(first) = iter.next() {
        let mut cur_cid = first.collection_id.0;
        let mut run_len: u32 = 1;
        for entry in iter {
            if entry.collection_id.0 == cur_cid {
                run_len = run_len
                    .checked_add(1)
                    .ok_or_else(|| Error::Format("TimeIndexSegment RLE run too long".into()))?;
            } else {
                runs.push((cur_cid, run_len));
                cur_cid = entry.collection_id.0;
                run_len = 1;
            }
        }
        runs.push((cur_cid, run_len));
    }
    let run_count = u32::try_from(runs.len())
        .map_err(|_| Error::Format("TimeIndexSegment run count overflow".into()))?;
    let mut coll_blob: Vec<u8> = Vec::with_capacity(runs.len() * 5);
    coll_blob.extend_from_slice(&run_count.to_le_bytes());
    for (cid, run_len) in &runs {
        coll_blob.extend_from_slice(&cid.to_le_bytes());
        write_uvarint(&mut coll_blob, u64::from(*run_len));
    }
    write_blob(&mut buf, &coll_blob)?;

    Ok(buf)
}

pub(crate) fn decode_time_index_segment(bytes: &[u8]) -> Result<TimeIndexSegment> {
    let mut cur = Cursor::new(bytes);
    let magic = cur.read_array::<4>()?;
    if magic != MAGIC {
        return Err(Error::Format(format!(
            "TimeIndexSegment magic mismatch: expected NXTI, got {magic:?}"
        )));
    }
    let version = cur.read_u16()?;
    if version != VERSION {
        return Err(Error::Format(format!(
            "TimeIndexSegment version mismatch: expected {VERSION}, got {version}"
        )));
    }
    let partition_flag = cur.read_u8()?;
    let collection_id = match partition_flag {
        0 => {
            let _ = cur.read_array::<4>()?;
            None
        }
        1 => Some(CollectionId(cur.read_u32()?)),
        other => {
            return Err(Error::Format(format!(
                "TimeIndexSegment partition flag must be 0 or 1, got {other}"
            )));
        }
    };
    let count = cur.read_u32()? as usize;
    let previous_root = read_optional_segment_ref(&mut cur)?;

    if count == 0 {
        if !cur.is_empty() {
            return Err(Error::Format(format!(
                "TimeIndexSegment: {} trailing bytes (count=0)",
                cur.remaining()
            )));
        }
        return Ok(TimeIndexSegment {
            collection_id,
            entries: Vec::new(),
            previous_root,
        });
    }

    // --- frame_id column ---
    let frame_blob = cur.read_blob()?;
    let mut fc = Cursor::new(frame_blob);
    let first_frame = fc.read_u64()?;
    let mut frame_ids: Vec<u64> = Vec::with_capacity(count);
    frame_ids.push(first_frame);
    let mut prev = first_frame;
    for _ in 1..count {
        let delta = fc.read_uvarint()?;
        let next = prev
            .checked_add(delta)
            .ok_or_else(|| Error::Format("TimeIndexSegment frame_id delta overflow".into()))?;
        frame_ids.push(next);
        prev = next;
    }
    if !fc.is_empty() {
        return Err(Error::Format(format!(
            "TimeIndexSegment: {} trailing bytes in frame_id column",
            fc.remaining()
        )));
    }

    // --- created_at column ---
    let created_blob = cur.read_blob()?;
    let mut cc = Cursor::new(created_blob);
    let first_created = cc.read_i64()?;
    let mut created: Vec<i64> = Vec::with_capacity(count);
    created.push(first_created);
    let mut prev_delta: i64 = 0;
    let mut prev_value: i64 = first_created;
    for _ in 1..count {
        let dod = cc.read_ivarint()?;
        let delta = prev_delta
            .checked_add(dod)
            .ok_or_else(|| Error::Format("TimeIndexSegment created delta overflow".into()))?;
        let value = prev_value
            .checked_add(delta)
            .ok_or_else(|| Error::Format("TimeIndexSegment created value overflow".into()))?;
        created.push(value);
        prev_value = value;
        prev_delta = delta;
    }
    if !cc.is_empty() {
        return Err(Error::Format(format!(
            "TimeIndexSegment: {} trailing bytes in created_at column",
            cc.remaining()
        )));
    }

    // --- updated_at column ---
    let updated_blob = cur.read_blob()?;
    let mut uc = Cursor::new(updated_blob);
    let mut updated: Vec<i64> = Vec::with_capacity(count);
    for c in &created {
        let offset = uc.read_ivarint()?;
        let value = c
            .checked_add(offset)
            .ok_or_else(|| Error::Format("TimeIndexSegment updated overflow".into()))?;
        updated.push(value);
    }
    if !uc.is_empty() {
        return Err(Error::Format(format!(
            "TimeIndexSegment: {} trailing bytes in updated_at column",
            uc.remaining()
        )));
    }

    // --- collection_id column (RLE) ---
    let coll_blob = cur.read_blob()?;
    let mut rc = Cursor::new(coll_blob);
    let run_count = rc.read_u32()? as usize;
    let mut collections: Vec<u32> = Vec::with_capacity(count);
    let mut total: u64 = 0;
    for _ in 0..run_count {
        let cid = rc.read_u32()?;
        let run_len = rc.read_uvarint()?;
        total = total.checked_add(run_len).ok_or_else(|| {
            Error::Format("TimeIndexSegment collection RLE total overflow".into())
        })?;
        for _ in 0..run_len {
            collections.push(cid);
        }
    }
    if total != count as u64 {
        return Err(Error::Format(format!(
            "TimeIndexSegment collection RLE covers {total} entries but count={count}"
        )));
    }
    if !rc.is_empty() {
        return Err(Error::Format(format!(
            "TimeIndexSegment: {} trailing bytes in collection column",
            rc.remaining()
        )));
    }

    if !cur.is_empty() {
        return Err(Error::Format(format!(
            "TimeIndexSegment: {} trailing bytes",
            cur.remaining()
        )));
    }

    let mut entries = Vec::with_capacity(count);
    for i in 0..count {
        entries.push(TimeIndexEntry {
            frame_id: FrameId(frame_ids[i]),
            collection_id: CollectionId(collections[i]),
            created_at: created[i],
            updated_at: updated[i],
        });
    }

    Ok(TimeIndexSegment {
        collection_id,
        entries,
        previous_root,
    })
}

fn write_optional_segment_ref(buf: &mut Vec<u8>, maybe_ref: Option<SegmentRef>) {
    if let Some(r) = maybe_ref {
        buf.push(1u8);
        buf.extend_from_slice(&r.segment_id.0.to_le_bytes());
        buf.extend_from_slice(&r.offset.to_le_bytes());
        buf.extend_from_slice(&r.length.to_le_bytes());
        buf.extend_from_slice(&r.checksum);
    } else {
        buf.push(0u8);
    }
}

fn read_optional_segment_ref(cur: &mut Cursor<'_>) -> Result<Option<SegmentRef>> {
    let flag = cur.read_u8()?;
    match flag {
        0 => Ok(None),
        1 => {
            let segment_id = crate::format::SegmentId(cur.read_u64()?);
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
            "TimeIndexSegment previous_root flag must be 0 or 1, got {other}"
        ))),
    }
}

fn write_blob(buf: &mut Vec<u8>, blob: &[u8]) -> Result<()> {
    let len = u32::try_from(blob.len())
        .map_err(|_| Error::Format("TimeIndexSegment blob too large".into()))?;
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
                "TimeIndexSegment truncated: need {N} bytes at offset {}",
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
                "TimeIndexSegment blob truncated: declared {len} but only {} bytes left",
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
                .ok_or_else(|| Error::Format("TimeIndexSegment varint overflow".into()))?;
            if b & 0x80 == 0 {
                return Ok(result);
            }
            shift += 7;
            if shift >= 64 {
                return Err(Error::Format("TimeIndexSegment varint too long".into()));
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

    fn entry(frame: u64, coll: u32, created: i64, updated: i64) -> TimeIndexEntry {
        TimeIndexEntry {
            frame_id: FrameId(frame),
            collection_id: CollectionId(coll),
            created_at: created,
            updated_at: updated,
        }
    }

    #[test]
    fn roundtrip_global_empty() {
        let seg = TimeIndexSegment {
            collection_id: None,
            entries: Vec::new(),
            previous_root: None,
        };
        let bytes = encode_time_index_segment(&seg).unwrap();
        let decoded = decode_time_index_segment(&bytes).unwrap();
        assert_eq!(decoded, seg);
    }

    #[test]
    fn roundtrip_global_with_entries() {
        let seg = TimeIndexSegment {
            collection_id: None,
            entries: vec![
                entry(1, 1, 1_000, 1_000),
                entry(2, 1, 1_500, 2_000),
                entry(3, 2, 1_500, 1_500),
                entry(4, 1, 2_000, 2_500),
            ],
            previous_root: None,
        };
        let bytes = encode_time_index_segment(&seg).unwrap();
        let decoded = decode_time_index_segment(&bytes).unwrap();
        assert_eq!(decoded, seg);
    }

    #[test]
    fn roundtrip_partitioned() {
        let seg = TimeIndexSegment {
            collection_id: Some(CollectionId(7)),
            entries: vec![entry(1, 7, 100, 100), entry(2, 7, 200, 250)],
            previous_root: None,
        };
        let bytes = encode_time_index_segment(&seg).unwrap();
        let decoded = decode_time_index_segment(&bytes).unwrap();
        assert_eq!(decoded, seg);
    }

    #[test]
    fn rejects_unsorted_created_at() {
        let seg = TimeIndexSegment {
            collection_id: None,
            entries: vec![entry(1, 1, 200, 200), entry(2, 1, 100, 100)],
            previous_root: None,
        };
        let err = encode_time_index_segment(&seg).expect_err("unsorted");
        assert!(err.to_string().contains("sorted ascending"));
    }

    #[test]
    fn rejects_magic() {
        let seg = TimeIndexSegment {
            collection_id: None,
            entries: vec![entry(1, 1, 100, 100)],
            previous_root: None,
        };
        let mut bytes = encode_time_index_segment(&seg).unwrap();
        bytes[0] = b'X';
        let err = decode_time_index_segment(&bytes).expect_err("magic");
        assert!(err.to_string().contains("magic"));
    }

    #[test]
    fn rejects_trailing() {
        let seg = TimeIndexSegment {
            collection_id: None,
            entries: vec![entry(1, 1, 100, 100)],
            previous_root: None,
        };
        let mut bytes = encode_time_index_segment(&seg).unwrap();
        bytes.push(0xAB);
        let err = decode_time_index_segment(&bytes).expect_err("trailing");
        assert!(err.to_string().contains("trailing"));
    }

    /// On a synthetic Quora-shaped load (522 k entries, sequential
    /// frame_ids, single collection, same created_at second, no
    /// updates), v2 must produce ≤ 4 B/entry — a >7× reduction from
    /// the 28 B/entry v1 layout.
    #[test]
    fn columnar_compresses_quora_shape() {
        let now: i64 = 1_700_000_000;
        let entries: Vec<TimeIndexEntry> = (1..=522_931u64)
            .map(|fid| entry(fid, 1, now, now))
            .collect();
        let seg = TimeIndexSegment {
            collection_id: None,
            entries,
            previous_root: None,
        };
        let bytes = encode_time_index_segment(&seg).unwrap();
        let per_entry = bytes.len() as f64 / 522_931.0;
        assert!(
            per_entry < 4.0,
            "Quora-shaped v2 encode is {per_entry:.2} B/entry, expected < 4",
        );
        let decoded = decode_time_index_segment(&bytes).unwrap();
        assert_eq!(decoded.entries.len(), 522_931);
        assert_eq!(decoded.entries[0].frame_id.0, 1);
        assert_eq!(decoded.entries[522_930].frame_id.0, 522_931);
    }
}
