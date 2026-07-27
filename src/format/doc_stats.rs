// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! `DocStatsSegment` payload codec, spec §12.4 (magic `VLTS`).
//!
//! Per-document statistics scoped to one `text_space_id`. Each delta
//! segment carries new or updated rows; running state is reconstructed
//! by chain replay (newest-wins per `frame_id`).
//!
//! v2 wire form (columnar):
//!
//!   MAGIC (4) | VERSION (2) = 2
//!   TEXT_SPACE_ID (u32) | RECORD_COUNT (u32)
//!   FIELD_COUNT (u32)   — single value, every row in a segment has the
//!                         same schema (BM25 / TF-IDF / Jaccard all carry
//!                         one field in practice; v1's per-row variable
//!                         length was uniformly the same value anyway).
//!   PREVIOUS_ROOT (variable, optional SegmentRef)
//!   -- columns (only when record_count > 0) --
//!   FRAME_ID_BYTES   (u32 len + bytes)     unsigned-LEB128 deltas vs prev
//!   COLLECTION_RUNS  (u32 run_count + (collection_id u32, run_len varint) × runs)
//!   FIELD_LEN_BYTES  (u32 len + bytes)     packed u8/u16/u32 per row × field_count
//!     prefixed by a 1-byte width tag per column (1 → u8, 2 → u16, 4 → u32)
//!   TOTAL_TERMS      (u32 len + bytes)     varint per row
//!   UNIQUE_TERMS     (u32 len + bytes)     varint per row
//!
//! Rationale: every observed Quora row has `field_count = 1`, a single
//! collection_id, sequential frame_ids, and field_lengths/total_terms
//! that fit in a u8 (avg corpus doc ~8 tokens, max ~100). v1's fixed
//! 28-byte row collapses to ~3 B/row.
//!
//! v1 was a row-by-row dump (28 B/entry). v2 is rejected by v1 readers
//! and vice-versa; consistent with v0.1-alpha's free-iteration policy.

use crate::{
    error::{Error, Result},
    format::{CollectionId, FrameId, TextSpaceId, segment::SegmentRef, text::DocStatsEntry},
};

const MAGIC: [u8; 4] = *b"VLTS";
const VERSION: u16 = 2;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct DocStatsSegment {
    pub(crate) text_space_id: TextSpaceId,
    pub(crate) previous_root: Option<SegmentRef>,
    pub(crate) records: Vec<DocStatsEntry>,
}

pub(crate) fn encode_doc_stats_segment(seg: &DocStatsSegment) -> Result<Vec<u8>> {
    for entry in &seg.records {
        if entry.text_space_id != seg.text_space_id {
            return Err(Error::Format(format!(
                "DocStatsSegment entry text_space_id {} does not match header {}",
                entry.text_space_id.0, seg.text_space_id.0
            )));
        }
    }
    // Validate field_count uniformity. The columnar layout requires
    // every row in the segment to carry the same number of fields.
    let field_count = if let Some(first) = seg.records.first() {
        let fc = first.field_lengths.len();
        for entry in &seg.records {
            if entry.field_lengths.len() != fc {
                return Err(Error::Format(format!(
                    "DocStatsSegment field_lengths width drift: row {} has {} fields, \
                     expected {fc} (every row in a segment must share its schema)",
                    entry.frame_id.0,
                    entry.field_lengths.len()
                )));
            }
        }
        fc
    } else {
        0
    };

    let mut buf = Vec::new();
    buf.extend_from_slice(&MAGIC);
    buf.extend_from_slice(&VERSION.to_le_bytes());
    buf.extend_from_slice(&seg.text_space_id.0.to_le_bytes());
    let record_count = u32::try_from(seg.records.len())
        .map_err(|_| Error::Format("DocStatsSegment record count overflow".into()))?;
    buf.extend_from_slice(&record_count.to_le_bytes());
    let field_count_u32 = u32::try_from(field_count)
        .map_err(|_| Error::Format("DocStatsSegment field_count overflow".into()))?;
    buf.extend_from_slice(&field_count_u32.to_le_bytes());
    write_optional_segment_ref(&mut buf, seg.previous_root);

    if seg.records.is_empty() {
        return Ok(buf);
    }

    // --- frame_id column ---
    let mut frame_bytes: Vec<u8> = Vec::with_capacity(seg.records.len() + 8);
    frame_bytes.extend_from_slice(&seg.records[0].frame_id.0.to_le_bytes());
    let mut prev = seg.records[0].frame_id.0;
    for entry in &seg.records[1..] {
        let cur = entry.frame_id.0;
        if cur < prev {
            return Err(Error::Format(
                "DocStatsSegment frame_ids must be monotone ascending".into(),
            ));
        }
        write_uvarint(&mut frame_bytes, cur - prev);
        prev = cur;
    }
    write_blob(&mut buf, &frame_bytes)?;

    // --- collection_id column (RLE) ---
    let mut runs: Vec<(u32, u32)> = Vec::new();
    let mut iter = seg.records.iter();
    if let Some(first) = iter.next() {
        let mut cur_cid = first.collection_id.0;
        let mut run_len: u32 = 1;
        for entry in iter {
            if entry.collection_id.0 == cur_cid {
                run_len = run_len.checked_add(1).ok_or_else(|| {
                    Error::Format("DocStatsSegment collection RLE run overflow".into())
                })?;
            } else {
                runs.push((cur_cid, run_len));
                cur_cid = entry.collection_id.0;
                run_len = 1;
            }
        }
        runs.push((cur_cid, run_len));
    }
    let mut coll_blob: Vec<u8> = Vec::with_capacity(runs.len() * 8);
    let run_count = u32::try_from(runs.len())
        .map_err(|_| Error::Format("DocStatsSegment collection run count overflow".into()))?;
    coll_blob.extend_from_slice(&run_count.to_le_bytes());
    for (cid, run_len) in &runs {
        coll_blob.extend_from_slice(&cid.to_le_bytes());
        write_uvarint(&mut coll_blob, u64::from(*run_len));
    }
    write_blob(&mut buf, &coll_blob)?;

    // --- field_lengths column ---
    // Per-column width tag (1, 2, 4) lets us emit u8 / u16 / u32
    // packed arrays without per-row overhead. Quora's avg field
    // length is ~8 tokens and the long tail tops out under 256,
    // so the width tag for the single field is u8 → 1 B/row.
    let mut field_blob: Vec<u8> = Vec::new();
    for f in 0..field_count {
        let max = seg
            .records
            .iter()
            .map(|r| r.field_lengths[f])
            .max()
            .unwrap_or(0);
        let width: u8 = if max <= u32::from(u8::MAX) {
            1
        } else if max <= u32::from(u16::MAX) {
            2
        } else {
            4
        };
        field_blob.push(width);
        for entry in &seg.records {
            let value = entry.field_lengths[f];
            match width {
                1 => field_blob.push(value as u8),
                2 => field_blob.extend_from_slice(&(value as u16).to_le_bytes()),
                _ => field_blob.extend_from_slice(&value.to_le_bytes()),
            }
        }
    }
    write_blob(&mut buf, &field_blob)?;

    // --- total_terms column ---
    let mut total_blob: Vec<u8> = Vec::with_capacity(seg.records.len());
    for entry in &seg.records {
        write_uvarint(&mut total_blob, u64::from(entry.total_terms));
    }
    write_blob(&mut buf, &total_blob)?;

    // --- unique_terms column ---
    let mut unique_blob: Vec<u8> = Vec::with_capacity(seg.records.len());
    for entry in &seg.records {
        write_uvarint(&mut unique_blob, u64::from(entry.unique_terms));
    }
    write_blob(&mut buf, &unique_blob)?;

    Ok(buf)
}

pub(crate) fn decode_doc_stats_segment(bytes: &[u8]) -> Result<DocStatsSegment> {
    let mut cur = Cursor::new(bytes);
    let magic = cur.read_array::<4>()?;
    if magic != MAGIC {
        return Err(Error::Format(format!(
            "DocStatsSegment magic mismatch: expected VLTS, got {magic:?}"
        )));
    }
    let version = cur.read_u16()?;
    if version != VERSION {
        return Err(Error::Format(format!(
            "DocStatsSegment version mismatch: expected {VERSION}, got {version}"
        )));
    }
    let text_space_id = TextSpaceId(cur.read_u32()?);
    let record_count = cur.read_u32()? as usize;
    let field_count = cur.read_u32()? as usize;
    let previous_root = read_optional_segment_ref(&mut cur)?;

    if record_count == 0 {
        if !cur.is_empty() {
            return Err(Error::Format(format!(
                "DocStatsSegment: {} trailing bytes (count=0)",
                cur.remaining()
            )));
        }
        return Ok(DocStatsSegment {
            text_space_id,
            previous_root,
            records: Vec::new(),
        });
    }

    // --- frame_id column ---
    let frame_blob = cur.read_blob()?;
    let mut fc = Cursor::new(frame_blob);
    let first_frame = fc.read_u64()?;
    let mut frame_ids: Vec<u64> = Vec::with_capacity(record_count);
    frame_ids.push(first_frame);
    let mut prev = first_frame;
    for _ in 1..record_count {
        let delta = fc.read_uvarint()?;
        let next = prev
            .checked_add(delta)
            .ok_or_else(|| Error::Format("DocStatsSegment frame_id delta overflow".into()))?;
        frame_ids.push(next);
        prev = next;
    }
    if !fc.is_empty() {
        return Err(Error::Format(format!(
            "DocStatsSegment: {} trailing bytes in frame_id column",
            fc.remaining()
        )));
    }

    // --- collection_id column ---
    let coll_blob = cur.read_blob()?;
    let mut cc = Cursor::new(coll_blob);
    let run_count = cc.read_u32()? as usize;
    let mut collections: Vec<u32> = Vec::with_capacity(record_count);
    let mut total: u64 = 0;
    for _ in 0..run_count {
        let cid = cc.read_u32()?;
        let run_len = cc.read_uvarint()?;
        total = total
            .checked_add(run_len)
            .ok_or_else(|| Error::Format("DocStatsSegment collection RLE total overflow".into()))?;
        for _ in 0..run_len {
            collections.push(cid);
        }
    }
    if total != record_count as u64 {
        return Err(Error::Format(format!(
            "DocStatsSegment collection RLE covers {total} entries but count={record_count}"
        )));
    }
    if !cc.is_empty() {
        return Err(Error::Format(format!(
            "DocStatsSegment: {} trailing bytes in collection column",
            cc.remaining()
        )));
    }

    // --- field_lengths column ---
    let field_blob = cur.read_blob()?;
    let mut fl = Cursor::new(field_blob);
    let mut field_lengths_per_row: Vec<Vec<u32>> = (0..record_count)
        .map(|_| Vec::with_capacity(field_count))
        .collect();
    for _ in 0..field_count {
        let width = fl.read_u8()?;
        for row_vec in &mut field_lengths_per_row {
            let value = match width {
                1 => u32::from(fl.read_u8()?),
                2 => u32::from(fl.read_u16()?),
                4 => fl.read_u32()?,
                other => {
                    return Err(Error::Format(format!(
                        "DocStatsSegment field width must be 1, 2, or 4; got {other}"
                    )));
                }
            };
            row_vec.push(value);
        }
    }
    if !fl.is_empty() {
        return Err(Error::Format(format!(
            "DocStatsSegment: {} trailing bytes in field_lengths column",
            fl.remaining()
        )));
    }

    // --- total_terms column ---
    let total_blob = cur.read_blob()?;
    let mut tc = Cursor::new(total_blob);
    let mut totals: Vec<u32> = Vec::with_capacity(record_count);
    for _ in 0..record_count {
        let value = u32::try_from(tc.read_uvarint()?)
            .map_err(|_| Error::Format("DocStatsSegment total_terms exceeds u32".into()))?;
        totals.push(value);
    }
    if !tc.is_empty() {
        return Err(Error::Format(format!(
            "DocStatsSegment: {} trailing bytes in total_terms column",
            tc.remaining()
        )));
    }

    // --- unique_terms column ---
    let unique_blob = cur.read_blob()?;
    let mut uc = Cursor::new(unique_blob);
    let mut uniques: Vec<u32> = Vec::with_capacity(record_count);
    for _ in 0..record_count {
        let value = u32::try_from(uc.read_uvarint()?)
            .map_err(|_| Error::Format("DocStatsSegment unique_terms exceeds u32".into()))?;
        uniques.push(value);
    }
    if !uc.is_empty() {
        return Err(Error::Format(format!(
            "DocStatsSegment: {} trailing bytes in unique_terms column",
            uc.remaining()
        )));
    }

    if !cur.is_empty() {
        return Err(Error::Format(format!(
            "DocStatsSegment: {} trailing bytes",
            cur.remaining()
        )));
    }

    let mut records = Vec::with_capacity(record_count);
    for i in 0..record_count {
        records.push(DocStatsEntry {
            frame_id: FrameId(frame_ids[i]),
            collection_id: CollectionId(collections[i]),
            text_space_id,
            field_lengths: std::mem::take(&mut field_lengths_per_row[i]),
            total_terms: totals[i],
            unique_terms: uniques[i],
        });
    }

    Ok(DocStatsSegment {
        text_space_id,
        previous_root,
        records,
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
            "DocStatsSegment previous_root presence flag must be 0 or 1, got {other}"
        ))),
    }
}

fn write_blob(buf: &mut Vec<u8>, blob: &[u8]) -> Result<()> {
    let len = u32::try_from(blob.len())
        .map_err(|_| Error::Format("DocStatsSegment blob too large".into()))?;
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
                "DocStatsSegment truncated: need {N} bytes at offset {}",
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

    fn read_blob(&mut self) -> Result<&'a [u8]> {
        let len = self.read_u32()? as usize;
        if self.pos + len > self.bytes.len() {
            return Err(Error::Format(format!(
                "DocStatsSegment blob truncated: declared {len} but only {} bytes left",
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
                .ok_or_else(|| Error::Format("DocStatsSegment varint overflow".into()))?;
            if b & 0x80 == 0 {
                return Ok(result);
            }
            shift += 7;
            if shift >= 64 {
                return Err(Error::Format("DocStatsSegment varint too long".into()));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::SegmentId;

    fn entry(
        space: u32,
        frame_id: u64,
        collection_id: u32,
        lengths: Vec<u32>,
        total: u32,
        unique: u32,
    ) -> DocStatsEntry {
        DocStatsEntry {
            frame_id: FrameId(frame_id),
            collection_id: CollectionId(collection_id),
            text_space_id: TextSpaceId(space),
            field_lengths: lengths,
            total_terms: total,
            unique_terms: unique,
        }
    }

    #[test]
    fn roundtrip_empty() {
        let seg = DocStatsSegment {
            text_space_id: TextSpaceId(7),
            previous_root: None,
            records: Vec::new(),
        };
        let bytes = encode_doc_stats_segment(&seg).unwrap();
        let decoded = decode_doc_stats_segment(&bytes).unwrap();
        assert_eq!(decoded, seg);
    }

    #[test]
    fn roundtrip_with_records_and_previous_root() {
        let seg = DocStatsSegment {
            text_space_id: TextSpaceId(7),
            previous_root: Some(SegmentRef {
                segment_id: SegmentId(99),
                offset: 4096,
                length: 256,
                checksum: [0xAB; 32],
            }),
            records: vec![
                entry(7, 1, 1, vec![10, 20, 30], 60, 45),
                entry(7, 2, 1, vec![5, 7, 11], 23, 18),
                entry(7, 3, 2, vec![100, 200, 4], 100, 50),
            ],
        };
        let bytes = encode_doc_stats_segment(&seg).unwrap();
        let decoded = decode_doc_stats_segment(&bytes).unwrap();
        assert_eq!(decoded, seg);
    }

    #[test]
    fn roundtrip_zero_field_lengths() {
        let seg = DocStatsSegment {
            text_space_id: TextSpaceId(1),
            previous_root: None,
            records: vec![entry(1, 1, 1, vec![], 0, 0)],
        };
        let bytes = encode_doc_stats_segment(&seg).unwrap();
        let decoded = decode_doc_stats_segment(&bytes).unwrap();
        assert_eq!(decoded, seg);
    }

    #[test]
    fn rejects_text_space_id_mismatch() {
        let seg = DocStatsSegment {
            text_space_id: TextSpaceId(7),
            previous_root: None,
            records: vec![DocStatsEntry {
                frame_id: FrameId(1),
                collection_id: CollectionId(1),
                text_space_id: TextSpaceId(999),
                field_lengths: vec![1, 2],
                total_terms: 3,
                unique_terms: 2,
            }],
        };
        let err = encode_doc_stats_segment(&seg).expect_err("text_space_id mismatch");
        assert!(err.to_string().contains("text_space_id"));
    }

    #[test]
    fn rejects_field_count_drift() {
        let seg = DocStatsSegment {
            text_space_id: TextSpaceId(7),
            previous_root: None,
            records: vec![
                entry(7, 1, 1, vec![3], 3, 3),
                entry(7, 2, 1, vec![3, 4], 7, 5),
            ],
        };
        let err = encode_doc_stats_segment(&seg).expect_err("field_count drift");
        assert!(err.to_string().contains("field_lengths width"));
    }

    #[test]
    fn rejects_magic_mismatch() {
        let seg = DocStatsSegment {
            text_space_id: TextSpaceId(1),
            previous_root: None,
            records: vec![entry(1, 1, 1, vec![1], 1, 1)],
        };
        let mut bytes = encode_doc_stats_segment(&seg).unwrap();
        bytes[0] = b'X';
        let err = decode_doc_stats_segment(&bytes).expect_err("magic");
        assert!(err.to_string().contains("magic"));
    }

    #[test]
    fn rejects_version_mismatch() {
        let seg = DocStatsSegment {
            text_space_id: TextSpaceId(1),
            previous_root: None,
            records: vec![entry(1, 1, 1, vec![1], 1, 1)],
        };
        let mut bytes = encode_doc_stats_segment(&seg).unwrap();
        bytes[4..6].copy_from_slice(&99u16.to_le_bytes());
        let err = decode_doc_stats_segment(&bytes).expect_err("version");
        assert!(err.to_string().contains("version"));
    }

    #[test]
    fn rejects_trailing_bytes() {
        let seg = DocStatsSegment {
            text_space_id: TextSpaceId(1),
            previous_root: None,
            records: vec![entry(1, 1, 1, vec![1, 2], 3, 2)],
        };
        let mut bytes = encode_doc_stats_segment(&seg).unwrap();
        bytes.push(0xAB);
        let err = decode_doc_stats_segment(&bytes).expect_err("trailing");
        assert!(err.to_string().contains("trailing"));
    }

    #[test]
    fn rejects_truncated() {
        let seg = DocStatsSegment {
            text_space_id: TextSpaceId(1),
            previous_root: None,
            records: vec![entry(1, 1, 1, vec![1, 2], 3, 2)],
        };
        let bytes = encode_doc_stats_segment(&seg).unwrap();
        let truncated = &bytes[..bytes.len() - 5];
        let err = decode_doc_stats_segment(truncated).expect_err("truncated");
        assert!(err.to_string().contains("truncated"));
    }

    /// On a synthetic Quora-shaped DocStats block (522 k rows,
    /// sequential frame_ids, single collection, single field, avg
    /// field_length under 256), v2 must produce ≤ 5 B/row — at
    /// least a 5× win over the 28 B/row v1 layout. Most of that
    /// budget goes into total_terms + unique_terms varints (~2 B
    /// each at ~50-token docs).
    #[test]
    fn columnar_compresses_quora_shape() {
        let records: Vec<DocStatsEntry> = (1..=522_931u64)
            .map(|fid| entry(1, fid, 1, vec![50], 50, 30))
            .collect();
        let seg = DocStatsSegment {
            text_space_id: TextSpaceId(1),
            previous_root: None,
            records,
        };
        let bytes = encode_doc_stats_segment(&seg).unwrap();
        let per_row = bytes.len() as f64 / 522_931.0;
        assert!(
            per_row < 5.0,
            "Quora-shaped DocStats v2 encode is {per_row:.2} B/row, expected < 5",
        );
        let decoded = decode_doc_stats_segment(&bytes).unwrap();
        assert_eq!(decoded.records.len(), 522_931);
        assert_eq!(decoded.records[0].frame_id.0, 1);
        assert_eq!(decoded.records[522_930].frame_id.0, 522_931);
    }
}
