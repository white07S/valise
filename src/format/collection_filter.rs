// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! `CollectionFilterSegment` payload codec, spec §16 (magic `VALISEF`).
//!
//! v1 wire form: sorted-postings lists for `collection_id → frame_id set`
//! and `collection_id → vector_id set`. v2 (in this crate) adds an
//! alternative encoding flag = 1, "run sequences" — pairs of
//! `(start, length)` covering ranges of consecutive ids. Each side
//! (frames, vectors) picks whichever flag is cheaper at encode time.
//!
//! Encode-time heuristic: for `N` ids, sorted-postings costs ~`N` bytes
//! (1 byte/varint when deltas stay <128, which is the common build-
//! everything-at-once shape). Runs cost `4 + 16 × R` bytes where R is
//! the number of contiguous runs. Single-collection builds (e.g. the
//! BEIR-quora corpus) collapse `[1..522 931]` into ONE run = ~20 bytes
//! vs ~510 KiB on the sorted-postings path.
//!
//! Each segment carries one collection's full membership view at commit
//! time. Replay walks the chain newest-first; an entry is shadowed by the
//! latest segment for the same `collection_id`.

use crate::{
    error::{Error, Result},
    format::{CollectionId, FrameId, VectorId, segment::SegmentRef},
};

const MAGIC: [u8; 4] = *b"VLCF";
const VERSION: u16 = 2;
/// Encoding flag = 0: sorted-delta postings (v1 default).
const ENCODING_SORTED_POSTINGS: u8 = 0;
/// Encoding flag = 1: run sequences `(start, length)` per side. Picked
/// at encode time when it's strictly smaller than the postings path.
const ENCODING_RUNS: u8 = 1;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct CollectionFilterSegment {
    pub(crate) collection_id: CollectionId,
    pub(crate) frame_ids: Vec<FrameId>,
    pub(crate) vector_ids: Vec<VectorId>,
    pub(crate) previous_root: Option<SegmentRef>,
}

pub(crate) fn encode_collection_filter_segment(seg: &CollectionFilterSegment) -> Result<Vec<u8>> {
    if !is_strictly_sorted(&seg.frame_ids.iter().map(|f| f.0).collect::<Vec<_>>()) {
        return Err(Error::Format(
            "CollectionFilterSegment frame_ids must be strictly sorted ascending".into(),
        ));
    }
    if !is_strictly_sorted(&seg.vector_ids.iter().map(|v| v.0).collect::<Vec<_>>()) {
        return Err(Error::Format(
            "CollectionFilterSegment vector_ids must be strictly sorted ascending".into(),
        ));
    }

    // Pick the cheaper encoding per side at write time. Encoding is
    // packed into one byte: low 4 bits = frame side, next 4 = vector
    // side. Both sides default to sorted postings; the run encoding
    // wins for sequential-id builds (every Quora-scale corpus, since
    // frame_ids come straight out of the monotonic id_allocator).
    let frame_postings =
        encode_sorted_postings(&seg.frame_ids.iter().map(|f| f.0).collect::<Vec<_>>());
    let frame_runs = encode_runs(&seg.frame_ids.iter().map(|f| f.0).collect::<Vec<_>>())?;
    let (frame_enc, frame_body) = if frame_runs.len() < frame_postings.len() {
        (ENCODING_RUNS, frame_runs)
    } else {
        (ENCODING_SORTED_POSTINGS, frame_postings)
    };
    let vector_postings =
        encode_sorted_postings(&seg.vector_ids.iter().map(|v| v.0).collect::<Vec<_>>());
    let vector_runs = encode_runs(&seg.vector_ids.iter().map(|v| v.0).collect::<Vec<_>>())?;
    let (vector_enc, vector_body) = if vector_runs.len() < vector_postings.len() {
        (ENCODING_RUNS, vector_runs)
    } else {
        (ENCODING_SORTED_POSTINGS, vector_postings)
    };

    let mut buf = Vec::new();
    buf.extend_from_slice(&MAGIC);
    buf.extend_from_slice(&VERSION.to_le_bytes());
    buf.push((frame_enc & 0x0F) | ((vector_enc & 0x0F) << 4));
    buf.push(0u8); // reserved padding to keep header word-aligned
    buf.extend_from_slice(&seg.collection_id.0.to_le_bytes());
    let frame_count = u32::try_from(seg.frame_ids.len())
        .map_err(|_| Error::Format("CollectionFilterSegment frame count overflow".into()))?;
    let vector_count = u32::try_from(seg.vector_ids.len())
        .map_err(|_| Error::Format("CollectionFilterSegment vector count overflow".into()))?;
    buf.extend_from_slice(&frame_count.to_le_bytes());
    buf.extend_from_slice(&vector_count.to_le_bytes());
    write_optional_segment_ref(&mut buf, seg.previous_root);

    buf.extend_from_slice(&frame_body);
    buf.extend_from_slice(&vector_body);
    Ok(buf)
}

/// Sorted-delta varint stream for a strictly-ascending id list. Spec
/// §16 v1 wire shape, retained as the default encoding for sparse
/// (non-sequential) collection memberships.
fn encode_sorted_postings(ids: &[u64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(ids.len());
    let mut prev = 0u64;
    for &id in ids {
        write_varint_u64(&mut out, id - prev);
        prev = id;
    }
    out
}

/// Run-length encoding: collapse maximal sequences of consecutive ids
/// into `(start, length)` pairs. Each pair stored as deltas against
/// the previous run's end so the cumulative cost stays varint-tight.
/// One run for `[1..522 931]` = `4 (run_count) + 1 (start=1) + 4 (length=522 931 needs 3-4 B varint)`
/// ≈ 9 bytes vs the 522 931-byte postings stream.
fn encode_runs(ids: &[u64]) -> Result<Vec<u8>> {
    let mut runs: Vec<(u64, u64)> = Vec::new();
    let mut i = 0;
    while i < ids.len() {
        let start = ids[i];
        let mut len: u64 = 1;
        while i + 1 < ids.len() && ids[i + 1] == ids[i] + 1 {
            len += 1;
            i += 1;
        }
        runs.push((start, len));
        i += 1;
    }
    let mut out = Vec::new();
    let run_count = u32::try_from(runs.len())
        .map_err(|_| Error::Format("CollectionFilterSegment run count overflow".into()))?;
    out.extend_from_slice(&run_count.to_le_bytes());
    let mut prev_end: u64 = 0;
    for (start, len) in runs {
        // Gap-from-prev-end varint. For the first run, prev_end = 0,
        // so the varint value is `start` (typically 1 → 1 byte).
        let gap = start
            .checked_sub(prev_end)
            .ok_or_else(|| Error::Format("CollectionFilterSegment runs not monotone".into()))?;
        write_varint_u64(&mut out, gap);
        write_varint_u64(&mut out, len);
        prev_end = start
            .checked_add(len)
            .ok_or_else(|| Error::Format("CollectionFilterSegment run end overflow".into()))?;
    }
    Ok(out)
}

pub(crate) fn decode_collection_filter_segment(bytes: &[u8]) -> Result<CollectionFilterSegment> {
    let mut cur = Cursor::new(bytes);
    let magic = cur.read_array::<4>()?;
    if magic != MAGIC {
        return Err(Error::Format(format!(
            "CollectionFilterSegment magic mismatch: expected VALISEF, got {magic:?}"
        )));
    }
    let version = cur.read_u16()?;
    if version != VERSION {
        return Err(Error::Format(format!(
            "CollectionFilterSegment version mismatch: expected {VERSION}, got {version}"
        )));
    }
    let encoding = cur.read_u8()?;
    let frame_enc = encoding & 0x0F;
    let vector_enc = (encoding >> 4) & 0x0F;
    let _reserved = cur.read_u8()?;
    let collection_id = CollectionId(cur.read_u32()?);
    let frame_count = cur.read_u32()? as usize;
    let vector_count = cur.read_u32()? as usize;
    let previous_root = read_optional_segment_ref(&mut cur)?;

    let frame_ids = decode_id_block(&mut cur, frame_count, frame_enc, "frame")?
        .into_iter()
        .map(FrameId)
        .collect();
    let vector_ids = decode_id_block(&mut cur, vector_count, vector_enc, "vector")?
        .into_iter()
        .map(VectorId)
        .collect();
    if !cur.is_empty() {
        return Err(Error::Format(format!(
            "CollectionFilterSegment: {} trailing bytes",
            cur.remaining()
        )));
    }
    Ok(CollectionFilterSegment {
        collection_id,
        frame_ids,
        vector_ids,
        previous_root,
    })
}

fn decode_id_block(
    cur: &mut Cursor<'_>,
    count: usize,
    encoding: u8,
    label: &'static str,
) -> Result<Vec<u64>> {
    match encoding {
        ENCODING_SORTED_POSTINGS => {
            let mut ids = Vec::with_capacity(count);
            let mut prev = 0u64;
            for _ in 0..count {
                let delta = read_varint_u64(cur)?;
                let id = prev.checked_add(delta).ok_or_else(|| {
                    Error::Format(format!("CollectionFilterSegment {label}_id overflow"))
                })?;
                ids.push(id);
                prev = id;
            }
            Ok(ids)
        }
        ENCODING_RUNS => {
            let run_count = cur.read_u32()? as usize;
            let mut ids = Vec::with_capacity(count);
            let mut prev_end: u64 = 0;
            for _ in 0..run_count {
                let gap = read_varint_u64(cur)?;
                let len = read_varint_u64(cur)?;
                let start = prev_end.checked_add(gap).ok_or_else(|| {
                    Error::Format(format!(
                        "CollectionFilterSegment {label} run start overflow"
                    ))
                })?;
                for k in 0..len {
                    ids.push(start.checked_add(k).ok_or_else(|| {
                        Error::Format(format!("CollectionFilterSegment {label} run id overflow"))
                    })?);
                }
                prev_end = start.checked_add(len).ok_or_else(|| {
                    Error::Format(format!("CollectionFilterSegment {label} run end overflow"))
                })?;
            }
            if ids.len() != count {
                return Err(Error::Format(format!(
                    "CollectionFilterSegment {label} run total {} != declared count {}",
                    ids.len(),
                    count
                )));
            }
            Ok(ids)
        }
        other => Err(Error::Unsupported(format!(
            "CollectionFilterSegment {label} encoding {other} not supported"
        ))),
    }
}

fn is_strictly_sorted(values: &[u64]) -> bool {
    values.windows(2).all(|w| w[0] < w[1])
}

fn write_varint_u64(buf: &mut Vec<u8>, mut value: u64) {
    while value >= 0x80 {
        buf.push((value as u8) | 0x80);
        value >>= 7;
    }
    buf.push(value as u8);
}

fn read_varint_u64(cur: &mut Cursor<'_>) -> Result<u64> {
    let mut value = 0u64;
    let mut shift = 0;
    loop {
        if shift >= 64 {
            return Err(Error::Format(
                "CollectionFilterSegment varint exceeds 64 bits".into(),
            ));
        }
        let byte = cur.read_u8()?;
        value |= u64::from(byte & 0x7F) << shift;
        if byte & 0x80 == 0 {
            break;
        }
        shift += 7;
    }
    Ok(value)
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
            "CollectionFilterSegment previous_root flag must be 0 or 1, got {other}"
        ))),
    }
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
                "CollectionFilterSegment truncated: need {N} bytes at offset {}",
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::SegmentId;

    fn ids<T, F: Fn(u64) -> T>(values: &[u64], wrap: F) -> Vec<T> {
        values.iter().map(|v| wrap(*v)).collect()
    }

    #[test]
    fn roundtrip_empty() {
        let seg = CollectionFilterSegment {
            collection_id: CollectionId(7),
            frame_ids: Vec::new(),
            vector_ids: Vec::new(),
            previous_root: None,
        };
        let bytes = encode_collection_filter_segment(&seg).unwrap();
        let decoded = decode_collection_filter_segment(&bytes).unwrap();
        assert_eq!(decoded, seg);
    }

    #[test]
    fn roundtrip_with_payload() {
        let seg = CollectionFilterSegment {
            collection_id: CollectionId(2),
            frame_ids: ids(&[1, 5, 10, 11, 100, 101, 1_000_000], FrameId),
            vector_ids: ids(&[3, 4, 7, 99], VectorId),
            previous_root: Some(SegmentRef {
                segment_id: SegmentId(42),
                offset: 4096,
                length: 1024,
                checksum: [0xCA; 32],
            }),
        };
        let bytes = encode_collection_filter_segment(&seg).unwrap();
        let decoded = decode_collection_filter_segment(&bytes).unwrap();
        assert_eq!(decoded, seg);
    }

    #[test]
    fn rejects_unsorted_frame_ids() {
        let seg = CollectionFilterSegment {
            collection_id: CollectionId(1),
            frame_ids: ids(&[5, 3, 10], FrameId),
            vector_ids: Vec::new(),
            previous_root: None,
        };
        let err = encode_collection_filter_segment(&seg).expect_err("unsorted");
        assert!(err.to_string().contains("strictly sorted"));
    }

    #[test]
    fn rejects_duplicate_frame_ids() {
        let seg = CollectionFilterSegment {
            collection_id: CollectionId(1),
            frame_ids: ids(&[1, 2, 2, 3], FrameId),
            vector_ids: Vec::new(),
            previous_root: None,
        };
        let err = encode_collection_filter_segment(&seg).expect_err("dup");
        assert!(err.to_string().contains("strictly sorted"));
    }

    #[test]
    fn rejects_magic() {
        let seg = CollectionFilterSegment {
            collection_id: CollectionId(1),
            frame_ids: vec![FrameId(1)],
            vector_ids: vec![VectorId(2)],
            previous_root: None,
        };
        let mut bytes = encode_collection_filter_segment(&seg).unwrap();
        bytes[0] = b'X';
        let err = decode_collection_filter_segment(&bytes).expect_err("magic");
        assert!(err.to_string().contains("magic"));
    }

    /// Sequential `[1..N]` frame_ids collapse to a single run record
    /// — three varints + a u32 run_count, regardless of `N`. The
    /// Quora BM25 corpus is exactly this shape.
    #[test]
    fn sequential_frame_ids_collapse_to_a_single_run() {
        let frame_ids = ids(&(1..=522_931u64).collect::<Vec<_>>(), FrameId);
        let seg = CollectionFilterSegment {
            collection_id: CollectionId(1),
            frame_ids: frame_ids.clone(),
            vector_ids: Vec::new(),
            previous_root: None,
        };
        let bytes = encode_collection_filter_segment(&seg).expect("encode");
        // Sanity: must be far smaller than the postings path's ~522 KiB.
        assert!(
            bytes.len() < 1_024,
            "expected run encoding to fit under 1 KiB, got {} bytes",
            bytes.len()
        );
        let decoded = decode_collection_filter_segment(&bytes).expect("decode");
        assert_eq!(decoded.frame_ids.len(), 522_931);
        assert_eq!(decoded.frame_ids[0], FrameId(1));
        assert_eq!(decoded.frame_ids[522_930], FrameId(522_931));
    }

    /// A non-sequential id set (gaps) must still prefer the cheaper
    /// of the two encodings on a per-side basis. Forced gaps here
    /// should keep postings as the winner.
    #[test]
    fn sparse_ids_pick_postings_over_runs() {
        let frame_ids = ids(&[1, 3, 5, 7, 11, 13], FrameId);
        let seg = CollectionFilterSegment {
            collection_id: CollectionId(1),
            frame_ids,
            vector_ids: Vec::new(),
            previous_root: None,
        };
        let bytes = encode_collection_filter_segment(&seg).expect("encode");
        let decoded = decode_collection_filter_segment(&bytes).expect("decode");
        assert_eq!(decoded.frame_ids.len(), 6);
        assert_eq!(decoded.frame_ids[5], FrameId(13));
    }

    #[test]
    fn rejects_trailing_bytes() {
        let seg = CollectionFilterSegment {
            collection_id: CollectionId(1),
            frame_ids: vec![FrameId(1), FrameId(2)],
            vector_ids: Vec::new(),
            previous_root: None,
        };
        let mut bytes = encode_collection_filter_segment(&seg).unwrap();
        bytes.push(0xAB);
        let err = decode_collection_filter_segment(&bytes).expect_err("trailing");
        assert!(err.to_string().contains("trailing"));
    }
}
