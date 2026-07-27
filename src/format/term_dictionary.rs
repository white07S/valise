// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! `TermDictionarySegment` payload codec, spec §12.1 (magic `VLTD`).
//!
//! Each delta segment carries only newly-introduced `term_id` records for one
//! `text_space_id`; running `df` / `cf` are recomputed by walking the chain
//! at read time (spec §12.0.4). The `previous_root` field links to the prior
//! head of the chain.

use crate::{
    error::{Error, Result},
    format::{TextSpaceId, segment::SegmentRef, text::TermDictionaryEntry},
};

#[cfg(test)]
use crate::format::TermId;

const MAGIC: [u8; 4] = *b"VLTD";
const VERSION: u16 = 1;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct TermDictionarySegment {
    pub(crate) text_space_id: TextSpaceId,
    pub(crate) previous_root: Option<SegmentRef>,
    pub(crate) records: Vec<TermDictionaryEntry>,
}

pub(crate) fn encode_term_dictionary_segment(seg: &TermDictionarySegment) -> Result<Vec<u8>> {
    let mut buf = Vec::new();
    buf.extend_from_slice(&MAGIC);
    buf.extend_from_slice(&VERSION.to_le_bytes());
    buf.extend_from_slice(&seg.text_space_id.0.to_le_bytes());
    let record_count = u32::try_from(seg.records.len())
        .map_err(|_| Error::Format("TermDictionarySegment record count overflow".into()))?;
    buf.extend_from_slice(&record_count.to_le_bytes());
    write_optional_segment_ref(&mut buf, seg.previous_root)?;
    for entry in &seg.records {
        buf.extend_from_slice(&entry.term_id.0.to_le_bytes());
        let term_bytes_len = u32::try_from(entry.term_bytes.len()).map_err(|_| {
            Error::Format(format!(
                "TermDictionaryEntry term_bytes length {} exceeds u32",
                entry.term_bytes.len()
            ))
        })?;
        buf.extend_from_slice(&term_bytes_len.to_le_bytes());
        buf.extend_from_slice(&entry.term_bytes);
        buf.extend_from_slice(&entry.collection_freq.to_le_bytes());
        buf.extend_from_slice(&entry.doc_freq.to_le_bytes());
    }
    Ok(buf)
}

#[cfg(test)]
pub(crate) fn decode_term_dictionary_segment(bytes: &[u8]) -> Result<TermDictionarySegment> {
    let mut cur = Cursor::new(bytes);
    let magic = cur.read_array::<4>()?;
    if magic != MAGIC {
        return Err(Error::Format(format!(
            "TermDictionarySegment magic mismatch: expected VLTD, got {magic:?}"
        )));
    }
    let version = cur.read_u16()?;
    if version != VERSION {
        return Err(Error::Format(format!(
            "TermDictionarySegment version mismatch: expected {VERSION}, got {version}"
        )));
    }
    let text_space_id = TextSpaceId(cur.read_u32()?);
    let record_count = cur.read_u32()? as usize;
    let previous_root = read_optional_segment_ref(&mut cur)?;

    let mut records = Vec::with_capacity(record_count);
    for _ in 0..record_count {
        let term_id = TermId(cur.read_u32()?);
        let term_bytes_len = cur.read_u32()? as usize;
        let term_bytes = cur.read_slice(term_bytes_len)?.to_vec();
        let collection_freq = cur.read_u64()?;
        let doc_freq = cur.read_u32()?;
        records.push(TermDictionaryEntry {
            term_id,
            term_bytes,
            collection_freq,
            doc_freq,
        });
    }

    if !cur.is_empty() {
        return Err(Error::Format(format!(
            "TermDictionarySegment: {} trailing bytes",
            cur.remaining()
        )));
    }

    Ok(TermDictionarySegment {
        text_space_id,
        previous_root,
        records,
    })
}

fn write_optional_segment_ref(buf: &mut Vec<u8>, maybe_ref: Option<SegmentRef>) -> Result<()> {
    if let Some(r) = maybe_ref {
        buf.push(1u8);
        buf.extend_from_slice(&r.segment_id.0.to_le_bytes());
        buf.extend_from_slice(&r.offset.to_le_bytes());
        buf.extend_from_slice(&r.length.to_le_bytes());
        buf.extend_from_slice(&r.checksum);
    } else {
        buf.push(0u8);
    }
    Ok(())
}

#[cfg(test)]
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
            "previous_root presence flag must be 0 or 1, got {other}"
        ))),
    }
}

#[cfg(test)]
struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

#[cfg(test)]
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
                "TermDictionarySegment truncated: need {N} bytes at offset {}",
                self.pos
            )));
        }
        let mut out = [0u8; N];
        out.copy_from_slice(&self.bytes[self.pos..self.pos + N]);
        self.pos += N;
        Ok(out)
    }

    fn read_slice(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.pos + n > self.bytes.len() {
            return Err(Error::Format(format!(
                "TermDictionarySegment truncated: need {n} bytes at offset {}",
                self.pos
            )));
        }
        let s = &self.bytes[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
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

    fn sample_entry(term_id: u32, term: &str, cf: u64, df: u32) -> TermDictionaryEntry {
        TermDictionaryEntry {
            term_id: TermId(term_id),
            term_bytes: term.as_bytes().to_vec(),
            collection_freq: cf,
            doc_freq: df,
        }
    }

    fn sample_seg_ref() -> SegmentRef {
        SegmentRef {
            segment_id: SegmentId(42),
            offset: 4096,
            length: 1024,
            checksum: [0xAB; 32],
        }
    }

    #[test]
    fn roundtrip_empty_segment_no_previous_root() {
        let seg = TermDictionarySegment {
            text_space_id: TextSpaceId(1),
            previous_root: None,
            records: Vec::new(),
        };
        let bytes = encode_term_dictionary_segment(&seg).expect("encode");
        let decoded = decode_term_dictionary_segment(&bytes).expect("decode");
        assert_eq!(decoded, seg);
    }

    #[test]
    fn roundtrip_with_records_and_previous_root() {
        let seg = TermDictionarySegment {
            text_space_id: TextSpaceId(7),
            previous_root: Some(sample_seg_ref()),
            records: vec![
                sample_entry(1, "alpha", 100, 50),
                sample_entry(2, "beta", 200, 75),
                sample_entry(3, "café", 30, 20), // multi-byte UTF-8
            ],
        };
        let bytes = encode_term_dictionary_segment(&seg).expect("encode");
        let decoded = decode_term_dictionary_segment(&bytes).expect("decode");
        assert_eq!(decoded, seg);
    }

    #[test]
    fn rejects_magic_mismatch() {
        let seg = TermDictionarySegment {
            text_space_id: TextSpaceId(1),
            previous_root: None,
            records: vec![sample_entry(1, "a", 1, 1)],
        };
        let mut bytes = encode_term_dictionary_segment(&seg).unwrap();
        bytes[0] = b'X';
        let err = decode_term_dictionary_segment(&bytes).expect_err("magic");
        assert!(err.to_string().contains("magic"));
    }

    #[test]
    fn rejects_version_mismatch() {
        let seg = TermDictionarySegment {
            text_space_id: TextSpaceId(1),
            previous_root: None,
            records: vec![sample_entry(1, "a", 1, 1)],
        };
        let mut bytes = encode_term_dictionary_segment(&seg).unwrap();
        bytes[4..6].copy_from_slice(&99u16.to_le_bytes());
        let err = decode_term_dictionary_segment(&bytes).expect_err("version");
        assert!(err.to_string().contains("version"));
    }

    #[test]
    fn rejects_trailing_bytes() {
        let seg = TermDictionarySegment {
            text_space_id: TextSpaceId(1),
            previous_root: None,
            records: vec![sample_entry(1, "a", 1, 1)],
        };
        let mut bytes = encode_term_dictionary_segment(&seg).unwrap();
        bytes.push(0xAB);
        let err = decode_term_dictionary_segment(&bytes).expect_err("trailing");
        assert!(err.to_string().contains("trailing"));
    }

    #[test]
    fn rejects_truncated() {
        let seg = TermDictionarySegment {
            text_space_id: TextSpaceId(1),
            previous_root: None,
            records: vec![sample_entry(1, "abc", 1, 1)],
        };
        let bytes = encode_term_dictionary_segment(&seg).unwrap();
        let truncated = &bytes[..bytes.len() - 5];
        let err = decode_term_dictionary_segment(truncated).expect_err("truncated");
        assert!(err.to_string().contains("truncated"));
    }

    #[test]
    fn empty_term_bytes_roundtrip() {
        // Edge: term_bytes of zero length. Spec doesn't reject; codec must
        // round-trip cleanly.
        let seg = TermDictionarySegment {
            text_space_id: TextSpaceId(1),
            previous_root: None,
            records: vec![sample_entry(1, "", 1, 1)],
        };
        let bytes = encode_term_dictionary_segment(&seg).unwrap();
        let decoded = decode_term_dictionary_segment(&bytes).unwrap();
        assert_eq!(decoded, seg);
    }
}
