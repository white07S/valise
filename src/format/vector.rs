// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! `VectorDataSegment` payload codec, spec §14.2 (magic `NXVD`).
//!
//! The header is fixed size; the two blocks (`ids`, `base`) follow at
//! offsets declared in the header. Each block is a fixed-stride array
//! indexed by `ordinal_in_segment`, so readers seek to a specific vector
//! by direct offset arithmetic with no scan.
//!
//! Post-consolidation: residual storage is gone. The single production
//! codec (`QamLloydMax`) has no residual, and the trailing
//! `residual_block_offset` field in the header is reserved as zero.

use crate::{
    error::{Error, Result},
    format::{CodecId, EmbeddingSpaceId, VectorId},
};

const MAGIC: [u8; 4] = *b"VLVD";
const VERSION: u16 = 1;
const HEADER_BYTES: usize = 34;
const VECTOR_ID_BYTES: usize = 8;

pub(crate) struct VectorDataSegmentWriter {
    embedding_space_id: EmbeddingSpaceId,
    codec_id: CodecId,
    base_bytes_per_vector: usize,
    ids: Vec<VectorId>,
    base_block: Vec<u8>,
}

impl VectorDataSegmentWriter {
    pub(crate) fn new(
        embedding_space_id: EmbeddingSpaceId,
        codec_id: CodecId,
        base_bytes_per_vector: usize,
    ) -> Self {
        Self {
            embedding_space_id,
            codec_id,
            base_bytes_per_vector,
            ids: Vec::new(),
            base_block: Vec::new(),
        }
    }

    pub(crate) fn item_count(&self) -> u32 {
        self.ids.len() as u32
    }

    pub(crate) fn append(&mut self, vector_id: VectorId, base: &[u8]) -> Result<u32> {
        if base.len() != self.base_bytes_per_vector {
            return Err(Error::Format(format!(
                "VectorDataSegment append: base length mismatch (expected {}, got {})",
                self.base_bytes_per_vector,
                base.len()
            )));
        }
        let ordinal = u32::try_from(self.ids.len())
            .map_err(|_| Error::Format("VectorDataSegment item_count overflow".into()))?;
        self.ids.push(vector_id);
        self.base_block.extend_from_slice(base);
        Ok(ordinal)
    }

    pub(crate) fn finish(self) -> Result<Vec<u8>> {
        let item_count = u32::try_from(self.ids.len())
            .map_err(|_| Error::Format("VectorDataSegment item_count overflow".into()))?;

        let ids_block_offset = HEADER_BYTES;
        let ids_block_len = (item_count as usize) * VECTOR_ID_BYTES;
        let base_block_offset = ids_block_offset + ids_block_len;
        let base_block_len = (item_count as usize) * self.base_bytes_per_vector;
        let total_len = base_block_offset + base_block_len;

        let ids_block_offset_u32 = u32::try_from(ids_block_offset)
            .map_err(|_| Error::Format("ids_block_offset overflow".into()))?;
        let base_block_offset_u32 = u32::try_from(base_block_offset)
            .map_err(|_| Error::Format("base_block_offset overflow".into()))?;
        let flags: u32 = 0;
        let residual_block_offset_u32: u32 = 0;

        let mut buf = Vec::with_capacity(total_len);
        buf.extend_from_slice(&MAGIC);
        buf.extend_from_slice(&VERSION.to_le_bytes());
        buf.extend_from_slice(&self.embedding_space_id.0.to_le_bytes());
        buf.extend_from_slice(&self.codec_id.0.to_le_bytes());
        buf.extend_from_slice(&item_count.to_le_bytes());
        buf.extend_from_slice(&flags.to_le_bytes());
        buf.extend_from_slice(&ids_block_offset_u32.to_le_bytes());
        buf.extend_from_slice(&base_block_offset_u32.to_le_bytes());
        buf.extend_from_slice(&residual_block_offset_u32.to_le_bytes());
        debug_assert_eq!(buf.len(), HEADER_BYTES);

        for vid in &self.ids {
            buf.extend_from_slice(&vid.0.to_le_bytes());
        }
        buf.extend_from_slice(&self.base_block);
        debug_assert_eq!(buf.len(), total_len);
        Ok(buf)
    }
}

#[derive(Debug)]
pub(crate) struct VectorDataSegmentReader<'a> {
    payload: &'a [u8],
    item_count: u32,
    base_bytes_per_vector: usize,
    ids_offset: usize,
    base_offset: usize,
}

impl<'a> VectorDataSegmentReader<'a> {
    pub(crate) fn open(payload: &'a [u8], base_bytes_per_vector: usize) -> Result<Self> {
        if payload.len() < HEADER_BYTES {
            return Err(Error::Format(format!(
                "VectorDataSegment payload truncated (need {HEADER_BYTES} bytes for header, got {})",
                payload.len()
            )));
        }
        if payload[0..4] != MAGIC {
            return Err(Error::Format("VectorDataSegment magic mismatch".into()));
        }
        let version = u16::from_le_bytes(payload[4..6].try_into().unwrap());
        if version != VERSION {
            return Err(Error::Format(format!(
                "VectorDataSegment version mismatch: expected {VERSION}, got {version}"
            )));
        }
        // Bytes 6..14 carry (embedding_space_id, codec_id); the reader's
        // callers resolve both from the catalog SegmentRef, so they are
        // not re-validated here.
        let item_count = u32::from_le_bytes(payload[14..18].try_into().unwrap());
        let flags = u32::from_le_bytes(payload[18..22].try_into().unwrap());
        let ids_offset = u32::from_le_bytes(payload[22..26].try_into().unwrap()) as usize;
        let base_offset = u32::from_le_bytes(payload[26..30].try_into().unwrap()) as usize;
        let residual_offset = u32::from_le_bytes(payload[30..34].try_into().unwrap()) as usize;

        if flags != 0 {
            return Err(Error::Format(format!(
                "VectorDataSegment flags reserved (must be 0), got {flags:#x}"
            )));
        }
        if residual_offset != 0 {
            return Err(Error::Format(
                "VectorDataSegment residual_block_offset reserved (must be 0)".into(),
            ));
        }

        let n = item_count as usize;
        let expected_ids_offset = HEADER_BYTES;
        let expected_base_offset = expected_ids_offset + n * VECTOR_ID_BYTES;
        if ids_offset != expected_ids_offset {
            return Err(Error::Format(format!(
                "VectorDataSegment ids_block_offset mismatch: expected {expected_ids_offset}, got {ids_offset}"
            )));
        }
        if base_offset != expected_base_offset {
            return Err(Error::Format(format!(
                "VectorDataSegment base_block_offset mismatch: expected {expected_base_offset}, got {base_offset}"
            )));
        }

        let total_expected = expected_base_offset + n * base_bytes_per_vector;
        if payload.len() != total_expected {
            return Err(Error::Format(format!(
                "VectorDataSegment payload length mismatch: expected {total_expected}, got {}",
                payload.len()
            )));
        }

        Ok(Self {
            payload,
            item_count,
            base_bytes_per_vector,
            ids_offset,
            base_offset,
        })
    }

    pub(crate) fn vector_id(&self, ordinal: u32) -> Result<VectorId> {
        self.check_ordinal(ordinal)?;
        let offset = self.ids_offset + (ordinal as usize) * VECTOR_ID_BYTES;
        let bytes: [u8; 8] = self.payload[offset..offset + 8]
            .try_into()
            .map_err(|_| Error::Format("vector id slice truncated".into()))?;
        Ok(VectorId(u64::from_le_bytes(bytes)))
    }

    pub(crate) fn base(&self, ordinal: u32) -> Result<&[u8]> {
        self.check_ordinal(ordinal)?;
        let offset = self.base_offset + (ordinal as usize) * self.base_bytes_per_vector;
        Ok(&self.payload[offset..offset + self.base_bytes_per_vector])
    }

    fn check_ordinal(&self, ordinal: u32) -> Result<()> {
        if ordinal >= self.item_count {
            return Err(Error::Format(format!(
                "VectorDataSegment ordinal {ordinal} out of range (item_count = {})",
                self.item_count
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip_n_vectors(n: u32, base_bytes: usize) {
        let mut writer = VectorDataSegmentWriter::new(EmbeddingSpaceId(2), CodecId(7), base_bytes);
        let mut expected: Vec<(VectorId, Vec<u8>)> = Vec::new();
        for i in 0..n {
            let base: Vec<u8> = (0..base_bytes).map(|k| (k as u8) ^ (i as u8)).collect();
            let id = VectorId((i as u64) + 100);
            let ordinal = writer.append(id, &base).expect("append");
            assert_eq!(ordinal, i);
            expected.push((id, base));
        }
        let payload = writer.finish().expect("finish");

        let reader = VectorDataSegmentReader::open(&payload, base_bytes).expect("open");
        // Header bytes 6..18: embedding_space_id, codec_id, item_count (LE).
        assert_eq!(u32::from_le_bytes(payload[6..10].try_into().unwrap()), 2);
        assert_eq!(u32::from_le_bytes(payload[10..14].try_into().unwrap()), 7);
        assert_eq!(u32::from_le_bytes(payload[14..18].try_into().unwrap()), n);
        for (i, (id, base)) in expected.iter().enumerate() {
            let i = i as u32;
            assert_eq!(reader.vector_id(i).unwrap(), *id);
            assert_eq!(reader.base(i).unwrap(), base.as_slice());
        }
    }

    #[test]
    fn roundtrip_basic() {
        roundtrip_n_vectors(100, 128);
    }

    #[test]
    fn empty_segment_writes_header_only() {
        let writer = VectorDataSegmentWriter::new(EmbeddingSpaceId(1), CodecId(1), 16);
        let payload = writer.finish().expect("finish");
        assert_eq!(payload.len(), HEADER_BYTES);
        let reader = VectorDataSegmentReader::open(&payload, 16).expect("open");
        assert_eq!(u32::from_le_bytes(payload[14..18].try_into().unwrap()), 0);
        assert!(reader.base(0).is_err(), "no ordinals in an empty segment");
    }

    #[test]
    fn append_rejects_base_length_mismatch() {
        let mut writer = VectorDataSegmentWriter::new(EmbeddingSpaceId(1), CodecId(1), 16);
        let err = writer
            .append(VectorId(1), &[0u8; 10])
            .expect_err("base mismatch");
        assert!(err.to_string().contains("base length"));
    }

    #[test]
    fn reader_rejects_magic_mismatch() {
        let writer = VectorDataSegmentWriter::new(EmbeddingSpaceId(1), CodecId(1), 16);
        let mut payload = writer.finish().expect("finish");
        payload[0] = b'X';
        let err = VectorDataSegmentReader::open(&payload, 16).expect_err("magic");
        assert!(err.to_string().contains("magic"));
    }

    #[test]
    fn reader_rejects_version_mismatch() {
        let writer = VectorDataSegmentWriter::new(EmbeddingSpaceId(1), CodecId(1), 16);
        let mut payload = writer.finish().expect("finish");
        payload[4..6].copy_from_slice(&99u16.to_le_bytes());
        let err = VectorDataSegmentReader::open(&payload, 16).expect_err("version");
        assert!(err.to_string().contains("version"));
    }

    #[test]
    fn reader_rejects_ordinal_out_of_range() {
        let mut writer = VectorDataSegmentWriter::new(EmbeddingSpaceId(1), CodecId(1), 16);
        writer.append(VectorId(1), &[0u8; 16]).unwrap();
        let payload = writer.finish().expect("finish");
        let reader = VectorDataSegmentReader::open(&payload, 16).expect("open");
        let err = reader.vector_id(5).expect_err("ordinal out of range");
        assert!(err.to_string().contains("ordinal"));
    }

    #[test]
    fn reader_rejects_inconsistent_offsets() {
        let mut writer = VectorDataSegmentWriter::new(EmbeddingSpaceId(1), CodecId(1), 16);
        writer.append(VectorId(1), &[0u8; 16]).unwrap();
        let mut payload = writer.finish().expect("finish");
        // Stomp on base_block_offset (offset 26..30 in header).
        payload[26..30].copy_from_slice(&999u32.to_le_bytes());
        let err = VectorDataSegmentReader::open(&payload, 16).expect_err("offset");
        assert!(err.to_string().contains("base_block_offset"));
    }

    #[test]
    fn reader_rejects_truncated_payload() {
        let mut writer = VectorDataSegmentWriter::new(EmbeddingSpaceId(1), CodecId(1), 16);
        writer.append(VectorId(1), &[0u8; 16]).unwrap();
        let payload = writer.finish().expect("finish");
        let err = VectorDataSegmentReader::open(&payload[..payload.len() - 5], 16)
            .expect_err("truncated");
        assert!(err.to_string().contains("length"));
    }

    #[test]
    fn reader_rejects_nonzero_flags() {
        let writer = VectorDataSegmentWriter::new(EmbeddingSpaceId(1), CodecId(1), 16);
        let mut payload = writer.finish().expect("finish");
        payload[18..22].copy_from_slice(&1u32.to_le_bytes());
        let err = VectorDataSegmentReader::open(&payload, 16).expect_err("flags");
        assert!(err.to_string().contains("flags"));
    }
}
