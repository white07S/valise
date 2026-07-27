// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Generic append-only segment model, spec §9.

use crate::{
    error::{Error, Result},
    format::{Checksum, SegmentId, blake3_checksum},
};

pub const SEGMENT_MAGIC: [u8; 4] = *b"VLSG";
pub const SEGMENT_HEADER_SIZE: usize = 76;
pub const SEGMENT_VERSION: u16 = 1;

#[derive(
    Clone, Copy, Debug, Eq, PartialEq, Hash, serde::Deserialize, serde::Serialize, Ord, PartialOrd,
)]
#[repr(u16)]
pub enum SegmentType {
    Payload = 0x0001,
    FrameCatalog = 0x0002,
    CollectionCatalog = 0x0003,
    TextSpaceCatalog = 0x0004,
    AnalyzerCatalog = 0x0005,
    FieldSchemaCatalog = 0x0006,
    RetrievalProfileCatalog = 0x0007,
    EmbeddingSpaceCatalog = 0x0008,
    CodecCatalog = 0x0009,
    VectorCatalog = 0x000A,
    FusionProfileCatalog = 0x000C,
    TermDictionary = 0x0010,
    Postings = 0x0011,
    Positions = 0x0012,
    DocStats = 0x0013,
    TokenSet = 0x0014,
    MinHashSignature = 0x0015,
    ForwardIndex = 0x0016,
    RawText = 0x0017,
    RawTextDictionary = 0x0018,
    VectorData = 0x0020,
    CollectionFilter = 0x0022,
    CodecParams = 0x0023,
    /// Phase 3: persisted vote-index CSR (`VVI1` magic). One per
    /// `(embedding_space, qam_codec)` pair; pointed at by a
    TimeIndex = 0x0030,
    SegmentRegistry = 0x00FE,
    Extension = 0x00FF,
    /// Batched per-frame metadata blobs (external identity key / URI,
    /// title, arbitrary metadata) referenced by `FrameDesc.uri_ref` /
    /// `title_ref` / `metadata_ref` via a `MetadataRef`. Written like
    /// `Payload`: one segment per flush batch, not one per frame.
    ///
    /// **Declared last on purpose.** `SegmentType` derives `serde` and is
    /// encoded by *declaration-order variant index* (e.g. inside the
    /// SegmentRegistry), so a new variant MUST be appended here — never
    /// inserted next to its `0x0019` discriminant — or every later
    /// variant's serde index shifts and existing files fail to decode.
    /// The explicit discriminant keeps the on-the-wire segment-header
    /// `u16` value at `0x0019` regardless of declaration position.
    Metadata = 0x0019,
}

impl TryFrom<u16> for SegmentType {
    type Error = Error;

    fn try_from(value: u16) -> Result<Self> {
        match value {
            0x0001 => Ok(Self::Payload),
            0x0002 => Ok(Self::FrameCatalog),
            0x0003 => Ok(Self::CollectionCatalog),
            0x0004 => Ok(Self::TextSpaceCatalog),
            0x0005 => Ok(Self::AnalyzerCatalog),
            0x0006 => Ok(Self::FieldSchemaCatalog),
            0x0007 => Ok(Self::RetrievalProfileCatalog),
            0x0008 => Ok(Self::EmbeddingSpaceCatalog),
            0x0009 => Ok(Self::CodecCatalog),
            0x000A => Ok(Self::VectorCatalog),
            0x000C => Ok(Self::FusionProfileCatalog),
            0x0010 => Ok(Self::TermDictionary),
            0x0011 => Ok(Self::Postings),
            0x0012 => Ok(Self::Positions),
            0x0013 => Ok(Self::DocStats),
            0x0014 => Ok(Self::TokenSet),
            0x0015 => Ok(Self::MinHashSignature),
            0x0016 => Ok(Self::ForwardIndex),
            0x0017 => Ok(Self::RawText),
            0x0018 => Ok(Self::RawTextDictionary),
            0x0019 => Ok(Self::Metadata),
            0x0020 => Ok(Self::VectorData),
            0x0022 => Ok(Self::CollectionFilter),
            0x0023 => Ok(Self::CodecParams),
            0x0030 => Ok(Self::TimeIndex),
            0x00FE => Ok(Self::SegmentRegistry),
            0x00FF => Ok(Self::Extension),
            _ => Err(Error::Format(format!("unknown segment type: {value:#06x}"))),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
#[repr(u16)]
pub enum Compression {
    None = 0,
    Zstd = 1,
    /// Reserved discriminant. **No encoder emits it and no decoder can
    /// read it** — see `Compression::ensure_decodable`. It exists so the
    /// wire value 2 stays claimed for a future LZ4 codec rather than being
    /// handed to something else.
    Lz4 = 2,
}

impl Compression {
    /// Reject a segment this build cannot actually decompress.
    ///
    /// The read paths are written as `if compression == Zstd { decompress }
    /// else { use the bytes as-is }`. Without this guard an `Lz4` segment
    /// would take the else branch and compressed bytes would be returned as
    /// though they were plaintext — silent corruption rather than an error.
    pub(crate) fn ensure_decodable(self) -> Result<()> {
        match self {
            Self::None | Self::Zstd => Ok(()),
            Self::Lz4 => Err(Error::Unsupported(
                "segment is LZ4-compressed; this build only decodes zstd (wire code 2 is reserved but unimplemented)"
                    .into(),
            )),
        }
    }
}

impl TryFrom<u16> for Compression {
    type Error = Error;

    fn try_from(value: u16) -> Result<Self> {
        match value {
            0 => Ok(Self::None),
            1 => Ok(Self::Zstd),
            2 => Ok(Self::Lz4),
            _ => Err(Error::Format(format!("unknown compression code: {value}"))),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct SegmentRef {
    pub segment_id: SegmentId,
    pub offset: u64,
    pub length: u64,
    pub checksum: Checksum,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SegmentHeader {
    pub segment_version: u16,
    pub segment_type: SegmentType,
    pub segment_id: SegmentId,
    pub flags: u32,
    pub item_count: u32,
    pub compression: Compression,
    pub uncompressed_length: u64,
    pub payload_length: u64,
    pub payload_checksum: Checksum,
}

impl SegmentHeader {
    #[must_use]
    pub fn new(
        segment_type: SegmentType,
        segment_id: SegmentId,
        item_count: u32,
        compression: Compression,
        uncompressed_length: u64,
        payload: &[u8],
    ) -> Self {
        Self {
            segment_version: SEGMENT_VERSION,
            segment_type,
            segment_id,
            flags: 0,
            item_count,
            compression,
            uncompressed_length,
            payload_length: payload.len() as u64,
            payload_checksum: blake3_checksum(payload),
        }
    }

    pub fn validate_payload(&self, payload: &[u8]) -> Result<()> {
        self.validate_header()?;
        if self.payload_length != payload.len() as u64 {
            return Err(Error::Integrity(format!(
                "segment payload length mismatch: expected {}, got {}",
                self.payload_length,
                payload.len()
            )));
        }
        let actual = blake3_checksum(payload);
        if self.payload_checksum != actual {
            return Err(Error::Integrity("segment payload checksum mismatch".into()));
        }
        Ok(())
    }

    pub fn validate_header(&self) -> Result<()> {
        if self.compression == Compression::None && self.uncompressed_length != self.payload_length
        {
            return Err(Error::Format(format!(
                "uncompressed length must equal payload length when compression is None: {} != {}",
                self.uncompressed_length, self.payload_length
            )));
        }
        Ok(())
    }
}

pub struct SegmentHeaderCodec;

impl SegmentHeaderCodec {
    pub fn encode(header: &SegmentHeader) -> [u8; SEGMENT_HEADER_SIZE] {
        let mut buf = [0u8; SEGMENT_HEADER_SIZE];
        buf[0..4].copy_from_slice(&SEGMENT_MAGIC);
        put_u16(&mut buf, 4, header.segment_version);
        put_u16(&mut buf, 6, header.segment_type as u16);
        put_u64(&mut buf, 8, header.segment_id.0);
        put_u32(&mut buf, 16, header.flags);
        put_u32(&mut buf, 20, header.item_count);
        put_u16(&mut buf, 24, header.compression as u16);
        put_u16(&mut buf, 26, 0);
        put_u64(&mut buf, 28, header.uncompressed_length);
        put_u64(&mut buf, 36, header.payload_length);
        buf[44..76].copy_from_slice(&header.payload_checksum);
        buf
    }

    pub fn decode(bytes: &[u8; SEGMENT_HEADER_SIZE]) -> Result<SegmentHeader> {
        if bytes[0..4] != SEGMENT_MAGIC {
            return Err(Error::Format("segment magic mismatch".into()));
        }
        let reserved = get_u16(bytes, 26)?;
        if reserved != 0 {
            return Err(Error::Format("segment reserved field must be zero".into()));
        }
        let version = get_u16(bytes, 4)?;
        if version != SEGMENT_VERSION {
            return Err(Error::Format(format!(
                "unsupported segment version: {version}"
            )));
        }
        let payload_checksum: Checksum = bytes[44..76]
            .try_into()
            .map_err(|_| Error::Format("segment checksum truncated".into()))?;

        let header = SegmentHeader {
            segment_version: version,
            segment_type: SegmentType::try_from(get_u16(bytes, 6)?)?,
            segment_id: SegmentId(get_u64(bytes, 8)?),
            flags: get_u32(bytes, 16)?,
            item_count: get_u32(bytes, 20)?,
            compression: Compression::try_from(get_u16(bytes, 24)?)?,
            uncompressed_length: get_u64(bytes, 28)?,
            payload_length: get_u64(bytes, 36)?,
            payload_checksum,
        };
        header.validate_header()?;
        Ok(header)
    }
}

fn put_u16(buf: &mut [u8], offset: usize, value: u16) {
    buf[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}

fn put_u32(buf: &mut [u8], offset: usize, value: u32) {
    buf[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn put_u64(buf: &mut [u8], offset: usize, value: u64) {
    buf[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

fn get_u16(bytes: &[u8], offset: usize) -> Result<u16> {
    Ok(u16::from_le_bytes(
        bytes[offset..offset + 2]
            .try_into()
            .map_err(|_| Error::Format("segment u16 field truncated".into()))?,
    ))
}

fn get_u32(bytes: &[u8], offset: usize) -> Result<u32> {
    Ok(u32::from_le_bytes(
        bytes[offset..offset + 4]
            .try_into()
            .map_err(|_| Error::Format("segment u32 field truncated".into()))?,
    ))
}

fn get_u64(bytes: &[u8], offset: usize) -> Result<u64> {
    Ok(u64::from_le_bytes(
        bytes[offset..offset + 8]
            .try_into()
            .map_err(|_| Error::Format("segment u64 field truncated".into()))?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segment_header_roundtrips() {
        let payload = b"payload bytes";
        let header = SegmentHeader::new(
            SegmentType::Payload,
            SegmentId(42),
            1,
            Compression::None,
            payload.len() as u64,
            payload,
        );

        let encoded = SegmentHeaderCodec::encode(&header);
        let decoded = SegmentHeaderCodec::decode(&encoded).expect("decode segment header");

        assert_eq!(decoded, header);
        decoded
            .validate_payload(payload)
            .expect("checksum should validate");
    }

    #[test]
    fn rejects_bad_payload_checksum() {
        let payload = b"payload bytes";
        let header = SegmentHeader::new(
            SegmentType::Payload,
            SegmentId(42),
            1,
            Compression::None,
            payload.len() as u64,
            payload,
        );

        let err = header
            .validate_payload(b"other bytes")
            .expect_err("checksum should fail");
        assert!(err.to_string().contains("length") || err.to_string().contains("checksum"));
    }

    #[test]
    fn rejects_unknown_segment_type() {
        let payload = b"payload bytes";
        let header = SegmentHeader::new(
            SegmentType::Payload,
            SegmentId(42),
            1,
            Compression::None,
            payload.len() as u64,
            payload,
        );
        let mut encoded = SegmentHeaderCodec::encode(&header);
        encoded[6..8].copy_from_slice(&0xBEEFu16.to_le_bytes());

        let err = SegmentHeaderCodec::decode(&encoded).expect_err("unknown type should fail");
        assert!(err.to_string().contains("unknown segment type"));
    }

    #[test]
    fn segment_type_discriminants_match_spec() {
        assert_eq!(SegmentType::Payload as u16, 0x0001);
        assert_eq!(SegmentType::FrameCatalog as u16, 0x0002);
        assert_eq!(SegmentType::CollectionCatalog as u16, 0x0003);
        assert_eq!(SegmentType::TextSpaceCatalog as u16, 0x0004);
        assert_eq!(SegmentType::AnalyzerCatalog as u16, 0x0005);
        assert_eq!(SegmentType::FieldSchemaCatalog as u16, 0x0006);
        assert_eq!(SegmentType::RetrievalProfileCatalog as u16, 0x0007);
        assert_eq!(SegmentType::EmbeddingSpaceCatalog as u16, 0x0008);
        assert_eq!(SegmentType::CodecCatalog as u16, 0x0009);
        assert_eq!(SegmentType::VectorCatalog as u16, 0x000A);
        assert_eq!(SegmentType::TermDictionary as u16, 0x0010);
        assert_eq!(SegmentType::Postings as u16, 0x0011);
        assert_eq!(SegmentType::Positions as u16, 0x0012);
        assert_eq!(SegmentType::DocStats as u16, 0x0013);
        assert_eq!(SegmentType::TokenSet as u16, 0x0014);
        assert_eq!(SegmentType::MinHashSignature as u16, 0x0015);
        assert_eq!(SegmentType::ForwardIndex as u16, 0x0016);
        assert_eq!(SegmentType::RawText as u16, 0x0017);
        assert_eq!(SegmentType::RawTextDictionary as u16, 0x0018);
        assert_eq!(SegmentType::Metadata as u16, 0x0019);
        assert_eq!(SegmentType::VectorData as u16, 0x0020);
        assert_eq!(SegmentType::CollectionFilter as u16, 0x0022);
        assert_eq!(SegmentType::CodecParams as u16, 0x0023);
        assert_eq!(SegmentType::TimeIndex as u16, 0x0030);
        assert_eq!(SegmentType::SegmentRegistry as u16, 0x00FE);
        assert_eq!(SegmentType::Extension as u16, 0x00FF);
    }

    #[test]
    fn all_segment_types_roundtrip() {
        let all = [
            SegmentType::Payload,
            SegmentType::FrameCatalog,
            SegmentType::CollectionCatalog,
            SegmentType::TextSpaceCatalog,
            SegmentType::AnalyzerCatalog,
            SegmentType::FieldSchemaCatalog,
            SegmentType::RetrievalProfileCatalog,
            SegmentType::EmbeddingSpaceCatalog,
            SegmentType::CodecCatalog,
            SegmentType::VectorCatalog,
            SegmentType::TermDictionary,
            SegmentType::Postings,
            SegmentType::Positions,
            SegmentType::DocStats,
            SegmentType::TokenSet,
            SegmentType::MinHashSignature,
            SegmentType::ForwardIndex,
            SegmentType::RawText,
            SegmentType::RawTextDictionary,
            SegmentType::Metadata,
            SegmentType::VectorData,
            SegmentType::CollectionFilter,
            SegmentType::CodecParams,
            SegmentType::TimeIndex,
            SegmentType::SegmentRegistry,
            SegmentType::Extension,
        ];
        let payload = b"p";
        for ty in all {
            let header = SegmentHeader::new(
                ty,
                SegmentId(7),
                0,
                Compression::None,
                payload.len() as u64,
                payload,
            );
            let encoded = SegmentHeaderCodec::encode(&header);
            let decoded = SegmentHeaderCodec::decode(&encoded).expect("decode");
            assert_eq!(decoded.segment_type, ty);
        }
    }

    #[test]
    fn rejects_invalid_none_compression_lengths() {
        let payload = b"payload bytes";
        let header = SegmentHeader {
            uncompressed_length: payload.len() as u64 + 1,
            ..SegmentHeader::new(
                SegmentType::Payload,
                SegmentId(42),
                1,
                Compression::None,
                payload.len() as u64,
                payload,
            )
        };
        let encoded = SegmentHeaderCodec::encode(&header);

        let err = SegmentHeaderCodec::decode(&encoded).expect_err("invalid length should fail");
        assert!(err.to_string().contains("uncompressed length"));
    }
}
