//! Snapshot segment registry and allocator state.

use crate::{
    error::{Error, Result},
    format::{
        AnalyzerId, CodecId, CollectionId, EmbeddingSpaceId, FieldSchemaId, FrameId,
        FusionProfileId, RetrievalProfileId, SegmentId, TextSpaceId, VectorId,
        segment::{Compression, SEGMENT_HEADER_SIZE, SegmentHeader, SegmentRef, SegmentType},
    },
};

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
pub enum SegmentCatalogStatus {
    Live,
    Superseded,
    Tombstoned,
}

#[derive(Clone, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct SegmentCatalogEntry {
    pub segment_id: SegmentId,
    pub segment_type: SegmentType,
    pub root: SegmentRef,
    pub segment_version: u16,
    pub flags: u32,
    pub item_count: u32,
    pub compression: Compression,
    pub uncompressed_length: u64,
    pub payload_length: u64,
    pub status: SegmentCatalogStatus,
}

impl SegmentCatalogEntry {
    pub fn live(root: SegmentRef, header: &SegmentHeader) -> Result<Self> {
        Self::new(root, header, SegmentCatalogStatus::Live)
    }

    pub fn new(
        root: SegmentRef,
        header: &SegmentHeader,
        status: SegmentCatalogStatus,
    ) -> Result<Self> {
        let entry = Self {
            segment_id: header.segment_id,
            segment_type: header.segment_type,
            root,
            segment_version: header.segment_version,
            flags: header.flags,
            item_count: header.item_count,
            compression: header.compression,
            uncompressed_length: header.uncompressed_length,
            payload_length: header.payload_length,
            status,
        };
        entry.validate_header(header)?;
        Ok(entry)
    }

    pub fn validate(&self) -> Result<()> {
        if self.segment_id.0 == 0 {
            return Err(Error::Format("segment id must be non-zero".into()));
        }
        if self.root.segment_id != self.segment_id {
            return Err(Error::Integrity("segment catalog root id mismatch".into()));
        }
        let expected_len = (SEGMENT_HEADER_SIZE as u64)
            .checked_add(self.payload_length)
            .ok_or_else(|| Error::Format("segment catalog length overflow".into()))?;
        if self.root.length != expected_len {
            return Err(Error::Integrity(
                "segment catalog root length mismatch".into(),
            ));
        }
        if self.compression == Compression::None && self.uncompressed_length != self.payload_length
        {
            return Err(Error::Format(format!(
                "uncompressed length must equal payload length when compression is None: {} != {}",
                self.uncompressed_length, self.payload_length
            )));
        }
        Ok(())
    }

    pub fn validate_header(&self, header: &SegmentHeader) -> Result<()> {
        self.validate()?;
        if self.segment_id != header.segment_id
            || self.segment_type != header.segment_type
            || self.segment_version != header.segment_version
            || self.flags != header.flags
            || self.item_count != header.item_count
            || self.compression != header.compression
            || self.uncompressed_length != header.uncompressed_length
            || self.payload_length != header.payload_length
            || self.root.checksum != header.payload_checksum
        {
            return Err(Error::Integrity(
                "segment catalog entry/header mismatch".into(),
            ));
        }
        Ok(())
    }
}

/// In-memory + on-disk allocator state. Single canonical shape — no
/// legacy wire variants. Field order is wire-stable; new ID types
/// append at the end so a future minor can extend without breaking
/// the existing fields.
#[derive(Clone, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct IdAllocator {
    pub next_collection_id: CollectionId,
    pub next_frame_id: FrameId,
    pub next_text_space_id: TextSpaceId,
    pub next_analyzer_id: AnalyzerId,
    pub next_field_schema_id: FieldSchemaId,
    pub next_retrieval_profile_id: RetrievalProfileId,
    pub next_embedding_space_id: EmbeddingSpaceId,
    pub next_codec_id: CodecId,
    pub next_vector_id: VectorId,
    pub next_segment_id: SegmentId,
    pub next_fusion_profile_id: FusionProfileId,
}

impl Default for IdAllocator {
    fn default() -> Self {
        Self {
            next_collection_id: CollectionId(1),
            next_frame_id: FrameId(1),
            next_text_space_id: TextSpaceId(1),
            next_analyzer_id: AnalyzerId(1),
            next_field_schema_id: FieldSchemaId(1),
            next_retrieval_profile_id: RetrievalProfileId(1),
            next_embedding_space_id: EmbeddingSpaceId(1),
            next_codec_id: CodecId(1),
            next_vector_id: VectorId(1),
            next_segment_id: SegmentId(1),
            next_fusion_profile_id: FusionProfileId(1),
        }
    }
}

impl IdAllocator {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn validate(&self) -> Result<()> {
        validate_next_u32(self.next_collection_id.0, "collection")?;
        validate_next_u64(self.next_frame_id.0, "frame")?;
        validate_next_u32(self.next_text_space_id.0, "text space")?;
        validate_next_u32(self.next_analyzer_id.0, "analyzer")?;
        validate_next_u32(self.next_field_schema_id.0, "field schema")?;
        validate_next_u32(self.next_retrieval_profile_id.0, "retrieval profile")?;
        validate_next_u32(self.next_embedding_space_id.0, "embedding space")?;
        validate_next_u32(self.next_codec_id.0, "codec")?;
        validate_next_u64(self.next_vector_id.0, "vector")?;
        validate_next_u64(self.next_segment_id.0, "segment")?;
        validate_next_u32(self.next_fusion_profile_id.0, "fusion profile")?;
        Ok(())
    }

    pub fn allocate_collection_id(&mut self) -> Result<CollectionId> {
        take_u32(&mut self.next_collection_id.0, "collection").map(CollectionId)
    }

    pub fn allocate_frame_id(&mut self) -> Result<FrameId> {
        take_u64(&mut self.next_frame_id.0, "frame").map(FrameId)
    }

    pub fn allocate_segment_id(&mut self) -> Result<SegmentId> {
        take_u64(&mut self.next_segment_id.0, "segment").map(SegmentId)
    }

    pub fn allocate_text_space_id(&mut self) -> Result<TextSpaceId> {
        take_u32(&mut self.next_text_space_id.0, "text space").map(TextSpaceId)
    }

    pub fn allocate_analyzer_id(&mut self) -> Result<AnalyzerId> {
        take_u32(&mut self.next_analyzer_id.0, "analyzer").map(AnalyzerId)
    }

    pub fn allocate_field_schema_id(&mut self) -> Result<FieldSchemaId> {
        take_u32(&mut self.next_field_schema_id.0, "field schema").map(FieldSchemaId)
    }

    pub fn allocate_retrieval_profile_id(&mut self) -> Result<RetrievalProfileId> {
        take_u32(&mut self.next_retrieval_profile_id.0, "retrieval profile").map(RetrievalProfileId)
    }

    pub fn allocate_embedding_space_id(&mut self) -> Result<EmbeddingSpaceId> {
        take_u32(&mut self.next_embedding_space_id.0, "embedding space").map(EmbeddingSpaceId)
    }

    pub fn allocate_codec_id(&mut self) -> Result<CodecId> {
        take_u32(&mut self.next_codec_id.0, "codec").map(CodecId)
    }

    pub fn allocate_vector_id(&mut self) -> Result<VectorId> {
        take_u64(&mut self.next_vector_id.0, "vector").map(VectorId)
    }

    pub fn allocate_fusion_profile_id(&mut self) -> Result<FusionProfileId> {
        take_u32(&mut self.next_fusion_profile_id.0, "fusion profile").map(FusionProfileId)
    }

    pub fn observe_collection_id(&mut self, id: CollectionId) -> Result<()> {
        observe_u32(&mut self.next_collection_id.0, id.0, "collection")
    }

    pub fn observe_frame_id(&mut self, id: FrameId) -> Result<()> {
        observe_u64(&mut self.next_frame_id.0, id.0, "frame")
    }

    pub fn observe_segment_id(&mut self, id: SegmentId) -> Result<()> {
        observe_u64(&mut self.next_segment_id.0, id.0, "segment")
    }

    pub fn observe_text_space_id(&mut self, id: TextSpaceId) -> Result<()> {
        observe_u32(&mut self.next_text_space_id.0, id.0, "text space")
    }

    pub fn observe_analyzer_id(&mut self, id: AnalyzerId) -> Result<()> {
        observe_u32(&mut self.next_analyzer_id.0, id.0, "analyzer")
    }

    pub fn observe_field_schema_id(&mut self, id: FieldSchemaId) -> Result<()> {
        observe_u32(&mut self.next_field_schema_id.0, id.0, "field schema")
    }

    pub fn observe_retrieval_profile_id(&mut self, id: RetrievalProfileId) -> Result<()> {
        observe_u32(
            &mut self.next_retrieval_profile_id.0,
            id.0,
            "retrieval profile",
        )
    }

    pub fn observe_embedding_space_id(&mut self, id: EmbeddingSpaceId) -> Result<()> {
        observe_u32(&mut self.next_embedding_space_id.0, id.0, "embedding space")
    }

    pub fn observe_codec_id(&mut self, id: CodecId) -> Result<()> {
        observe_u32(&mut self.next_codec_id.0, id.0, "codec")
    }

    pub fn observe_vector_id(&mut self, id: VectorId) -> Result<()> {
        observe_u64(&mut self.next_vector_id.0, id.0, "vector")
    }

    pub fn observe_fusion_profile_id(&mut self, id: FusionProfileId) -> Result<()> {
        observe_u32(&mut self.next_fusion_profile_id.0, id.0, "fusion profile")
    }
}

fn take_u32(next: &mut u32, label: &str) -> Result<u32> {
    validate_next_u32(*next, label)?;
    let id = *next;
    *next = next_u32(id, label)?;
    Ok(id)
}

fn take_u64(next: &mut u64, label: &str) -> Result<u64> {
    validate_next_u64(*next, label)?;
    let id = *next;
    *next = next_u64(id, label)?;
    Ok(id)
}

fn observe_u32(next: &mut u32, observed: u32, label: &str) -> Result<()> {
    if observed == 0 {
        return Err(Error::Format(format!("{label} id must be non-zero")));
    }
    let required_next = next_u32(observed, label)?;
    if *next < required_next {
        *next = required_next;
    }
    Ok(())
}

fn observe_u64(next: &mut u64, observed: u64, label: &str) -> Result<()> {
    if observed == 0 {
        return Err(Error::Format(format!("{label} id must be non-zero")));
    }
    let required_next = next_u64(observed, label)?;
    if *next < required_next {
        *next = required_next;
    }
    Ok(())
}

fn validate_next_u32(next: u32, label: &str) -> Result<()> {
    if next == 0 {
        return Err(Error::Format(format!("next {label} id must be non-zero")));
    }
    Ok(())
}

fn validate_next_u64(next: u64, label: &str) -> Result<()> {
    if next == 0 {
        return Err(Error::Format(format!("next {label} id must be non-zero")));
    }
    Ok(())
}

fn next_u32(id: u32, label: &str) -> Result<u32> {
    id.checked_add(1)
        .ok_or_else(|| Error::Format(format!("{label} id space exhausted")))
}

fn next_u64(id: u64, label: &str) -> Result<u64> {
    id.checked_add(1)
        .ok_or_else(|| Error::Format(format!("{label} id space exhausted")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::{
        catalog_codec::{CatalogTableKind, decode_catalog_records, encode_catalog_records},
        segment::{Compression, SegmentHeader},
    };

    #[test]
    fn segment_catalog_entry_validates_against_header() {
        let payload = b"catalog payload";
        let header = SegmentHeader::new(
            SegmentType::FrameCatalog,
            SegmentId(9),
            2,
            Compression::None,
            payload.len() as u64,
            payload,
        );
        let root = SegmentRef {
            segment_id: header.segment_id,
            offset: 4096,
            length: SEGMENT_HEADER_SIZE as u64 + payload.len() as u64,
            checksum: header.payload_checksum,
        };

        let entry = SegmentCatalogEntry::live(root, &header).expect("entry");

        entry
            .validate_header(&header)
            .expect("entry should match header");
        assert_eq!(entry.segment_id, SegmentId(9));
        assert_eq!(entry.status, SegmentCatalogStatus::Live);
    }

    #[test]
    fn segment_catalog_entry_rejects_bad_root_length() {
        let payload = b"catalog payload";
        let header = SegmentHeader::new(
            SegmentType::FrameCatalog,
            SegmentId(9),
            2,
            Compression::None,
            payload.len() as u64,
            payload,
        );
        let root = SegmentRef {
            segment_id: header.segment_id,
            offset: 4096,
            length: SEGMENT_HEADER_SIZE as u64 + payload.len() as u64 + 1,
            checksum: header.payload_checksum,
        };

        let err =
            SegmentCatalogEntry::live(root, &header).expect_err("root length mismatch should fail");

        assert!(err.to_string().contains("length mismatch"));
    }

    #[test]
    fn allocates_and_observes_ids_monotonically() {
        let mut allocator = IdAllocator::new();

        assert_eq!(
            allocator.allocate_collection_id().expect("collection id"),
            CollectionId(1)
        );
        assert_eq!(allocator.allocate_frame_id().expect("frame id"), FrameId(1));
        allocator
            .observe_collection_id(CollectionId(41))
            .expect("observe collection");
        allocator
            .observe_collection_id(CollectionId(4))
            .expect("observe older collection");

        assert_eq!(allocator.next_collection_id, CollectionId(42));
        assert_eq!(allocator.next_frame_id, FrameId(2));
    }

    #[test]
    fn rejects_zero_or_exhausted_allocator_state() {
        let mut allocator = IdAllocator {
            next_collection_id: CollectionId(0),
            ..IdAllocator::default()
        };
        assert!(
            allocator
                .validate()
                .expect_err("zero next id should fail")
                .to_string()
                .contains("non-zero")
        );

        allocator.next_collection_id = CollectionId(u32::MAX);
        assert!(
            allocator
                .allocate_collection_id()
                .expect_err("exhausted id space should fail")
                .to_string()
                .contains("exhausted")
        );
    }

    #[test]
    fn registry_records_roundtrip_through_catalog_codec() {
        let mut allocator = IdAllocator::new();
        allocator
            .observe_segment_id(SegmentId(100))
            .expect("observe segment");
        let encoded = encode_catalog_records(CatalogTableKind::IdAllocator, &[allocator.clone()])
            .expect("encode allocator");
        let decoded: Vec<IdAllocator> =
            decode_catalog_records(CatalogTableKind::IdAllocator, &encoded)
                .expect("decode allocator");

        assert_eq!(decoded, vec![allocator]);
    }
}
