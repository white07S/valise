use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io::{Read, Seek, SeekFrom, Write},
};

use parking_lot::RwLock;

use crate::{
    error::{Error, Result},
    format::{
        Checksum, SegmentId, blake3_checksum,
        payload::PayloadRef,
        registry::{IdAllocator, SegmentCatalogEntry},
        segment::{
            Compression, SEGMENT_HEADER_SIZE, SegmentHeader, SegmentHeaderCodec, SegmentRef,
            SegmentType,
        },
    },
};

/// In-memory mirror of the on-disk segment registry. Bundles the catalog
/// vector (sorted by id at commit), the by-id lookup, and a `loaded` flag
/// so readers can distinguish "ghost-deferred at open" from "loaded but
/// empty". Mutated under `&mut ValiseFile` via `RwLock::get_mut`; read under
/// `&ValiseFile` via `RwLock::read`. `Clone` so commit can publish an
/// `Arc<SegmentRegistry>` snapshot (Stage 2).
#[derive(Clone, Default)]
pub(crate) struct SegmentRegistry {
    pub(crate) catalog: Vec<SegmentCatalogEntry>,
    pub(crate) by_id: HashMap<SegmentId, SegmentRef>,
    /// `false` when the registry was deferred at open (Exploit A2 ghost
    /// registry: not read until a code path needs the by-id index).
    /// Reader-side lazy load via `RwLock::write` flips this to `true`.
    pub(crate) loaded: bool,
}

/// Bundled mutable borrow handed to writer helpers. The `registry` field
/// is the canonical in-memory segment registry (catalog + by-id index +
/// loaded flag). `dirty_ids` is held separately because it tracks
/// commit-time accumulation, not registry state.
pub(super) struct SegmentRegistryMut<'a> {
    pub(super) registry: &'a mut SegmentRegistry,
    pub(super) dirty_ids: &'a mut HashSet<SegmentId>,
}

pub(super) fn append_segment_at_end(
    file: &mut File,
    segment_id: SegmentId,
    segment_type: SegmentType,
    item_count: u32,
    payload: &[u8],
) -> Result<SegmentRef> {
    let root = segment_ref_at_end(file, segment_id, segment_type, item_count, payload)?;
    file.seek(SeekFrom::Start(root.offset))?;
    let header = SegmentHeader::new(
        segment_type,
        segment_id,
        item_count,
        Compression::None,
        payload.len() as u64,
        payload,
    );
    let encoded_header = SegmentHeaderCodec::encode(&header);
    file.write_all(&encoded_header)?;
    file.write_all(payload)?;
    Ok(root)
}

/// Append a segment whose payload is already-compressed bytes.
/// `wire_payload` is what hits disk (the post-compression stream);
/// `uncompressed_length` is the original byte count and is recorded
/// in the segment header so readers know how big the decoded buffer
/// must be. The segment's BLAKE3 checksum is computed over
/// `wire_payload`, identical to the uncompressed path.
pub(super) fn append_compressed_segment_at_end(
    file: &mut File,
    segment_id: SegmentId,
    segment_type: SegmentType,
    item_count: u32,
    wire_payload: &[u8],
    uncompressed_length: u64,
    compression: Compression,
) -> Result<SegmentRef> {
    if compression == Compression::None {
        return Err(Error::Format(
            "append_compressed_segment_at_end: compression must not be None".into(),
        ));
    }
    let offset = file.metadata()?.len();
    let header = SegmentHeader::new(
        segment_type,
        segment_id,
        item_count,
        compression,
        uncompressed_length,
        wire_payload,
    );
    let encoded_header = SegmentHeaderCodec::encode(&header);
    file.seek(SeekFrom::Start(offset))?;
    file.write_all(&encoded_header)?;
    file.write_all(wire_payload)?;
    Ok(SegmentRef {
        segment_id: header.segment_id,
        offset,
        length: SEGMENT_HEADER_SIZE as u64 + wire_payload.len() as u64,
        checksum: header.payload_checksum,
    })
}

/// Build a `SegmentCatalogEntry` for a pre-compressed segment. Differs
/// from [`segment_catalog_entry`] in that `wire_payload` is the post-
/// compression bytes (i.e. what's on disk and what was hashed in the
/// header) and `uncompressed_length` is recorded explicitly.
pub(super) fn compressed_segment_catalog_entry(
    segment_id: SegmentId,
    segment_type: SegmentType,
    item_count: u32,
    wire_payload: &[u8],
    uncompressed_length: u64,
    compression: Compression,
    root: SegmentRef,
) -> Result<SegmentCatalogEntry> {
    let payload_length = root
        .length
        .checked_sub(SEGMENT_HEADER_SIZE as u64)
        .ok_or_else(|| Error::Format("segment root length is smaller than header".into()))?;
    let checksum = if wire_payload.is_empty() {
        root.checksum
    } else {
        blake3_checksum(wire_payload)
    };
    let entry = SegmentCatalogEntry {
        segment_id,
        segment_type,
        root,
        segment_version: crate::format::segment::SEGMENT_VERSION,
        flags: 0,
        item_count,
        compression,
        uncompressed_length,
        payload_length,
        status: crate::format::registry::SegmentCatalogStatus::Live,
    };
    if entry.root.checksum != checksum {
        return Err(Error::Integrity(
            "compressed segment catalog checksum mismatch".into(),
        ));
    }
    entry.validate()?;
    Ok(entry)
}

pub(super) fn segment_ref_at_end(
    file: &File,
    segment_id: SegmentId,
    segment_type: SegmentType,
    item_count: u32,
    payload: &[u8],
) -> Result<SegmentRef> {
    let offset = file.metadata()?.len();
    let header = SegmentHeader::new(
        segment_type,
        segment_id,
        item_count,
        Compression::None,
        payload.len() as u64,
        payload,
    );
    Ok(SegmentRef {
        segment_id: header.segment_id,
        offset,
        length: SEGMENT_HEADER_SIZE as u64 + payload.len() as u64,
        checksum: header.payload_checksum,
    })
}

pub(super) fn append_registered_segment(
    file: &mut File,
    id_allocator: &mut IdAllocator,
    segments: SegmentRegistryMut<'_>,
    segment_type: SegmentType,
    item_count: u32,
    payload: &[u8],
) -> Result<SegmentRef> {
    let segment_id = id_allocator.allocate_segment_id()?;
    let root = append_segment_at_end(file, segment_id, segment_type, item_count, payload)?;
    register_segment(
        segments,
        segment_catalog_entry(segment_id, segment_type, item_count, payload, root)?,
    );
    Ok(root)
}

pub(super) fn read_segment_payload(file: &mut File, root: SegmentRef) -> Result<Vec<u8>> {
    let (_, payload) = read_segment(file, root, None)?;
    Ok(payload)
}

pub(super) fn read_segment_payload_typed(
    file: &mut File,
    root: SegmentRef,
    expected: SegmentType,
) -> Result<Vec<u8>> {
    let (_, payload) = read_segment(file, root, Some(expected))?;
    Ok(payload)
}

fn read_segment(
    file: &mut File,
    root: SegmentRef,
    expected: Option<SegmentType>,
) -> Result<(SegmentHeader, Vec<u8>)> {
    file.seek(SeekFrom::Start(root.offset))?;
    let mut header_bytes = [0u8; SEGMENT_HEADER_SIZE];
    file.read_exact(&mut header_bytes)?;
    let header = SegmentHeaderCodec::decode(&header_bytes)?;
    if header.segment_id != root.segment_id {
        return Err(Error::Integrity("segment id mismatch".into()));
    }
    if header.payload_checksum != root.checksum {
        return Err(Error::Integrity("segment root checksum mismatch".into()));
    }
    if root.length != SEGMENT_HEADER_SIZE as u64 + header.payload_length {
        return Err(Error::Integrity("segment root length mismatch".into()));
    }
    if let Some(expected) = expected
        && header.segment_type != expected
    {
        return Err(Error::Integrity(format!(
            "segment type mismatch: expected {:?}, got {:?}",
            expected, header.segment_type
        )));
    }
    let mut payload = vec![0u8; header.payload_length as usize];
    file.read_exact(&mut payload)?;
    header.validate_payload(&payload)?;
    header.compression.ensure_decodable()?;
    if header.compression == Compression::Zstd {
        let decompressed = zstd::decode_all(payload.as_slice())
            .map_err(|err| Error::Format(format!("zstd decode failed: {err}")))?;
        if decompressed.len() as u64 != header.uncompressed_length {
            return Err(Error::Integrity(format!(
                "zstd decompressed length {} does not match header.uncompressed_length {}",
                decompressed.len(),
                header.uncompressed_length
            )));
        }
        return Ok((header, decompressed));
    }
    if header.compression != Compression::None {
        return Err(Error::Unsupported(format!(
            "segment compression {:?} not supported",
            header.compression
        )));
    }
    Ok((header, payload))
}

pub(super) fn read_payload_ref(
    file: &mut File,
    segment: SegmentRef,
    payload_ref: PayloadRef,
    verified_segments: &RwLock<HashSet<SegmentId>>,
) -> Result<Vec<u8>> {
    file.seek(SeekFrom::Start(segment.offset))?;
    let mut header_bytes = [0u8; SEGMENT_HEADER_SIZE];
    file.read_exact(&mut header_bytes)?;
    let header = SegmentHeaderCodec::decode(&header_bytes)?;
    if header.segment_id != payload_ref.segment_id {
        return Err(Error::Integrity("payload segment id mismatch".into()));
    }
    if header.segment_type != SegmentType::Payload {
        return Err(Error::Integrity(
            "payload ref points to non-payload segment".into(),
        ));
    }
    if segment.checksum != [0; 32] && segment.checksum != header.payload_checksum {
        return Err(Error::Integrity("payload segment checksum mismatch".into()));
    }
    // Root-cause guard (crash-campaign finding): `header.payload_length`
    // is raw wire data protected by NO checksum of its own (the segment
    // header only stores the PAYLOAD's BLAKE3), so a single flipped bit
    // can declare a petabyte-scale length. Allocating from it
    // unvalidated aborts the whole process — allocation failure is not
    // unwindable, so `catch_unwind` cannot contain it. Validate the
    // length against the registry's SegmentRef (exact match when
    // present) and the file's true size BEFORE any allocation sized
    // from it.
    if segment.length != 0 {
        let expected = segment
            .length
            .checked_sub(SEGMENT_HEADER_SIZE as u64)
            .ok_or_else(|| Error::Integrity("payload segment ref shorter than header".into()))?;
        if header.payload_length != expected {
            return Err(Error::Integrity(
                "payload segment length disagrees with registry".into(),
            ));
        }
    }
    let file_len = file.metadata()?.len();
    let payload_end = segment
        .offset
        .checked_add(SEGMENT_HEADER_SIZE as u64)
        .and_then(|v| v.checked_add(header.payload_length))
        .ok_or_else(|| Error::Format("payload segment extent overflow".into()))?;
    if payload_end > file_len {
        return Err(Error::Integrity(
            "payload segment length exceeds file size".into(),
        ));
    }
    header.compression.ensure_decodable()?;
    if header.compression == Compression::Zstd {
        // Compressed Payload: read the full segment payload off disk,
        // decompress, then slice. Less efficient than the uncompressed
        // mmap fast path, but payload reads are rare relative to BM25
        // query hot loop, and the compression buys ~2-3× on Quora-shaped
        // English text.
        let end = payload_ref
            .bytes_offset
            .checked_add(payload_ref.bytes_length)
            .ok_or_else(|| Error::Format("payload ref range overflow".into()))?;
        if end > header.uncompressed_length {
            return Err(Error::Integrity(
                "payload ref exceeds segment uncompressed length".into(),
            ));
        }
        let mut compressed = vec![0u8; header.payload_length as usize];
        file.seek(SeekFrom::Start(segment.offset + SEGMENT_HEADER_SIZE as u64))?;
        file.read_exact(&mut compressed)?;
        // Integrity: re-hash the wire bytes against the stored BLAKE3
        // before handing them to zstd. The registry/header cross-check
        // above only proves the two STORED copies agree with each
        // other; a flipped bit in the committed stream itself would
        // otherwise decompress "successfully" (frames carry no zstd
        // content checksum) and corrupt every payload in the batch.
        verify_payload_wire_bytes(
            header.segment_id,
            &header.payload_checksum,
            &compressed,
            verified_segments,
        )?;
        let decompressed = zstd::decode_all(compressed.as_slice())
            .map_err(|err| Error::Format(format!("zstd decode failed: {err}")))?;
        if decompressed.len() as u64 != header.uncompressed_length {
            return Err(Error::Integrity(format!(
                "zstd decompressed length {} does not match header.uncompressed_length {}",
                decompressed.len(),
                header.uncompressed_length
            )));
        }
        let start = payload_ref.bytes_offset as usize;
        let end = end as usize;
        return Ok(decompressed[start..end].to_vec());
    }
    if header.compression != Compression::None {
        return Err(Error::Unsupported(format!(
            "payload segment compression {:?} not supported",
            header.compression
        )));
    }
    let end = payload_ref
        .bytes_offset
        .checked_add(payload_ref.bytes_length)
        .ok_or_else(|| Error::Format("payload ref range overflow".into()))?;
    if end > header.payload_length {
        return Err(Error::Integrity(
            "payload ref exceeds segment length".into(),
        ));
    }
    if !verified_segments.read().contains(&header.segment_id) {
        // First touch of this uncompressed segment this open: read the
        // WHOLE wire payload so its BLAKE3 can be checked, then slice
        // the requested range out of the buffer already in hand. Later
        // reads of a verified segment take the ranged fast path below.
        let mut wire = vec![0u8; header.payload_length as usize];
        file.seek(SeekFrom::Start(segment.offset + SEGMENT_HEADER_SIZE as u64))?;
        file.read_exact(&mut wire)?;
        verify_payload_wire_bytes(
            header.segment_id,
            &header.payload_checksum,
            &wire,
            verified_segments,
        )?;
        return Ok(wire[payload_ref.bytes_offset as usize..end as usize].to_vec());
    }
    file.seek(SeekFrom::Start(
        segment.offset + SEGMENT_HEADER_SIZE as u64 + payload_ref.bytes_offset,
    ))?;
    let mut payload = vec![0u8; payload_ref.bytes_length as usize];
    file.read_exact(&mut payload)?;
    Ok(payload)
}

/// Verify a Payload segment's wire bytes against its stored BLAKE3, at
/// most once per segment per file handle. `verified` is shared across
/// concurrent readers AND across every `Snapshot` published by the same
/// handle (it is `Arc`-cloned into snapshots at publish time): the
/// common case is a read-lock lookup; only the first reader of a
/// segment pays the hash and takes the write lock. Committed segments
/// are immutable, so a verified id never needs re-checking.
///
/// The all-zero checksum is the legacy-registry sentinel
/// (`legacy_offset_segment_ref`) — there is no stored hash to verify
/// against, so it is skipped rather than false-rejected.
pub(crate) fn verify_payload_wire_bytes(
    segment_id: SegmentId,
    expected_checksum: &Checksum,
    wire: &[u8],
    verified: &RwLock<HashSet<SegmentId>>,
) -> Result<()> {
    if *expected_checksum == [0u8; 32] {
        return Ok(());
    }
    if verified.read().contains(&segment_id) {
        return Ok(());
    }
    if blake3_checksum(wire) != *expected_checksum {
        return Err(Error::Integrity(
            "payload segment wire bytes checksum mismatch".into(),
        ));
    }
    verified.write().insert(segment_id);
    Ok(())
}

pub(super) fn build_segment_map(
    segment_catalog: &[SegmentCatalogEntry],
) -> HashMap<SegmentId, SegmentRef> {
    segment_catalog
        .iter()
        .map(|entry| (entry.segment_id, entry.root))
        .collect()
}

pub(super) fn register_segment(segments: SegmentRegistryMut<'_>, entry: SegmentCatalogEntry) {
    // Same pattern as the catalog upserts: keep the catalog vector sorted
    // by segment_id and use binary_search_by_key. Monotonic id allocation
    // means we hit Err(len) and Vec::insert(len, …) is amortized O(1).
    match segments
        .registry
        .catalog
        .binary_search_by_key(&entry.segment_id.0, |e| e.segment_id.0)
    {
        Ok(idx) => segments.registry.catalog[idx] = entry.clone(),
        Err(idx) => segments.registry.catalog.insert(idx, entry.clone()),
    }
    segments.registry.by_id.insert(entry.segment_id, entry.root);
    segments.dirty_ids.insert(entry.segment_id);
}

pub(super) fn segment_catalog_entry(
    segment_id: SegmentId,
    segment_type: SegmentType,
    item_count: u32,
    payload: &[u8],
    root: SegmentRef,
) -> Result<SegmentCatalogEntry> {
    let payload_length = root
        .length
        .checked_sub(SEGMENT_HEADER_SIZE as u64)
        .ok_or_else(|| Error::Format("segment root length is smaller than header".into()))?;
    let checksum = if payload.is_empty() {
        root.checksum
    } else {
        blake3_checksum(payload)
    };
    Ok(SegmentCatalogEntry {
        segment_id,
        segment_type,
        root,
        segment_version: crate::format::segment::SEGMENT_VERSION,
        flags: 0,
        item_count,
        compression: Compression::None,
        uncompressed_length: payload_length,
        payload_length,
        status: crate::format::registry::SegmentCatalogStatus::Live,
    })
    .and_then(|entry| {
        if entry.root.checksum != checksum {
            return Err(Error::Integrity("segment catalog checksum mismatch".into()));
        }
        entry.validate()?;
        Ok(entry)
    })
}

pub(super) fn legacy_offset_segment_ref(segment_id: SegmentId) -> SegmentRef {
    SegmentRef {
        segment_id,
        offset: segment_id.0,
        length: 0,
        checksum: [0; 32],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempfile;

    #[test]
    fn typed_read_rejects_wrong_segment_type() {
        let mut file = tempfile().expect("tempfile");
        let payload = b"x";
        let segment_id = SegmentId(1);
        let header = SegmentHeader::new(
            SegmentType::Extension,
            segment_id,
            1,
            Compression::None,
            payload.len() as u64,
            payload,
        );
        let encoded_header = SegmentHeaderCodec::encode(&header);
        file.write_all(&encoded_header).unwrap();
        file.write_all(payload).unwrap();

        let root = SegmentRef {
            segment_id,
            offset: 0,
            length: SEGMENT_HEADER_SIZE as u64 + payload.len() as u64,
            checksum: header.payload_checksum,
        };

        let err = read_segment_payload_typed(&mut file, root, SegmentType::SegmentRegistry)
            .expect_err("expected type mismatch error");
        assert!(
            err.to_string().contains("segment type mismatch"),
            "unexpected error: {err}"
        );

        let payload_back = read_segment_payload_typed(&mut file, root, SegmentType::Extension)
            .expect("matching expected type should succeed");
        assert_eq!(payload_back, payload);
    }
}
