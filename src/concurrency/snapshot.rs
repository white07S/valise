// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Immutable view of an Valise file at a specific
//! `(snapshot_generation, toc_offset)`.
//!
//! Held by readers; published by writers via `ArcSwap`. Stage 2 establishes
//! the type and the publish protocol; Stage 4 will move all reader paths to
//! drive directly off the snapshot.
//!
//! See `docs/CONCURRENCY_PLAN.md` §3.2 for the full design.
//!
//! # Pinning
//!
//! Each snapshot owns `Arc`-refcounted handles to:
//! - the read-only `Mmap` covering the file at publish time,
//! - the `CatalogSnapshot`,
//! - the slim `frame_locators` map,
//! - the `SegmentRegistry`,
//! - the `vector_by_id` lookup.
//!
//! The `vector_base_ptrs` cache is per-snapshot but lazy: an `OnceLock`
//! that computes against the snapshot's pinned mmap on first need. Snapshot
//! identity is by `generation`; we deliberately do not derive `PartialEq`
//! to keep deep equality from compiling accidentally.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, OnceLock};

use parking_lot::RwLock;

use crate::error::{Error, Result};
use crate::file::VectorBasePtrs;
use crate::file::catalog_io::FrameLocator;
use crate::file::segment_io::{SegmentRegistry, verify_payload_wire_bytes};
use crate::format::FrameId;
use crate::format::SegmentId;
use crate::format::VectorId;
use crate::format::catalog::{CatalogSnapshot, FrameDesc, PayloadEncoding, VectorDesc};
use crate::format::segment::SegmentType;

/// Immutable, refcounted view of an Valise file as published by the most
/// recent successful `commit()`. See module docs.
#[allow(
    dead_code,
    reason = "non-mmap fields consumed by Stage 4 read paths; Stage 2 ships only publish + pin"
)]
pub struct Snapshot {
    /// Monotonic generation number, equals `header.snapshot_generation`
    /// at publish time.
    pub(crate) generation: u64,

    /// Authoritative TOC offset for this snapshot. Reads cap byte access at
    /// this offset; the reader never observes bytes appended after publish.
    pub(crate) toc_offset: u64,

    /// Pinned mmap covering bytes `[0, toc_offset)`. `None` for an empty
    /// (pre-first-commit) file. The `Arc` keeps the mapping alive even if
    /// `commit()` rotates `ValiseFile::file_mmap` to a fresh mapping.
    pub(crate) mmap: Option<Arc<memmap2::Mmap>>,

    /// Catalog at publish time. Cheap to clone (the inner is shared).
    pub(crate) catalog: Arc<CatalogSnapshot>,

    /// Slim disk locators for committed frames. Built at open from the
    /// committed catalog chain; refreshed at every commit publish.
    pub(crate) frame_locators: Arc<HashMap<FrameId, FrameLocator>>,

    /// In-memory segment registry at publish time. The `loaded` flag inside
    /// reflects whether the open path eagerly decoded the chain.
    pub(crate) segment_registry: Arc<SegmentRegistry>,

    /// `VectorId → VectorDesc` lookup for active vectors at publish time.
    /// Used by the ANN reranker to dereference graph hits.
    pub(crate) vector_by_id: Arc<HashMap<VectorId, VectorDesc>>,

    /// Lazy per-snapshot vector-base-pointer cache. Computed against
    /// `self.mmap` on first need; the lifetime of the resulting raw
    /// pointers is structurally tied to the snapshot's pinned mmap.
    pub(crate) vector_base_ptrs: OnceLock<Arc<VectorBasePtrs>>,

    /// Handle-level verified-set for payload wire bytes, `Arc`-shared
    /// with the owning `ValiseFile` (and every other snapshot it
    /// publishes). [`Self::read_payload`] re-hashes a Payload segment's
    /// on-disk bytes against the stored BLAKE3 once per segment per
    /// handle before trusting them; snapshots are immutable and
    /// `Arc`-shared across readers, so the set uses interior mutability
    /// (`RwLock`) with a read-lock fast path.
    pub(crate) verified_payload_segments: Arc<RwLock<HashSet<SegmentId>>>,
}

impl Snapshot {
    /// Monotonic generation number — equals `header.snapshot_generation`
    /// at publish time. Never decreases over the lifetime of an `ValiseFile`.
    #[must_use]
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Authoritative TOC offset for this snapshot. Reads cap byte access
    /// at this offset; the pinned mmap may extend further (a later commit
    /// rotated the underlying mapping) but readers honour this bound.
    #[must_use]
    pub fn toc_offset(&self) -> u64 {
        self.toc_offset
    }

    /// Pinned mmap covering bytes `[0, toc_offset)`. `None` for an empty
    /// (pre-first-commit) file. Returned as `Option<&memmap2::Mmap>` so
    /// callers can peek at bytes without taking on the `Arc` clone cost.
    #[must_use]
    pub fn mmap(&self) -> Option<&memmap2::Mmap> {
        self.mmap.as_deref()
    }

    /// Catalog at publish time. Internally `Arc`-shared.
    #[must_use]
    pub fn catalog(&self) -> &CatalogSnapshot {
        &self.catalog
    }

    /// Resolve a `FrameId` into a full `FrameDesc` by going directly
    /// to this snapshot's pinned mmap + frame-locator table. Bypasses
    /// the `Database`'s outer `RwLock` entirely — readers calling this
    /// method do not block on, and are not blocked by, a concurrent
    /// writer's `commit()` window. The committed state at this
    /// snapshot's generation is what's returned.
    pub fn frame_full(&self, frame_id: FrameId) -> Result<FrameDesc> {
        // Hot path: heavy frame already in the snapshot's catalog
        // (when the open path eagerly decoded it).
        if let Ok(idx) = self
            .catalog
            .frames
            .binary_search_by_key(&frame_id.0, |f| f.frame_id.0)
        {
            return Ok(self.catalog.frames[idx].clone());
        }
        let locator =
            self.frame_locators.get(&frame_id).copied().ok_or_else(|| {
                Error::Format(format!("frame_full: unknown frame_id {}", frame_id.0))
            })?;
        let mmap = self
            .mmap
            .as_ref()
            .ok_or_else(|| Error::Integrity("frame_full: snapshot has no pinned mmap".into()))?;
        let payload = crate::file::mmap_segment_payload(
            mmap,
            locator.catalog_segment,
            SegmentType::FrameCatalog,
        )?;
        let end = locator
            .payload_offset
            .checked_add(locator.payload_len)
            .ok_or_else(|| Error::Integrity("frame_full: locator offset overflow".into()))?;
        if end > payload.len() {
            return Err(Error::Integrity(format!(
                "frame_full: locator out of range (payload len {}, locator end {end})",
                payload.len()
            )));
        }
        let bytes = &payload[locator.payload_offset..end];
        let (desc, _): (FrameDesc, _) =
            bincode::serde::decode_from_slice(bytes, crate::format::bincode_config())
                .map_err(|e| Error::Serde(e.to_string()))?;
        if desc.frame_id != frame_id {
            return Err(Error::Integrity(format!(
                "frame_full: locator pointed at frame_id {} but expected {}",
                desc.frame_id.0, frame_id.0
            )));
        }
        Ok(desc)
    }

    /// Read a frame's raw payload bytes via the snapshot. Same lock-
    /// free guarantee as [`Self::frame_full`] — drives directly off
    /// the pinned mmap + segment registry.
    pub fn read_payload(&self, frame_id: FrameId) -> Result<Vec<u8>> {
        let frame = self.frame_full(frame_id)?;
        if frame.payload_encoding != PayloadEncoding::Raw {
            return Err(Error::Unsupported(
                "compressed frame payloads are not implemented in v0.1-alpha".into(),
            ));
        }
        let payload_ref = frame
            .payload_ref
            .ok_or_else(|| Error::Format("frame has no materialized payload".into()))?;
        let segment = self
            .segment_registry
            .by_id
            .get(&payload_ref.segment_id)
            .copied()
            .ok_or_else(|| {
                Error::Integrity(format!(
                    "read_payload: data segment {} not in snapshot's registry",
                    payload_ref.segment_id.0
                ))
            })?;
        let mmap = self
            .mmap
            .as_ref()
            .ok_or_else(|| Error::Integrity("read_payload: snapshot has no pinned mmap".into()))?;
        let segment_payload =
            crate::file::mmap_segment_payload(mmap, segment, SegmentType::Payload)?;
        // Integrity: re-hash the segment's wire bytes against the
        // stored BLAKE3 before trusting them (once per segment per
        // handle — the verified-set is shared with the owning
        // `ValiseFile`). `mmap_segment_payload` above only cross-checked
        // the two STORED checksum copies against each other; a flipped
        // bit in the committed bytes themselves would otherwise be
        // served (uncompressed path) or decompress "successfully"
        // (zstd path, no content checksum) and corrupt every payload
        // in the batch.
        verify_payload_wire_bytes(
            segment.segment_id,
            &segment.checksum,
            segment_payload,
            &self.verified_payload_segments,
        )?;
        // Inspect compression by peeking the segment header bytes
        // (offset 24..26 inside the segment header). For Zstd-encoded
        // Payload segments we decompress the full segment and then
        // slice. Allocation cost is bounded by the segment size
        // (≤ ~4 MiB per batch) and the read path is cold.
        let start = segment.offset as usize;
        let compression_code = u16::from_le_bytes([mmap[start + 24], mmap[start + 25]]);
        crate::format::segment::Compression::try_from(compression_code)?.ensure_decodable()?;
        let payload = if compression_code == crate::format::segment::Compression::Zstd as u16 {
            let end = payload_ref
                .bytes_offset
                .checked_add(payload_ref.bytes_length)
                .ok_or_else(|| Error::Format("payload ref range overflow".into()))?;
            let decompressed = zstd::decode_all(segment_payload).map_err(|err| {
                Error::Format(format!("zstd decode failed in snapshot read: {err}"))
            })?;
            if end > decompressed.len() as u64 {
                return Err(Error::Integrity(
                    "payload ref exceeds segment uncompressed length".into(),
                ));
            }
            decompressed[payload_ref.bytes_offset as usize..end as usize].to_vec()
        } else {
            let end = payload_ref
                .bytes_offset
                .checked_add(payload_ref.bytes_length)
                .ok_or_else(|| Error::Format("payload ref range overflow".into()))?;
            if end > segment_payload.len() as u64 {
                return Err(Error::Integrity(
                    "payload ref exceeds segment length".into(),
                ));
            }
            segment_payload[payload_ref.bytes_offset as usize..end as usize].to_vec()
        };
        Ok(payload)
    }

    /// Read a frame's payload as UTF-8 text via the snapshot.
    pub fn read_raw_text(&self, frame_id: FrameId) -> Result<String> {
        let bytes = self.read_payload(frame_id)?;
        String::from_utf8(bytes).map_err(|err| {
            Error::Format(format!(
                "read_raw_text: frame {} payload is not valid UTF-8: {err}",
                frame_id.0
            ))
        })
    }

    /// The empty snapshot — gen 0, no mmap, default catalog. Used as the
    /// initial `ArcSwap` payload before the first `Self::from_state` is
    /// built. Stable identity in tests where comparing generations
    /// against zero is meaningful.
    #[allow(dead_code, reason = "constructor scaffolding for Stage 4 reuse")]
    pub(crate) fn empty() -> Self {
        Self {
            generation: 0,
            toc_offset: 0,
            mmap: None,
            catalog: Arc::new(CatalogSnapshot::default()),
            frame_locators: Arc::new(HashMap::new()),
            segment_registry: Arc::new(SegmentRegistry::default()),
            vector_by_id: Arc::new(HashMap::new()),
            vector_base_ptrs: OnceLock::new(),
            verified_payload_segments: Arc::new(RwLock::new(HashSet::new())),
        }
    }
}

impl std::fmt::Debug for Snapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Snapshot")
            .field("generation", &self.generation)
            .field("toc_offset", &self.toc_offset)
            .field("has_mmap", &self.mmap.is_some())
            .field("frames", &self.catalog.frames.len())
            .field("frame_stubs", &self.catalog.frame_stubs.len())
            .field("vectors", &self.catalog.vectors.len())
            .finish()
    }
}
