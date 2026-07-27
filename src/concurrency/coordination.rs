//! Typed view of the in-header coordination region (spec §7.1).
//!
//! Stage 3a: persistent layout + encode/decode + recognition. The
//! cross-process atomic CAS / OFD-lock acquisition lives in `locks.rs`
//! and `database.rs` (Stage 3b).
//!
//! ## Layout (832 bytes, starting at file offset 128)
//!
//! ```text
//!   128..192   CoordinationHeader      (64 B; magic, version, atomics)
//!   192..256   WriterSlot              (64 B; 1 lock byte + padding)
//!   256..320   CheckpointerSlot        (64 B; reserved, unused in v0.2)
//!   320..832   ReaderSlot[8]           (64 B each)
//! ```
//!
//! All multi-byte integers are little-endian. Cross-process atomics use
//! `std::sync::atomic::Ordering::{Acquire, Release}` once Stage 3b
//! activates writes; Stage 3a only encodes the initial zero-filled state.
//!
//! See `docs/FORMAT.md` §7.1 and `docs/CONCURRENCY_PLAN.md` §5.

use std::sync::atomic::{AtomicU32, AtomicU64};

use crate::format::{
    COORD_CACHE_LINE, COORD_CHECKPOINTER_SLOT_OFFSET, COORD_MAGIC, COORD_READER_SLOT_COUNT,
    COORD_READER_SLOTS_OFFSET, COORD_REGION_OFFSET, COORD_REGION_SIZE, COORD_VERSION,
    COORD_WRITER_SLOT_OFFSET,
};

// ---- Public-API constants surfaced for downstream use --------------------

/// Length of the coordination header sub-record.
pub(crate) const COORD_HEADER_SIZE: usize = 64;
/// Length of the writer slot.
pub(crate) const COORD_WRITER_SLOT_SIZE: usize = 64;
/// Length of the checkpointer slot.
pub(crate) const COORD_CHECKPOINTER_SLOT_SIZE: usize = 64;
/// Length of one reader slot.
pub(crate) const COORD_READER_SLOT_SIZE: usize = 64;

// ---- Static layout assertions --------------------------------------------

const _: () = {
    assert!(COORD_HEADER_SIZE == COORD_CACHE_LINE);
    assert!(COORD_WRITER_SLOT_SIZE == COORD_CACHE_LINE);
    assert!(COORD_CHECKPOINTER_SLOT_SIZE == COORD_CACHE_LINE);
    assert!(COORD_READER_SLOT_SIZE == COORD_CACHE_LINE);
    assert!(COORD_WRITER_SLOT_OFFSET == COORD_HEADER_SIZE);
    assert!(COORD_CHECKPOINTER_SLOT_OFFSET == COORD_WRITER_SLOT_OFFSET + COORD_WRITER_SLOT_SIZE);
    assert!(
        COORD_READER_SLOTS_OFFSET == COORD_CHECKPOINTER_SLOT_OFFSET + COORD_CHECKPOINTER_SLOT_SIZE
    );
    let total =
        COORD_READER_SLOTS_OFFSET + COORD_READER_SLOT_COUNT as usize * COORD_READER_SLOT_SIZE;
    assert!(total == COORD_REGION_SIZE);
};

// ---- Field offsets within each sub-record --------------------------------

pub(crate) mod header_offset {
    pub(crate) const MAGIC: usize = 0;
    pub(crate) const VERSION: usize = 8;
    pub(crate) const READER_SLOT_COUNT: usize = 12;
    pub(crate) const PUBLISHED_TOC_OFFSET: usize = 16;
    pub(crate) const PUBLISHED_SNAPSHOT_GENERATION: usize = 24;
    // 32..64: reserved (zero-filled padding to 64).
}

#[allow(
    dead_code,
    reason = "Stage 3b consumers — OWNER_* are stamped at slot acquire"
)]
pub(crate) mod reader_slot_offset {
    pub(crate) const PINNED_TOC_OFFSET: usize = 0;
    pub(crate) const PINNED_SNAPSHOT_GENERATION: usize = 8;
    pub(crate) const OWNER_PID: usize = 16;
    pub(crate) const OWNER_INSTANCE: usize = 20;
    // 24..64: reserved padding.
}

// ---- Encode / decode -----------------------------------------------------

/// Stamp the initial coordination-region bytes into a freshly-zeroed
/// header buffer. Idempotent — overwrites the entire region with the
/// canonical empty state (magic, version, slot count, all atomics
/// zero/`u64::MAX`-sentineled as appropriate).
pub(crate) fn stamp_initial(header_buf: &mut [u8]) {
    debug_assert!(header_buf.len() >= COORD_REGION_OFFSET + COORD_REGION_SIZE);

    let region = &mut header_buf[COORD_REGION_OFFSET..COORD_REGION_OFFSET + COORD_REGION_SIZE];

    // Zero everything first — defensive against partial reuse of the buffer.
    region.fill(0);

    // Coordination header: magic, version, slot count.
    region[header_offset::MAGIC..header_offset::MAGIC + 8].copy_from_slice(&COORD_MAGIC);
    region[header_offset::VERSION..header_offset::VERSION + 4]
        .copy_from_slice(&COORD_VERSION.to_le_bytes());
    region[header_offset::READER_SLOT_COUNT..header_offset::READER_SLOT_COUNT + 4]
        .copy_from_slice(&COORD_READER_SLOT_COUNT.to_le_bytes());
    // PUBLISHED_TOC_OFFSET and PUBLISHED_SNAPSHOT_GENERATION are u64 zeros
    // until the first commit publishes them (Stage 3b).

    // Reader slots: pinned_toc_offset = u64::MAX (free sentinel) on each.
    for n in 0..COORD_READER_SLOT_COUNT as usize {
        let slot_start = COORD_READER_SLOTS_OFFSET + n * COORD_READER_SLOT_SIZE;
        let pinned_off = slot_start + reader_slot_offset::PINNED_TOC_OFFSET;
        region[pinned_off..pinned_off + 8].copy_from_slice(&u64::MAX.to_le_bytes());
        // PINNED_SNAPSHOT_GENERATION, OWNER_PID, OWNER_INSTANCE remain 0.
    }
}

/// Lightweight value type returned by `read_header_view` — captures the
/// non-atomic descriptive fields plus a snapshot of the two atomic u64s
/// at decode time. Cross-process readers that need *live* atomic values
/// must access them via the mmap; this type is intentionally inert.
#[allow(
    dead_code,
    reason = "Stage 3b consumers use this for open-time coord region detection"
)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct CoordinationHeaderView {
    pub(crate) magic: [u8; 8],
    pub(crate) version: u32,
    pub(crate) reader_slot_count: u32,
    pub(crate) published_toc_offset: u64,
    pub(crate) published_snapshot_generation: u64,
}

impl CoordinationHeaderView {
    #[allow(
        dead_code,
        reason = "Stage 3b open path uses this for region detection"
    )]
    pub(crate) fn is_valid(&self) -> bool {
        self.magic == COORD_MAGIC && self.version == COORD_VERSION
    }
}

/// Decode the coordination header from a header byte buffer. Returns
/// `None` when the buffer is too short to contain the region.
#[allow(dead_code, reason = "Stage 3b open path consumes this")]
pub(crate) fn read_header_view(header_buf: &[u8]) -> Option<CoordinationHeaderView> {
    if header_buf.len() < COORD_REGION_OFFSET + COORD_HEADER_SIZE {
        return None;
    }
    let region = &header_buf[COORD_REGION_OFFSET..COORD_REGION_OFFSET + COORD_HEADER_SIZE];
    let mut magic = [0u8; 8];
    magic.copy_from_slice(&region[header_offset::MAGIC..header_offset::MAGIC + 8]);
    let version = u32::from_le_bytes(
        region[header_offset::VERSION..header_offset::VERSION + 4]
            .try_into()
            .expect("4 bytes"),
    );
    let reader_slot_count = u32::from_le_bytes(
        region[header_offset::READER_SLOT_COUNT..header_offset::READER_SLOT_COUNT + 4]
            .try_into()
            .expect("4 bytes"),
    );
    let published_toc_offset = u64::from_le_bytes(
        region[header_offset::PUBLISHED_TOC_OFFSET..header_offset::PUBLISHED_TOC_OFFSET + 8]
            .try_into()
            .expect("8 bytes"),
    );
    let published_snapshot_generation = u64::from_le_bytes(
        region[header_offset::PUBLISHED_SNAPSHOT_GENERATION
            ..header_offset::PUBLISHED_SNAPSHOT_GENERATION + 8]
            .try_into()
            .expect("8 bytes"),
    );
    Some(CoordinationHeaderView {
        magic,
        version,
        reader_slot_count,
        published_toc_offset,
        published_snapshot_generation,
    })
}

// ---- Live atomic accessors (Stage 3b consumers) --------------------------
//
// Stage 3a defines the typed accessors but does not yet wire them into
// the writer/reader paths. The atomic loads/stores are placed here in a
// single module so the protocol's memory ordering can be reviewed in one
// shot when Stage 3b activates them.

// ---- Atomic accessors over the mmap-backed coord region (Stage 3b) -------
//
// SAFETY contract for everything below: the caller must keep the mmap
// alive (`Arc<Mmap>` pin or equivalent) for the duration of any atomic
// reference returned. Field offsets are 8-byte (or 4-byte) multiples
// within a 64-byte-aligned region that starts at file offset 128, so
// the alignment requirements are satisfied by construction.

/// Produce a typed `&AtomicU64` over the bytes at the given offset
/// within the coord-region mmap slice. Internal helper.
///
/// SAFETY: caller must hold the mmap alive and pass an offset that is a
/// multiple of 8 inside `region.len()`.
unsafe fn atomic_u64(region: &[u8], offset: usize) -> &AtomicU64 {
    debug_assert!(offset + 8 <= region.len());
    debug_assert!(
        offset.is_multiple_of(8),
        "AtomicU64 requires 8-byte alignment"
    );
    // SAFETY: AtomicU64 has the same layout as u64 per std::sync::atomic
    // docs; Acquire/Release atomic ops on shared mmap pages are
    // well-defined on every supported target (aarch64-apple-darwin,
    // x86_64) when properly aligned. The bytes are inside the mmap; the
    // mmap's lifetime is enforced by the caller's pin.
    unsafe { &*(region.as_ptr().add(offset) as *const AtomicU64) }
}

#[allow(dead_code, reason = "consumed by reader-slot owner stamping")]
unsafe fn atomic_u32(region: &[u8], offset: usize) -> &AtomicU32 {
    debug_assert!(offset + 4 <= region.len());
    debug_assert!(
        offset.is_multiple_of(4),
        "AtomicU32 requires 4-byte alignment"
    );
    // SAFETY: see atomic_u64.
    unsafe { &*(region.as_ptr().add(offset) as *const AtomicU32) }
}

/// Slice the coordination region out of an mmap that maps the file from
/// offset 0. Returns `None` if the mmap is too short.
pub(crate) fn region_slice(mmap: &memmap2::Mmap) -> Option<&[u8]> {
    if mmap.len() < COORD_REGION_OFFSET + COORD_REGION_SIZE {
        return None;
    }
    Some(&mmap[COORD_REGION_OFFSET..COORD_REGION_OFFSET + COORD_REGION_SIZE])
}

/// Atomic accessor for `coord_published_toc_offset`.
///
/// SAFETY: caller must hold the mmap alive for the lifetime of the
/// returned reference.
#[allow(
    dead_code,
    reason = "Stage 3b writes via pwrite; readers may use this for atomic load"
)]
pub(crate) unsafe fn published_toc_offset(region: &[u8]) -> &AtomicU64 {
    // SAFETY: forwarded to `atomic_u64`; offset is at a fixed
    // 8-byte-aligned location within the region.
    unsafe { atomic_u64(region, header_offset::PUBLISHED_TOC_OFFSET) }
}

/// Atomic accessor for `coord_published_snapshot_generation`.
///
/// SAFETY: same as `published_toc_offset`.
pub(crate) unsafe fn published_snapshot_generation(region: &[u8]) -> &AtomicU64 {
    // SAFETY: same as `published_toc_offset`.
    unsafe { atomic_u64(region, header_offset::PUBLISHED_SNAPSHOT_GENERATION) }
}

/// Validate that the coord region's magic + version match this build.
/// Returns `true` when the region is recognizable; callers fall back to
/// legacy whole-file `flock` arbitration when this returns `false`.
pub(crate) fn is_active(region: &[u8]) -> bool {
    if region.len() < COORD_HEADER_SIZE {
        return false;
    }
    if region[header_offset::MAGIC..header_offset::MAGIC + 8] != COORD_MAGIC {
        return false;
    }
    let version = u32::from_le_bytes(
        region[header_offset::VERSION..header_offset::VERSION + 4]
            .try_into()
            .expect("4 bytes"),
    );
    version == COORD_VERSION
}

/// Reader slot accessors. Slot index `n` in `0..reader_slot_count`.
#[allow(dead_code, reason = "consumed by reader-slot pin (Stage 4)")]
pub(crate) struct ReaderSlot<'r> {
    region: &'r [u8],
    base: usize,
}

#[allow(dead_code, reason = "consumed by reader-slot pin (Stage 4)")]
impl<'r> ReaderSlot<'r> {
    pub(crate) fn nth(region: &'r [u8], n: usize) -> Option<Self> {
        // Region-relative offset: reader slots start at region offset 192
        // (file offset 320 = COORD_REGION_OFFSET 128 + 192).
        let base = COORD_READER_SLOTS_OFFSET + n * COORD_READER_SLOT_SIZE;
        if base + COORD_READER_SLOT_SIZE > region.len() {
            return None;
        }
        Some(Self { region, base })
    }

    pub(crate) fn pinned_toc_offset(&self) -> &AtomicU64 {
        // SAFETY: region offset arithmetic verified by `nth`.
        unsafe {
            atomic_u64(
                self.region,
                self.base + reader_slot_offset::PINNED_TOC_OFFSET,
            )
        }
    }

    pub(crate) fn pinned_snapshot_generation(&self) -> &AtomicU64 {
        // SAFETY: same.
        unsafe {
            atomic_u64(
                self.region,
                self.base + reader_slot_offset::PINNED_SNAPSHOT_GENERATION,
            )
        }
    }

    pub(crate) fn owner_pid(&self) -> &AtomicU32 {
        // SAFETY: same.
        unsafe { atomic_u32(self.region, self.base + reader_slot_offset::OWNER_PID) }
    }

    pub(crate) fn owner_instance(&self) -> &AtomicU32 {
        // SAFETY: same.
        unsafe { atomic_u32(self.region, self.base + reader_slot_offset::OWNER_INSTANCE) }
    }

    /// Byte offset of this slot's lock byte within the **file** (not the
    /// region) — i.e. the value to pass to `try_acquire_byte_lock`.
    pub(crate) fn lock_byte_in_file(slot_index: usize) -> u64 {
        (COORD_REGION_OFFSET + COORD_READER_SLOTS_OFFSET + slot_index * COORD_READER_SLOT_SIZE)
            as u64
    }
}

/// Byte offset of the writer slot's lock byte within the file.
pub(crate) const WRITER_LOCK_BYTE: u64 = (COORD_REGION_OFFSET + COORD_WRITER_SLOT_OFFSET) as u64;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::HEADER_SIZE;

    #[test]
    fn stamp_initial_round_trips() {
        let mut buf = vec![0u8; HEADER_SIZE];
        stamp_initial(&mut buf);
        let view = read_header_view(&buf).expect("decode");
        assert!(view.is_valid());
        assert_eq!(view.magic, COORD_MAGIC);
        assert_eq!(view.version, COORD_VERSION);
        assert_eq!(view.reader_slot_count, COORD_READER_SLOT_COUNT);
        assert_eq!(view.published_toc_offset, 0);
        assert_eq!(view.published_snapshot_generation, 0);
    }

    #[test]
    fn untouched_buffer_does_not_validate() {
        let buf = vec![0u8; HEADER_SIZE];
        let view = read_header_view(&buf).expect("decode");
        assert!(!view.is_valid(), "all-zero magic must not validate");
    }

    #[test]
    fn reader_slots_initialize_to_free_sentinel() {
        let mut buf = vec![0u8; HEADER_SIZE];
        stamp_initial(&mut buf);
        for n in 0..COORD_READER_SLOT_COUNT as usize {
            let slot_start =
                COORD_REGION_OFFSET + COORD_READER_SLOTS_OFFSET + n * COORD_READER_SLOT_SIZE;
            let pinned_off = slot_start + reader_slot_offset::PINNED_TOC_OFFSET;
            let val = u64::from_le_bytes(buf[pinned_off..pinned_off + 8].try_into().unwrap());
            assert_eq!(
                val,
                u64::MAX,
                "reader slot {n} must initialize to u64::MAX (free sentinel)"
            );
        }
    }

    #[test]
    fn region_does_not_overflow_header() {
        // Compile-time check duplicated at runtime for safety.
        const {
            assert!(COORD_REGION_OFFSET + COORD_REGION_SIZE <= HEADER_SIZE);
        }
    }
}
