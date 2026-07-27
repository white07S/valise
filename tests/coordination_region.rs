//! Stage 3a: prove the coordination region is stamped on disk and
//! survives a write+reopen cycle.
//!
//! Stage 3b will activate the cross-process atomics + OFD-lock byte
//! protocol on top of these persisted bytes; that's a separate test
//! file (`tests/coordination_locking.rs`) when it lands.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

use tempfile::tempdir;
use valise::ValiseFile;

const COORD_MAGIC: &[u8; 8] = b"VLSCOORD";
const COORD_REGION_OFFSET: u64 = 128;
const COORD_VERSION_OFFSET: u64 = COORD_REGION_OFFSET + 8;
const COORD_SLOT_COUNT_OFFSET: u64 = COORD_REGION_OFFSET + 12;
const FEATURE_BITMAP_OFFSET: u64 = 112;
const FEATURE_COORDINATION_REGION: u64 = 0x0040;

fn read_at(path: &std::path::Path, offset: u64, len: usize) -> Vec<u8> {
    let mut f = File::open(path).unwrap();
    f.seek(SeekFrom::Start(offset)).unwrap();
    let mut buf = vec![0u8; len];
    f.read_exact(&mut buf).unwrap();
    buf
}

fn read_u32_le(path: &std::path::Path, offset: u64) -> u32 {
    u32::from_le_bytes(read_at(path, offset, 4).try_into().unwrap())
}

fn read_u64_le(path: &std::path::Path, offset: u64) -> u64 {
    u64::from_le_bytes(read_at(path, offset, 8).try_into().unwrap())
}

#[test]
fn create_stamps_coordination_region_on_disk() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("coord.vls");
    let valise = ValiseFile::create(&path).unwrap();
    drop(valise);

    // Magic is at file offset 128.
    let magic = read_at(&path, COORD_REGION_OFFSET, 8);
    assert_eq!(&magic[..], COORD_MAGIC);

    // Version = 1.
    assert_eq!(read_u32_le(&path, COORD_VERSION_OFFSET), 1);
    // Reader slot count = 8.
    assert_eq!(read_u32_le(&path, COORD_SLOT_COUNT_OFFSET), 8);
    // Feature bit set.
    let bits = read_u64_le(&path, FEATURE_BITMAP_OFFSET);
    assert_ne!(
        bits & FEATURE_COORDINATION_REGION,
        0,
        "feature_bitmap must announce the coord region (got {bits:#x})"
    );
}

#[test]
fn reader_slots_initialize_to_free_sentinel() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("coord-slots.vls");
    let _valise = ValiseFile::create(&path).unwrap();

    // Reader slots start at file offset 320 (region offset 192). Each is
    // 64 bytes; the first 8 bytes are `pinned_toc_offset` which must be
    // u64::MAX (free sentinel).
    let reader_slots_start = 320u64;
    for n in 0..8 {
        let pinned_off = reader_slots_start + n as u64 * 64;
        let val = read_u64_le(&path, pinned_off);
        assert_eq!(
            val,
            u64::MAX,
            "reader slot {n} pinned_toc_offset must initialize to u64::MAX"
        );
    }
}

#[test]
fn coord_region_survives_commit_cycle() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("coord-commit.vls");

    {
        let mut valise = ValiseFile::create(&path).unwrap();
        let cid = valise.create_collection("c").unwrap();
        let _ = valise
            .put_frame(valise::PutFrame::document(cid, b"x"))
            .unwrap();
        valise.commit().unwrap();
    }

    // After a commit, the header is rewritten — but the coord region
    // must still validate. Stage 3a re-stamps the empty region every
    // header write; Stage 3b will preserve mutated atomics. For Stage
    // 3a (no atomics yet activated) the empty layout is the canonical
    // state, so this assertion is meaningful.
    let magic = read_at(&path, COORD_REGION_OFFSET, 8);
    assert_eq!(&magic[..], COORD_MAGIC, "magic preserved across commit");

    let valise = ValiseFile::open_read_only(&path).unwrap();
    let bits = valise.header().feature_bitmap;
    assert_ne!(bits & FEATURE_COORDINATION_REGION, 0);
}

#[test]
fn coord_publish_advances_generation_at_commit() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("publish.vls");

    let outcome_a;
    let outcome_b;
    {
        let mut valise = ValiseFile::create(&path).unwrap();
        let cid = valise.create_collection("c").unwrap();

        // Pre-first-commit: published generation is 0.
        assert_eq!(valise.coord_published_generation(), Some(0));

        let _ = valise
            .put_frame(valise::PutFrame::document(cid, b"first"))
            .unwrap();
        outcome_a = valise.commit().unwrap();
        assert_eq!(
            valise.coord_published_generation(),
            Some(outcome_a.snapshot_generation),
            "post-commit coord generation must equal the commit's outcome",
        );

        let _ = valise
            .put_frame(valise::PutFrame::document(cid, b"second"))
            .unwrap();
        outcome_b = valise.commit().unwrap();
        assert!(
            outcome_b.snapshot_generation > outcome_a.snapshot_generation,
            "generation must advance",
        );
        assert_eq!(
            valise.coord_published_generation(),
            Some(outcome_b.snapshot_generation),
        );
    }

    // Persistence: reopen and verify the latest published generation
    // round-trips through the coord region atomics on disk.
    let valise = ValiseFile::open_read_only(&path).unwrap();
    assert_eq!(
        valise.coord_published_generation(),
        Some(outcome_b.snapshot_generation),
        "coord generation must persist across open cycles",
    );
    // And the published_toc_offset on disk must equal the file's footer
    // offset. We check via raw bytes since `coord_published_toc_offset`
    // is not exposed through the public API.
    let coord_published_toc_offset = u64::from_le_bytes(
        read_at(&path, COORD_REGION_OFFSET + 16, 8)
            .try_into()
            .unwrap(),
    );
    assert_eq!(coord_published_toc_offset, valise.header().footer_offset);
}

#[test]
fn open_rejects_files_without_coordination_region() {
    // Post-consolidation: legacy (no coord region) files are no longer
    // supported. open() MUST refuse them with a clear "lacks
    // FEATURE_COORDINATION_REGION" error rather than silently
    // falling back to whole-file flock arbitration.
    use std::io::Write;
    let dir = tempdir().unwrap();
    let path = dir.path().join("legacy.vls");
    {
        let mut valise = ValiseFile::create(&path).unwrap();
        let cid = valise.create_collection("c").unwrap();
        let _ = valise
            .put_frame(valise::PutFrame::document(cid, b"x"))
            .unwrap();
        valise.commit().unwrap();
    }
    // Clear the coordination feature bit to simulate a pre-v1 file.
    {
        let mut f = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .unwrap();
        f.seek(SeekFrom::Start(FEATURE_BITMAP_OFFSET)).unwrap();
        let mut bits_buf = [0u8; 8];
        f.read_exact(&mut bits_buf).unwrap();
        let bits = u64::from_le_bytes(bits_buf);
        let cleared = bits & !FEATURE_COORDINATION_REGION;
        f.seek(SeekFrom::Start(FEATURE_BITMAP_OFFSET)).unwrap();
        f.write_all(&cleared.to_le_bytes()).unwrap();
    }
    // Reopen MUST fail with the explicit feature-bit error.
    let err = ValiseFile::open_read_only(&path)
        .err()
        .expect("legacy file must be refused");
    let msg = err.to_string();
    assert!(
        msg.contains("FEATURE_COORDINATION_REGION") || msg.contains("legacy files"),
        "expected legacy-rejection error, got: {msg}"
    );
}
