//! Stage 2: prove the published `Snapshot` works.
//!
//! Three guarantees:
//! 1. `valise.snapshot()` returns a fresh `Arc<Snapshot>` whose `generation`
//!    matches `header.snapshot_generation` at publish time.
//! 2. Generation advances monotonically with each `commit()`.
//! 3. Pinning an old snapshot keeps its mmap alive even after a fresh
//!    `commit()` rotates the file's current mmap. The pinned mmap's bytes
//!    are still readable; the snapshot's catalog still reflects the old
//!    state.

use std::sync::Arc;

use tempfile::tempdir;
use valise::{CollectionId, PutFrame, ValiseFile};

#[test]
fn snapshot_pins_state_across_commits() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("snapshot.vls");

    let mut valise = ValiseFile::create(&path).unwrap();
    let collection_id = valise.create_collection("c").unwrap();
    let _ = valise
        .put_frame(PutFrame::document(collection_id, b"first"))
        .unwrap();
    let outcome_a = valise.commit().unwrap();
    let snap_a = valise.snapshot();
    assert_eq!(snap_a.generation(), outcome_a.snapshot_generation);
    assert_eq!(snap_a.catalog().frames.len(), 1);
    assert!(
        snap_a.mmap().is_some(),
        "post-commit snapshot must pin a non-empty mmap"
    );
    let mmap_arc_a = snap_a.mmap().unwrap();
    let frames_at_a = snap_a.catalog().frames.len();

    // Capture the byte-range the old mmap covers — we'll verify it's still
    // readable after the next commit rotates `file_mmap`.
    let mmap_a_len = mmap_arc_a.len();
    assert!(mmap_a_len > 0);
    let _byte0_a = mmap_arc_a[0];

    // Drive a second commit that adds a new frame. After this, the file
    // is larger and `valise.file_mmap` rotates to a fresh `Arc<Mmap>`. The
    // pinned `snap_a` still owns its older `Arc<Mmap>`, which the OS keeps
    // mapped because the refcount stays > 0.
    let _ = valise
        .put_frame(PutFrame::document(collection_id, b"second"))
        .unwrap();
    let outcome_b = valise.commit().unwrap();
    let snap_b = valise.snapshot();

    // 1. Generation advanced monotonically.
    assert!(
        outcome_b.snapshot_generation > outcome_a.snapshot_generation,
        "generation must advance across commits ({} -> {})",
        outcome_a.snapshot_generation,
        outcome_b.snapshot_generation,
    );
    assert_eq!(snap_b.generation(), outcome_b.snapshot_generation);

    // 2. Old snapshot retains its old view: still 1 frame.
    assert_eq!(snap_a.catalog().frames.len(), frames_at_a);
    // New snapshot reflects the addition.
    assert!(snap_b.catalog().frames.len() >= 2);

    // 3. Old mmap stays alive via the pinned `Arc`. Direct byte access via
    //    the held mmap reference works without segfault.
    assert_eq!(mmap_arc_a.len(), mmap_a_len);
    let _last_byte_a = mmap_arc_a[mmap_a_len - 1];

    // 4. Snapshots are cheap clones — `Arc` semantics. Cloning twice is a
    //    refcount bump.
    let _snap_a_clone1 = Arc::clone(&snap_a);
    let _snap_a_clone2 = Arc::clone(&snap_a);
}

#[test]
fn snapshot_after_open_matches_header_generation() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("snap-open.vls");

    let outcome = {
        let mut valise = ValiseFile::create(&path).unwrap();
        let cid: CollectionId = valise.create_collection("c").unwrap();
        let _ = valise.put_frame(PutFrame::document(cid, b"x")).unwrap();
        valise.commit().unwrap()
    };

    // Reopen and confirm the published snapshot matches the persisted
    // generation. This is what Stage 4 readers will rely on at handle-open
    // time to pin a consistent view from the very first read.
    let valise = ValiseFile::open_read_only(&path).unwrap();
    let snap = valise.snapshot();
    assert_eq!(snap.generation(), outcome.snapshot_generation);
    assert_eq!(snap.toc_offset(), valise.header().footer_offset);
    assert!(snap.mmap().is_some());
    // v2 columnar FrameCatalog decodes the full FrameDescs eagerly at
    // open. The v1 ghost-catalog (slim-decode + lazy heavy load via
    // `frame_full`) is preserved only as a fallback for legacy bincode
    // catalogs. Both shapes still publish `frame_stubs`.
    assert_eq!(snap.catalog().frames.len(), 1);
    assert_eq!(snap.catalog().frame_stubs.len(), 1);
}
