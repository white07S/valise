// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use std::{fs::OpenOptions, io::Read};

use tempfile::tempdir;
use valise::{PutFrame, ValiseFile};

#[test]
fn create_commit_reopen_and_read_payload() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("sample.vls");

    let mut valise = ValiseFile::create(&path).expect("create Valise file");
    let collection_id = valise.create_collection("docs").expect("create collection");
    let frame_id = valise
        .put_frame(PutFrame::document(collection_id, b"hello world"))
        .expect("put frame");

    assert_eq!(
        valise.read_payload(frame_id).expect("read overlay payload"),
        b"hello world"
    );

    let outcome = valise.commit().expect("commit");
    assert!(outcome.changed);
    drop(valise);

    let reopened = ValiseFile::open_read_only(&path).expect("open read-only");
    assert_eq!(reopened.collections().len(), 1);
    assert_eq!(reopened.frame_stubs().len(), 1);
    assert_eq!(
        reopened.read_payload(frame_id).expect("read payload"),
        b"hello world"
    );
}

#[test]
fn frame_uri_persists_across_reopen() {
    // Hook #1: PutFrame.uri is persisted in-file via a batched Metadata
    // segment + FrameDesc.uri_ref, and read back with read_frame_uri.
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("uri.vls");

    let mut valise = ValiseFile::create(&path).expect("create");
    let cid = valise.create_collection("docs").expect("collection");

    // Two keyed frames + one anonymous, so the batched Metadata segment
    // holds multiple keys and per-frame slicing must be correct.
    let keyed = valise
        .put_frame(PutFrame::document(cid, b"doc one").with_uri(b"key-alpha"))
        .expect("put keyed");
    let anon = valise
        .put_frame(PutFrame::document(cid, b"doc two"))
        .expect("put anon");
    let keyed2 = valise
        .put_frame(PutFrame::document(cid, b"doc three").with_uri(b"urn:example:42"))
        .expect("put keyed2");

    valise.commit().expect("commit");
    drop(valise);

    let reopened = ValiseFile::open_read_only(&path).expect("reopen");
    assert_eq!(
        reopened.read_frame_uri(keyed).expect("read uri"),
        Some(b"key-alpha".to_vec())
    );
    assert_eq!(reopened.read_frame_uri(anon).expect("read uri"), None);
    assert_eq!(
        reopened.read_frame_uri(keyed2).expect("read uri"),
        Some(b"urn:example:42".to_vec())
    );
    // The payload side-channel is unaffected by the uri.
    assert_eq!(reopened.read_payload(anon).expect("payload"), b"doc two");
}

#[test]
fn commit_persists_state_across_reopen() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("sample.vls");

    let mut valise = ValiseFile::create(&path).expect("create Valise file");
    let collection_id = valise.create_collection("docs").expect("create collection");
    let frame_id = valise
        .put_frame(PutFrame::document(collection_id, b"frame payload"))
        .expect("put frame");
    valise.commit().expect("commit");
    drop(valise);

    let reopened = ValiseFile::open_read_only(&path).expect("open read-only");
    assert_eq!(reopened.collections().len(), 1);
    assert_eq!(reopened.frame_stubs().len(), 1);
    assert_eq!(
        reopened.read_payload(frame_id).expect("read payload"),
        b"frame payload"
    );
}

#[test]
fn no_op_commit_does_not_advance_snapshot() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("sample.vls");

    let mut valise = ValiseFile::create(&path).expect("create Valise file");
    let first_generation = valise.header().snapshot_generation;
    let outcome = valise.commit().expect("commit");

    assert!(!outcome.changed);
    assert_eq!(outcome.snapshot_generation, first_generation);
}

#[test]
fn operations_do_not_create_sidecars() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("sample.vls");

    let mut valise = ValiseFile::create(&path).expect("create Valise file");
    let collection_id = valise.create_collection("docs").expect("create collection");
    valise
        .put_frame(PutFrame::document(collection_id, b"payload"))
        .expect("put frame");
    valise.commit().expect("commit");
    drop(valise);

    let entries = std::fs::read_dir(dir.path())
        .expect("read dir")
        .collect::<Result<Vec<_>, _>>()
        .expect("collect entries");
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].path(), path);
}

#[test]
fn segment_ids_are_small_integers_not_file_offsets() {
    // Segment IDs are sequentially-allocated u64s starting at 1, NOT
    // byte positions in the file. A small N-frame fixture should emit
    // segment IDs well under 1000 even though the file is many KB
    // long.
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("sample.vls");

    let mut valise = ValiseFile::create(&path).expect("create Valise file");
    let collection_id = valise.create_collection("docs").expect("create collection");
    valise.commit().expect("commit collection");
    let frame_id = valise
        .put_frame(PutFrame::document(collection_id, b"payload"))
        .expect("put frame");
    // Payload batching: `payload_ref` is None until the batch flushes
    // at commit time. Commit, then inspect the materialized ref.
    valise.commit().expect("commit frame");

    let frame = valise
        .frames()
        .iter()
        .find(|frame| frame.frame_id == frame_id)
        .expect("frame exists");
    let payload_ref = frame.payload_ref.expect("payload ref");
    assert!(payload_ref.segment_id.0 < 1_000);
    drop(valise);

    let reopened = ValiseFile::open_read_only(&path).expect("open read-only");
    assert_eq!(reopened.read_payload(frame_id).expect("read"), b"payload");
}

#[test]
fn catalog_deltas_reopen_after_many_commits() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("sample.vls");

    let mut valise = ValiseFile::create(&path).expect("create Valise file");
    let collection_id = valise.create_collection("docs").expect("create collection");
    valise.commit().expect("commit collection");
    for i in 0..25 {
        let payload = [i; 32];
        valise
            .put_frame(PutFrame::document(collection_id, &payload))
            .expect("put frame");
        valise.commit().expect("commit frame delta");
    }
    drop(valise);

    let reopened = ValiseFile::open_read_only(&path).expect("open read-only");
    assert_eq!(reopened.collections().len(), 1);
    assert_eq!(reopened.frame_stubs().len(), 25);
}

#[test]
fn read_payload_rejects_corrupt_payload_bytes() {
    // After the per-frame BLAKE3 in `FrameDesc` was retired in favor
    // of the segment-level checksum (one BLAKE3 per Payload segment
    // in both `SegmentHeader` and the registry's `SegmentRef`), this
    // test covers the surviving integrity contract: corrupting the
    // segment's stored hash so it no longer matches the registry's
    // copy must fail the read. Corrupting raw payload bytes alone
    // without also rewriting the matching segment header / registry
    // checksums is now caught at the slow read path (via
    // `header.validate_payload`) — the mmap fast path trusts the
    // header/registry pair (intentional, see the segment-header
    // codec docs).
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("sample.vls");
    let payload = b"checksum protected payload";

    let mut valise = ValiseFile::create(&path).expect("create Valise file");
    let collection_id = valise.create_collection("docs").expect("create collection");
    let frame_id = valise
        .put_frame(PutFrame::document(collection_id, payload))
        .expect("put frame");
    let _ = payload;
    valise.commit().expect("commit");
    drop(valise);

    corrupt_segment_header_checksum_field(&path, payload);

    // Opening the file under read-only must surface the integrity
    // failure either at open time (segment registry self-check) or
    // at first read (mmap_segment_payload's header/registry
    // cross-check). Either path must reject.
    let open_result = ValiseFile::open_read_only(&path);
    let error_text = match open_result {
        Err(e) => e.to_string(),
        Ok(reopened) => reopened
            .read_payload(frame_id)
            .expect_err("corrupt header must fail validation")
            .to_string(),
    };
    assert!(
        error_text.contains("checksum") || error_text.contains("mismatch"),
        "expected an integrity failure, got: {error_text}",
    );
}

/// Locate the Payload segment carrying `payload` and flip a byte of
/// the segment header's `payload_checksum` field (offset 44..76 of
/// the 76-byte segment header). That diverges the header's hash from
/// the registry's hash without touching the bytes the mmap path
/// returns, so the cross-check in `mmap_segment_payload` is what
/// fails.
fn corrupt_segment_header_checksum_field(path: &std::path::Path, payload: &[u8]) {
    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .expect("open file");
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).expect("read file");
    let payload_offset = bytes
        .windows(payload.len())
        .position(|window| window == payload)
        .expect("payload bytes present in file");
    // The segment header is 76 bytes immediately before the payload
    // bytes (no padding between header and payload in v2). The
    // payload_checksum field lives at header offset 44..76.
    let checksum_offset = payload_offset - 76 + 44;
    use std::io::{Seek, SeekFrom, Write};
    file.seek(SeekFrom::Start(checksum_offset as u64))
        .expect("seek");
    file.write_all(&[0xAB]).expect("write");
    file.sync_all().expect("fsync");
}
