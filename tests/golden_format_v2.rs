// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Format-drift detection: BLAKE3 of a fully deterministic v2 fixture.
//!
//! What this catches: any inadvertent change to the on-disk byte layout
//! — segment headers, catalog encoding, payload framing, TOC body
//! shape, header layout, coord region. Even a single-byte shift in any
//! of those structures will flip the hash and fail the test.
//!
//! Why we mask the UUID: `Header::new` calls `Uuid::new_v4()`, which
//! is non-deterministic. The UUID is opaque to recovery semantics —
//! it's a debug aid, never read at hot paths — so masking it before
//! hashing gives us a stable contract while still pinning every
//! format-meaningful byte.
//!
//! Why we pin timestamps: `put_frame` records `current_unix_timestamp`
//! by default. We pass explicit `Some(0)` so the corpus is built at
//! "epoch zero" and the catalog encoding stays deterministic.
//!
//! When this test fails: that's the signal that some persisted
//! structure changed. Re-generate the expected hash ONLY after
//! confirming the change is intentional and corresponds to a format
//! version bump.

use std::{fs, path::Path};

use tempfile::tempdir;
use valise::{PutFrame, ValiseFile, format::CollectionId};

/// Build the deterministic fixture and return the masked BLAKE3 hash.
fn fixture_hash(path: &Path) -> [u8; 32] {
    let mut valise = ValiseFile::create(path).expect("create");
    let cid = valise.create_collection_at("docs", 0).expect("collection");
    assert_eq!(cid, CollectionId(1));

    // Two frames with explicit timestamps so the catalog encoding is
    // bit-stable. Payloads chosen short enough to keep the file small
    // (~16 KB) and the diff readable when this test fails.
    valise
        .put_frame(PutFrame {
            collection_id: cid,
            role: valise::format::catalog::FrameRole::Document,
            payload: b"alpha frame payload",
            created_at: Some(0),
            updated_at: Some(0),
            parent_frame_id: None,
            chunk_index: None,
            chunk_count: None,
            uri: None,
        })
        .expect("put alpha");
    valise
        .put_frame(PutFrame {
            collection_id: cid,
            role: valise::format::catalog::FrameRole::Document,
            payload: b"beta frame payload",
            created_at: Some(0),
            updated_at: Some(0),
            parent_frame_id: None,
            chunk_index: None,
            chunk_count: None,
            uri: None,
        })
        .expect("put beta");
    valise.commit().expect("commit");
    drop(valise);

    let mut bytes = fs::read(path).expect("read fixture");
    // Mask UUID (16 bytes at offset 12..28).
    for b in &mut bytes[12..28] {
        *b = 0;
    }
    *blake3::hash(&bytes).as_bytes()
}

#[test]
fn v2_format_is_byte_stable() {
    // Update this hash ONLY when the change is intentional and
    // corresponds to a format major bump or a documented minor bump.
    // The blanket re-encoding rule: hash flip ⇒ format change ⇒ spec
    // amendment + version bump. Don't update silently.
    // Updated 2026-05-15: format drift from three optimizations bundled
    // for the storage-overhead audit on the BEIR-quora BM25 build —
    //   (1) `FrameDesc.payload_checksum` removed (segment-level BLAKE3
    //       still protects the bytes via SegmentHeader + SegmentRef),
    //   (2) `TimeIndexSegment` v1 → v2 columnar wire (frame_id varint
    //       deltas, created_at delta-of-delta, updated_at offset,
    //       collection_id RLE),
    //   (3) `Payload` segments are now batched at commit time instead
    //       of one segment per `put_frame`, collapsing 522k-segment
    //       fan-out to one segment per ≤4 MiB batch.
    // Format MAJOR stays at 2 because no header field shape moved; the
    // wire-incompatible changes are inside the TimeIndex codec
    // (version bumped to 2 inside the segment payload) and inside
    // `FrameDesc` (field removed). v0.1-alpha allows free iteration
    // of the canonical wire shape — see docs/FORMAT.md.
    // Updated 2026-05-15 (round 2): format drift from four additional
    // optimizations bundled for the storage-overhead audit:
    //   (5) `CollectionFilter` gains a run-encoding (flag 1) that
    //       collapses `[1..N]` sequential id sets into one record.
    //   (6) `DocStats` v1 → v2 columnar (frame_id deltas, collection
    //       RLE, packed field_lengths with per-column width tag).
    //   (7) Payload segments are now zstd-compressed at level 3.
    //   (8) `FrameCatalog` v1 (bincode-per-row) → v2 (`NXF2` magic,
    //       columnar layout with RLE for nearly-constant fields and
    //       delta encoding for monotonic columns).
    // No FORMAT_MAJOR bump — the wire shapes are all inside individual
    // segment payloads, header layout is unchanged. v0.1-alpha free-
    // iteration policy applies; existing files require a rebuild.
    // Updated 2026-05-22 (v2.2): ANN profile burial. Dropped
    //   - `TocFooterBody.ann_profile_catalog_root` (Option<SegmentRef>)
    //   - `TocFooterBody.vector_ann_roots` (Vec<VectorAnnRoot>)
    //   - `EmbeddingSpaceDesc.ann_profile_id` (Option<AnnProfileId>)
    //   - `IdAllocator.next_ann_profile_id`
    //   - `SegmentType::{AnnProfileCatalog, VectorAnn}`
    //   - `CatalogTableKind::AnnProfile`
    //   - `AnnProfileDesc` / `AnnEngine` / `AnnParams` types
    // `FORMAT_MINOR` bumped 1 → 2 (v2.3, vote-profile burial). Header
    // validate now rejects any minor that doesn't match the binary's.
    // See MIGRATION.md.
    // Updated 2026-06-11 (v2.4): UPQ codec family. `FORMAT_MINOR`
    // bumped 2 → 3 (the header minor byte is the only change this
    // fixture sees — the fixture is frames-only and registers no
    // codec; `CodecFamily::Upq` + the `NXUP` params blob only appear
    // in files that register a UPQ codec). See MIGRATION.md.
    const EXPECTED_HASH: &str = "91a80686990eae59970095d4cb38e3a79ae1221914c64db88e97f7f85f029801";

    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("golden_v2.vls");
    let actual = fixture_hash(&path);
    let actual_hex = hex::encode(actual);

    if EXPECTED_HASH == "REGENERATE_ME" {
        // First run: the test FAILS loudly so the developer notices,
        // but the failure prints the hash to paste into EXPECTED_HASH.
        panic!("[golden] expected hash sentinel — paste this into EXPECTED_HASH:\n  {actual_hex}");
    }

    assert_eq!(
        actual_hex, EXPECTED_HASH,
        "v2 file format drifted. If this is intentional, re-generate the \
         hash and document the change with a format-version bump."
    );
}

#[test]
fn fixture_is_reproducible_across_runs() {
    // Sanity check: running the fixture twice produces the same hash.
    // Catches accidental sources of non-determinism (timestamps,
    // randomness, hash-map iteration order in the encoder, etc.) that
    // would make `v2_format_is_byte_stable` flaky.
    let dir = tempdir().expect("tempdir");
    let h1 = fixture_hash(&dir.path().join("a.vls"));
    let h2 = fixture_hash(&dir.path().join("b.vls"));
    assert_eq!(
        h1, h2,
        "fixture is non-deterministic — find and remove the source of variance"
    );
}
