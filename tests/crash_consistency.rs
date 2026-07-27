// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Crash-consistency fault-injection matrix for the Valise commit protocol.
//!
//! The commit path (`ValiseFile::commit_phase_writes`, src/file/commit.rs)
//! appends payload + catalog segments at EOF, appends a self-validating
//! TOC footer (magic `VLTC`, embedded body + footer BLAKE3, see
//! src/format/toc.rs), rewrites the logical header prefix (bytes
//! 0..120, `footer_offset` at bytes 32..40), then issues ONE
//! `F_FULLFSYNC`.
//!
//! Open (src/file/lifecycle.rs) reads `header.footer_offset` and
//! validates the referenced footer PLUS every snapshot root reachable
//! from it, and cross-checks `header.snapshot_generation` against the
//! footer's own generation. If any of that fails — torn/absent footer,
//! bad checksum, offset past EOF, unloadable roots, or a
//! generation-mismatched (torn) header — open falls back to SCAN-BACK
//! RECOVERY: it streams the file for `VLTC` footer candidates, fully
//! re-validates each (double BLAKE3, create-contract anchor, and a
//! complete snapshot-root load), and re-anchors on the
//! highest-generation valid snapshot. The previous commit's footer is
//! always durable (that commit's own barrier covered it), so a file
//! that ever committed successfully is always recoverable.
//!
//! Every test here operates on a byte-level mutated COPY of a pristine
//! two-generation fixture file and classifies the reopen outcome as one
//! of:
//!
//!   - `OldSnapshot`  — opens, data matches the gen1 commit exactly
//!   - `NewSnapshot`  — opens, data matches the gen2 commit exactly
//!   - `CleanReject`  — `open_read_only` returned a typed [`valise::Error`]
//!   - `RejectedAtRead` — opened, but a payload read surfaced a typed
//!     error in a pattern matching neither snapshot (fail-safe: no
//!     wrong bytes were returned)
//!   - `Anomalous`    — opened and returned bytes that belong to
//!     NEITHER committed snapshot (silent wrong data — always a bug)
//!   - `Panicked`     — open/read panicked (always a bug)
//!
//! Contract encoded by this matrix (post scan-back fix):
//!
//!   - A fault that invalidates the NEWEST commit's footer or its
//!     reachable roots recovers the previous ACKNOWLEDGED snapshot
//!     (`OldSnapshot`), never fails the open, and never serves torn
//!     state. This intentionally covers detection-only faults too
//!     (e.g. a media bitflip on the current footer): open serves the
//!     newest FULLY-VALID snapshot and logs a recovery warning rather
//!     than rejecting the whole file.
//!   - The reported generation is always the anchoring footer's own
//!     (self-consistent), even after a torn header rewrite.
//!   - Committed payload bytes are re-hashed against their stored
//!     BLAKE3 before first use per open, so a flipped bit in a
//!     committed payload region surfaces as a typed integrity error
//!     (`RejectedAtRead`), never as silently wrong data.
//!   - Wire-embedded lengths are validated against the registry and the
//!     real file size BEFORE any allocation sized from them: a flipped
//!     high bit in a segment header's `payload_length` is a typed
//!     error, never a process-killing allocation abort.
//!   - If the header's byte-832 create-contract digest matches NO valid
//!     footer (the anchor region itself is corrupt), candidate footers
//!     are validated by contract consensus instead: 2+ agreeing valid
//!     footers (or the single footer of a fresh file) establish the
//!     lineage; all-disagreeing footers still fail closed.
//!
//! Tier A tests pin the per-fault-class outcome under that contract.
//! Tier B tests re-state the two recovery guarantees that motivated the
//! scan-back fix (formerly `#[ignore]`d as the target contract).
//!
//! Note: the existing test
//! `tests/qam_vector.rs::crash_safety_open_after_partial_segment_truncation`
//! is mislabeled — it performs no truncation (it drops an uncommitted
//! writer and asserts the file length never grew). The truncation
//! matrix below is the real coverage.

use std::{
    fs,
    path::{Path, PathBuf},
    sync::OnceLock,
};

use tempfile::TempDir;
use valise::{Database, FrameId, OpenMode, PutFrame, ValiseFile};

// ---------------------------------------------------------------------------
// On-disk layout constants, mirrored from src/format (header.rs / toc.rs).
// ---------------------------------------------------------------------------

/// Logical header prefix rewritten on every commit
/// (`HeaderCodec::write_logical_prefix`, bytes 0..120).
const HEADER_LOGICAL_LEN: usize = 120;
/// `Header.footer_offset` lives at bytes 32..40 (LE u64).
const HEADER_FOOTER_OFFSET_AT: usize = 32;
/// Fixed TOC footer header length (src/format/toc.rs `TOC_HEADER_LEN`).
const TOC_HEADER_LEN: u64 = 88;
/// `body_len` field inside the TOC footer header (LE u64 at 16..24).
const TOC_BODY_LEN_AT: u64 = 16;
const TOC_MAGIC: &[u8; 4] = b"VLTC";
/// Segment header magic + layout (src/format/segment.rs):
/// `segment_type` at bytes 6..8 (LE u16), `payload_checksum` at 44..76.
const SEGMENT_MAGIC: &[u8; 4] = b"VLSG";
const SEGMENT_HEADER_LEN: u64 = 76;
const SEGMENT_TYPE_AT: u64 = 6;
const SEGMENT_TYPE_PAYLOAD: u16 = 0x0001;
const SEGMENT_CHECKSUM_AT: u64 = 44;

// Distinguishable per-generation payloads. NOTE: payload batch
// segments are zstd-compressed on disk; these ASCII strings are still
// byte-locatable in the image because zstd stores first occurrences as
// raw literals (the sibling payload of the same batch is then encoded
// as a back-reference into that literal).
const PAYLOAD_A1: &[u8] = b"crash-matrix-gen1-payload-A1";
const PAYLOAD_A2: &[u8] = b"crash-matrix-gen1-payload-A2";
const PAYLOAD_B1: &[u8] = b"crash-matrix-gen2-payload-B1";
const PAYLOAD_B2: &[u8] = b"crash-matrix-gen2-payload-B2";

// ---------------------------------------------------------------------------
// Deterministic PRNG (no time-derived seeds anywhere in this file).
// xorshift64* — the multiply finalizer matters because consumers take
// the value modulo small ranges and plain xorshift has weak low bits.
// ---------------------------------------------------------------------------

struct XorShift64(u64);

impl XorShift64 {
    fn new(seed: u64) -> Self {
        assert_ne!(seed, 0, "xorshift seed must be nonzero");
        Self(seed)
    }

    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
}

// ---------------------------------------------------------------------------
// Fixture: a pristine two-generation file image, built once.
// ---------------------------------------------------------------------------

struct Fixture {
    /// Full bytes of the file after the gen2 commit.
    pristine: Vec<u8>,
    /// Logical header prefix (0..120) as of the gen1 commit.
    gen1_header: Vec<u8>,
    /// File length right after the gen1 commit — the gen2 appends
    /// (payload segment, catalog segments, footer) all live past this.
    gen1_len: u64,
    gen1_generation: u64,
    gen2_generation: u64,
    /// `header.footer_offset` of the pristine file (the gen2 footer).
    footer_offset: u64,
    /// Total encoded length of the gen2 footer (88-byte header + body).
    footer_len: u64,
    /// Offset of frame B1's payload bytes (a zstd literal inside the
    /// compressed gen2 payload segment) inside `pristine`.
    payload_b1_offset: u64,
    /// Offset of the gen2 payload segment's 76-byte header.
    payload_seg_b_header: u64,
    frame_a1: FrameId,
    frame_a2: FrameId,
    frame_b1: FrameId,
    frame_b2: FrameId,
}

fn le_u64(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(
        bytes[offset..offset + 8]
            .try_into()
            .expect("u64 slice in bounds"),
    )
}

/// Find the unique occurrence of `needle` in `haystack`.
fn find_unique(haystack: &[u8], needle: &[u8]) -> u64 {
    let mut hits = haystack
        .windows(needle.len())
        .enumerate()
        .filter(|(_, w)| *w == needle)
        .map(|(i, _)| i as u64);
    let first = hits.next().expect("needle present in file image");
    assert!(hits.next().is_none(), "needle occurs more than once");
    first
}

fn fixture() -> &'static Fixture {
    static FIXTURE: OnceLock<Fixture> = OnceLock::new();
    FIXTURE.get_or_init(build_fixture)
}

fn build_fixture() -> Fixture {
    let dir = tempfile::tempdir().expect("fixture tempdir");
    let path = dir.path().join("pristine.vls");

    // Generation 1: two committed frames.
    let mut valise = ValiseFile::create(&path).expect("create fixture file");
    let collection = valise.create_collection("docs").expect("create collection");
    let frame_a1 = valise
        .put_frame(PutFrame::document(collection, PAYLOAD_A1))
        .expect("put frame A1");
    let frame_a2 = valise
        .put_frame(PutFrame::document(collection, PAYLOAD_A2))
        .expect("put frame A2");
    let gen1 = valise.commit().expect("commit gen1");
    assert!(gen1.changed);
    drop(valise);
    let gen1_bytes = fs::read(&path).expect("read gen1 image");

    // Generation 2: two more frames on top.
    let mut valise = ValiseFile::open_read_write(&path).expect("reopen read-write");
    let frame_b1 = valise
        .put_frame(PutFrame::document(collection, PAYLOAD_B1))
        .expect("put frame B1");
    let frame_b2 = valise
        .put_frame(PutFrame::document(collection, PAYLOAD_B2))
        .expect("put frame B2");
    let gen2 = valise.commit().expect("commit gen2");
    assert!(gen2.changed);
    drop(valise);
    let pristine = fs::read(&path).expect("read pristine image");

    // Decode the geometry we need straight from the bytes so faults can
    // be placed deterministically.
    let footer_offset = le_u64(&pristine, HEADER_FOOTER_OFFSET_AT);
    assert!(
        footer_offset >= gen1_bytes.len() as u64,
        "gen2 footer must be appended past the gen1 image \
         (footer_offset={footer_offset}, gen1_len={})",
        gen1_bytes.len()
    );
    assert_eq!(
        &pristine[footer_offset as usize..footer_offset as usize + 4],
        TOC_MAGIC,
        "header.footer_offset must point at the VLTC magic"
    );
    let body_len = le_u64(&pristine, (footer_offset + TOC_BODY_LEN_AT) as usize);
    let footer_len = TOC_HEADER_LEN + body_len;
    assert_eq!(
        footer_offset + footer_len,
        pristine.len() as u64,
        "the committed footer is expected to be the last bytes of the file"
    );

    let payload_b1_offset = find_unique(&pristine, PAYLOAD_B1);
    assert!(
        payload_b1_offset >= gen1_bytes.len() as u64,
        "frame B1 payload bytes must live in the gen2 append region"
    );
    // The payload segment's header is the closest VLSG magic at or
    // before the payload bytes. Verify its type field so a fault aimed
    // at the segment header cannot silently target the wrong structure.
    let payload_seg_b_header = pristine[..payload_b1_offset as usize]
        .windows(SEGMENT_MAGIC.len())
        .rposition(|w| w == SEGMENT_MAGIC)
        .expect("segment header magic before payload bytes") as u64;
    let seg_type = u16::from_le_bytes(
        pristine[(payload_seg_b_header + SEGMENT_TYPE_AT) as usize
            ..(payload_seg_b_header + SEGMENT_TYPE_AT + 2) as usize]
            .try_into()
            .expect("segment type field"),
    );
    assert_eq!(
        seg_type, SEGMENT_TYPE_PAYLOAD,
        "closest segment header before frame B1 bytes must be the Payload segment"
    );
    assert!(
        payload_seg_b_header + SEGMENT_HEADER_LEN <= payload_b1_offset,
        "payload bytes must start past the 76-byte segment header"
    );

    Fixture {
        gen1_header: gen1_bytes[..HEADER_LOGICAL_LEN].to_vec(),
        gen1_len: gen1_bytes.len() as u64,
        gen1_generation: gen1.snapshot_generation,
        gen2_generation: gen2.snapshot_generation,
        footer_offset,
        footer_len,
        payload_b1_offset,
        payload_seg_b_header,
        pristine,
        frame_a1,
        frame_a2,
        frame_b1,
        frame_b2,
    }
}

// ---------------------------------------------------------------------------
// Fault application: every fault mutates a fresh COPY of the pristine
// image in its own tempdir. The fixture itself is never touched.
// ---------------------------------------------------------------------------

struct FaultFile {
    _dir: TempDir,
    path: PathBuf,
}

fn apply_fault(fx: &Fixture, mutate: impl FnOnce(&mut Vec<u8>)) -> FaultFile {
    let dir = tempfile::tempdir().expect("fault tempdir");
    let path = dir.path().join("faulted.vls");
    let mut bytes = fx.pristine.clone();
    mutate(&mut bytes);
    fs::write(&path, &bytes).expect("write fault copy");
    FaultFile { _dir: dir, path }
}

// ---------------------------------------------------------------------------
// Outcome classification.
// ---------------------------------------------------------------------------

#[derive(Debug)]
enum Outcome {
    /// Opened; visible data is exactly the gen1 commit.
    OldSnapshot { generation: u64 },
    /// Opened; visible data is exactly the gen2 commit.
    NewSnapshot { generation: u64 },
    /// `open_read_only` returned a typed error. No data was served.
    CleanReject(String),
    /// Opened, but payload reads errored in a pattern matching neither
    /// snapshot. Fail-safe (no wrong bytes served), but the file opened.
    RejectedAtRead(String),
    /// Opened and served bytes matching NEITHER committed snapshot.
    ///
    /// The detail string is only ever read through the derived `Debug`
    /// when an assertion fails, which dead-code analysis does not count —
    /// but it is the whole diagnostic for a corruption escape, so keep it.
    #[expect(dead_code, reason = "surfaced via Debug in assertion failures")]
    Anomalous(String),
    /// Open or read panicked.
    Panicked(String),
}

/// Assert that a fault against the gen2 commit recovered the previous
/// acknowledged (gen1) snapshot with a self-consistent generation label.
fn assert_recovers_gen1(fx: &Fixture, outcome: &Outcome, context: &str) {
    match outcome {
        Outcome::OldSnapshot { generation } => assert_eq!(
            *generation, fx.gen1_generation,
            "{context}: recovered snapshot must report gen1's own generation"
        ),
        other => {
            panic!("{context}: must scan-back to the acknowledged gen1 snapshot, got: {other:?}")
        }
    }
}

fn classify(path: &Path, fx: &Fixture) -> Outcome {
    let probe = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| probe_file(path, fx)));
    // Set VALISE_CRASH_MATRIX_TRACE=1 (with --nocapture) to see the raw
    // classification of every fault while iterating on the matrix.
    if std::env::var_os("VALISE_CRASH_MATRIX_TRACE").is_some() {
        #[allow(clippy::print_stderr)]
        {
            eprintln!("[crash-matrix] {probe:?}");
        }
    }
    match probe {
        Ok(outcome) => outcome,
        Err(payload) => {
            let text = payload
                .downcast_ref::<&str>()
                .map(|s| (*s).to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "non-string panic payload".to_string());
            Outcome::Panicked(text)
        }
    }
}

fn probe_file(path: &Path, fx: &Fixture) -> Outcome {
    let file = match ValiseFile::open_read_only(path) {
        Ok(file) => file,
        Err(err) => return Outcome::CleanReject(err.to_string()),
    };
    let generation = file.header().snapshot_generation;
    let frames = file.frame_stubs().len();

    let reads = [
        (file.read_payload(fx.frame_a1), PAYLOAD_A1, "A1"),
        (file.read_payload(fx.frame_a2), PAYLOAD_A2, "A2"),
        (file.read_payload(fx.frame_b1), PAYLOAD_B1, "B1"),
        (file.read_payload(fx.frame_b2), PAYLOAD_B2, "B2"),
    ];

    let matches_gen2 = frames == 4
        && reads
            .iter()
            .all(|(read, want, _)| matches!(read, Ok(bytes) if bytes == want));
    if matches_gen2 {
        return Outcome::NewSnapshot { generation };
    }

    let matches_gen1 = frames == 2
        && matches!(&reads[0].0, Ok(bytes) if bytes == PAYLOAD_A1)
        && matches!(&reads[1].0, Ok(bytes) if bytes == PAYLOAD_A2)
        && reads[2].0.is_err()
        && reads[3].0.is_err();
    if matches_gen1 {
        return Outcome::OldSnapshot { generation };
    }

    // Neither snapshot. Wrong bytes anywhere = Anomalous; all-typed
    // errors = fail-safe RejectedAtRead.
    let mut detail = format!("generation={generation} frames={frames}");
    let mut wrong_bytes = false;
    for (read, want, name) in &reads {
        match read {
            Ok(bytes) if bytes == want => detail.push_str(&format!(" {name}=ok")),
            Ok(bytes) => {
                wrong_bytes = true;
                detail.push_str(&format!(
                    " {name}=WRONG({:?})",
                    String::from_utf8_lossy(bytes)
                ));
            }
            Err(err) => detail.push_str(&format!(" {name}=err({err})")),
        }
    }
    if wrong_bytes {
        Outcome::Anomalous(detail)
    } else {
        Outcome::RejectedAtRead(detail)
    }
}

// ---------------------------------------------------------------------------
// Tier A — per-fault-class outcomes under the scan-back contract. Every
// case must be fail-safe AND recover the newest fully-valid snapshot:
// never a panic, never wrong bytes, never an unopenable file when a
// durable snapshot still exists.
// ---------------------------------------------------------------------------

/// Case 1: tear the last byte off the committed gen2 footer, as a crash
/// during the footer append would after the header pointer already
/// became durable (the reorder hole). Scan-back contract: the gen2
/// footer fails its embedded checksums, so open re-anchors on the
/// still-durable gen1 footer. (Was `CleanReject` before scan-back.)
#[test]
fn truncate_tail_1_byte() {
    let fx = fixture();
    let fault = apply_fault(fx, |bytes| {
        bytes.truncate(bytes.len() - 1);
    });
    let outcome = classify(&fault.path, fx);
    assert_recovers_gen1(fx, &outcome, "truncating 1 byte off EOF");
}

/// Case 2a: truncate inside the footer's 88-byte header scalar fields.
/// (Was `CleanReject` before scan-back.)
#[test]
fn truncate_mid_footer_header() {
    let fx = fixture();
    let cut = fx.footer_offset + 16;
    let fault = apply_fault(fx, |bytes| bytes.truncate(cut as usize));
    let outcome = classify(&fault.path, fx);
    assert_recovers_gen1(fx, &outcome, "truncating inside the footer header");
}

/// Case 2b: truncate inside the footer's embedded checksum region
/// (footer_checksum at footer bytes 56..88).
/// (Was `CleanReject` before scan-back.)
#[test]
fn truncate_mid_footer_checksum() {
    let fx = fixture();
    let cut = fx.footer_offset + 60;
    let fault = apply_fault(fx, |bytes| bytes.truncate(cut as usize));
    let outcome = classify(&fault.path, fx);
    assert_recovers_gen1(fx, &outcome, "truncating inside the footer checksum");
}

/// Case 2c: truncate midway through the footer body.
/// (Was `CleanReject` before scan-back.)
#[test]
fn truncate_mid_footer_body() {
    let fx = fixture();
    let body_len = fx.footer_len - TOC_HEADER_LEN;
    let cut = fx.footer_offset + TOC_HEADER_LEN + body_len / 2;
    let fault = apply_fault(fx, |bytes| bytes.truncate(cut as usize));
    let outcome = classify(&fault.path, fx);
    assert_recovers_gen1(fx, &outcome, "truncating mid footer body");
}

/// Case 3: truncate inside the gen2 segment region (payload + catalog
/// segments appended before the footer). The header points past EOF;
/// scan-back finds the intact gen1 footer below the cut.
/// (Was `CleanReject` before scan-back.)
#[test]
fn truncate_mid_last_segment() {
    let fx = fixture();
    let region = fx.footer_offset - fx.gen1_len;
    assert!(region > 0, "gen2 must have appended segments");
    let cut = fx.gen1_len + region / 2;
    let fault = apply_fault(fx, |bytes| bytes.truncate(cut as usize));
    let outcome = classify(&fault.path, fx);
    assert_recovers_gen1(fx, &outcome, "truncating mid last segment");
}

/// Case 4: simulated write reorder — the header pointer flip became
/// durable but the footer bytes it points at did not (all zero). This
/// is the reachable "header durable, footer torn" state: there is no
/// barrier between the footer/segment appends and the header rewrite,
/// only the single trailing F_FULLFSYNC. Scan-back closes exactly this
/// hole. (Was `CleanReject`; Tier B case 9 states the same guarantee.)
#[test]
fn zero_footer_bytes_header_points_at_it() {
    let fx = fixture();
    let fault = apply_fault(fx, |bytes| {
        let start = fx.footer_offset as usize;
        let end = (fx.footer_offset + fx.footer_len) as usize;
        bytes[start..end].fill(0);
    });
    let outcome = classify(&fault.path, fx);
    assert_recovers_gen1(fx, &outcome, "zeroed footer under a durable header");
}

/// Case 5: same reorder, but the footer region holds garbage instead of
/// zeros (e.g. stale page contents). Deterministic seeded bytes.
/// (Was `CleanReject` before scan-back.)
#[test]
fn garbage_footer_bytes() {
    let fx = fixture();
    let mut rng = XorShift64::new(0x9E37_79B9_7F4A_7C15);
    let fault = apply_fault(fx, |bytes| {
        let start = fx.footer_offset as usize;
        let end = (fx.footer_offset + fx.footer_len) as usize;
        for byte in &mut bytes[start..end] {
            *byte = rng.next() as u8;
        }
    });
    let outcome = classify(&fault.path, fx);
    assert_recovers_gen1(fx, &outcome, "garbage footer under a durable header");
}

/// Case 6a: single-byte XOR flips at deterministic offsets covering
/// every structural region of the committed gen2 footer: magic,
/// version, snapshot_generation, body_len, body_checksum,
/// footer_checksum, first body byte, last body byte. Every flip is
/// caught by the footer's embedded self-validation; the scan-back
/// contract then serves the newest remaining valid snapshot (gen1)
/// instead of rejecting the file. This is a detection-only fault class
/// (media corruption of an acknowledged commit): the gen2 data loss is
/// surfaced via the recovery warning log, not an open failure.
/// (Was `CleanReject` before scan-back.)
#[test]
fn bitflip_committed_footer() {
    let fx = fixture();
    let rel_offsets: [u64; 8] = [
        0,                 // magic
        5,                 // version
        12,                // snapshot_generation
        21,                // body_len (high-ish byte -> declared len past EOF)
        30,                // body_checksum field
        60,                // footer_checksum field
        TOC_HEADER_LEN,    // first body byte
        fx.footer_len - 1, // last body byte
    ];
    for rel in rel_offsets {
        let target = (fx.footer_offset + rel) as usize;
        let fault = apply_fault(fx, |bytes| bytes[target] ^= 0xFF);
        let outcome = classify(&fault.path, fx);
        assert_recovers_gen1(
            fx,
            &outcome,
            &format!("footer bit flip at relative offset {rel}"),
        );
    }
}

/// Case 6b: single-byte flips inside the gen2 CATALOG segment bytes
/// (the appended segments immediately before the footer). The gen2
/// footer itself still validates, but its reachable catalog roots fail
/// their checksums during the open-time load — scan-back treats an
/// anchor whose roots don't load exactly like a torn footer and falls
/// back to gen1. Must never surface as a different-but-valid snapshot.
/// (Was `CleanReject` before scan-back.)
#[test]
fn bitflip_committed_catalog_segment() {
    let fx = fixture();
    // The bytes directly before the footer are the tail of the last
    // catalog segment appended by the gen2 commit (segments are packed
    // with no padding; the footer starts at the byte after the last
    // segment ends).
    let rel_offsets: [u64; 3] = [1, 9, 33];
    for rel in rel_offsets {
        let target = (fx.footer_offset - rel) as usize;
        let fault = apply_fault(fx, |bytes| bytes[target] ^= 0xFF);
        let outcome = classify(&fault.path, fx);
        assert_recovers_gen1(
            fx,
            &outcome,
            &format!("catalog segment bit flip at footer-{rel}"),
        );
    }
}

/// Case 6c-1: flip inside the gen2 payload segment HEADER's stored
/// checksum field (bytes 44..76 of the 76-byte header). Payload
/// segments are not touched at open (only their registry entries are),
/// so under scan-back the file opens on gen2; the read path then
/// cross-checks the header's stored checksum against the registry's
/// stored copy (src/file/segment_io.rs::read_payload_ref) and the
/// divergence fails closed — a typed integrity error at read, never
/// wrong bytes, never a panic.
#[test]
fn bitflip_committed_payload_segment_header_checksum() {
    let fx = fixture();
    let target = (fx.payload_seg_b_header + SEGMENT_CHECKSUM_AT) as usize;
    let fault = apply_fault(fx, |bytes| bytes[target] ^= 0xAB);
    let outcome = classify(&fault.path, fx);
    match &outcome {
        Outcome::CleanReject(reason) | Outcome::RejectedAtRead(reason) => {
            assert!(
                reason.contains("checksum") || reason.contains("mismatch"),
                "expected an integrity rejection, got: {reason}"
            );
        }
        other => panic!("segment-header checksum flip must fail closed, got: {other:?}"),
    }
}

/// Case 6c-2: flip a single byte of frame B1's payload bytes inside
/// the committed, checksummed gen2 payload segment.
///
/// FIXED (was the critical silent-corruption gap): payload batch
/// segments are zstd-compressed, and the read path used to cross-check
/// only the two STORED checksum copies (registry SegmentRef vs segment
/// header) against each other, never re-hashing the wire bytes —
/// `zstd::decode_all` then accepted the tampered stream (frames carry
/// no zstd content checksum) and one flipped literal corrupted EVERY
/// frame in the batch via back-references.
///
/// `read_payload_ref` now re-hashes the segment's wire bytes against
/// the stored BLAKE3 before first decompression (once per segment per
/// open), so the flip surfaces as a typed integrity error on both gen2
/// frames: no wrong bytes are ever served, and the gen1 batch is
/// untouched. (Old pinned outcome: `Anomalous` with B1+B2 both wrong.)
#[test]
fn bitflip_committed_payload_segment_bytes_rejected_at_read() {
    let fx = fixture();
    let target = (fx.payload_b1_offset + 3) as usize;
    let fault = apply_fault(fx, |bytes| bytes[target] ^= 0x01);
    let outcome = classify(&fault.path, fx);
    match &outcome {
        Outcome::RejectedAtRead(detail) => {
            assert!(
                detail.contains("A1=ok") && detail.contains("A2=ok"),
                "gen1 payloads must still read back intact, got: {detail}"
            );
            assert!(
                detail.contains("B1=err") && detail.contains("B2=err"),
                "both frames of the tampered gen2 batch must fail closed, got: {detail}"
            );
            assert!(
                detail.contains("checksum"),
                "the rejection must be a typed integrity (checksum) error, got: {detail}"
            );
            assert!(
                !detail.contains("WRONG"),
                "no wrong bytes may ever be served, got: {detail}"
            );
        }
        Outcome::Panicked(msg) => panic!("payload bit flip must never panic: {msg}"),
        other => panic!(
            "payload byte flip must surface as a typed integrity error at \
             read (RejectedAtRead), got: {other:?}"
        ),
    }
}

/// Case 6c-3: the same payload bitflip as 6c-2, read through the
/// CONCURRENCY layer (`Database` → `ReadConnection` → `Snapshot::read_payload`,
/// src/concurrency/snapshot.rs) instead of the engine path. The
/// snapshot's lock-free mmap read used to share the old trust model —
/// it cross-checked only the two stored checksum copies and handed the
/// tampered zstd stream to `decode_all`. `Snapshot::read_payload` now
/// re-hashes the wire bytes against the stored BLAKE3 (once per segment
/// per handle, via the verified-set shared with the owning `ValiseFile`),
/// so the flip surfaces as a typed integrity error on both gen2 frames
/// while the gen1 batch reads back intact.
#[test]
fn bitflip_committed_payload_bytes_rejected_via_snapshot_read() {
    let fx = fixture();
    let target = (fx.payload_b1_offset + 3) as usize;
    let fault = apply_fault(fx, |bytes| bytes[target] ^= 0x01);

    let db = Database::open(&fault.path, OpenMode::ReadOnly).expect("open database read-only");
    let conn = db.reader();
    let snapshot = conn.snapshot();
    assert_eq!(snapshot.generation(), fx.gen2_generation);

    // gen1 batch: untouched, must read back intact through the snapshot.
    assert_eq!(
        snapshot.read_payload(fx.frame_a1).expect("A1 via snapshot"),
        PAYLOAD_A1
    );
    assert_eq!(
        snapshot.read_payload(fx.frame_a2).expect("A2 via snapshot"),
        PAYLOAD_A2
    );

    // gen2 batch: both frames fail closed with a typed checksum error;
    // no wrong bytes may ever be served.
    for (frame, name) in [(fx.frame_b1, "B1"), (fx.frame_b2, "B2")] {
        match snapshot.read_payload(frame) {
            Err(err) => assert!(
                err.to_string().contains("checksum"),
                "{name}: expected a typed integrity (checksum) error, got: {err}"
            ),
            Ok(bytes) => panic!(
                "{name}: snapshot read served bytes from a tampered payload \
                 segment: {:?}",
                String::from_utf8_lossy(&bytes)
            ),
        }
    }
}

/// Case 7: torn header rewrite — only the first half (0..60) of the
/// 120-byte logical header rewrite landed, leaving gen1's
/// footer_offset paired with gen2's snapshot_generation label (bytes
/// 72..80 live in the second half). Open cross-checks the label
/// against the referenced footer's own generation; on mismatch the
/// header is suspect and scan-back re-derives the anchor from the
/// footers alone. Both footers are intact here, so the recovered
/// snapshot is the highest-generation one — gen2 — reported with its
/// own (self-consistent) generation. Before the fix this state served
/// gen1 data mislabeled with gen2's generation.
#[test]
fn torn_header_partial_rewrite() {
    let fx = fixture();
    let fault = apply_fault(fx, |bytes| {
        bytes[..HEADER_LOGICAL_LEN / 2].copy_from_slice(&fx.gen1_header[..HEADER_LOGICAL_LEN / 2]);
    });
    let outcome = classify(&fault.path, fx);
    match &outcome {
        Outcome::NewSnapshot { generation } => {
            assert_eq!(
                *generation, fx.gen2_generation,
                "recovered snapshot must be self-consistent: gen2 data under gen2's label"
            );
        }
        other => panic!(
            "torn header must recover the newest valid snapshot with a \
             self-consistent generation, got: {other:?}"
        ),
    }
}

/// Case 7b: the opposite tear — the second half (60..120) of the
/// logical header rewrite is stale (gen1's snapshot_generation label)
/// while the first half (gen2's footer_offset) landed. The referenced
/// gen2 footer is valid but disagrees with the stale label, so the
/// same cross-check + scan re-anchors on gen2 with its own label.
#[test]
fn torn_header_stale_second_half() {
    let fx = fixture();
    let fault = apply_fault(fx, |bytes| {
        bytes[HEADER_LOGICAL_LEN / 2..HEADER_LOGICAL_LEN]
            .copy_from_slice(&fx.gen1_header[HEADER_LOGICAL_LEN / 2..HEADER_LOGICAL_LEN]);
    });
    let outcome = classify(&fault.path, fx);
    match &outcome {
        Outcome::NewSnapshot { generation } => {
            assert_eq!(
                *generation, fx.gen2_generation,
                "recovered snapshot must be self-consistent: gen2 data under gen2's label"
            );
        }
        other => panic!(
            "torn header (stale second half) must recover the newest valid \
             snapshot with a self-consistent generation, got: {other:?}"
        ),
    }
}

/// Case 8: orphan bytes appended past the committed footer (a crash
/// after segment appends of a NEXT commit, before its footer/header
/// landed). The header still points at the valid gen2 footer, so the
/// tail must be invisible: the file opens on the new (gen2) snapshot.
#[test]
fn uncommitted_tail_ignored() {
    let fx = fixture();
    let mut rng = XorShift64::new(0xDEAD_BEEF_CAFE_F00D);
    let fault = apply_fault(fx, |bytes| {
        let tail: Vec<u8> = (0..4096).map(|_| rng.next() as u8).collect();
        bytes.extend_from_slice(&tail);
    });
    let outcome = classify(&fault.path, fx);
    match &outcome {
        Outcome::NewSnapshot { generation } => {
            assert_eq!(*generation, fx.gen2_generation);
        }
        other => {
            panic!("orphan tail past the footer must be ignored (NewSnapshot), got: {other:?}")
        }
    }
}

// ---------------------------------------------------------------------------
// Tier B — the recovery contract that motivated the scan-back fix: a
// crash that tears the NEW commit must recover the previous
// ACKNOWLEDGED snapshot (gen1), not fail closed. Formerly `#[ignore]`d
// as the target end state; now live.
// ---------------------------------------------------------------------------

/// Case 9: the reorder state of case 4 (header durable, footer torn)
/// must recover gen1 — every byte of the gen1 snapshot is still intact
/// in the file; only the pointer chain to it was lost.
#[test]
fn reorder_recovers_previous_snapshot() {
    let fx = fixture();
    let fault = apply_fault(fx, |bytes| {
        let start = fx.footer_offset as usize;
        let end = (fx.footer_offset + fx.footer_len) as usize;
        bytes[start..end].fill(0);
    });
    let outcome = classify(&fault.path, fx);
    match &outcome {
        Outcome::OldSnapshot { generation } => assert_eq!(*generation, fx.gen1_generation),
        other => panic!(
            "target contract: torn gen2 footer must recover the acknowledged \
             gen1 snapshot, got: {other:?}"
        ),
    }
}

/// Case 10: a truncated gen2 footer (case 1) must likewise recover gen1.
#[test]
fn truncated_new_footer_recovers_previous_snapshot() {
    let fx = fixture();
    let fault = apply_fault(fx, |bytes| {
        bytes.truncate(bytes.len() - 1);
    });
    let outcome = classify(&fault.path, fx);
    match &outcome {
        Outcome::OldSnapshot { generation } => assert_eq!(*generation, fx.gen1_generation),
        other => panic!(
            "target contract: truncated gen2 footer must recover the acknowledged \
             gen1 snapshot, got: {other:?}"
        ),
    }
}

// ---------------------------------------------------------------------------
// Tier C — scan-back robustness: decoy magics, multi-generation tears,
// the post-recovery write path, and the zero-commit degenerate case.
// ---------------------------------------------------------------------------

/// Fixed 4 KB header size (src/format.rs `HEADER_SIZE`). Footers always
/// live at or past this offset.
const HEADER_SIZE: u64 = 4096;

/// Encode a forged TOC footer header: valid magic / version /
/// header_len and a small body_len, a maximally tempting generation
/// label, but garbage checksums and body. A scan that trusts anything
/// short of the full double-BLAKE3 validation would anchor on this.
/// The garbage regions use seeded high-entropy bytes so zstd stores
/// them as raw literals (a repeated-byte fill would be run-length
/// encoded and never appear verbatim in the compressed payload).
fn forged_footer_bytes() -> Vec<u8> {
    let mut rng = XorShift64::new(0xF00D_FACE_D00D_5EED);
    let mut forged = Vec::new();
    forged.extend_from_slice(TOC_MAGIC); // magic
    forged.extend_from_slice(&1u16.to_le_bytes()); // TOC_VERSION
    forged.extend_from_slice(&(TOC_HEADER_LEN as u16).to_le_bytes()); // header_len
    forged.extend_from_slice(&u64::MAX.to_le_bytes()); // "snapshot_generation"
    forged.extend_from_slice(&8u64.to_le_bytes()); // body_len
    // Garbage body checksum + footer checksum + 8-byte "body".
    forged.extend((0..72).map(|_| rng.next() as u8));
    forged
}

/// Case 12: `VLTC` decoy bytes — one forged footer embedded in a
/// COMMITTED gen1 payload, and one planted directly over the torn gen2
/// footer region — must be skipped by the scan (their checksums cannot
/// validate) without aborting it, and the real gen1 snapshot behind
/// them must be recovered with its payload intact.
#[test]
fn scanback_skips_nxtc_decoys() {
    let dir = tempfile::tempdir().expect("decoy tempdir");
    let path = dir.path().join("decoy.vls");
    let forged = forged_footer_bytes();
    // Payload batches are zstd-compressed; small/compressible inputs
    // get entropy-coded literals and the decoy would not appear
    // byte-verbatim in the image. A high-entropy carrier forces zstd's
    // raw-block fallback so the planted footer bytes land on disk
    // exactly as written — which is what the scan must walk past.
    let mut carrier_rng = XorShift64::new(0xDEC0_7B17_E5CA_11ED);
    let mut decoy_payload: Vec<u8> = (0..2048).map(|_| carrier_rng.next() as u8).collect();
    decoy_payload.extend_from_slice(&forged);
    decoy_payload.extend((0..2048).map(|_| carrier_rng.next() as u8));

    let mut valise = ValiseFile::create(&path).expect("create decoy file");
    let collection = valise.create_collection("docs").expect("create collection");
    let frame_decoy = valise
        .put_frame(PutFrame::document(collection, &decoy_payload))
        .expect("put decoy frame");
    let gen1 = valise.commit().expect("commit gen1");
    drop(valise);

    let mut valise = ValiseFile::open_read_write(&path).expect("reopen for gen2");
    valise
        .put_frame(PutFrame::document(collection, b"gen2-doomed-frame"))
        .expect("put gen2 frame");
    valise.commit().expect("commit gen2");
    drop(valise);

    let mut bytes = fs::read(&path).expect("read image");
    // The payload decoy must be byte-visible in the committed image
    // (zstd stores first occurrences as raw literals), otherwise this
    // test would not exercise the scan at all.
    let occurrences = bytes
        .windows(forged.len())
        .filter(|w| *w == &forged[..])
        .count();
    assert_eq!(
        occurrences, 1,
        "forged footer bytes must appear verbatim inside committed payload data"
    );

    // Fault: zero the gen2 footer (the reorder hole), then plant a
    // SECOND decoy inside the zeroed region — right where the scan
    // walks past the torn commit.
    let footer_offset = le_u64(&bytes, HEADER_FOOTER_OFFSET_AT);
    let body_len = le_u64(&bytes, (footer_offset + TOC_BODY_LEN_AT) as usize);
    let footer_len = TOC_HEADER_LEN + body_len;
    assert!(
        footer_len as usize >= 8 + forged.len(),
        "footer region must fit the planted decoy"
    );
    bytes[footer_offset as usize..(footer_offset + footer_len) as usize].fill(0);
    let plant_at = (footer_offset + 8) as usize;
    bytes[plant_at..plant_at + forged.len()].copy_from_slice(&forged);
    fs::write(&path, &bytes).expect("write faulted image");

    let reopened =
        ValiseFile::open_read_only(&path).expect("scan-back must recover gen1 despite decoys");
    assert_eq!(
        reopened.header().snapshot_generation,
        gen1.snapshot_generation,
        "recovery must anchor on the real gen1 footer, not a forged generation"
    );
    assert_eq!(reopened.frame_stubs().len(), 1);
    assert_eq!(
        reopened
            .read_payload(frame_decoy)
            .expect("decoy-carrying payload must read back"),
        decoy_payload
    );
}

/// Case 13: deep scan-back — three committed generations; the gen3
/// footer is torn by truncation AND the gen2 footer is zeroed (two
/// successive torn commits). Recovery must walk all the way back to
/// gen1, the newest fully-valid snapshot.
#[test]
fn deep_scanback_recovers_two_generations_back() {
    let dir = tempfile::tempdir().expect("deep tempdir");
    let path = dir.path().join("deep.vls");

    let mut valise = ValiseFile::create(&path).expect("create deep file");
    let collection = valise.create_collection("docs").expect("create collection");
    let frame_d1 = valise
        .put_frame(PutFrame::document(collection, b"deep-gen1-D1"))
        .expect("put D1");
    let gen1 = valise.commit().expect("commit gen1");
    drop(valise);

    let mut valise = ValiseFile::open_read_write(&path).expect("reopen for gen2");
    let frame_d2 = valise
        .put_frame(PutFrame::document(collection, b"deep-gen2-D2"))
        .expect("put D2");
    valise.commit().expect("commit gen2");
    drop(valise);
    let image2 = fs::read(&path).expect("gen2 image");

    let mut valise = ValiseFile::open_read_write(&path).expect("reopen for gen3");
    let frame_d3 = valise
        .put_frame(PutFrame::document(collection, b"deep-gen3-D3"))
        .expect("put D3");
    valise.commit().expect("commit gen3");
    drop(valise);
    let mut bytes = fs::read(&path).expect("gen3 image");

    // Footer geometry comes from each generation's own image (offsets
    // are stable — the file is append-only).
    let gen2_footer = le_u64(&image2, HEADER_FOOTER_OFFSET_AT);
    let gen2_footer_len =
        TOC_HEADER_LEN + le_u64(&image2, (gen2_footer + TOC_BODY_LEN_AT) as usize);
    let gen3_footer = le_u64(&bytes, HEADER_FOOTER_OFFSET_AT);
    assert!(
        gen3_footer > gen2_footer,
        "gen3 footer must be appended past gen2's"
    );

    // Zero the gen2 footer, then truncate mid-way through the gen3
    // footer's 88-byte header.
    bytes[gen2_footer as usize..(gen2_footer + gen2_footer_len) as usize].fill(0);
    bytes.truncate((gen3_footer + 40) as usize);
    fs::write(&path, &bytes).expect("write faulted image");

    let reopened = ValiseFile::open_read_only(&path).expect("deep scan-back must recover gen1");
    assert_eq!(
        reopened.header().snapshot_generation,
        gen1.snapshot_generation,
        "with gen3 and gen2 both torn, gen1 is the newest valid snapshot"
    );
    assert_eq!(reopened.frame_stubs().len(), 1);
    assert_eq!(
        reopened.read_payload(frame_d1).expect("D1 must read back"),
        b"deep-gen1-D1"
    );
    assert!(
        reopened.read_payload(frame_d2).is_err(),
        "gen2 frame unreachable"
    );
    assert!(
        reopened.read_payload(frame_d3).is_err(),
        "gen3 frame unreachable"
    );
}

/// Case 14: the write path after a recovery. A ReadWrite open runs the
/// same scan-back as read-only; the next commit appends at EOF (the
/// torn gen2 bytes stay orphaned, never reclaimed or referenced) and
/// publishes generation = recovered + 1; a clean reopen then sees the
/// healed header/footer pair without any scan.
#[test]
fn recovered_file_accepts_new_commits() {
    let fx = fixture();
    // The reorder fault of case 4: header durable, gen2 footer zeroed.
    let fault = apply_fault(fx, |bytes| {
        let start = fx.footer_offset as usize;
        let end = (fx.footer_offset + fx.footer_len) as usize;
        bytes[start..end].fill(0);
    });

    let mut valise = ValiseFile::open_read_write(&fault.path).expect("read-write recovery open");
    assert_eq!(
        valise.header().snapshot_generation,
        fx.gen1_generation,
        "read-write open must recover the same snapshot as read-only"
    );

    let collection = valise.collections()[0].collection_id;
    let frame_c = valise
        .put_frame(PutFrame::document(collection, b"post-recovery-frame-C"))
        .expect("put frame after recovery");
    let outcome = valise.commit().expect("commit after recovery");
    assert!(outcome.changed);
    assert_eq!(
        outcome.snapshot_generation,
        fx.gen1_generation + 1,
        "post-recovery commit must publish generation = recovered + 1"
    );
    drop(valise);

    let reopened = ValiseFile::open_read_only(&fault.path).expect("reopen after healing commit");
    assert_eq!(
        reopened.header().snapshot_generation,
        fx.gen1_generation + 1
    );
    assert_eq!(reopened.frame_stubs().len(), 3);
    assert_eq!(reopened.read_payload(fx.frame_a1).expect("A1"), PAYLOAD_A1);
    assert_eq!(reopened.read_payload(fx.frame_a2).expect("A2"), PAYLOAD_A2);
    assert_eq!(
        reopened.read_payload(frame_c).expect("frame C"),
        b"post-recovery-frame-C"
    );
}

/// Case 15a: a freshly created file with zero commits opens exactly as
/// before the scan-back work — `create()` durably writes the empty
/// generation-1 footer, so the fast path anchors on it with no scan.
#[test]
fn zero_commit_file_reopens_cleanly() {
    let dir = tempfile::tempdir().expect("zero-commit tempdir");
    let path = dir.path().join("empty.vls");
    let valise = ValiseFile::create(&path).expect("create");
    drop(valise);

    let reopened = ValiseFile::open_read_only(&path).expect("zero-commit file must open");
    assert_eq!(
        reopened.header().snapshot_generation,
        1,
        "create() stamps the empty generation-1 footer"
    );
    assert!(reopened.frame_stubs().is_empty());
}

/// Case 15b: a zero-commit file whose ONLY footer is torn has no older
/// durable snapshot to scan back to — open must still fail closed with
/// a typed error, preserving pre-scan-back behavior for the degenerate
/// case. (Unreachable via the commit protocol: create() is FullSync-
/// durable before it returns. This is a hand-torn file.)
#[test]
fn zero_commit_file_with_torn_footer_still_rejects() {
    let dir = tempfile::tempdir().expect("torn-empty tempdir");
    let path = dir.path().join("torn-empty.vls");
    let valise = ValiseFile::create(&path).expect("create");
    drop(valise);

    // Truncate mid-way through the sole (generation-1) footer's header.
    let file = fs::OpenOptions::new()
        .write(true)
        .open(&path)
        .expect("open for truncation");
    file.set_len(HEADER_SIZE + 40).expect("truncate mid-footer");
    drop(file);

    let err = match ValiseFile::open_read_only(&path) {
        Err(err) => err,
        Ok(_) => panic!("no valid footer anywhere: open must fail closed"),
    };
    assert!(
        err.to_string().contains("no recoverable snapshot"),
        "expected the scan-back exhaustion error, got: {err}"
    );
}

// ---------------------------------------------------------------------------
// Tier D — hardening findings from the randomized campaign: the
// unvalidated-length allocation guard and the footer-consensus
// contract-digest fallback.
// ---------------------------------------------------------------------------

/// 32-byte create-contract BLAKE3 digest, stamped once at create
/// (src/format.rs `CREATE_CONTRACT_DIGEST_OFFSET`), outside the 0..120
/// logical-prefix rewrite range.
const CONTRACT_DIGEST_AT: usize = 832;
const CONTRACT_DIGEST_LEN: usize = 32;
/// `payload_length` (LE u64) lives at bytes 36..44 of the 76-byte
/// segment header (src/format/segment.rs `SegmentHeaderCodec`). It is
/// raw wire data: the segment header carries no checksum of its own.
const SEGMENT_PAYLOAD_LEN_AT: u64 = 36;

/// Garble the header's create-contract digest region so it matches no
/// footer: invert the first byte (guaranteed mismatch) and randomize
/// the rest.
fn garble_contract_digest(bytes: &mut [u8], rng: &mut XorShift64) {
    bytes[CONTRACT_DIGEST_AT] ^= 0xFF;
    for byte in &mut bytes[CONTRACT_DIGEST_AT + 1..CONTRACT_DIGEST_AT + CONTRACT_DIGEST_LEN] {
        *byte = rng.next() as u8;
    }
}

/// Case 17 (crash-campaign regression, random-bitflip iteration 1905):
/// a single bit flip that sets bit 54 of a committed Payload segment
/// header's `payload_length` used to make `read_payload_ref` allocate
/// 2^54+N bytes BEFORE any validation — and allocation failure ABORTS
/// the process (not unwindable, so even `catch_unwind` cannot contain
/// it). The read path now cross-checks `payload_length` against the
/// registry's `SegmentRef` and the real file size before any
/// allocation, so the flip surfaces as a typed integrity error. This
/// test's own survival is the assertion that no abort happens.
#[test]
fn bitflip_payload_length_high_bit_rejected_without_abort() {
    let fx = fixture();
    // Bit 54 of the LE u64 at header+36..44 = byte header+42, bit 6.
    let target = (fx.payload_seg_b_header + SEGMENT_PAYLOAD_LEN_AT + 6) as usize;
    let fault = apply_fault(fx, |bytes| bytes[target] ^= 1 << 6);
    let outcome = classify(&fault.path, fx);
    match &outcome {
        Outcome::RejectedAtRead(detail) => {
            assert!(
                detail.contains("A1=ok") && detail.contains("A2=ok"),
                "gen1 payloads must still read back intact, got: {detail}"
            );
            assert!(
                detail.contains("B1=err") && detail.contains("B2=err"),
                "reads through the poisoned length must fail closed, got: {detail}"
            );
            assert!(
                detail.contains("length"),
                "the rejection must be the typed length-validation error, got: {detail}"
            );
            assert!(!detail.contains("WRONG"), "no wrong bytes, got: {detail}");
        }
        other => panic!(
            "petabyte-scale declared payload length must fail closed at read \
             (RejectedAtRead), got: {other:?}"
        ),
    }
}

/// Case 18a: the header's byte-832 contract digest region is randomized
/// while every footer is intact. The header anchor matches no valid
/// footer, so recovery falls back to footer consensus: the create-time,
/// gen1 and gen2 footers all embed the same contract (3 agreeing
/// votes), and the newest fully-valid snapshot (gen2) is recovered with
/// a self-consistent label. Before the consensus fallback this state
/// bricked the file (fail-closed on every open).
#[test]
fn corrupted_contract_digest_recovers_via_footer_consensus() {
    let fx = fixture();
    let mut rng = XorShift64::new(0x0C07_7AC7_D16E_57ED);
    let fault = apply_fault(fx, |bytes| garble_contract_digest(bytes, &mut rng));
    let outcome = classify(&fault.path, fx);
    match &outcome {
        Outcome::NewSnapshot { generation } => assert_eq!(
            *generation, fx.gen2_generation,
            "consensus recovery must serve the newest valid snapshot self-consistently"
        ),
        other => panic!(
            "corrupt header digest with intact footers must recover via \
             footer consensus, got: {other:?}"
        ),
    }
}

/// Case 18b: corrupt digest AND a zeroed newest footer — consensus must
/// come from the remaining two agreeing footers (create-time + gen1)
/// and recovery lands on gen1.
#[test]
fn corrupted_contract_digest_with_torn_footer_recovers_previous() {
    let fx = fixture();
    let mut rng = XorShift64::new(0xD16E_57ED_0C07_7AC7);
    let fault = apply_fault(fx, |bytes| {
        garble_contract_digest(bytes, &mut rng);
        let start = fx.footer_offset as usize;
        let end = (fx.footer_offset + fx.footer_len) as usize;
        bytes[start..end].fill(0);
    });
    let outcome = classify(&fault.path, fx);
    assert_recovers_gen1(fx, &outcome, "corrupt digest + torn newest footer");
}

/// Case 18c: a fresh zero-commit file with a corrupted digest region —
/// a single valid footer is its own consensus, so the file still opens
/// on the empty generation-1 snapshot.
#[test]
fn corrupted_contract_digest_fresh_file_single_footer_consensus() {
    let dir = tempfile::tempdir().expect("fresh-digest tempdir");
    let path = dir.path().join("fresh-digest.vls");
    let valise = ValiseFile::create(&path).expect("create");
    drop(valise);

    let mut rng = XorShift64::new(0x517E_C0DE_D16E_57ED);
    let mut bytes = fs::read(&path).expect("read image");
    garble_contract_digest(&mut bytes, &mut rng);
    fs::write(&path, &bytes).expect("write faulted image");

    let reopened = ValiseFile::open_read_only(&path)
        .expect("single-footer consensus must recover a fresh file");
    assert_eq!(reopened.header().snapshot_generation, 1);
    assert!(reopened.frame_stubs().is_empty());
}

/// Case 18d: corrupt digest AND every footer destroyed — no candidate
/// exists, consensus has nothing to vote with, and the open must still
/// fail closed with a typed error.
#[test]
fn corrupted_contract_digest_with_no_valid_footer_fails_closed() {
    let fx = fixture();
    let mut rng = XorShift64::new(0x0BAD_D16E_0BAD_F007);
    // Footer geometry: gen2 from the fixture fields; gen1 via its
    // header prefix; the create-time footer sits at HEADER_SIZE. The
    // pristine image still holds every footer's bytes (append-only),
    // so lengths can be read off the image directly.
    let gen1_footer = le_u64(&fx.gen1_header, HEADER_FOOTER_OFFSET_AT);
    let gen1_footer_len =
        TOC_HEADER_LEN + le_u64(&fx.pristine, (gen1_footer + TOC_BODY_LEN_AT) as usize);
    let create_footer = HEADER_SIZE;
    let create_footer_len =
        TOC_HEADER_LEN + le_u64(&fx.pristine, (create_footer + TOC_BODY_LEN_AT) as usize);
    let fault = apply_fault(fx, |bytes| {
        garble_contract_digest(bytes, &mut rng);
        for (offset, len) in [
            (fx.footer_offset, fx.footer_len),
            (gen1_footer, gen1_footer_len),
            (create_footer, create_footer_len),
        ] {
            bytes[offset as usize..(offset + len) as usize].fill(0);
        }
    });
    let outcome = classify(&fault.path, fx);
    match &outcome {
        Outcome::CleanReject(reason) => assert!(
            reason.contains("no recoverable snapshot"),
            "expected the scan-back exhaustion error, got: {reason}"
        ),
        other => panic!(
            "no valid footer anywhere: consensus cannot apply and open must \
             fail closed, got: {other:?}"
        ),
    }
}

// ---------------------------------------------------------------------------
// Case 11: SIGKILL-mid-commit campaign. A child PROCESS loops small
// commits against a real file and records every ACKNOWLEDGED
// generation (write happens only after commit() returns). The parent
// SIGKILLs it at a seeded pseudo-random point, reopens the file, and
// checks that no acknowledged commit was lost. A process kill (unlike
// power loss) preserves the OS page cache, so with the current
// one-fsync protocol the file must reopen cleanly on every iteration.
// ---------------------------------------------------------------------------

#[cfg(unix)]
mod kill_campaign {
    use std::{
        fs,
        io::Write,
        process::{Command, Stdio},
        time::{Duration, Instant},
    };

    use valise::{PutFrame, ValiseFile};

    use super::XorShift64;

    const CHILD_STORE_ENV: &str = "VALISE_CRASH_CONSISTENCY_CHILD_STORE";
    const CHILD_ACK_ENV: &str = "VALISE_CRASH_CONSISTENCY_CHILD_ACK";
    /// Safety valve so an unkilled child terminates on its own.
    const CHILD_MAX_COMMITS: usize = 500;
    const CAMPAIGN_ITERATIONS: usize = 5;

    /// Child worker. Runs as a no-op passing test unless spawned by
    /// `sigkill_mid_commit_campaign` with the env vars set, in which
    /// case it opens the parent-created store and loops: put_frame →
    /// commit → append the acknowledged generation to the ack file.
    /// The ack write is a single unbuffered `write_all` syscall issued
    /// strictly AFTER commit() returned, so every recorded generation
    /// was acknowledged to the "application".
    #[test]
    fn kill_campaign_child_worker() {
        let Some(store) = std::env::var_os(CHILD_STORE_ENV) else {
            return; // normal test run: nothing to do
        };
        let ack_path = std::env::var_os(CHILD_ACK_ENV).expect("child ack path env");

        let mut ack = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&ack_path)
            .expect("child: open ack file");
        let mut valise = ValiseFile::open_read_write(&store).expect("child: open store read-write");
        let collection = valise.collections()[0].collection_id;

        for i in 0..CHILD_MAX_COMMITS {
            let payload = format!("kill-campaign-frame-{i}");
            valise
                .put_frame(PutFrame::document(collection, payload.as_bytes()))
                .expect("child: put_frame");
            let outcome = valise.commit().expect("child: commit");
            // Acknowledged: commit returned. Record it durably enough
            // for a process kill (kernel write, no userspace buffer).
            let line = format!("{}\n", outcome.snapshot_generation);
            ack.write_all(line.as_bytes()).expect("child: record ack");
        }
    }

    #[test]
    fn sigkill_mid_commit_campaign() {
        let mut rng = XorShift64::new(0x5EED_4B11_CA3B_A167);
        let mut clean_rejects: Vec<String> = Vec::new();

        for iteration in 0..CAMPAIGN_ITERATIONS {
            let dir = tempfile::tempdir().expect("campaign tempdir");
            let store = dir.path().join("campaign.vls");
            let ack_path = dir.path().join("acks.txt");

            // Seed store: one collection, one committed frame.
            let mut valise = ValiseFile::create(&store).expect("campaign: create store");
            let collection = valise
                .create_collection("kill-campaign")
                .expect("campaign: collection");
            valise
                .put_frame(PutFrame::document(collection, b"seed-frame"))
                .expect("campaign: seed frame");
            let base = valise.commit().expect("campaign: seed commit");
            drop(valise);
            let base_generation = base.snapshot_generation;

            // Spawn the child worker inside this same test binary.
            let exe = std::env::current_exe().expect("campaign: current_exe");
            let mut child = Command::new(exe)
                .arg("kill_campaign::kill_campaign_child_worker")
                .arg("--exact")
                .arg("--test-threads")
                .arg("1")
                .env(CHILD_STORE_ENV, &store)
                .env(CHILD_ACK_ENV, &ack_path)
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .expect("campaign: spawn child");

            // Wait for the first acknowledged commit so the kill lands
            // inside the commit loop, then kill after a seeded delay.
            let deadline = Instant::now() + Duration::from_secs(60);
            loop {
                let has_ack = fs::read_to_string(&ack_path)
                    .map(|s| s.contains('\n'))
                    .unwrap_or(false);
                if has_ack {
                    break;
                }
                if Instant::now() > deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    panic!("campaign iteration {iteration}: child produced no ack within 60s");
                }
                std::thread::sleep(Duration::from_millis(2));
            }
            let delay_ms = 1 + rng.next() % 40;
            std::thread::sleep(Duration::from_millis(delay_ms));
            child.kill().expect("campaign: SIGKILL child");
            child.wait().expect("campaign: reap child");

            // Last acknowledged generation. A torn final line (killed
            // mid-ack-write) is ignored — that commit was acked to the
            // child but not fully recorded, which only weakens the
            // check, never falsely fails it.
            let acked = fs::read_to_string(&ack_path).expect("campaign: read acks");
            let last_acked = acked
                .lines()
                .filter_map(|l| l.parse::<u64>().ok())
                .max()
                .expect("campaign: at least one acknowledged commit");
            assert!(last_acked > base_generation);

            // Reopen and verify no acknowledged commit was lost.
            match ValiseFile::open_read_only(&store) {
                Err(err) => clean_rejects.push(format!(
                    "iteration {iteration}: open failed after SIGKILL: {err}"
                )),
                Ok(file) => {
                    let generation = file.header().snapshot_generation;
                    assert!(
                        generation >= last_acked,
                        "iteration {iteration}: LOST ACKNOWLEDGED COMMIT — \
                         last acked generation {last_acked}, visible {generation}"
                    );
                    assert!(
                        generation <= last_acked + 1,
                        "iteration {iteration}: visible generation {generation} is more \
                         than one ahead of last recorded ack {last_acked}"
                    );
                    // Structural coherence: the seed commit had 1 frame
                    // and every later commit added exactly one.
                    let expected_frames = 1 + (generation - base_generation) as usize;
                    assert_eq!(
                        file.frame_stubs().len(),
                        expected_frames,
                        "iteration {iteration}: frame count does not match the visible generation"
                    );
                }
            }
        }

        // Process kill preserves the page cache: with the current
        // protocol every iteration must reopen cleanly.
        assert!(
            clean_rejects.is_empty(),
            "SIGKILL campaign produced unexpected open failures:\n{}",
            clean_rejects.join("\n")
        );
    }
}
