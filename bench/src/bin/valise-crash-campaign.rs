// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Randomized crash-consistency STRESS CAMPAIGN for the Valise commit +
//! scan-back-recovery protocol.
//!
//! The deterministic fault matrix (`tests/crash_consistency.rs`) pins
//! the per-fault-class outcome contract; this binary provides VOLUME:
//! thousands of seeded randomized faults against a pristine
//! multi-generation `.vls` image, with every reopen classified into the
//! same recovery-outcome taxonomy and timed. Results land as one JSON
//! document that feeds the paper's recovery-outcomes table.
//!
//! Fault classes (all placement/extent decisions come from a seeded
//! xorshift64* stream — no time-derived randomness anywhere):
//!
//! 1. `random-truncation`: truncate to a uniform length in (4 KiB, len).
//! 2. `random-bitflip`: flip 1 random bit past the 4 KiB header.
//! 3. `random-multi-bitflip`: flip k ~ U[2,8] distinct random bits.
//! 4. `torn-footer`: zero/garble a prefix/suffix/window of the final
//!    TOC footer.
//! 5. `torn-header`: overwrite a random sub-range of the logical header
//!    (bytes 0..120) with the previous generation's header bytes or
//!    random bytes; half the time ALSO corrupt the byte-832
//!    create-contract digest (which must fail closed).
//! 6. `reorder-sim`: header points at the new footer, but random
//!    windows of the last commit's appended bytes (segments + footer)
//!    are zeroed/garbled — arbitrary writeback subsets of the final
//!    commit.
//! 7. `random-garbage-window`: overwrite a random 64 B – 4 KiB window
//!    anywhere past the header with random bytes.
//!
//! Stateful / process-level classes (same JSON counters):
//!
//! 8. `crash-recover-cycle`: 3 rounds per iteration of fault (from the
//!    torn-footer / reorder-sim repertoire, confined to the newest
//!    commit's bytes) → ReadWrite recovery open → byte-exact content
//!    verification → heal commit appending one frame → close. Enforced
//!    across rounds: never wrong bytes, the served generation loses at
//!    most the single faulted commit (never regresses below the
//!    previous round's acknowledged commit minus one), the healing
//!    commit publishes served+1, and the final reopen serves every
//!    surviving appended frame byte-exact.
//! 9. `sigkill-storm`: a child process (this binary re-invoked in a
//!    hidden worker mode) streams small commits, acking each generation
//!    on stdout after `commit()` returns; the parent SIGKILLs it at a
//!    seeded 1–40 ms delay past the first ack, reopens, and requires
//!    visible generation >= last ack (zero lost acknowledged commits)
//!    with exact frame content. Kill delays are seeded, but real
//!    scheduling makes the outcome distribution machine-dependent.
//! 10. `compaction-install`: the atomic-replace crash states of
//!     `db::compact` (temp at `src.vls.compacting`, install = rename +
//!     parent-dir fsync), constructed SYNTHETICALLY from two complete
//!     images (see the JSON `notes` field): (A) temp fully committed +
//!     original intact, (B) rename completed, (C) torn temp + original
//!     intact. Opening the MAIN path must always yield the complete old
//!     (A, C) or complete new (B) store — never torn, never influenced
//!     by a leftover `.vls.compacting` sibling.
//!
//! Per-iteration classification (mirrors `tests/crash_consistency.rs`;
//! content is verified byte-for-byte, not just "open succeeded"):
//!
//! - `recovered_newest`: opens; the newest generation's frames all read
//!   back exactly; generation self-consistent.
//! - `recovered_previous`: opens at an earlier generation; that
//!   generation's frames all read back exactly and no newer frame is
//!   visible.
//! - `rejected_at_read`: opens; some payload read returns a typed
//!   error; NO wrong bytes anywhere.
//! - `clean_reject`: `open_read_only` returns a typed error.
//! - `wrong_data`: any read returns bytes differing from the pristine
//!   expected content for the served generation (MUST be zero).
//! - `panic`: open/read panicked (MUST be zero).
//!
//! Scan-back activation is not observable through the public API, so
//! the campaign records the served-generation distribution per class
//! (`served_generation_hist`): any open that serves a generation below
//! the newest necessarily went through scan-back recovery.
//!
//! With `--scale-sweep`, recovery timing is additionally measured on
//! ~1/16/64/256 MiB files (incompressible payloads): the final footer
//! is zeroed and the scan-back open time (median of 5) is reported
//! against the clean open time.
//!
//! ## Reproduce
//!
//! ```bash
//! cargo build --release -p valise-bench --bin valise-crash-campaign
//! target/release/valise-crash-campaign \
//!     --iters-per-class 15000 \
//!     --cycle-iters 2000 \
//!     --kill-iters 200 \
//!     --scale-sweep \
//!     --out bench/results/paper/crash_campaign_full.json
//! ```

use std::{
    collections::{BTreeMap, HashSet},
    fs,
    io::{BufRead, BufReader, Write as _},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::{Arc, Mutex},
    thread,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, bail, ensure};
use clap::Parser;
use serde::Serialize;

use valise::{PutFrame, ValiseFile};

// ---------------------------------------------------------------------------
// On-disk layout constants, mirrored from src/format (header.rs / toc.rs)
// exactly as in tests/crash_consistency.rs.
// ---------------------------------------------------------------------------

/// Fixed 4 KB header size (src/format.rs `HEADER_SIZE`).
const HEADER_SIZE: u64 = 4096;
/// Logical header prefix rewritten on every commit (bytes 0..120).
const HEADER_LOGICAL_LEN: usize = 120;
/// `Header.footer_offset` lives at bytes 32..40 (LE u64).
const HEADER_FOOTER_OFFSET_AT: usize = 32;
/// `Header.snapshot_generation` lives at bytes 72..80 (LE u64).
const HEADER_GENERATION_AT: usize = 72;
/// 32-byte create-contract BLAKE3 digest, stamped once at create
/// (src/format.rs `CREATE_CONTRACT_DIGEST_OFFSET`).
const CONTRACT_DIGEST_AT: usize = 832;
const CONTRACT_DIGEST_LEN: usize = 32;
/// Fixed TOC footer header length (src/format/toc.rs `TOC_HEADER_LEN`).
const TOC_HEADER_LEN: u64 = 88;
/// `body_len` field inside the TOC footer header (LE u64 at 16..24).
const TOC_BODY_LEN_AT: u64 = 16;
const TOC_MAGIC: &[u8; 4] = b"VLTC";

// ---------------------------------------------------------------------------
// Deterministic PRNG: SplitMix64 for stream derivation, xorshift64* for
// draws (same generator as tests/crash_consistency.rs).
// ---------------------------------------------------------------------------

fn splitmix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

struct XorShift64(u64);

impl XorShift64 {
    fn new(seed: u64) -> Self {
        // A zero state would lock the generator; the |1 keeps every
        // derived stream live without hurting distribution.
        Self(seed | 1)
    }

    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform draw in `[0, n)`.
    fn below(&mut self, n: u64) -> u64 {
        assert!(n > 0, "BUG: below(0)");
        self.next() % n
    }

    /// Uniform draw in `[lo, hi)`.
    fn range(&mut self, lo: u64, hi: u64) -> u64 {
        assert!(lo < hi, "BUG: empty range {lo}..{hi}");
        lo + self.below(hi - lo)
    }

    fn coin(&mut self) -> bool {
        self.next() & 1 == 1
    }

    fn fill(&mut self, buf: &mut [u8]) {
        for byte in buf.iter_mut() {
            *byte = self.next() as u8;
        }
    }
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(
    name = "valise-crash-campaign",
    about = "Randomized crash-consistency stress campaign: seeded faults → reopen → \
             recovery-outcome classification → JSON"
)]
struct Cli {
    /// Randomized fault iterations per fault class.
    #[arg(long, default_value_t = 500)]
    iters_per_class: u64,
    /// Campaign master seed. Every fault placement derives from it —
    /// the same seed reproduces the identical campaign bit-for-bit.
    #[arg(long, default_value_t = 0x5EED_2026_0C4A_A167)]
    seed: u64,
    /// Committed generations in the fixture image (>= 2 so the fault
    /// classes that need a previous generation are well-defined).
    #[arg(long, default_value_t = 3)]
    generations: u64,
    /// Frames committed per generation. Odd slots carry high-entropy
    /// payloads (stored as raw zstd blocks), even slots ASCII markers.
    #[arg(long, default_value_t = 4)]
    frames_per_gen: u64,
    /// Iterations for the `crash-recover-cycle` class (each runs 3
    /// fault → recover → heal-commit rounds, so its per-iteration cost
    /// is dominated by 3 durable commits — it gets its own budget).
    #[arg(long, default_value_t = 2000)]
    cycle_iters: u64,
    /// Iterations for the `sigkill-storm` class (each spawns + SIGKILLs
    /// a real child process, so it gets its own budget).
    #[arg(long, default_value_t = 200)]
    kill_iters: u64,
    /// Output JSON path (parent directories are created).
    #[arg(long)]
    out: PathBuf,
    /// Also run the recovery-timing scale sweep on ~1/16/64/256 MiB
    /// files (torn final footer: scan-back open vs clean open).
    #[arg(long, default_value_t = false)]
    scale_sweep: bool,
}

// ---------------------------------------------------------------------------
// Fixture: a pristine multi-generation image, built once, kept in memory.
// ---------------------------------------------------------------------------

/// Byte-level geometry of the file image as of one committed snapshot.
struct GenInfo {
    /// The snapshot generation label (header + footer agree).
    generation: u64,
    /// Number of fixture frames visible in this snapshot (fixture
    /// frames are committed in order, so frame `i` is visible iff
    /// `i < visible_frames`).
    visible_frames: usize,
    /// Logical header prefix (bytes 0..120) as of this commit.
    header_prefix: Vec<u8>,
    /// File length right after this commit.
    file_len: u64,
    /// `header.footer_offset` as of this commit.
    footer_offset: u64,
    /// Total encoded footer length (88-byte header + body).
    footer_len: u64,
}

struct ExpectedFrame {
    id: valise::FrameId,
    payload: Vec<u8>,
}

struct Fixture {
    /// Full pristine bytes after the final commit.
    pristine: Vec<u8>,
    /// Snapshot geometry per recoverable generation, oldest → newest.
    /// Index 0 is the empty post-create snapshot (0 frames).
    gens: Vec<GenInfo>,
    /// All committed frames with their exact expected payload bytes.
    frames: Vec<ExpectedFrame>,
}

fn le_u64(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(
        bytes[offset..offset + 8]
            .try_into()
            .expect("BUG: u64 slice in bounds"),
    )
}

/// Decode the snapshot geometry from a raw file image and sanity-check
/// the invariants every fault placement below relies on.
fn snapshot_geometry(image: &[u8], visible_frames: usize) -> Result<GenInfo> {
    let footer_offset = le_u64(image, HEADER_FOOTER_OFFSET_AT);
    ensure!(
        footer_offset >= HEADER_SIZE,
        "footer offset {footer_offset} inside header region"
    );
    ensure!(
        &image[footer_offset as usize..footer_offset as usize + 4] == TOC_MAGIC,
        "header.footer_offset must point at the VLTC magic"
    );
    let body_len = le_u64(image, (footer_offset + TOC_BODY_LEN_AT) as usize);
    let footer_len = TOC_HEADER_LEN + body_len;
    ensure!(
        footer_offset + footer_len == image.len() as u64,
        "the committed footer must be the last bytes of the file \
         (footer end {}, file len {})",
        footer_offset + footer_len,
        image.len()
    );
    Ok(GenInfo {
        generation: le_u64(image, HEADER_GENERATION_AT),
        visible_frames,
        header_prefix: image[..HEADER_LOGICAL_LEN].to_vec(),
        file_len: image.len() as u64,
        footer_offset,
        footer_len,
    })
}

fn build_fixture(cli: &Cli, work_dir: &Path) -> Result<Fixture> {
    let path = work_dir.join("pristine.vls");
    let mut rng = XorShift64::new(splitmix64(cli.seed ^ 0xF1C7_0000_0000_0001));

    // Post-create empty snapshot: create() durably writes the empty
    // generation-1 footer, which is itself a scan-back anchor (deep
    // truncations can legitimately land on it).
    let valise = ValiseFile::create(&path).context("create fixture file")?;
    drop(valise);
    let image = fs::read(&path)?;
    let mut gens = vec![snapshot_geometry(&image, 0).context("post-create geometry")?];
    ensure!(
        image[CONTRACT_DIGEST_AT..CONTRACT_DIGEST_AT + CONTRACT_DIGEST_LEN]
            .iter()
            .any(|b| *b != 0),
        "create-contract digest at byte {CONTRACT_DIGEST_AT} must be stamped"
    );

    let mut frames: Vec<ExpectedFrame> = Vec::new();
    let mut collection = None;
    for g in 0..cli.generations {
        let mut valise = ValiseFile::open_read_write(&path).context("reopen fixture read-write")?;
        let coll = match collection {
            Some(c) => c,
            None => {
                let c = valise
                    .create_collection("campaign")
                    .context("create collection")?;
                collection = Some(c);
                c
            }
        };
        for f in 0..cli.frames_per_gen {
            // Even slots: distinguishable ASCII markers. Odd slots:
            // high-entropy bytes — zstd stores those as raw blocks, so
            // the image carries payload regions that faults can hit
            // byte-verbatim (mirrors the deterministic fixture's mix).
            let payload: Vec<u8> = if f % 2 == 0 {
                format!(
                    "valise-campaign-generation{g:02}-frame{f:02}-{:016x}",
                    rng.next()
                )
                .into_bytes()
            } else {
                let len = 1024 + rng.below(3072) as usize;
                let mut buf = vec![0u8; len];
                rng.fill(&mut buf);
                buf
            };
            let id = valise
                .put_frame(PutFrame::document(coll, &payload))
                .context("put fixture frame")?;
            frames.push(ExpectedFrame { id, payload });
        }
        let outcome = valise.commit().context("commit fixture generation")?;
        ensure!(outcome.changed, "fixture commit must change the snapshot");
        drop(valise);
        let image = fs::read(&path)?;
        let info = snapshot_geometry(&image, frames.len())
            .with_context(|| format!("geometry after commit {g}"))?;
        ensure!(
            info.generation == outcome.snapshot_generation,
            "header generation label {} != acknowledged commit generation {}",
            info.generation,
            outcome.snapshot_generation
        );
        gens.push(info);
    }

    let pristine = fs::read(&path)?;
    fs::remove_file(&path)?;
    Ok(Fixture {
        pristine,
        gens,
        frames,
    })
}

// ---------------------------------------------------------------------------
// Fault classes.
// ---------------------------------------------------------------------------

/// Injects one randomized fault into `bytes` (a fresh copy of the
/// pristine image) and returns a short human-readable placement detail
/// for incident reports.
type FaultFn = fn(&Fixture, &mut Vec<u8>, &mut XorShift64) -> String;

const CLASSES: &[(&str, FaultFn)] = &[
    ("random-truncation", fault_random_truncation),
    ("random-bitflip", fault_random_bitflip),
    ("random-multi-bitflip", fault_random_multi_bitflip),
    ("torn-footer", fault_torn_footer),
    ("torn-header", fault_torn_header),
    ("reorder-sim", fault_reorder_sim),
    ("random-garbage-window", fault_random_garbage_window),
];

fn fill_zero_or_random(bytes: &mut [u8], rng: &mut XorShift64) -> &'static str {
    if rng.coin() {
        bytes.fill(0);
        "zeros"
    } else {
        rng.fill(bytes);
        "random"
    }
}

/// Class 1: Truncate to a uniformly random length in (4096, file_len).
fn fault_random_truncation(_fx: &Fixture, bytes: &mut Vec<u8>, rng: &mut XorShift64) -> String {
    let new_len = rng.range(HEADER_SIZE + 1, bytes.len() as u64);
    bytes.truncate(new_len as usize);
    format!("truncate to {new_len}")
}

/// Class 2: Flip 1 random bit anywhere past the 4 KiB header.
#[allow(clippy::ptr_arg)] // signature fixed by `FaultFn` (truncation needs the Vec)
fn fault_random_bitflip(_fx: &Fixture, bytes: &mut Vec<u8>, rng: &mut XorShift64) -> String {
    let bit = rng.range(HEADER_SIZE * 8, bytes.len() as u64 * 8);
    bytes[(bit / 8) as usize] ^= 1 << (bit % 8);
    format!("flip bit {bit} (byte {})", bit / 8)
}

/// Class 3: Flip k ~ U[2,8] distinct random bits past the header.
#[allow(clippy::ptr_arg)] // signature fixed by `FaultFn` (truncation needs the Vec)
fn fault_random_multi_bitflip(_fx: &Fixture, bytes: &mut Vec<u8>, rng: &mut XorShift64) -> String {
    let k = 2 + rng.below(7);
    let mut flipped = HashSet::new();
    while (flipped.len() as u64) < k {
        let bit = rng.range(HEADER_SIZE * 8, bytes.len() as u64 * 8);
        if flipped.insert(bit) {
            bytes[(bit / 8) as usize] ^= 1 << (bit % 8);
        }
    }
    format!("flip {k} bits {flipped:?}")
}

/// Class 4: Overwrite a random prefix/suffix/window of the final footer with
/// zeros or random bytes.
#[allow(clippy::ptr_arg)] // signature fixed by `FaultFn` (truncation needs the Vec)
fn fault_torn_footer(fx: &Fixture, bytes: &mut Vec<u8>, rng: &mut XorShift64) -> String {
    let newest = fx.gens.last().expect("BUG: fixture has generations");
    let footer_at = newest.footer_offset as usize;
    let footer_len = newest.footer_len as usize;
    let (start, len, shape) = match rng.below(3) {
        0 => (0, 1 + rng.below(footer_len as u64) as usize, "prefix"),
        1 => {
            let len = 1 + rng.below(footer_len as u64) as usize;
            (footer_len - len, len, "suffix")
        }
        _ => {
            let start = rng.below(footer_len as u64) as usize;
            let len = 1 + rng.below((footer_len - start) as u64) as usize;
            (start, len, "window")
        }
    };
    let target = footer_at + start..footer_at + start + len;
    let fill = fill_zero_or_random(&mut bytes[target], rng);
    format!("footer {shape} +{start} len {len} fill {fill}")
}

/// Class 5: Overwrite a random sub-range of the logical header (bytes 0..120)
/// with the previous generation's header bytes or random bytes; half
/// the time also corrupt the byte-832 create-contract digest (which
/// must fail closed — that is fail-correct too).
#[allow(clippy::ptr_arg)] // signature fixed by `FaultFn` (truncation needs the Vec)
fn fault_torn_header(fx: &Fixture, bytes: &mut Vec<u8>, rng: &mut XorShift64) -> String {
    let prev = &fx.gens[fx.gens.len() - 2];
    let start = rng.below(HEADER_LOGICAL_LEN as u64) as usize;
    let len = 1 + rng.below((HEADER_LOGICAL_LEN - start) as u64) as usize;
    let source = if rng.coin() {
        bytes[start..start + len].copy_from_slice(&prev.header_prefix[start..start + len]);
        "prev-gen-header"
    } else {
        rng.fill(&mut bytes[start..start + len]);
        "random"
    };
    let digest = if rng.coin() {
        let at = CONTRACT_DIGEST_AT + rng.below(CONTRACT_DIGEST_LEN as u64) as usize;
        bytes[at] ^= (rng.next() as u8) | 1;
        "corrupted"
    } else {
        "intact"
    };
    format!(
        "header [{start}..{}) <- {source}, digest {digest}",
        start + len
    )
}

/// Class 6: Reorder simulation: the header keeps pointing at the new footer,
/// but 1–3 random windows of the final commit's appended bytes
/// (segments + footer, everything past the previous generation's
/// footer) are zeroed or garbled — arbitrary writeback subsets of the
/// last commit reaching the platter.
#[allow(clippy::ptr_arg)] // signature fixed by `FaultFn` (truncation needs the Vec)
fn fault_reorder_sim(fx: &Fixture, bytes: &mut Vec<u8>, rng: &mut XorShift64) -> String {
    let prev = &fx.gens[fx.gens.len() - 2];
    let region_lo = prev.file_len;
    let region_hi = bytes.len() as u64;
    let windows = 1 + rng.below(3);
    let mut detail = format!("{windows} windows in [{region_lo}..{region_hi}):");
    for _ in 0..windows {
        let start = rng.range(region_lo, region_hi);
        let len = 1 + rng.below(region_hi - start);
        let target = start as usize..(start + len) as usize;
        let fill = fill_zero_or_random(&mut bytes[target], rng);
        detail.push_str(&format!(" +{start} len {len} fill {fill}"));
    }
    detail
}

/// Class 7: Overwrite a random 64 B – 4 KiB window anywhere past the header
/// with random bytes.
#[allow(clippy::ptr_arg)] // signature fixed by `FaultFn` (truncation needs the Vec)
fn fault_random_garbage_window(_fx: &Fixture, bytes: &mut Vec<u8>, rng: &mut XorShift64) -> String {
    let body_len = bytes.len() as u64 - HEADER_SIZE;
    let max_len = body_len.min(4096);
    let len = if max_len <= 64 {
        max_len
    } else {
        rng.range(64, max_len + 1)
    };
    let start = rng.range(HEADER_SIZE, bytes.len() as u64 - len + 1);
    rng.fill(&mut bytes[start as usize..(start + len) as usize]);
    format!("garbage window +{start} len {len}")
}

// ---------------------------------------------------------------------------
// Outcome classification.
// ---------------------------------------------------------------------------

#[derive(Debug)]
enum Outcome {
    RecoveredNewest,
    RecoveredPrevious,
    RejectedAtRead,
    CleanReject,
    /// Silent wrong data OR a self-inconsistent served snapshot.
    /// Must be zero. Carries the full detail for the incident report.
    WrongData(String),
    Panicked(String),
}

impl Outcome {
    /// Severity rank for merging multiple sub-checks of one iteration
    /// (compaction states, cycle rounds): the worst sub-outcome is the
    /// iteration's outcome.
    fn severity(&self) -> u8 {
        match self {
            Outcome::Panicked(_) => 5,
            Outcome::WrongData(_) => 4,
            Outcome::CleanReject => 3,
            Outcome::RejectedAtRead => 2,
            Outcome::RecoveredPrevious => 1,
            Outcome::RecoveredNewest => 0,
        }
    }

    fn worst(self, other: Outcome) -> Outcome {
        if other.severity() > self.severity() {
            other
        } else {
            self
        }
    }
}

struct Probe {
    outcome: Outcome,
    /// `open_read_only` wall time in µs (whether it succeeded or not).
    open_us: u64,
    /// Generation label served by a successful open.
    served_generation: Option<u64>,
}

/// Open the faulted file and verify CONTENT, not just success: every
/// fixture frame is read and byte-compared against the pristine
/// expected payload for the served generation.
fn probe_file(path: &Path, fx: &Fixture) -> Probe {
    let mark = Instant::now();
    let opened = ValiseFile::open_read_only(path);
    let open_us = mark.elapsed().as_micros() as u64;
    let file = match opened {
        Ok(file) => file,
        Err(_) => {
            return Probe {
                outcome: Outcome::CleanReject,
                open_us,
                served_generation: None,
            };
        }
    };

    let served = file.header().snapshot_generation;
    let Some(gen_index) = fx.gens.iter().position(|g| g.generation == served) else {
        return Probe {
            outcome: Outcome::WrongData(format!(
                "opened at generation {served}, which matches NO committed snapshot"
            )),
            open_us,
            served_generation: Some(served),
        };
    };
    let visible = fx.gens[gen_index].visible_frames;
    let stub_count = file.frame_stubs().len();

    let mut wrong: Vec<String> = Vec::new();
    let mut visible_read_errors = 0usize;
    let mut invisible_reads_ok = 0usize;
    for (i, frame) in fx.frames.iter().enumerate() {
        match file.read_payload(frame.id) {
            Ok(bytes) => {
                if bytes != frame.payload {
                    wrong.push(format!(
                        "frame {i}: {} bytes served, expected {} (first divergence at {:?})",
                        bytes.len(),
                        frame.payload.len(),
                        bytes
                            .iter()
                            .zip(frame.payload.iter())
                            .position(|(a, b)| a != b)
                    ));
                } else if i >= visible {
                    // Correct bytes, but the frame does not belong to
                    // the served snapshot: the anchor is incoherent.
                    invisible_reads_ok += 1;
                }
            }
            Err(_) if i < visible => visible_read_errors += 1,
            Err(_) => {} // invisible frame unreadable: expected
        }
    }

    let outcome = if !wrong.is_empty() {
        Outcome::WrongData(format!(
            "served generation {served}: wrong bytes — {}",
            wrong.join("; ")
        ))
    } else if invisible_reads_ok > 0 {
        Outcome::WrongData(format!(
            "served generation {served} but {invisible_reads_ok} frame(s) from newer \
             generations read back — self-inconsistent snapshot"
        ))
    } else if visible_read_errors > 0 {
        Outcome::RejectedAtRead
    } else if stub_count != visible {
        Outcome::WrongData(format!(
            "served generation {served}: {stub_count} frame stubs visible, snapshot \
             committed {visible} — self-inconsistent catalog"
        ))
    } else if gen_index == fx.gens.len() - 1 {
        Outcome::RecoveredNewest
    } else {
        Outcome::RecoveredPrevious
    };
    Probe {
        outcome,
        open_us,
        served_generation: Some(served),
    }
}

fn classify(path: &Path, fx: &Fixture) -> Probe {
    let probe = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| probe_file(path, fx)));
    match probe {
        Ok(probe) => probe,
        Err(payload) => {
            let text = payload
                .downcast_ref::<&str>()
                .map(|s| (*s).to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "non-string panic payload".to_string());
            Probe {
                outcome: Outcome::Panicked(text),
                open_us: 0,
                served_generation: None,
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Aggregation + JSON shape.
// ---------------------------------------------------------------------------

#[derive(Default, Serialize)]
struct ClassStats {
    cases: u64,
    recovered_newest: u64,
    recovered_previous: u64,
    rejected_at_read: u64,
    clean_reject: u64,
    wrong_data: u64,
    panic: u64,
    open_p50_us: u64,
    open_p95_us: u64,
    /// Distribution of the generation label served by successful opens.
    /// Any count on a generation below the newest is a lower bound on
    /// scan-back activations (scan-back can also re-select the newest).
    served_generation_hist: BTreeMap<u64, u64>,
    #[serde(skip)]
    open_times_us: Vec<u64>,
}

impl ClassStats {
    fn count_outcome(&mut self, outcome: &Outcome) {
        self.cases += 1;
        match outcome {
            Outcome::RecoveredNewest => self.recovered_newest += 1,
            Outcome::RecoveredPrevious => self.recovered_previous += 1,
            Outcome::RejectedAtRead => self.rejected_at_read += 1,
            Outcome::CleanReject => self.clean_reject += 1,
            Outcome::WrongData(_) => self.wrong_data += 1,
            Outcome::Panicked(_) => self.panic += 1,
        }
    }

    fn record(&mut self, probe: &Probe) {
        self.count_outcome(&probe.outcome);
        if !matches!(probe.outcome, Outcome::Panicked(_)) {
            self.open_times_us.push(probe.open_us);
        }
        if let Some(g) = probe.served_generation {
            *self.served_generation_hist.entry(g).or_insert(0) += 1;
        }
    }

    /// Record one stateful-driver iteration (several opens + served
    /// generations per iteration).
    fn record_driver(&mut self, probe: &DriverProbe) {
        self.count_outcome(&probe.outcome);
        self.open_times_us.extend_from_slice(&probe.opens_us);
        for g in &probe.served {
            *self.served_generation_hist.entry(*g).or_insert(0) += 1;
        }
    }

    /// Fold another class's counters into this one (for `totals`).
    fn absorb(&mut self, other: &ClassStats) {
        self.cases += other.cases;
        self.recovered_newest += other.recovered_newest;
        self.recovered_previous += other.recovered_previous;
        self.rejected_at_read += other.rejected_at_read;
        self.clean_reject += other.clean_reject;
        self.wrong_data += other.wrong_data;
        self.panic += other.panic;
        self.open_times_us.extend_from_slice(&other.open_times_us);
        for (g, n) in &other.served_generation_hist {
            *self.served_generation_hist.entry(*g).or_insert(0) += n;
        }
    }

    fn finalize(&mut self) {
        self.open_times_us.sort_unstable();
        self.open_p50_us = percentile(&self.open_times_us, 50.0);
        self.open_p95_us = percentile(&self.open_times_us, 95.0);
    }
}

/// Nearest-rank percentile over an ascending-sorted slice.
fn percentile(sorted: &[u64], p: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let rank = ((p / 100.0) * sorted.len() as f64).ceil() as usize;
    sorted[rank.clamp(1, sorted.len()) - 1]
}

#[derive(Serialize)]
struct Incident {
    class: String,
    iter: u64,
    iter_seed: String,
    kind: String,
    fault: String,
    detail: String,
}

#[derive(Serialize)]
struct ScalePoint {
    mib: u64,
    scanback_open_ms: f64,
    clean_open_ms: f64,
}

#[derive(Serialize)]
struct ConfigOut {
    iters_per_class: u64,
    seed: u64,
    generations: u64,
    frames_per_gen: u64,
    cycle_iters: u64,
    kill_iters: u64,
}

#[derive(Serialize)]
struct Report {
    config: ConfigOut,
    classes: BTreeMap<String, ClassStats>,
    totals: ClassStats,
    scale_sweep: Vec<ScalePoint>,
    incidents: Vec<Incident>,
    notes: Vec<String>,
    campaign_wall_seconds: f64,
}

// ---------------------------------------------------------------------------
// Stateful drivers (crash-recover-cycle / sigkill-storm /
// compaction-install): one iteration performs several opens and content
// verifications; the iteration's outcome is the worst sub-outcome.
// ---------------------------------------------------------------------------

/// Result of one stateful-class iteration.
struct DriverProbe {
    outcome: Outcome,
    /// Wall time of every `open_read_only` / `open_read_write` issued.
    opens_us: Vec<u64>,
    /// Generation label of every successful open.
    served: Vec<u64>,
}

impl DriverProbe {
    fn new() -> Self {
        Self {
            outcome: Outcome::RecoveredNewest,
            opens_us: Vec::new(),
            served: Vec::new(),
        }
    }
}

/// Run a stateful class: seeded per-iteration streams, panic capture,
/// incident logging. `clean_reject` is ALSO incident-logged for these
/// classes — every image they open is guaranteed recoverable, so an
/// open failure is a lost-durable-state event even though it only
/// gates the run via the wrong_data/panic counters.
fn run_stateful_class<F>(
    name: &str,
    class_idx: u64,
    iters: u64,
    seed: u64,
    incidents: &mut Vec<Incident>,
    mut iteration: F,
) -> Result<ClassStats>
where
    F: FnMut(&mut XorShift64, &mut String) -> Result<DriverProbe>,
{
    let class_base = splitmix64(seed ^ (class_idx << 32));
    let mut stats = ClassStats::default();
    for iter in 0..iters {
        let iter_seed = splitmix64(class_base.wrapping_add(iter));
        let mut rng = XorShift64::new(iter_seed);
        let mut fault_log = String::new();
        let probe = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            iteration(&mut rng, &mut fault_log)
        })) {
            Ok(result) => result?, // harness/setup error: abort the campaign
            Err(payload) => {
                let text = payload
                    .downcast_ref::<&str>()
                    .map(|s| (*s).to_string())
                    .or_else(|| payload.downcast_ref::<String>().cloned())
                    .unwrap_or_else(|| "non-string panic payload".to_string());
                let mut probe = DriverProbe::new();
                probe.outcome = Outcome::Panicked(text);
                probe
            }
        };
        stats.record_driver(&probe);
        let kind = match &probe.outcome {
            Outcome::WrongData(_) => Some("wrong_data"),
            Outcome::Panicked(_) => Some("panic"),
            Outcome::CleanReject => Some("clean_reject"),
            _ => None,
        };
        if let Some(kind) = kind {
            let detail = match &probe.outcome {
                Outcome::WrongData(d) | Outcome::Panicked(d) => d.clone(),
                _ => "open returned a typed error on a guaranteed-recoverable image".to_string(),
            };
            incidents.push(Incident {
                class: name.to_string(),
                iter,
                iter_seed: format!("{iter_seed:#018x}"),
                kind: kind.to_string(),
                fault: fault_log,
                detail,
            });
        }
    }
    Ok(stats)
}

// ---------------------------------------------------------------------------
// Class 8: crash-recover-cycle.
// ---------------------------------------------------------------------------

const CYCLE_ROUNDS: usize = 3;

/// Per-frame expected state carried across cycle rounds.
struct CycleFrame {
    id: valise::FrameId,
    payload: Vec<u8>,
    /// Generation of the commit that (re-)published this frame, in the
    /// CURRENT live chain (entries of dropped commits are pruned as
    /// soon as a recovery drops them, so numeric reuse of generation
    /// labels by healing commits stays unambiguous).
    commit_gen: u64,
    /// Set once a fault damaged this frame's committed payload bytes
    /// (detection-only media damage): later reads may keep returning
    /// the typed integrity error; wrong bytes stay forbidden.
    may_error: bool,
}

/// Commit generation of each fixture frame, by frame index.
fn fixture_frame_generations(fx: &Fixture) -> Vec<u64> {
    let mut out = vec![0u64; fx.frames.len()];
    for pair in fx.gens.windows(2) {
        for slot in &mut out[pair[0].visible_frames..pair[1].visible_frames] {
            *slot = pair[1].generation;
        }
    }
    out
}

/// Fault from the torn-footer / reorder-sim repertoire, confined to the
/// newest commit's bytes (`[commit_start, len)`): the previous
/// generation's snapshot stays physically intact, so recovery may lose
/// at most the single faulted commit.
fn cycle_inject_fault(bytes: &mut [u8], commit_start: u64, rng: &mut XorShift64) -> String {
    let file_len = bytes.len() as u64;
    assert!(commit_start < file_len, "BUG: empty commit region");
    if rng.coin() {
        let footer_at = le_u64(bytes, HEADER_FOOTER_OFFSET_AT);
        let body_len = le_u64(bytes, (footer_at + TOC_BODY_LEN_AT) as usize);
        let footer_len = (TOC_HEADER_LEN + body_len) as usize;
        let footer_at = footer_at as usize;
        let (start, len, shape) = match rng.below(3) {
            0 => (0, 1 + rng.below(footer_len as u64) as usize, "prefix"),
            1 => {
                let len = 1 + rng.below(footer_len as u64) as usize;
                (footer_len - len, len, "suffix")
            }
            _ => {
                let start = rng.below(footer_len as u64) as usize;
                let len = 1 + rng.below((footer_len - start) as u64) as usize;
                (start, len, "window")
            }
        };
        let fill = fill_zero_or_random(&mut bytes[footer_at + start..footer_at + start + len], rng);
        format!("torn-footer {shape} +{start} len {len} {fill}")
    } else {
        let windows = 1 + rng.below(3);
        let mut detail = format!("reorder {windows}w in [{commit_start}..{file_len}):");
        for _ in 0..windows {
            let start = rng.range(commit_start, file_len);
            let len = 1 + rng.below(file_len - start);
            let fill = fill_zero_or_random(&mut bytes[start as usize..(start + len) as usize], rng);
            detail.push_str(&format!(" +{start} len {len} {fill}"));
        }
        detail
    }
}

/// One crash-recover-cycle iteration. Any contract violation is
/// reported through `Outcome::WrongData` with a full detail string;
/// tolerated typed read errors (media damage to the faulted commit's
/// own payload) downgrade the iteration to `RejectedAtRead`.
fn cycle_iteration(
    path: &Path,
    fx: &Fixture,
    frame_gens: &[u64],
    rng: &mut XorShift64,
    fault_log: &mut String,
) -> Result<DriverProbe> {
    let mut probe = DriverProbe::new();
    fs::write(path, &fx.pristine).context("cycle: write pristine copy")?;

    let mut expected: Vec<CycleFrame> = fx
        .frames
        .iter()
        .zip(frame_gens)
        .map(|(f, commit_gen)| CycleFrame {
            id: f.id,
            payload: f.payload.clone(),
            commit_gen: *commit_gen,
            may_error: false,
        })
        .collect();
    let newest = fx.gens.last().expect("BUG: fixture has generations");
    let mut last_ack = newest.generation;
    let mut commit_start = fx.gens[fx.gens.len() - 2].file_len;
    let mut any_read_error = false;
    let mut any_scanback = false;

    macro_rules! violation {
        ($($arg:tt)*) => {{
            probe.outcome = Outcome::WrongData(format!($($arg)*));
            return Ok(probe);
        }};
    }

    for round in 0..CYCLE_ROUNDS {
        // Fault the newest commit's bytes on disk.
        let mut bytes = fs::read(path).context("cycle: read working image")?;
        let desc = cycle_inject_fault(&mut bytes, commit_start, rng);
        fault_log.push_str(&format!("r{round}[{desc}] "));
        fs::write(path, &bytes).context("cycle: write faulted image")?;

        // ReadWrite open: recovery (if needed) must always succeed —
        // the previous snapshot is physically intact.
        let mark = Instant::now();
        let opened = ValiseFile::open_read_write(path);
        probe.opens_us.push(mark.elapsed().as_micros() as u64);
        let mut file = match opened {
            Ok(file) => file,
            Err(_) => {
                probe.outcome = Outcome::CleanReject;
                return Ok(probe);
            }
        };
        let served = file.header().snapshot_generation;
        probe.served.push(served);
        if served > last_ack || served + 1 < last_ack {
            violation!(
                "round {round}: served generation {served} vs last acknowledged {last_ack} — \
                 recovery must lose at most the single faulted commit"
            );
        }
        if served < last_ack {
            any_scanback = true;
        }
        // Frames of a dropped commit are gone; their ids may be reused
        // by the healing commit below.
        expected.retain(|e| e.commit_gen <= served);

        // Byte-exact content verification of the served snapshot.
        let stub_count = file.frame_stubs().len();
        if stub_count != expected.len() {
            violation!(
                "round {round}: served generation {served} shows {stub_count} frames, \
                 expected {}",
                expected.len()
            );
        }
        for entry in &mut expected {
            match file.read_payload(entry.id) {
                Ok(bytes) if bytes == entry.payload => {}
                Ok(bytes) => violation!(
                    "round {round}: frame {:?} served {} wrong bytes (expected {})",
                    entry.id,
                    bytes.len(),
                    entry.payload.len()
                ),
                Err(err) => {
                    // Tolerated only for frames of the faulted commit
                    // (their payload bytes may carry the media damage)
                    // or frames already known-damaged.
                    if !(entry.may_error || entry.commit_gen == last_ack) {
                        violation!(
                            "round {round}: typed read error on frame {:?} of UNDAMAGED \
                             commit {}: {err}",
                            entry.id,
                            entry.commit_gen
                        );
                    }
                    entry.may_error = true;
                    any_read_error = true;
                }
            }
        }

        // Heal: append one frame and commit. The commit must publish
        // served+1 and is acknowledged the moment commit() returns.
        let coll = file.collections()[0].collection_id;
        let pre_commit_len = fs::metadata(path).context("cycle: stat")?.len();
        let payload = format!("cycle-heal-round{round}-{:016x}", rng.next()).into_bytes();
        let id = match file.put_frame(PutFrame::document(coll, &payload)) {
            Ok(id) => id,
            Err(err) => violation!("round {round}: put_frame after recovery failed: {err}"),
        };
        let outcome = match file.commit() {
            Ok(outcome) => outcome,
            Err(err) => violation!("round {round}: healing commit failed: {err}"),
        };
        if outcome.snapshot_generation != served + 1 {
            violation!(
                "round {round}: healing commit published generation {} (expected {})",
                outcome.snapshot_generation,
                served + 1
            );
        }
        expected.push(CycleFrame {
            id,
            payload,
            commit_gen: outcome.snapshot_generation,
            may_error: false,
        });
        last_ack = outcome.snapshot_generation;
        commit_start = pre_commit_len;
        drop(file);
    }

    // Final reopen: the image was cleanly committed by the last round's
    // heal, so this is a fast-path open that must serve every surviving
    // frame — including the final round's appended frame — byte-exact.
    let mark = Instant::now();
    let opened = ValiseFile::open_read_only(path);
    probe.opens_us.push(mark.elapsed().as_micros() as u64);
    let file = match opened {
        Ok(file) => file,
        Err(_) => {
            probe.outcome = Outcome::CleanReject;
            return Ok(probe);
        }
    };
    let served = file.header().snapshot_generation;
    probe.served.push(served);
    if served != last_ack {
        violation!("final reopen: served generation {served}, last acknowledged {last_ack}");
    }
    if file.frame_stubs().len() != expected.len() {
        violation!(
            "final reopen: {} frames visible, expected {}",
            file.frame_stubs().len(),
            expected.len()
        );
    }
    for entry in &expected {
        match file.read_payload(entry.id) {
            Ok(bytes) if bytes == entry.payload => {}
            Ok(_) => violation!("final reopen: wrong bytes on frame {:?}", entry.id),
            Err(err) => {
                if !entry.may_error {
                    violation!(
                        "final reopen: typed read error on undamaged frame {:?}: {err}",
                        entry.id
                    );
                }
                any_read_error = true;
            }
        }
    }

    probe.outcome = if any_read_error {
        Outcome::RejectedAtRead
    } else if any_scanback {
        Outcome::RecoveredPrevious
    } else {
        Outcome::RecoveredNewest
    };
    Ok(probe)
}

// ---------------------------------------------------------------------------
// Class 9: sigkill-storm.
// ---------------------------------------------------------------------------

/// Hidden worker mode: when set, the binary ignores its CLI and runs
/// the commit-loop worker against the store at the env var's path.
const KILL_WORKER_ENV: &str = "VALISE_CRASH_CAMPAIGN_KILL_WORKER";
/// Safety valve so an unkilled worker terminates on its own.
const KILL_WORKER_MAX_COMMITS: usize = 500;

/// Child: loop put_frame → commit → ack the ACKNOWLEDGED generation on
/// stdout (line-buffered pipe + explicit flush, written strictly after
/// `commit()` returned).
fn run_kill_worker(store: &Path) -> Result<()> {
    let mut valise = ValiseFile::open_read_write(store).context("kill worker: open store")?;
    let coll = valise.collections()[0].collection_id;
    let stdout = std::io::stdout();
    for i in 0..KILL_WORKER_MAX_COMMITS {
        let payload = format!("storm-frame-{i}");
        valise
            .put_frame(PutFrame::document(coll, payload.as_bytes()))
            .context("kill worker: put_frame")?;
        let outcome = valise.commit().context("kill worker: commit")?;
        let mut lock = stdout.lock();
        writeln!(lock, "{}", outcome.snapshot_generation)?;
        lock.flush()?;
    }
    Ok(())
}

/// One sigkill-storm iteration: seed a store, spawn the worker, SIGKILL
/// it at a seeded delay past its first ack, reopen, and require zero
/// lost acknowledged commits plus exact frame content. A process kill
/// preserves the page cache, so the reopen must always succeed.
fn kill_storm_iteration(
    store: &Path,
    rng: &mut XorShift64,
    fault_log: &mut String,
) -> Result<DriverProbe> {
    let mut probe = DriverProbe::new();

    // Fresh seeded store (create_new semantics: remove any leftover).
    if store.exists() {
        fs::remove_file(store).context("kill storm: remove stale store")?;
    }
    let mut valise = ValiseFile::create(store).context("kill storm: create store")?;
    let coll = valise.create_collection("storm")?;
    valise.put_frame(PutFrame::document(coll, b"storm-seed"))?;
    let base = valise.commit().context("kill storm: seed commit")?;
    drop(valise);

    let exe = std::env::current_exe().context("kill storm: current_exe")?;
    let mut child = Command::new(exe)
        .env(KILL_WORKER_ENV, store)
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .context("kill storm: spawn worker")?;
    let child_stdout = child.stdout.take().expect("BUG: piped stdout missing");
    let acks: Arc<Mutex<Vec<u64>>> = Arc::new(Mutex::new(Vec::new()));
    let acks_writer = Arc::clone(&acks);
    let reader = thread::spawn(move || {
        for line in BufReader::new(child_stdout).lines() {
            let Ok(line) = line else { break };
            // A line torn by the kill mid-write fails to parse and is
            // ignored — that only weakens the check, never fakes an ack.
            if let Ok(generation) = line.trim().parse::<u64>() {
                acks_writer
                    .lock()
                    .expect("BUG: ack mutex poisoned")
                    .push(generation);
            }
        }
    });

    // Wait for the first acknowledged commit, then kill after a seeded
    // delay so the SIGKILL lands at a random point of the commit loop.
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        if !acks.lock().expect("BUG: ack mutex poisoned").is_empty() {
            break;
        }
        if Instant::now() > deadline {
            let _ = child.kill();
            let _ = child.wait();
            let _ = reader.join();
            bail!("kill storm: worker produced no ack within 60s");
        }
        thread::sleep(Duration::from_millis(1));
    }
    let delay_ms = 1 + rng.below(40);
    fault_log.push_str(&format!("SIGKILL {delay_ms}ms past first ack"));
    thread::sleep(Duration::from_millis(delay_ms));
    let _ = child.kill();
    child.wait().context("kill storm: reap worker")?;
    reader
        .join()
        .map_err(|_| anyhow::anyhow!("kill storm: ack reader thread panicked (harness bug)"))?;

    let acks = acks.lock().expect("BUG: ack mutex poisoned");
    let last_acked = *acks.iter().max().expect("BUG: first ack was observed");
    drop(acks);

    macro_rules! violation {
        ($($arg:tt)*) => {{
            probe.outcome = Outcome::WrongData(format!($($arg)*));
            return Ok(probe);
        }};
    }

    let mark = Instant::now();
    let opened = ValiseFile::open_read_only(store);
    probe.opens_us.push(mark.elapsed().as_micros() as u64);
    let file = match opened {
        Ok(file) => file,
        Err(_) => {
            probe.outcome = Outcome::CleanReject;
            return Ok(probe);
        }
    };
    let served = file.header().snapshot_generation;
    probe.served.push(served);
    if served < last_acked {
        violation!(
            "LOST ACKNOWLEDGED COMMIT: last acked generation {last_acked}, visible {served}"
        );
    }
    if served > last_acked + 1 {
        violation!(
            "visible generation {served} is more than one ahead of the last recorded \
             ack {last_acked}"
        );
    }
    // Structural + content coherence: the seed commit had 1 frame and
    // every worker commit appended exactly one, in id order.
    let stubs = file.frame_stubs();
    let expected_frames = 1 + (served - base.snapshot_generation) as usize;
    if stubs.len() != expected_frames {
        violation!(
            "visible generation {served} implies {expected_frames} frames, catalog shows {}",
            stubs.len()
        );
    }
    for (idx, stub) in stubs.iter().enumerate() {
        let want: Vec<u8> = if idx == 0 {
            b"storm-seed".to_vec()
        } else {
            format!("storm-frame-{}", idx - 1).into_bytes()
        };
        match file.read_payload(stub.frame_id) {
            Ok(bytes) if bytes == want => {}
            Ok(bytes) => violation!(
                "frame {idx}: wrong bytes ({:?}, expected {:?})",
                String::from_utf8_lossy(&bytes),
                String::from_utf8_lossy(&want)
            ),
            Err(_) => {
                // Fail-safe but unexpected under a page-cache-preserving
                // kill; surfaces as a nonzero rejected_at_read count.
                probe.outcome = Outcome::RejectedAtRead;
                return Ok(probe);
            }
        }
    }
    probe.outcome = Outcome::RecoveredNewest;
    Ok(probe)
}

// ---------------------------------------------------------------------------
// Class 10: compaction-install.
// ---------------------------------------------------------------------------

/// A complete committed image with its expected content, standing in
/// for one side of the compaction swap.
struct ImageSpec {
    bytes: Vec<u8>,
    generation: u64,
    frames: Vec<ExpectedFrame>,
}

/// Synthesize the compaction OUTPUT image: the fixture's live set
/// streamed in frame order into a fresh single-commit file — the same
/// shape `db::compact` produces at the temp path before the rename.
fn build_compacted_image(fx: &Fixture, work_dir: &Path) -> Result<ImageSpec> {
    let path = work_dir.join("compact-new.vls");
    if path.exists() {
        fs::remove_file(&path)?;
    }
    let mut valise = ValiseFile::create(&path).context("compacted image: create")?;
    let coll = valise.create_collection("campaign")?;
    let mut frames = Vec::with_capacity(fx.frames.len());
    for frame in &fx.frames {
        let id = valise.put_frame(PutFrame::document(coll, &frame.payload))?;
        frames.push(ExpectedFrame {
            id,
            payload: frame.payload.clone(),
        });
    }
    let outcome = valise.commit().context("compacted image: commit")?;
    drop(valise);
    let bytes = fs::read(&path)?;
    fs::remove_file(&path)?;
    Ok(ImageSpec {
        bytes,
        generation: outcome.snapshot_generation,
        frames,
    })
}

/// Open `path` and require the COMPLETE image described by
/// `(generation, frames)`: exact generation label, exact frame count,
/// every payload byte-exact. Any deviation is a violation of the
/// atomic-replace contract — the main path here is always a complete
/// committed image, so even a typed (fail-safe) error counts.
fn probe_exact(path: &Path, generation: u64, frames: &[ExpectedFrame], what: &str) -> Probe {
    let mark = Instant::now();
    let opened = ValiseFile::open_read_only(path);
    let open_us = mark.elapsed().as_micros() as u64;
    let file = match opened {
        Ok(file) => file,
        Err(_) => {
            return Probe {
                outcome: Outcome::CleanReject,
                open_us,
                served_generation: None,
            };
        }
    };
    let served = file.header().snapshot_generation;
    let outcome = if served != generation {
        Outcome::WrongData(format!(
            "{what}: served generation {served}, expected the complete image's {generation}"
        ))
    } else if file.frame_stubs().len() != frames.len() {
        Outcome::WrongData(format!(
            "{what}: {} frames visible, complete image has {}",
            file.frame_stubs().len(),
            frames.len()
        ))
    } else {
        let mut outcome = Outcome::RecoveredNewest;
        for (idx, frame) in frames.iter().enumerate() {
            match file.read_payload(frame.id) {
                Ok(bytes) if bytes == frame.payload => {}
                Ok(_) => {
                    outcome = Outcome::WrongData(format!("{what}: wrong bytes on frame {idx}"));
                    break;
                }
                Err(err) => {
                    outcome = Outcome::WrongData(format!(
                        "{what}: typed read error on frame {idx} of a complete image: {err}"
                    ));
                    break;
                }
            }
        }
        outcome
    };
    Probe {
        outcome,
        open_us,
        served_generation: Some(served),
    }
}

/// One compaction-install iteration: replay the three crash states of
/// the temp-build + rename install sequence and require the main path
/// to serve a complete store in each.
fn compaction_iteration(
    main_path: &Path,
    old: &ImageSpec,
    new: &ImageSpec,
    rng: &mut XorShift64,
    fault_log: &mut String,
) -> Result<DriverProbe> {
    let temp_path = main_path.with_extension("valise.compacting");
    let mut probe = DriverProbe::new();
    let absorb = |probe: &mut DriverProbe, sub: Probe| {
        probe.opens_us.push(sub.open_us);
        if let Some(g) = sub.served_generation {
            probe.served.push(g);
        }
        let outcome = std::mem::replace(&mut probe.outcome, Outcome::RecoveredNewest);
        probe.outcome = outcome.worst(sub.outcome);
    };

    // State A: crash after the temp compacted store fully committed,
    // before the rename. Main = complete OLD; temp sibling present.
    fs::write(main_path, &old.bytes)?;
    fs::write(&temp_path, &new.bytes)?;
    let sub = probe_exact(
        main_path,
        old.generation,
        &old.frames,
        "state A (pre-rename)",
    );
    absorb(&mut probe, sub);

    // State B: crash after the rename completed. Main = complete NEW;
    // temp consumed by the rename.
    fs::remove_file(&temp_path)?;
    fs::write(main_path, &new.bytes)?;
    let sub = probe_exact(
        main_path,
        new.generation,
        &new.frames,
        "state B (post-rename)",
    );
    absorb(&mut probe, sub);

    // State C: crash mid-temp-build. Main = complete OLD; temp sibling
    // torn at a seeded random point (never a legal open target).
    let mut torn = new.bytes.clone();
    let desc = match rng.below(3) {
        0 => {
            let cut = rng.range(1, torn.len() as u64) as usize;
            torn.truncate(cut);
            format!("temp truncated to {cut}")
        }
        1 => {
            let len = rng.range(64, 4096.min(torn.len() as u64 - 1) + 1);
            let start = rng.range(0, torn.len() as u64 - len + 1);
            let mut window = vec![0u8; len as usize];
            rng.fill(&mut window);
            torn[start as usize..(start + len) as usize].copy_from_slice(&window);
            format!("temp garbage window +{start} len {len}")
        }
        _ => {
            let footer_at = le_u64(&torn, HEADER_FOOTER_OFFSET_AT);
            let body_len = le_u64(&torn, (footer_at + TOC_BODY_LEN_AT) as usize);
            torn[footer_at as usize..(footer_at + TOC_HEADER_LEN + body_len) as usize].fill(0);
            "temp footer zeroed".to_string()
        }
    };
    fault_log.push_str(&desc);
    fs::write(main_path, &old.bytes)?;
    fs::write(&temp_path, &torn)?;
    let sub = probe_exact(
        main_path,
        old.generation,
        &old.frames,
        "state C (torn temp)",
    );
    absorb(&mut probe, sub);

    fs::remove_file(&temp_path)?;
    Ok(probe)
}

// ---------------------------------------------------------------------------
// Scale sweep: scan-back open time vs clean open time by file size.
// ---------------------------------------------------------------------------

fn median_ms(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).expect("BUG: NaN open time"));
    samples[samples.len() / 2]
}

fn run_scale_sweep(work_dir: &Path, seed: u64) -> Result<Vec<ScalePoint>> {
    const FRAME_BYTES: usize = 256 * 1024;
    const OPEN_SAMPLES: usize = 5;
    let mut points = Vec::new();
    for mib in [1u64, 16, 64, 256] {
        eprintln!("[scale-sweep] building ~{mib} MiB file...");
        let path = work_dir.join(format!("sweep-{mib}mib.vls"));
        let torn_path = work_dir.join(format!("sweep-{mib}mib-torn.vls"));
        let mut rng = XorShift64::new(splitmix64(seed ^ 0x5CA1_E000 ^ mib));

        // Bulk generation: incompressible payloads so zstd stores raw
        // blocks and the on-disk size tracks the target. Then a final
        // small commit whose footer the fault will tear.
        let mut valise = ValiseFile::create(&path).context("create sweep file")?;
        let coll = valise.create_collection("sweep")?;
        let mut buf = vec![0u8; FRAME_BYTES];
        for _ in 0..(mib * 1024 * 1024) / FRAME_BYTES as u64 {
            rng.fill(&mut buf);
            valise.put_frame(PutFrame::document(coll, &buf))?;
        }
        let bulk = valise.commit().context("sweep bulk commit")?;
        valise.put_frame(PutFrame::document(coll, b"sweep-final-commit"))?;
        let last = valise.commit().context("sweep final commit")?;
        drop(valise);

        // Clean opens.
        let mut clean = Vec::with_capacity(OPEN_SAMPLES);
        for _ in 0..OPEN_SAMPLES {
            let mark = Instant::now();
            let file = ValiseFile::open_read_only(&path).context("sweep clean open")?;
            clean.push(mark.elapsed().as_secs_f64() * 1e3);
            ensure!(file.header().snapshot_generation == last.snapshot_generation);
            drop(file);
        }

        // Tear the final footer (zeroed, header still points at it) and
        // measure the scan-back open, which must recover the bulk
        // generation by streaming the file for VLTC candidates.
        let mut bytes = fs::read(&path)?;
        let footer_offset = le_u64(&bytes, HEADER_FOOTER_OFFSET_AT);
        let body_len = le_u64(&bytes, (footer_offset + TOC_BODY_LEN_AT) as usize);
        bytes[footer_offset as usize..(footer_offset + TOC_HEADER_LEN + body_len) as usize].fill(0);
        fs::write(&torn_path, &bytes)?;
        drop(bytes);

        let mut scanback = Vec::with_capacity(OPEN_SAMPLES);
        for _ in 0..OPEN_SAMPLES {
            let mark = Instant::now();
            let file = ValiseFile::open_read_only(&torn_path).context("sweep scan-back open")?;
            scanback.push(mark.elapsed().as_secs_f64() * 1e3);
            ensure!(
                file.header().snapshot_generation == bulk.snapshot_generation,
                "scan-back must recover the bulk generation {} (served {})",
                bulk.snapshot_generation,
                file.header().snapshot_generation
            );
            drop(file);
        }

        let point = ScalePoint {
            mib,
            scanback_open_ms: median_ms(scanback),
            clean_open_ms: median_ms(clean),
        };
        eprintln!(
            "[scale-sweep] {mib} MiB: clean {:.3} ms, scan-back {:.3} ms",
            point.clean_open_ms, point.scanback_open_ms
        );
        points.push(point);
        fs::remove_file(&path)?;
        fs::remove_file(&torn_path)?;
    }
    Ok(points)
}

// ---------------------------------------------------------------------------
// Campaign driver.
// ---------------------------------------------------------------------------

fn run_campaign(cli: &Cli, work_dir: &Path) -> Result<Report> {
    let campaign_start = Instant::now();
    eprintln!(
        "[fixture] building pristine image ({} generations x {} frames, seed {:#x})...",
        cli.generations, cli.frames_per_gen, cli.seed
    );
    let fx = build_fixture(cli, work_dir)?;
    eprintln!(
        "[fixture] {} bytes, generations {:?}, {} frames",
        fx.pristine.len(),
        fx.gens.iter().map(|g| g.generation).collect::<Vec<_>>(),
        fx.frames.len()
    );

    let fault_path = work_dir.join("faulted.vls");
    let mut classes: BTreeMap<String, ClassStats> = BTreeMap::new();
    let mut totals = ClassStats::default();
    let mut incidents: Vec<Incident> = Vec::new();

    // Panics are counted, not printed 3500 times: silence the hook for
    // the duration of the loop (catch_unwind still captures payloads).
    let prev_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));

    for (class_idx, (name, fault)) in CLASSES.iter().enumerate() {
        let class_base = splitmix64(cli.seed ^ ((class_idx as u64 + 1) << 32));
        let mut stats = ClassStats::default();
        for iter in 0..cli.iters_per_class {
            let iter_seed = splitmix64(class_base.wrapping_add(iter));
            let mut rng = XorShift64::new(iter_seed);
            let mut bytes = fx.pristine.clone();
            let fault_detail = fault(&fx, &mut bytes, &mut rng);
            fs::write(&fault_path, &bytes).context("write faulted copy")?;

            let probe = classify(&fault_path, &fx);
            stats.record(&probe);
            totals.record(&probe);
            match &probe.outcome {
                Outcome::WrongData(detail) | Outcome::Panicked(detail) => {
                    incidents.push(Incident {
                        class: (*name).to_string(),
                        iter,
                        iter_seed: format!("{iter_seed:#018x}"),
                        kind: if matches!(probe.outcome, Outcome::WrongData(_)) {
                            "wrong_data".to_string()
                        } else {
                            "panic".to_string()
                        },
                        fault: fault_detail,
                        detail: detail.clone(),
                    });
                }
                _ => {}
            }
        }
        stats.finalize();
        print_class_line(name, &stats);
        classes.insert((*name).to_string(), stats);
    }

    // Class 8: crash-recover-cycle (fault → RW recovery → heal commit,
    // 3 rounds per iteration, its own iteration budget).
    let frame_gens = fixture_frame_generations(&fx);
    let cycle_path = work_dir.join("cycle.vls");
    let mut stats = run_stateful_class(
        "crash-recover-cycle",
        8,
        cli.cycle_iters,
        cli.seed,
        &mut incidents,
        |rng, fault_log| cycle_iteration(&cycle_path, &fx, &frame_gens, rng, fault_log),
    )?;
    totals.absorb(&stats);
    stats.finalize();
    print_class_line("crash-recover-cycle", &stats);
    classes.insert("crash-recover-cycle".to_string(), stats);

    // Class 9: sigkill-storm (real process kills, its own budget).
    let storm_path = work_dir.join("storm.vls");
    let mut stats = run_stateful_class(
        "sigkill-storm",
        9,
        cli.kill_iters,
        cli.seed,
        &mut incidents,
        |rng, fault_log| kill_storm_iteration(&storm_path, rng, fault_log),
    )?;
    totals.absorb(&stats);
    stats.finalize();
    print_class_line("sigkill-storm", &stats);
    classes.insert("sigkill-storm".to_string(), stats);

    // Class 10: compaction-install (synthetic atomic-replace states).
    let compacted = build_compacted_image(&fx, work_dir)?;
    let main_store = work_dir.join("store.vls");
    let old_image = ImageSpec {
        bytes: fx.pristine.clone(),
        generation: fx.gens.last().expect("BUG: fixture generations").generation,
        frames: fx
            .frames
            .iter()
            .map(|f| ExpectedFrame {
                id: f.id,
                payload: f.payload.clone(),
            })
            .collect(),
    };
    let mut stats = run_stateful_class(
        "compaction-install",
        10,
        cli.iters_per_class,
        cli.seed,
        &mut incidents,
        |rng, fault_log| compaction_iteration(&main_store, &old_image, &compacted, rng, fault_log),
    )?;
    totals.absorb(&stats);
    stats.finalize();
    print_class_line("compaction-install", &stats);
    classes.insert("compaction-install".to_string(), stats);

    std::panic::set_hook(prev_hook);
    totals.finalize();

    let scale_sweep = if cli.scale_sweep {
        run_scale_sweep(work_dir, cli.seed)?
    } else {
        Vec::new()
    };

    Ok(Report {
        config: ConfigOut {
            iters_per_class: cli.iters_per_class,
            seed: cli.seed,
            generations: cli.generations,
            frames_per_gen: cli.frames_per_gen,
            cycle_iters: cli.cycle_iters,
            kill_iters: cli.kill_iters,
        },
        classes,
        totals,
        scale_sweep,
        incidents,
        notes: vec![
            "compaction-install states are constructed SYNTHETICALLY from two complete \
             images (old = campaign fixture; new = its live set re-streamed into a fresh \
             single-commit file, the shape db::compact builds at the temp path): capturing \
             a real Store::compact mid-flight from a bench binary would need engine hooks. \
             Temp naming and install order mirror src/db/compact.rs \
             (src_path.with_extension(\"valise.compacting\"), rename + parent-dir fsync)."
                .to_string(),
            "crash-recover-cycle enforces: wrong bytes never; recovery loses at most the \
             single faulted commit (served >= previous round's acknowledged generation - 1); \
             healing commits publish served+1; the final reopen serves every surviving \
             frame byte-exact. Typed read errors are tolerated only on frames whose own \
             commit's bytes carried the injected damage."
                .to_string(),
            "sigkill-storm kill delays are seeded, but real process scheduling makes the \
             outcome DISTRIBUTION machine-dependent; the zero-lost-acks and content \
             invariants themselves are exact."
                .to_string(),
        ],
        campaign_wall_seconds: campaign_start.elapsed().as_secs_f64(),
    })
}

fn print_class_line(name: &str, stats: &ClassStats) {
    eprintln!(
        "[{name}] cases={} newest={} previous={} rejected_at_read={} clean_reject={} \
         wrong_data={} panic={} open_p50={}us p95={}us",
        stats.cases,
        stats.recovered_newest,
        stats.recovered_previous,
        stats.rejected_at_read,
        stats.clean_reject,
        stats.wrong_data,
        stats.panic,
        stats.open_p50_us,
        stats.open_p95_us
    );
}

fn main() -> Result<()> {
    // Hidden worker mode for the sigkill-storm class: the parent
    // re-invokes this binary with the env var set; the worker never
    // parses the CLI.
    if let Some(store) = std::env::var_os(KILL_WORKER_ENV) {
        return run_kill_worker(Path::new(&store));
    }

    let cli = Cli::parse();
    ensure!(cli.iters_per_class > 0, "--iters-per-class must be > 0");
    ensure!(cli.cycle_iters > 0, "--cycle-iters must be > 0");
    ensure!(cli.kill_iters > 0, "--kill-iters must be > 0");
    ensure!(
        cli.generations >= 2,
        "--generations must be >= 2 (torn-header / reorder-sim need a previous generation)"
    );
    ensure!(cli.frames_per_gen > 0, "--frames-per-gen must be > 0");

    let work_dir = std::env::temp_dir().join(format!(
        "valise-crash-campaign-{}-{:x}",
        std::process::id(),
        cli.seed
    ));
    fs::create_dir_all(&work_dir).context("create campaign work dir")?;
    let result = run_campaign(&cli, &work_dir);
    let _ = fs::remove_dir_all(&work_dir);
    let report = result?;

    if let Some(parent) = cli.out.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent).context("create --out parent directory")?;
    }
    fs::write(&cli.out, serde_json::to_string_pretty(&report)?).context("write --out JSON")?;

    // `classes x --iters-per-class` is never the campaign total: two
    // classes carry their own budgets (--cycle-iters, --kill-iters), so
    // that phrasing under-reported the default run by 1,200 cases and
    // contradicted the per-class lines printed just above. `totals.cases`
    // counts one case per iteration of every class and cannot drift.
    println!(
        "campaign complete: {} cases across {} classes \
         (--iters-per-class {}, --cycle-iters {}, --kill-iters {}) in {:.1}s -> {}",
        report.totals.cases,
        report.classes.len(),
        report.config.iters_per_class,
        report.config.cycle_iters,
        report.config.kill_iters,
        report.campaign_wall_seconds,
        cli.out.display()
    );
    println!(
        "totals: newest={} previous={} rejected_at_read={} clean_reject={} \
         wrong_data={} panic={}",
        report.totals.recovered_newest,
        report.totals.recovered_previous,
        report.totals.rejected_at_read,
        report.totals.clean_reject,
        report.totals.wrong_data,
        report.totals.panic
    );
    if report.totals.wrong_data > 0 || report.totals.panic > 0 {
        for incident in &report.incidents {
            eprintln!(
                "INCIDENT [{}] iter {} seed {}: {} — fault: {} — {}",
                incident.class,
                incident.iter,
                incident.iter_seed,
                incident.kind,
                incident.fault,
                incident.detail
            );
        }
        bail!(
            "FAILURE: {} wrong-data and {} panic incidents — this is either a harness \
             bug or a real engine bug; see the incidents list in {}",
            report.totals.wrong_data,
            report.totals.panic,
            cli.out.display()
        );
    }
    Ok(())
}
