//! In-guest worker for the QEMU power-cut campaign (`bench/vmcrash/`).
//!
//! Runs as PID 1 inside a minimal Alpine-derived initramfs (exec'd by
//! `/init` after the data filesystem is mounted at `/data`). Two modes,
//! selected by `valisemode=` on the kernel command line:
//!
//! - `commit`: create `/data/store.vls`, then loop forever:
//!   put one deterministic seeded frame -> `commit()` -> only AFTER
//!   `commit()` returns, write `ACK <seq> <generation> <sha256/16>` to
//!   the console and flush. The host SIGKILLs qemu at a seeded moment;
//!   the set of COMPLETE ACK lines the host received before the kill is
//!   the acknowledged set (host-observable definition).
//! - `verify`: reopen the store read-only (scan-back recovery
//!   permitted), enumerate frames in order, byte-compare each against
//!   the same seeded payload derivation, and report
//!   `GEN` / `PRESENT` / `VERIFYDONE` lines, then power off.
//!
//! Console protocol lines (everything else is noise to the host parser):
//!   WORKER mode=<m> seed=<hex>
//!   ACK <seq> <generation> <sha256-first-16-hex>
//!   GEN <generation>
//!   PRESENT <seq> <sha256-first-16-hex> <ok|BAD|ERR:...>
//!   VERIFYDONE ok=<true|false> max_seq=<i64> gen=<u64> unopenable=<bool>
//!   WORKERFATAL <context>: <error>

use std::io::Write as _;

use sha2::{Digest, Sha256};

use valise::{ValiseFile, PutFrame};

// ---------------------------------------------------------------------------
// Deterministic payload derivation (mirrored in bench/vmcrash/campaign.py).
// ---------------------------------------------------------------------------

fn splitmix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// High-entropy payload, 512..4096 bytes, fully determined by
/// `(seed, seq)`. High entropy keeps zstd on the raw-block path so the
/// stored bytes carry the payload verbatim.
fn derive_payload(seed: u64, seq: u64) -> Vec<u8> {
    let mut x = splitmix64(seed ^ seq.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    let len = 512 + (x % 3584) as usize;
    let mut buf = vec![0u8; len];
    for chunk in buf.chunks_mut(8) {
        x = splitmix64(x);
        chunk.copy_from_slice(&x.to_le_bytes()[..chunk.len()]);
    }
    buf
}

fn sha256_16(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    hex::encode(&digest[..8])
}

// ---------------------------------------------------------------------------
// Console + cmdline plumbing.
// ---------------------------------------------------------------------------

/// Write one whole line + flush. stdout is /dev/console (the qemu
/// serial port); the host parses complete lines only, so a line torn by
/// the kill is simply not an ack.
fn emit(line: &str) {
    let mut out = std::io::stdout().lock();
    let _ = writeln!(out, "{line}");
    let _ = out.flush();
}

fn cmdline_arg(cmdline: &str, key: &str) -> Option<String> {
    cmdline
        .split_ascii_whitespace()
        .find_map(|tok| tok.strip_prefix(key).map(|v| v.to_string()))
}

/// Sync everything and power the VM off. PID 1 owns CAP_SYS_BOOT.
fn power_off() -> ! {
    unsafe {
        libc::sync();
        libc::reboot(libc::LINUX_REBOOT_CMD_POWER_OFF);
    }
    // If reboot(2) somehow returned, spin: the host has a hard timeout.
    loop {
        std::thread::sleep(std::time::Duration::from_secs(3600));
    }
}

fn fatal(context: &str, err: impl std::fmt::Display) -> ! {
    emit(&format!("WORKERFATAL {context}: {err}"));
    power_off()
}

// ---------------------------------------------------------------------------
// Modes.
// ---------------------------------------------------------------------------

const STORE: &str = "/data/store.vls";

fn run_commit(seed: u64) -> ! {
    let mut valise = match ValiseFile::create(STORE) {
        Ok(f) => f,
        Err(e) => fatal("create store", e),
    };
    let coll = match valise.create_collection("vmcrash") {
        Ok(c) => c,
        Err(e) => fatal("create collection", e),
    };
    for seq in 0u64.. {
        let payload = derive_payload(seed, seq);
        if let Err(e) = valise.put_frame(PutFrame::document(coll, &payload)) {
            fatal("put_frame", e);
        }
        let outcome = match valise.commit() {
            Ok(o) => o,
            Err(e) => fatal("commit", e),
        };
        // The ack is written strictly AFTER commit() returned: an ack
        // the host reads is a durability promise the store already made.
        emit(&format!(
            "ACK {seq} {} {}",
            outcome.snapshot_generation,
            sha256_16(&payload)
        ));
    }
    power_off()
}

fn run_verify(seed: u64) -> ! {
    let file = match ValiseFile::open_read_only(STORE) {
        Ok(f) => f,
        Err(e) => {
            emit(&format!("WORKERFATAL open store: {e}"));
            emit("VERIFYDONE ok=false max_seq=-1 gen=0 unopenable=true");
            power_off()
        }
    };
    let generation = file.header().snapshot_generation;
    emit(&format!("GEN {generation}"));
    let stubs = file.frame_stubs().to_vec();
    let mut all_ok = true;
    for (seq, stub) in stubs.iter().enumerate() {
        let expect = derive_payload(seed, seq as u64);
        match file.read_payload(stub.frame_id) {
            Ok(bytes) if bytes == expect => {
                emit(&format!("PRESENT {seq} {} ok", sha256_16(&bytes)));
            }
            Ok(bytes) => {
                all_ok = false;
                emit(&format!(
                    "PRESENT {seq} {} BAD expected {} ({} bytes served, {} expected)",
                    sha256_16(&bytes),
                    sha256_16(&expect),
                    bytes.len(),
                    expect.len()
                ));
            }
            Err(e) => {
                all_ok = false;
                emit(&format!("PRESENT {seq} - ERR:{e}"));
            }
        }
    }
    emit(&format!(
        "VERIFYDONE ok={all_ok} max_seq={} gen={generation} unopenable=false",
        stubs.len() as i64 - 1
    ));
    power_off()
}

fn main() {
    let cmdline = std::fs::read_to_string("/proc/cmdline").unwrap_or_default();
    let mode = cmdline_arg(&cmdline, "valisemode=").unwrap_or_default();
    let seed = cmdline_arg(&cmdline, "valiseseed=")
        .and_then(|s| u64::from_str_radix(s.trim_start_matches("0x"), 16).ok())
        .unwrap_or(0);
    emit(&format!("WORKER mode={mode} seed={seed:#018x}"));
    match mode.as_str() {
        "commit" => run_commit(seed),
        "verify" => run_verify(seed),
        other => fatal("mode", format!("unknown valisemode '{other}'")),
    }
}
