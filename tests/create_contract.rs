// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Phase 1 (consolidation plan §4): file-level create-time contract.
//!
//! These tests exercise the contract round-trip and the integrity guard
//! provided by the BLAKE3 digest stamped into the header reserved area
//! at byte 832. They also verify that `CreateOptions` validation runs
//! before any bytes hit disk.

use std::fs::OpenOptions;
use std::io::{Read, Seek, SeekFrom, Write};

use valise::{
    AutoPromote, CreateOptions, DtypeSet, Error, OpenMode, Result, TextMode, ValiseFile,
    VectorContract,
};

fn tmpfile(name: &str) -> std::path::PathBuf {
    let dir = tempfile::tempdir().expect("tempdir").keep();
    dir.join(name)
}

/// `ValiseFile` doesn't impl `Debug`, so `expect_err` won't compile when
/// the `Ok` branch carries one. Tiny helper that strips the `Ok` and
/// returns the `Error` directly.
fn expect_err<T>(r: Result<T>, ctx: &str) -> Error {
    match r {
        Ok(_) => panic!("expected error: {ctx}"),
        Err(e) => e,
    }
}

#[test]
fn default_create_carries_default_contract() {
    let path = tmpfile("default.vls");
    let file = ValiseFile::create(&path).expect("create");
    let contract = file.create_contract().clone();
    drop(file);

    assert!(contract.text_enabled);
    assert_eq!(contract.max_dim, 4096);
    assert_eq!(contract.allowed_dtypes, DtypeSet::ALL.bits());
    assert_eq!(contract.auto_promote_non_f8, 200_000);
    assert_eq!(contract.auto_promote_f8, 500_000);
}

#[test]
fn contract_round_trips_through_open() {
    let path = tmpfile("rt.vls");
    let opts = CreateOptions {
        text: TextMode::Disabled,
        vector: VectorContract {
            max_dim: 1024,
            allowed_dtypes: DtypeSet::F32.union(DtypeSet::BF16),
        },
        auto_promote: AutoPromote {
            non_f8_threshold: 50_000,
            f8_threshold: 200_000,
        },
        ..Default::default()
    };
    let file = ValiseFile::create_with_options(&path, opts).expect("create");
    drop(file);

    let reopened = ValiseFile::open(&path, OpenMode::ReadWrite).expect("reopen");
    let c = reopened.create_contract();
    assert!(!c.text_enabled);
    assert_eq!(c.max_dim, 1024);
    assert_eq!(c.allowed_dtypes, DtypeSet::F32.union(DtypeSet::BF16).bits());
    assert_eq!(c.auto_promote_non_f8, 50_000);
    assert_eq!(c.auto_promote_f8, 200_000);
    assert_eq!(reopened.max_dim(), 1024);
    assert!(!reopened.text_enabled());
    assert_eq!(
        reopened.allowed_dtypes(),
        DtypeSet::F32.union(DtypeSet::BF16)
    );
    assert_eq!(reopened.auto_promote().non_f8_threshold, 50_000);
    assert_eq!(reopened.auto_promote().f8_threshold, 200_000);
}

#[test]
fn rejects_zero_max_dim() {
    let path = tmpfile("bad_dim.vls");
    let opts = CreateOptions {
        vector: VectorContract {
            max_dim: 0,
            allowed_dtypes: DtypeSet::ALL,
        },
        ..Default::default()
    };
    let err = expect_err(
        ValiseFile::create_with_options(&path, opts),
        "zero dim must fail",
    );
    assert!(err.to_string().contains("max_dim"), "got: {err}");
    // Failure happens before file creation (`build_contract` validates
    // before `create_new(true)`).
    assert!(
        !path.exists(),
        "file MUST NOT be created on validation error"
    );
}

#[test]
fn rejects_empty_allowed_dtypes() {
    let path = tmpfile("empty_dt.vls");
    let opts = CreateOptions {
        vector: VectorContract {
            max_dim: 1024,
            allowed_dtypes: DtypeSet::empty(),
        },
        ..Default::default()
    };
    let err = expect_err(
        ValiseFile::create_with_options(&path, opts),
        "empty dtypes must fail",
    );
    assert!(err.to_string().contains("dtypes") || err.to_string().contains("Dtype"));
}

#[test]
fn rejects_low_non_f8_threshold() {
    let path = tmpfile("low_thresh.vls");
    let opts = CreateOptions {
        auto_promote: AutoPromote {
            non_f8_threshold: 100,
            f8_threshold: 100,
        },
        ..Default::default()
    };
    expect_err(
        ValiseFile::create_with_options(&path, opts),
        "threshold below 1024 must fail",
    );
}

#[test]
fn rejects_f8_threshold_below_non_f8() {
    let path = tmpfile("inverted.vls");
    let opts = CreateOptions {
        auto_promote: AutoPromote {
            non_f8_threshold: 200_000,
            f8_threshold: 100_000,
        },
        ..Default::default()
    };
    expect_err(
        ValiseFile::create_with_options(&path, opts),
        "f8 < non_f8 must fail",
    );
}

#[test]
fn tampered_header_digest_recovers_via_footer_consensus() {
    // Plan §4.3, amended by the footer-consensus recovery decision: the
    // 32-byte BLAKE3 digest of `CreateContractV1` stamped at byte 832
    // is the PRIMARY lineage anchor, but a corrupted anchor region no
    // longer bricks the file. When the header copy matches no
    // checksum-valid footer, `open()` re-derives the contract from the
    // footers themselves (2+ agreeing footers, or the single footer of
    // a fresh file like this one) and recovers — the served contract is
    // the footer's embedded, checksum-protected copy. The tamper is
    // still detected (it triggers the recovery path and its warn log);
    // it is just no longer fatal. Fail-closed behavior when NO valid
    // footer exists is pinned by
    // tests/crash_consistency.rs::corrupted_contract_digest_with_no_valid_footer_fails_closed.
    let path = tmpfile("tamper.vls");
    let file = ValiseFile::create(&path).expect("create");
    let pristine_contract = file.create_contract().clone();
    drop(file);

    // Corrupt the digest by flipping one byte at offset 832.
    {
        let mut f = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .expect("open for tamper");
        f.seek(SeekFrom::Start(832)).expect("seek");
        let mut byte = [0u8; 1];
        f.read_exact(&mut byte).expect("read");
        byte[0] ^= 0x01;
        f.seek(SeekFrom::Start(832)).expect("seek");
        f.write_all(&byte).expect("write");
        f.sync_all().expect("sync");
    }

    let reopened = ValiseFile::open(&path, OpenMode::ReadOnly)
        .expect("tampered header digest must recover via footer consensus");
    assert_eq!(
        reopened.create_contract(),
        &pristine_contract,
        "the served contract must be the footer's embedded copy, unchanged"
    );
    assert_eq!(reopened.header().snapshot_generation, 1);
}

#[test]
fn contract_survives_commit_round_trip() {
    // Phase 1: every commit re-stamps the contract unchanged, so a
    // committed file still validates after reopening.
    let path = tmpfile("commit_rt.vls");
    let opts = CreateOptions {
        text: TextMode::Disabled,
        ..Default::default()
    };
    let mut file = ValiseFile::create_with_options(&path, opts).expect("create");
    file.commit().expect("commit");
    drop(file);

    let reopened = ValiseFile::open(&path, OpenMode::ReadOnly).expect("reopen");
    assert!(!reopened.text_enabled());
}
