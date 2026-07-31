// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! A `VectorData` segment whose base bytes are corrupted on disk must be
//! rejected on read, not decoded into a wrong vector. `read_vector`
//! re-hashes the segment payload (once per open) against the stored
//! checksum, so a flipped base byte surfaces as `Error::Integrity`.

use valise::io::Durability;
use valise::{
    AutoPromote, CreateOptions, Dtype, EmbeddingSpaceSpec, Error, OpenMode, PutFrame, PutVector,
    QamLloydMaxParams, Reconstruct, ValiseFile, VectorContract, VectorMetric,
};

const DIM: usize = 64;
const NUM_PAIRS: u32 = (DIM / 2) as u32;

fn tmpfile(name: &str) -> std::path::PathBuf {
    let dir = tempfile::tempdir().expect("tempdir").keep();
    dir.join(name)
}

fn small_qam_params() -> QamLloydMaxParams {
    QamLloydMaxParams {
        dimension: DIM as u32,
        num_pairs: NUM_PAIRS,
        block_size: 16,
        rotation_seed: 0xC0DE,
        amp_bits: 4,
        phase_bits: 4,
        renormalize_at_decode: false,
        sigma_per_pair: vec![1.0_f32 / DIM as f32; NUM_PAIRS as usize],
        calibration_id: [0u8; 32],
    }
}

/// Deterministic L2-normalized vector.
fn unit_vec(seed: u64) -> Vec<f32> {
    let mut v: Vec<f32> = (0..DIM)
        .map(|i| {
            let phase = (i as f64) * 0.13 + (seed as f64) * 0.37;
            (phase.sin() + (phase * 1.7).cos() * 0.5) as f32
        })
        .collect();
    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
    for x in &mut v {
        *x /= n;
    }
    v
}

/// Scan the raw file bytes for the single `VectorData` (VLVD) segment and
/// return `(payload_start, payload_len)`. Matches a `VLSG` header whose
/// type is `0x0020` and whose payload begins with the `VLVD` magic (the
/// magic check rules out a coincidental `VLSG` byte pattern in payload
/// data).
fn find_vectordata_payload(bytes: &[u8]) -> (usize, usize) {
    const HEADER: usize = 76;
    let mut i = 0usize;
    while i + HEADER <= bytes.len() {
        if &bytes[i..i + 4] == b"VLSG" {
            let seg_type = u16::from_le_bytes(bytes[i + 6..i + 8].try_into().unwrap());
            let payload_len =
                u64::from_le_bytes(bytes[i + 36..i + 44].try_into().unwrap()) as usize;
            let payload_start = i + HEADER;
            if seg_type == 0x0020
                && payload_start + 4 <= bytes.len()
                && &bytes[payload_start..payload_start + 4] == b"VLVD"
            {
                return (payload_start, payload_len);
            }
        }
        i += 1;
    }
    panic!("no VectorData (VLVD) segment found in file");
}

#[test]
fn corrupted_vectordata_base_is_detected_on_read() {
    let path = tmpfile("fix1_vectordata_verify.vls");

    // Build a 1-vector QAM capsule via the public API.
    let opts = CreateOptions {
        auto_promote: AutoPromote {
            non_f8_threshold: 1_000_000,
            f8_threshold: 5_000_000,
        },
        vector: VectorContract {
            max_dim: DIM as u32,
            allowed_dtypes: valise::DtypeSet::ALL,
        },
        durability: Durability::Buffered,
        ..Default::default()
    };
    let mut valise = ValiseFile::create_with_options(&path, opts).expect("create");
    let collection = valise.create_collection("c").expect("collection");
    let codec_id = valise
        .register_codec_qam(small_qam_params())
        .expect("register_codec_qam");
    let space = valise
        .register_embedding_space(EmbeddingSpaceSpec {
            provider: "test".into(),
            model: "qam-test".into(),
            dimension: DIM as u32,
            metric: VectorMetric::Cosine,
            normalized: true,
            dtype: Dtype::F32,
            primary_codec_id: Some(codec_id),
            secondary_codec_id: None,
        })
        .expect("register_embedding_space");
    let frame = valise
        .put_frame(PutFrame::document(collection, b"doc-0"))
        .expect("put_frame");
    let vid = valise
        .put_vector(PutVector {
            owner_frame_id: frame,
            embedding_space_id: space,
            values: &unit_vec(0),
        })
        .expect("put_vector");
    valise.commit().expect("commit");
    drop(valise);

    // Sanity: an uncorrupted reopen reads the vector cleanly.
    {
        let valise = ValiseFile::open(&path, OpenMode::ReadOnly).expect("reopen");
        let ok = valise.read_vector(vid, Reconstruct::F32Vector);
        assert!(ok.is_ok(), "pre-corruption read should succeed: {ok:?}");
    }

    // Corrupt the first vector's base bytes on disk: XOR 0xFF over the
    // 8 bytes at (payload_start + base_block_offset). The header offsets
    // and payload length are untouched, so the VLVD reader still opens
    // the segment happily — only the stored BLAKE3 catches it.
    let mut bytes = std::fs::read(&path).expect("read file");
    let (payload_start, payload_len) = find_vectordata_payload(&bytes);
    let base_offset =
        u32::from_le_bytes(bytes[payload_start + 26..payload_start + 30].try_into().unwrap())
            as usize;
    let corrupt_at = payload_start + base_offset;
    assert!(
        corrupt_at + 8 <= payload_start + payload_len,
        "corruption window must stay inside the payload"
    );
    for b in &mut bytes[corrupt_at..corrupt_at + 8] {
        *b ^= 0xFF;
    }
    std::fs::write(&path, &bytes).expect("write corrupted file");

    // Reopen (fresh handle → empty verified set) and read: the fix turns
    // the previously-silent corruption into a detected integrity error.
    let valise = ValiseFile::open(&path, OpenMode::ReadOnly).expect("reopen corrupted");
    let result = valise.read_vector(vid, Reconstruct::F32Vector);
    match result {
        Err(Error::Integrity(_)) => {}
        other => panic!("expected Err(Error::Integrity), got {other:?}"),
    }
}
