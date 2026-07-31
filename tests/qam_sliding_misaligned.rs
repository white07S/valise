// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! QAM-sliding scoring over dims where `num_pairs` is not a multiple of 8.
//! The kernels only cover whole 8-pair groups; the trailing `num_pairs % 8`
//! pairs are scored by a scalar tail so they aren't dropped, and
//! `num_pairs < 8` no longer walks off the end. The multiple-of-8 fast path
//! is unchanged.

use valise::io::Durability;
use valise::{
    AutoPromote, CreateOptions, Dtype, EmbeddingSpaceSpec, OpenMode, PutFrame, PutVector,
    QamLloydMaxParams, ValiseFile, VectorContract, VectorFidelity, VectorMetric, VectorSearchQuery,
};

fn tmpfile(name: &str) -> std::path::PathBuf {
    let dir = tempfile::tempdir().expect("tempdir").keep();
    dir.join(name)
}

/// Build a QAM(5,6) codec's params for an arbitrary even `dim`, with a
/// `block_size` that divides `dim`. `num_pairs = dim / 2` is deliberately
/// NOT a multiple of 8 here — the case the scalar tail has to handle.
fn qam_5_6_params(dim: u32, block_size: u32) -> QamLloydMaxParams {
    let num_pairs = dim / 2;
    QamLloydMaxParams {
        dimension: dim,
        num_pairs,
        block_size,
        rotation_seed: 0xC0DE,
        amp_bits: 5,
        phase_bits: 6,
        renormalize_at_decode: false,
        // Uniform prior σ = 1/√dim; positive + finite, all the codec
        // validators require.
        sigma_per_pair: vec![(1.0_f32 / dim as f32).sqrt(); num_pairs as usize],
        calibration_id: [0u8; 32],
    }
}

fn make_file(
    path: &std::path::Path,
    dim: u32,
    block_size: u32,
) -> (
    ValiseFile,
    valise::format::CollectionId,
    valise::format::EmbeddingSpaceId,
) {
    let opts = CreateOptions {
        auto_promote: AutoPromote {
            non_f8_threshold: 1_000_000,
            f8_threshold: 5_000_000,
        },
        vector: VectorContract {
            max_dim: dim,
            allowed_dtypes: valise::DtypeSet::ALL,
        },
        durability: Durability::Buffered,
        ..Default::default()
    };
    let mut valise = ValiseFile::create_with_options(path, opts).expect("create");
    let collection = valise.create_collection("c").expect("collection");
    let codec_id = valise
        .register_codec_qam(qam_5_6_params(dim, block_size))
        .expect("register_codec_qam");
    let space = valise
        .register_embedding_space(EmbeddingSpaceSpec {
            provider: "test".into(),
            model: "qam-5-6".into(),
            dimension: dim,
            metric: VectorMetric::Cosine,
            normalized: true,
            dtype: Dtype::F32,
            primary_codec_id: Some(codec_id),
            secondary_codec_id: None,
        })
        .expect("register_embedding_space");
    (valise, collection, space)
}

/// (a) Tail-drop fixed. dim=100 QAM(5,6) → num_pairs=50 = 6·8 + 2, so
/// pairs 48 and 49 (rotated dims 96..=99, exactly the last block_size=4
/// Hadamard block) are the *tail* the old kernel dropped.
///
/// A and B are byte-identical in dims 0..96 and carry the same fixed
/// magnitude in the tail block — just parked in a different otherwise-zero
/// slot (dim 96 for A, dim 98 for B) — so their L2 norms match exactly and
/// they differ ONLY in the tail. With the tail dropped, `score(A)` and
/// `score(B)` against query = A were identical. Counting the tail, the
/// self-match A must score differently from the permuted B.
#[test]
fn tail_pairs_are_scored_dim100() {
    const DIM: usize = 100;
    let path = tmpfile("fix3_tail_dim100.vls");
    let (mut valise, collection, space) = make_file(&path, DIM as u32, 4);

    // Base pattern shared by A and B across dims 0..96 (the full groups).
    let mut a = vec![0.0_f32; DIM];
    for (i, x) in a.iter_mut().enumerate().take(96) {
        *x = ((i as f32) * 0.11).sin() * 0.4 + 0.05;
    }
    let mut b = a.clone();
    // Move a fixed magnitude between two otherwise-zero tail slots so the
    // L2 norms are identical but the tail block content differs.
    let tail_val = 0.9_f32;
    a[96] = tail_val; // A: energy at dim 96 (pair 48)
    b[98] = tail_val; // B: energy at dim 98 (pair 49)

    // Sanity: identical in the full-group region, equal norm overall.
    assert_eq!(&a[..96], &b[..96]);
    let na: f32 = a.iter().map(|v| v * v).sum();
    let nb: f32 = b.iter().map(|v| v * v).sum();
    assert!((na - nb).abs() < 1e-6, "A and B must share an L2 norm");

    let put = |valise: &mut ValiseFile, v: &[f32]| -> valise::format::VectorId {
        let frame = valise
            .put_frame(PutFrame::document(collection, b"d"))
            .expect("put_frame");
        valise
            .put_vector(PutVector {
                owner_frame_id: frame,
                embedding_space_id: space,
                values: v,
            })
            .expect("put_vector")
    };

    let vid_a = put(&mut valise, &a);
    let vid_b = put(&mut valise, &b);
    // A few distractors so the corpus isn't just the two crafted rows.
    for s in 0..6u64 {
        let mut d = vec![0.0_f32; DIM];
        for (i, x) in d.iter_mut().enumerate() {
            *x = ((i as f32) * 0.07 + (s as f32) * 1.3).cos() * 0.5;
        }
        put(&mut valise, &d);
    }
    valise.commit().expect("commit");
    drop(valise);

    let valise = ValiseFile::open(&path, OpenMode::ReadOnly).expect("reopen");
    let hits = valise
        .vector_search(VectorSearchQuery {
            embedding_space_id: space,
            query: a.clone(),
            k: 8,
            channel_k: Some(64),
            collection_filter: None,
            fidelity: VectorFidelity::Lossy,
        })
        .expect("vector_search");

    let score_of = |vid: valise::format::VectorId| -> f32 {
        hits.iter()
            .find(|h| h.vector_id == vid)
            .unwrap_or_else(|| panic!("vid {} missing from hits: {hits:?}", vid.0))
            .score
    };
    let sa = score_of(vid_a);
    let sb = score_of(vid_b);
    assert!(
        (sa - sb).abs() > 1e-9,
        "tail pairs were dropped: score(A)={sa} == score(B)={sb} even though A and B differ only in the tail block"
    );
}

/// (b) Underflow fixed. dim=8 QAM(5,6) → num_pairs=4 < 8, so the old
/// kernel computed `groups = 0` and then `groups - 1 = usize::MAX`,
/// indexing out of bounds. The search must now complete without a
/// crash/panic.
#[test]
fn small_num_pairs_does_not_underflow_dim8() {
    const DIM: usize = 8;
    let path = tmpfile("fix3_underflow_dim8.vls");
    let (mut valise, collection, space) = make_file(&path, DIM as u32, 4);

    for s in 0..16u64 {
        let mut v = vec![0.0_f32; DIM];
        for (i, x) in v.iter_mut().enumerate() {
            *x = ((i as f32) * 0.31 + (s as f32) * 0.53).sin() + 0.1;
        }
        let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
        for x in &mut v {
            *x /= n;
        }
        let frame = valise
            .put_frame(PutFrame::document(collection, b"d"))
            .expect("put_frame");
        valise
            .put_vector(PutVector {
                owner_frame_id: frame,
                embedding_space_id: space,
                values: &v,
            })
            .expect("put_vector");
    }
    valise.commit().expect("commit");
    drop(valise);

    let valise = ValiseFile::open(&path, OpenMode::ReadOnly).expect("reopen");
    let mut query = vec![0.0_f32; DIM];
    for (i, x) in query.iter_mut().enumerate() {
        *x = ((i as f32) * 0.31 + 0.53).sin() + 0.1;
    }
    let res = valise.vector_search(VectorSearchQuery::accurate(space, query, 5));
    assert!(res.is_ok(), "num_pairs=4 search must not crash: {res:?}");
    assert!(
        !res.unwrap().is_empty(),
        "num_pairs=4 search returned no hits"
    );
}
