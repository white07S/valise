// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! End-to-end UPQ codec-family flows (mirrors `qam_vector.rs`).
//!
//! - Register UPQ codec (from sample / from params) → register space →
//!   ingest → commit → reopen → search (sketch + decoded-i8 rerank).
//! - `accurate` vs `fast` fidelity both find the self-match.
//! - Rayleigh-designed (data-free) codec round-trips identically.
//! - Params accessor returns byte-equivalent params after reopen.
//! - Tombstones drop out of results; dim mismatches rejected.

use valise::io::Durability;
use valise::{
    AutoPromote, CreateOptions, Dtype, EmbeddingSpaceSpec, OpenMode, PutFrame, PutVector,
    UpqDesignSource, ValiseFile, VectorContract, VectorFidelity, VectorMetric, VectorSearchQuery,
};

const DIM: usize = 64;
const N: usize = 64;

fn tmpfile(name: &str) -> std::path::PathBuf {
    let dir = tempfile::tempdir().expect("tempdir").keep();
    dir.join(name)
}

/// Deterministic L2-normalized vector for tests.
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

fn sample_rows(n: usize) -> Vec<Vec<f32>> {
    (0..n as u64).map(unit_vec).collect()
}

fn fresh_file(
    path: &std::path::Path,
    source: UpqDesignSource,
) -> (
    ValiseFile,
    valise::format::CollectionId,
    valise::format::EmbeddingSpaceId,
) {
    let opts = CreateOptions {
        auto_promote: AutoPromote {
            non_f8_threshold: 1 << 20,
            f8_threshold: 1 << 22,
        },
        vector: VectorContract {
            max_dim: DIM as u32,
            allowed_dtypes: valise::DtypeSet::ALL,
        },
        durability: Durability::Buffered,
        ..Default::default()
    };
    let mut valise = ValiseFile::create_with_options(path, opts).expect("create");
    let collection = valise.create_collection("c").expect("collection");
    let codec_id = valise
        .register_codec_upq_from_sample_with_options(DIM, 256, source, &sample_rows(128))
        .expect("register_codec_upq");
    let space = valise
        .register_embedding_space(EmbeddingSpaceSpec {
            provider: "test".into(),
            model: "upq-test".into(),
            dimension: DIM as u32,
            metric: VectorMetric::Cosine,
            normalized: true,
            dtype: Dtype::F32,
            primary_codec_id: Some(codec_id),
            secondary_codec_id: None,
        })
        .expect("register_embedding_space");
    (valise, collection, space)
}

fn ingest(
    valise: &mut ValiseFile,
    collection: valise::format::CollectionId,
    space: valise::format::EmbeddingSpaceId,
    n: usize,
) -> Vec<valise::format::VectorId> {
    let mut vids = Vec::with_capacity(n);
    for i in 0..n {
        let v = unit_vec(i as u64);
        let frame = valise
            .put_frame(PutFrame::document(
                collection,
                format!("doc-{i}").as_bytes(),
            ))
            .expect("put_frame");
        let vid = valise
            .put_vector(PutVector {
                owner_frame_id: frame,
                embedding_space_id: space,
                values: &v,
            })
            .expect("put_vector");
        vids.push(vid);
    }
    valise.commit().expect("commit");
    vids
}

fn search(
    valise: &ValiseFile,
    space: valise::format::EmbeddingSpaceId,
    query: Vec<f32>,
    k: usize,
    fidelity: VectorFidelity,
) -> Vec<valise::VectorHit> {
    let mut q = VectorSearchQuery::accurate(space, query, k);
    q.fidelity = fidelity;
    valise.vector_search(q).expect("vector_search")
}

#[test]
fn upq_register_ingest_commit_reopen_search_roundtrip() {
    let path = tmpfile("upq_roundtrip.vls");
    let (mut valise, collection, space) = fresh_file(&path, UpqDesignSource::Empirical);
    let vids = ingest(&mut valise, collection, space, N);
    drop(valise);

    let valise = ValiseFile::open(&path, OpenMode::ReadOnly).expect("reopen");
    for probe in [0usize, 7, 33, N - 1] {
        let hits = search(
            &valise,
            space,
            unit_vec(probe as u64),
            4,
            VectorFidelity::Full,
        );
        assert!(!hits.is_empty(), "probe {probe}: no hits");
        assert_eq!(
            hits[0].vector_id, vids[probe],
            "probe {probe}: self-match not rank 1 (hits: {hits:?})"
        );
    }
}

#[test]
fn upq_fast_and_accurate_both_find_self() {
    let path = tmpfile("upq_fast.vls");
    let (mut valise, collection, space) = fresh_file(&path, UpqDesignSource::Empirical);
    let vids = ingest(&mut valise, collection, space, N);

    for fidelity in [VectorFidelity::Full, VectorFidelity::Lossy] {
        let hits = search(&valise, space, unit_vec(11), 4, fidelity);
        assert_eq!(
            hits[0].vector_id, vids[11],
            "{fidelity:?}: self-match not rank 1"
        );
    }
}

#[test]
fn upq_rayleigh_design_roundtrips() {
    let path = tmpfile("upq_ray.vls");
    let (mut valise, collection, space) = fresh_file(&path, UpqDesignSource::Rayleigh);
    let vids = ingest(&mut valise, collection, space, N);
    drop(valise);

    let valise = ValiseFile::open(&path, OpenMode::ReadOnly).expect("reopen");
    let hits = search(&valise, space, unit_vec(21), 4, VectorFidelity::Full);
    assert_eq!(hits[0].vector_id, vids[21]);
}

#[test]
fn upq_params_accessor_survives_reopen() {
    let path = tmpfile("upq_params.vls");
    let opts = CreateOptions {
        vector: VectorContract {
            max_dim: DIM as u32,
            allowed_dtypes: valise::DtypeSet::ALL,
        },
        durability: Durability::Buffered,
        ..Default::default()
    };
    let mut valise = ValiseFile::create_with_options(&path, opts).expect("create");
    let codec_id = valise
        .register_codec_upq_from_sample(DIM, &sample_rows(128))
        .expect("register");
    let before = valise.codec_upq_params(codec_id).expect("params before");
    assert_eq!(before.cells, 2048);
    assert_eq!(before.dimension, DIM as u32);
    valise.commit().expect("commit");
    drop(valise);

    let valise = ValiseFile::open(&path, OpenMode::ReadOnly).expect("reopen");
    let after = valise.codec_upq_params(codec_id).expect("params after");
    assert_eq!(before, after, "params must round-trip byte-equivalent");
}

#[test]
fn upq_delete_tombstones_vector_from_search() {
    let path = tmpfile("upq_tombstone.vls");
    let (mut valise, collection, space) = fresh_file(&path, UpqDesignSource::Empirical);
    let vids = ingest(&mut valise, collection, space, N);

    valise.delete_vector(vids[5]).expect("delete");
    valise.commit().expect("commit delete");

    let hits = search(&valise, space, unit_vec(5), 8, VectorFidelity::Full);
    assert!(
        hits.iter().all(|h| h.vector_id != vids[5]),
        "tombstoned vector still surfaced: {hits:?}"
    );
}

#[test]
fn upq_space_rejects_query_dim_mismatch() {
    let path = tmpfile("upq_dim.vls");
    let (mut valise, collection, space) = fresh_file(&path, UpqDesignSource::Empirical);
    ingest(&mut valise, collection, space, 8);
    let err = valise.vector_search(VectorSearchQuery::accurate(space, vec![0.0f32; DIM / 2], 4));
    assert!(err.is_err(), "short query must be rejected");
}
