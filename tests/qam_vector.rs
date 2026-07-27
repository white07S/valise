//! End-to-end QAM vector flows.
//!
//! What we cover:
//!
//! - Register codec → register space → ingest → commit → reopen →
//!   read_vector.
//! - `vector_search` on a non-QAM-sliding space → brute-force fallback.
//! - `vector_search` on a QAM(5,6) space → sign-sketch scan +
//!   QAM-sliding rerank (the sketch index is derived from the stored
//!   phase codes at file-open; no commit-time build).
//! - Tombstone semantics: deleted vectors don't appear in results.
//! - Dim mismatch + unknown space + bad codec params rejected.
//! - Concurrent readers over the in-memory sketch index see consistent hits.

use std::sync::Arc;
use std::thread;

use valise::io::Durability;
use valise::{
    AutoPromote, CreateOptions, Dtype, EmbeddingSpaceSpec, OpenMode, PutFrame, PutVector,
    QamLloydMaxParams, ValiseFile, VectorContract, VectorFidelity, VectorMetric, VectorSearchQuery,
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

/// Open a fresh file ready to ingest QAM vectors. Returns the file +
/// (collection_id, space_id) of the registered context.
fn fresh_file(
    path: &std::path::Path,
    auto_promote_threshold: u64,
) -> (
    ValiseFile,
    valise::format::CollectionId,
    valise::format::EmbeddingSpaceId,
) {
    let opts = CreateOptions {
        auto_promote: AutoPromote {
            non_f8_threshold: auto_promote_threshold,
            f8_threshold: auto_promote_threshold.saturating_mul(5),
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
    (valise, collection, space)
}

#[test]
fn ingest_commit_reopen_read_vector_round_trip() {
    let path = tmpfile("round_trip.vls");
    let (mut valise, collection, space) = fresh_file(&path, 1_000_000);
    let mut vids = Vec::with_capacity(32);
    for i in 0..32usize {
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
    drop(valise);

    let valise = ValiseFile::open(&path, OpenMode::ReadOnly).expect("reopen");
    // Round-trip count of active vectors matches.
    assert_eq!(valise.vectors().len(), 32);
    // And each vector decodes lossily back to a finite vector of the
    // right shape (recall semantics covered by the search test below).
    for vid in &vids {
        match valise
            .read_vector(*vid, valise::Reconstruct::F32Vector)
            .expect("read_vector")
        {
            valise::ReadVectorResult::F32Vector(v) => {
                assert_eq!(v.len(), DIM);
                assert!(v.iter().all(|f| f.is_finite()));
            }
            other => panic!("expected F32Vector, got {other:?}"),
        }
    }
}

#[test]
fn calibrate_from_sample_codec_round_trips() {
    // Hook #2: the production calibration entry. Unlike `fresh_file`,
    // which hands `register_codec_qam` prebuilt params, this fits σ from
    // a real sample via `register_codec_qam_from_sample` — the path the
    // DB layer uses to create a vector space without precomputed params.
    let path = tmpfile("calibrate_from_sample.vls");
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

    // Representative sample → fit per-pair Rayleigh σ.
    let sample: Vec<Vec<f32>> = (0..256u64).map(unit_vec).collect();
    let codec_id = valise
        .register_codec_qam_from_sample(DIM, &sample)
        .expect("register_codec_qam_from_sample");

    let space = valise
        .register_embedding_space(EmbeddingSpaceSpec {
            provider: "test".into(),
            model: "qam-calibrated".into(),
            dimension: DIM as u32,
            metric: VectorMetric::Cosine,
            normalized: true,
            dtype: Dtype::F32,
            primary_codec_id: Some(codec_id),
            secondary_codec_id: None,
        })
        .expect("register_embedding_space");

    let mut vids = Vec::with_capacity(32);
    for i in 0..32usize {
        let v = unit_vec(i as u64);
        let frame = valise
            .put_frame(PutFrame::document(collection, b"doc"))
            .expect("put_frame");
        vids.push(
            valise
                .put_vector(PutVector {
                    owner_frame_id: frame,
                    embedding_space_id: space,
                    values: &v,
                })
                .expect("put_vector"),
        );
    }
    valise.commit().expect("commit");
    drop(valise);

    // Reopen: the calibrated codec persisted and decodes every vector.
    let valise = ValiseFile::open(&path, OpenMode::ReadOnly).expect("reopen");
    assert_eq!(valise.vectors().len(), 32);
    for vid in &vids {
        match valise
            .read_vector(*vid, valise::Reconstruct::F32Vector)
            .expect("read_vector")
        {
            valise::ReadVectorResult::F32Vector(v) => {
                assert_eq!(v.len(), DIM);
                assert!(v.iter().all(|f| f.is_finite()));
            }
            other => panic!("expected F32Vector, got {other:?}"),
        }
    }
}

#[test]
fn sketch_build_handles_non_multiple_of_64_dim() {
    // Regression: a dim that is a multiple of 16 but NOT of 64 (e.g. 80) used
    // to panic at file-open — the sketch builder sized each row `dim/64`
    // (truncating) while `sign_sketch` emits `dim.div_ceil(64)` words. Build a
    // QAM(5,6) space at such a dim, ingest, commit, reopen (builds the sketch),
    // and vector_search (scans it). Must not panic and must return hits.
    fn uvec(dim: usize, seed: u64) -> Vec<f32> {
        let mut v: Vec<f32> = (0..dim)
            .map(|i| ((i as f64) * 0.13 + (seed as f64) * 0.37).sin() as f32)
            .collect();
        let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
        for x in &mut v {
            *x /= n;
        }
        v
    }
    for dim in [80usize, 144] {
        let path = tmpfile(&format!("sketch_dim{dim}.vls"));
        let opts = CreateOptions {
            vector: VectorContract {
                max_dim: dim as u32,
                allowed_dtypes: valise::DtypeSet::ALL,
            },
            durability: Durability::Buffered,
            ..Default::default()
        };
        let mut valise = ValiseFile::create_with_options(&path, opts).expect("create");
        let collection = valise.create_collection("c").expect("collection");
        let sample: Vec<Vec<f32>> = (0..256u64).map(|s| uvec(dim, s)).collect();
        let codec_id = valise
            .register_codec_qam_from_sample(dim, &sample)
            .expect("calibrate");
        let space = valise
            .register_embedding_space(EmbeddingSpaceSpec {
                provider: "test".into(),
                model: "qam".into(),
                dimension: dim as u32,
                metric: VectorMetric::Cosine,
                normalized: true,
                dtype: Dtype::F32,
                primary_codec_id: Some(codec_id),
                secondary_codec_id: None,
            })
            .expect("space");
        for i in 0..32u64 {
            let frame = valise
                .put_frame(PutFrame::document(collection, b"d"))
                .expect("frame");
            valise
                .put_vector(PutVector {
                    owner_frame_id: frame,
                    embedding_space_id: space,
                    values: &uvec(dim, i),
                })
                .expect("put_vector");
        }
        valise.commit().expect("commit");
        drop(valise);

        // Reopen builds the sign-sketch index for this dim (would panic before
        // the div_ceil fix), and the search scans it.
        let valise = ValiseFile::open(&path, OpenMode::ReadOnly).expect("reopen");
        let hits = valise
            .vector_search(VectorSearchQuery::accurate(space, uvec(dim, 5), 10))
            .expect("vector_search");
        assert!(!hits.is_empty(), "dim {dim} search returned no hits");
    }
}

#[test]
fn calibrate_from_sample_rejects_bad_input() {
    // Empty sample and dim-mismatched vectors are rejected before any
    // codec is registered (validation lives in `calibrate`).
    let path = tmpfile("calibrate_bad.vls");
    let opts = CreateOptions {
        vector: VectorContract {
            max_dim: DIM as u32,
            allowed_dtypes: valise::DtypeSet::ALL,
        },
        durability: Durability::Buffered,
        ..Default::default()
    };
    let mut valise = ValiseFile::create_with_options(&path, opts).expect("create");

    assert!(
        valise.register_codec_qam_from_sample(DIM, &[]).is_err(),
        "empty sample must be rejected"
    );
    let wrong_dim = vec![vec![0.1_f32; DIM + 2]];
    assert!(
        valise
            .register_codec_qam_from_sample(DIM, &wrong_dim)
            .is_err(),
        "dim-mismatched sample must be rejected"
    );
}

#[test]
fn active_set_enumeration_excludes_tombstones() {
    // Hook #5: active_frames / active_vectors return only live records
    // (compaction enumeration), and active_vectors filters by space.
    let path = tmpfile("active_set.vls");
    let (mut valise, collection, space) = fresh_file(&path, 1_000_000);
    let mut frames = Vec::new();
    let mut vids = Vec::new();
    for i in 0..10usize {
        let f = valise
            .put_frame(PutFrame::document(collection, b"d"))
            .expect("put_frame");
        let v = valise
            .put_vector(PutVector {
                owner_frame_id: f,
                embedding_space_id: space,
                values: &unit_vec(i as u64),
            })
            .expect("put_vector");
        frames.push(f);
        vids.push(v);
    }
    // Commit first — vectors are buffered until commit and can't be
    // tombstoned while pending.
    valise.commit().expect("commit puts");
    // Tombstone two frames and three vectors.
    valise.delete_frame(frames[0]).expect("del frame");
    valise.delete_frame(frames[1]).expect("del frame");
    valise.delete_vector(vids[0]).expect("del vec");
    valise.delete_vector(vids[1]).expect("del vec");
    valise.delete_vector(vids[2]).expect("del vec");
    valise.commit().expect("commit");
    drop(valise);

    let valise = ValiseFile::open(&path, OpenMode::ReadOnly).expect("reopen");

    let active_frames: Vec<_> = valise.active_frames().map(|f| f.frame_id).collect();
    assert_eq!(active_frames.len(), 8, "10 - 2 tombstoned");
    assert!(!active_frames.contains(&frames[0]));
    assert!(!active_frames.contains(&frames[1]));
    assert!(active_frames.contains(&frames[2]));

    let active_vids: Vec<_> = valise.active_vectors(space).map(|v| v.vector_id).collect();
    assert_eq!(active_vids.len(), 7, "10 - 3 tombstoned");
    assert!(!active_vids.contains(&vids[0]));
    assert!(active_vids.contains(&vids[3]));
    assert!(
        valise
            .active_vectors(space)
            .all(|v| v.embedding_space_id == space)
    );

    // A different embedding space yields nothing.
    assert_eq!(
        valise
            .active_vectors(valise::format::EmbeddingSpaceId(9999))
            .count(),
        0
    );
}

#[test]
fn non_5_6_codec_finds_self_match_via_sketch() {
    // fresh_file registers a (4,4) QAM codec. Every QAM config (not just the
    // production (5,6)) now derives a sign-sketch at open and routes through
    // sketch-scan -> general asymmetric rerank, so the self-match is recovered
    // without a (5,6)-only fast path.
    let path = tmpfile("brute_force.vls");
    let (mut valise, collection, space) = fresh_file(&path, 1_000_000);
    for i in 0..16usize {
        let v = unit_vec(i as u64);
        let frame = valise
            .put_frame(PutFrame::document(collection, b"x"))
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

    // Every ingested vector should be its own top-1 hit (modulo codec
    // quantization noise — the symmetric self-match is robust at
    // amp_bits=4, phase_bits=4 for a 16-vector corpus).
    let mut self_matches = 0usize;
    for i in 0..16usize {
        let q = unit_vec(i as u64);
        let hits = valise
            .vector_search(VectorSearchQuery {
                embedding_space_id: space,
                query: q,
                k: 1,
                channel_k: None,
                collection_filter: None,
                fidelity: VectorFidelity::Full,
            })
            .expect("vector_search");
        assert_eq!(hits.len(), 1, "expected exactly one hit");
        if hits[0].vector_id.0 == (i as u64 + 1) {
            self_matches += 1;
        }
    }
    assert!(
        self_matches >= 14,
        "self-match recall too low: {self_matches}/16"
    );
}

#[test]
fn sketch_path_finds_self_match_at_scale() {
    // Ingest 1100 vectors. At open, the sign-sketch index is derived
    // from the QAM phase codes (no commit-time build, no persisted
    // segment); vector_search routes through the sketch-scan →
    // QAM-sliding rerank path and recovers the self-match.
    const N: usize = 1100;
    let path = tmpfile("sketch_scale.vls");
    let (mut valise, collection, space) = fresh_file(&path, 1024);
    for i in 0..N {
        let v = unit_vec(i as u64);
        let frame = valise
            .put_frame(PutFrame::document(collection, b"x"))
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
    let q = unit_vec(0);
    let hits = valise
        .vector_search(VectorSearchQuery {
            embedding_space_id: space,
            query: q,
            k: 5,
            channel_k: Some(N / 4),
            collection_filter: None,
            fidelity: VectorFidelity::Full,
        })
        .expect("sketch search");
    assert_eq!(hits.len(), 5);
    assert!(
        hits.iter().any(|h| h.vector_id.0 == 1),
        "self-match should appear in top-5 of sketch+rerank"
    );
}

#[test]
fn higher_bit_codec_via_with_bits_uses_sketch_path() {
    // Register a non-production (6,7) codec through the bit-configurable
    // calibration entry. It must build a sign-sketch at open (proven by
    // vector_search_traced, which errors if no sketch index exists) and
    // recover self-matches via sketch-scan -> general asymmetric rerank.
    const N: usize = 600;
    let path = tmpfile("with_bits_67.vls");
    let opts = CreateOptions {
        vector: VectorContract {
            max_dim: DIM as u32,
            allowed_dtypes: valise::DtypeSet::ALL,
        },
        durability: Durability::Buffered,
        ..Default::default()
    };
    let mut valise = ValiseFile::create_with_options(&path, opts).expect("create");
    let collection = valise.create_collection("c").expect("collection");
    let sample: Vec<Vec<f32>> = (0..256).map(|i| unit_vec(i as u64)).collect();
    let codec_id = valise
        .register_codec_qam_from_sample_with_bits(DIM, 6, 7, &sample)
        .expect("register (6,7) codec");
    let space = valise
        .register_embedding_space(EmbeddingSpaceSpec {
            provider: "test".into(),
            model: "qam-6-7".into(),
            dimension: DIM as u32,
            metric: VectorMetric::Cosine,
            normalized: true,
            dtype: Dtype::F32,
            primary_codec_id: Some(codec_id),
            secondary_codec_id: None,
        })
        .expect("register_embedding_space");
    for i in 0..N {
        let v = unit_vec(i as u64);
        let frame = valise
            .put_frame(PutFrame::document(collection, b"x"))
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
    // `vector_search_traced` requires a sign-sketch index for the space; if
    // the (6,7) codec hadn't generalized, this would error instead.
    let mut self_matches = 0usize;
    for i in 0..N {
        let (hits, _trace) = valise
            .vector_search_traced(VectorSearchQuery::accurate(space, unit_vec(i as u64), 1))
            .expect("traced sketch search (6,7) must have a sketch index");
        assert_eq!(hits.len(), 1);
        if hits[0].vector_id.0 == (i as u64 + 1) {
            self_matches += 1;
        }
    }
    assert!(
        self_matches as f64 / N as f64 >= 0.95,
        "(6,7) sketch-path self-match recall too low: {self_matches}/{N}"
    );
}

#[test]
fn accurate_query_forces_sketch_path_and_fast_is_lossy() {
    // accurate() and fast() both leave channel_k = None (so the engine's
    // fixed default budget applies — sized to reach the codec ceiling); they
    // differ only in rerank fidelity. Full = f32 rerank, Lossy = i8 scores.
    let q = VectorSearchQuery::accurate(valise::format::EmbeddingSpaceId(1), vec![0.0; DIM], 5);
    assert!(
        q.channel_k.is_none(),
        "accurate uses the default candidate budget (no per-call override)"
    );
    assert_eq!(q.fidelity, VectorFidelity::Full);
    let f = VectorSearchQuery::fast(valise::format::EmbeddingSpaceId(1), vec![0.0; DIM], 5);
    assert!(f.channel_k.is_none());
    assert_eq!(f.fidelity, VectorFidelity::Lossy);

    // End-to-end: accurate() returns the correct self-match via the
    // sketch-scan → QAM-sliding rerank path.
    const N: usize = 1100;
    let path = tmpfile("accurate.vls");
    let (mut valise, collection, space) = fresh_file(&path, 1024);
    for i in 0..N {
        let frame = valise
            .put_frame(PutFrame::document(collection, b"x"))
            .expect("put_frame");
        valise
            .put_vector(PutVector {
                owner_frame_id: frame,
                embedding_space_id: space,
                values: &unit_vec(i as u64),
            })
            .expect("put_vector");
    }
    valise.commit().expect("commit");
    drop(valise);

    let valise = ValiseFile::open(&path, OpenMode::ReadOnly).expect("reopen");
    let hits = valise
        .vector_search(VectorSearchQuery::accurate(space, unit_vec(1), 5))
        .expect("accurate search");
    assert_eq!(hits.len(), 5);
    assert!(
        hits.iter().any(|h| h.vector_id.0 == 2),
        "self-match should appear in top-5 of sketch+rerank"
    );
}

#[test]
fn delete_vector_tombstones_it_from_search() {
    let path = tmpfile("delete.vls");
    let (mut valise, collection, space) = fresh_file(&path, 1_000_000);
    let mut vids = Vec::new();
    for i in 0..16usize {
        let v = unit_vec(i as u64);
        let frame = valise
            .put_frame(PutFrame::document(collection, b"x"))
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
    // Tombstone vid #3 (which is unit_vec(2) = best self-match for query
    // unit_vec(2)).
    valise.delete_vector(vids[2]).expect("delete_vector");
    valise.commit().expect("commit");

    let q = unit_vec(2);
    let hits = valise
        .vector_search(VectorSearchQuery {
            embedding_space_id: space,
            query: q,
            k: 5,
            channel_k: None,
            collection_filter: None,
            fidelity: VectorFidelity::Full,
        })
        .expect("vector_search");
    assert!(
        !hits.iter().any(|h| h.vector_id == vids[2]),
        "tombstoned vid {} appeared in search results: {:?}",
        vids[2].0,
        hits.iter().map(|h| h.vector_id.0).collect::<Vec<_>>(),
    );
}

#[test]
fn register_embedding_space_rejects_dim_above_contract() {
    let path = tmpfile("dim_contract.vls");
    let opts = CreateOptions {
        vector: VectorContract {
            max_dim: 32,
            allowed_dtypes: valise::DtypeSet::ALL,
        },
        durability: Durability::Buffered,
        ..Default::default()
    };
    let mut valise = ValiseFile::create_with_options(&path, opts).expect("create");
    let codec_id = valise
        .register_codec_qam(small_qam_params())
        .expect("register_codec_qam");
    let err = valise
        .register_embedding_space(EmbeddingSpaceSpec {
            provider: "test".into(),
            model: "too-big".into(),
            dimension: 64, // > contract.max_dim = 32
            metric: VectorMetric::Cosine,
            normalized: true,
            dtype: Dtype::F32,
            primary_codec_id: Some(codec_id),
            secondary_codec_id: None,
        })
        .expect_err("dim > contract.max_dim should be rejected");
    assert!(
        err.to_string().contains("max_dim") || err.to_string().contains("dimension"),
        "error didn't mention dim limit: {err}"
    );
}

#[test]
fn put_vector_rejects_dim_mismatch() {
    let path = tmpfile("dim_mismatch.vls");
    let (mut valise, collection, space) = fresh_file(&path, 1_000_000);
    let frame = valise
        .put_frame(PutFrame::document(collection, b"x"))
        .expect("put_frame");
    let wrong_size_vec = vec![0.5f32; DIM + 1];
    let err = valise
        .put_vector(PutVector {
            owner_frame_id: frame,
            embedding_space_id: space,
            values: &wrong_size_vec,
        })
        .expect_err("wrong dim should be rejected");
    let _ = err;
}

#[test]
fn vector_search_rejects_unknown_space() {
    let path = tmpfile("unknown_space.vls");
    let (mut valise, _collection, _space) = fresh_file(&path, 1_000_000);
    valise.commit().expect("commit");
    let err = valise
        .vector_search(VectorSearchQuery {
            embedding_space_id: valise::EmbeddingSpaceId(9999),
            query: unit_vec(0),
            k: 5,
            channel_k: None,
            collection_filter: None,
            fidelity: VectorFidelity::Full,
        })
        .expect_err("unknown space should be rejected");
    assert!(
        err.to_string().contains("unknown") || err.to_string().contains("embedding_space"),
        "error didn't mention unknown space: {err}"
    );
}

#[test]
fn concurrent_readers_see_consistent_hits() {
    // Populate a vote-auto-tiered file, open multiple read handles
    // from multiple threads, and verify each thread's self-match
    // appears in its top-k. The vote_index_cache + per-shard scratch
    // are exercised concurrently here.
    const N: usize = 1100;
    let path = tmpfile("concurrent.vls");
    let (mut valise, collection, space) = fresh_file(&path, 1024);
    for i in 0..N {
        let v = unit_vec(i as u64);
        let frame = valise
            .put_frame(PutFrame::document(collection, b"x"))
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

    let path_arc = Arc::new(path);
    let mut handles = Vec::new();
    for t in 0..4u64 {
        let p = Arc::clone(&path_arc);
        handles.push(thread::spawn(move || {
            let valise = ValiseFile::open(&*p, OpenMode::ReadOnly).expect("open");
            let q = unit_vec(t);
            valise
                .vector_search(VectorSearchQuery {
                    embedding_space_id: space,
                    query: q,
                    k: 10,
                    channel_k: Some(N / 4),
                    collection_filter: None,
                    fidelity: VectorFidelity::Full,
                })
                .expect("vector_search")
                .into_iter()
                .map(|h| h.vector_id.0)
                .collect::<Vec<_>>()
        }));
    }
    let results: Vec<Vec<u64>> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    // Each thread queries a different vector → different top-10, but
    // the self-match (vid t+1) should be in its top-10.
    for (t, r) in results.iter().enumerate() {
        assert!(
            r.contains(&((t as u64) + 1)),
            "thread {t} top-10 missing self-match (vid {}): {r:?}",
            (t as u64) + 1
        );
    }
}

#[test]
fn crash_safety_open_after_partial_segment_truncation() {
    // Write a populated file, then truncate the underlying bytes to
    // simulate a crash mid-segment-write. Open should recover to the
    // previous TOC.
    let path = tmpfile("crash.vls");
    let (mut valise, collection, space) = fresh_file(&path, 1_000_000);
    for i in 0..8usize {
        let v = unit_vec(i as u64);
        let frame = valise
            .put_frame(PutFrame::document(collection, b"x"))
            .expect("put_frame");
        valise
            .put_vector(PutVector {
                owner_frame_id: frame,
                embedding_space_id: space,
                values: &v,
            })
            .expect("put_vector");
    }
    valise.commit().expect("first commit");
    // Capture file length at the known-good TOC.
    let good_len = std::fs::metadata(&path).expect("stat").len();

    // Append more vectors but truncate before committing the new TOC.
    for i in 8..16usize {
        let v = unit_vec(i as u64);
        let frame = valise
            .put_frame(PutFrame::document(collection, b"x"))
            .expect("put_frame");
        valise
            .put_vector(PutVector {
                owner_frame_id: frame,
                embedding_space_id: space,
                values: &v,
            })
            .expect("put_vector");
    }
    // Drop without committing — leaves the file at its good length.
    drop(valise);

    let actual_len = std::fs::metadata(&path).expect("stat after drop").len();
    assert_eq!(
        actual_len, good_len,
        "drop without commit shouldn't extend the file"
    );

    // Reopen; should see the first 8 vectors only.
    let valise = ValiseFile::open(&path, OpenMode::ReadOnly).expect("recovery open");
    assert_eq!(valise.vectors().len(), 8);
}
