// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Concurrent search correctness.
//!
//! The writer-pipeline test (`tests/writer_pipeline.rs`) proves that a
//! long-lived `WriteConnection` does not block concurrent
//! `ReadConnection`s. This file covers the orthogonal axis: many
//! `ReadConnection`s issuing search queries simultaneously must
//! produce consistent, deterministic results and not race on shared
//! caches (sketch index, decoded postings, doc-normalizer, etc.).
//!
//! Three guarantees:
//! 1. **Vector search** — N reader threads × M queries each see
//!    identical top-k frame_ids for the same query input.
//! 2. **Text search** — same shape for BM25, exercising the shared
//!    `bm25_cache` (impact postings + decoded postings + doc_normalizer).
//! 3. **Snapshot isolation under writer** — readers that pin a
//!    snapshot before the writer starts must see a frozen view of the
//!    catalog regardless of intervening commits.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::Duration;

use tempfile::tempdir;
use valise::{
    AnalyzerDesc, AnalyzerId, AutoPromote, CreateOptions, Database, Dtype, DtypeSet,
    EmbeddingSpaceSpec, FieldDesc, FieldSchemaDesc, FieldSchemaId, FieldSource, IdfVariant,
    OpenMode, PunctuationPolicy, PutFrame, PutVector, QamLloydMaxParams, QueryAlgorithm,
    RetrievalProfileDesc, RetrievalProfileId, RetrievalProfileParams, RetrievalProfileType,
    Stemming, StopwordsPolicy, TextQuery, TextSpaceDesc, TextSpaceId, Tokenizer,
    UnicodeNormalization, ValiseFile, VectorContract, VectorFidelity, VectorMetric,
    VectorSearchQuery,
};

const DIM: u32 = 32;
const NUM_PAIRS: u32 = DIM / 2;

fn qam_params() -> QamLloydMaxParams {
    QamLloydMaxParams {
        dimension: DIM,
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

fn build_vector_file(path: &std::path::Path, n_vectors: usize, auto_tier: u64) {
    let opts = CreateOptions {
        auto_promote: AutoPromote {
            non_f8_threshold: auto_tier,
            f8_threshold: auto_tier.saturating_mul(5),
        },
        vector: VectorContract {
            max_dim: DIM,
            allowed_dtypes: DtypeSet::ALL,
        },
        ..Default::default()
    };
    let mut valise = ValiseFile::create_with_options(path, opts).expect("create");
    let collection = valise.create_collection("c").expect("collection");
    let codec_id = valise.register_codec_qam(qam_params()).expect("codec");
    let space = valise
        .register_embedding_space(EmbeddingSpaceSpec {
            provider: "test".into(),
            model: "qam".into(),
            dimension: DIM,
            metric: VectorMetric::Cosine,
            normalized: true,
            dtype: Dtype::F32,
            primary_codec_id: Some(codec_id),
            secondary_codec_id: None,
        })
        .expect("space");
    for i in 0..n_vectors {
        let v = unit_vec(i as u64);
        let frame = valise
            .put_frame(PutFrame::document(
                collection,
                format!("doc-{i}").as_bytes(),
            ))
            .expect("put_frame");
        valise
            .put_vector(PutVector {
                owner_frame_id: frame,
                embedding_space_id: space,
                values: &v,
            })
            .expect("put_vector");
        let _ = space;
    }
    valise.commit().expect("commit");
    drop(valise);
}

#[test]
fn concurrent_vector_readers_see_identical_top_k() {
    const N_VECTORS: usize = 2_000;
    const N_THREADS: usize = 8;
    const QUERIES_PER_THREAD: usize = 64;

    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("concurrent_vec.vls");
    build_vector_file(&path, N_VECTORS, 1024);

    let db = Database::open(&path, OpenMode::ReadOnly).expect("open db");
    let space = db.reader().embedding_spaces()[0].embedding_space_id;

    // Compute the baseline top-k once (single thread) so every spawned
    // thread can diff against it.
    let baseline_reader = db.reader();
    let mut baselines: HashMap<u64, Vec<u64>> = HashMap::new();
    for seed in 0..QUERIES_PER_THREAD as u64 {
        let q = unit_vec(seed);
        let hits = baseline_reader
            .vector_search(VectorSearchQuery {
                embedding_space_id: space,
                query: q,
                k: 10,
                channel_k: None,
                collection_filter: None,
                fidelity: VectorFidelity::Full,
            })
            .expect("baseline search");
        baselines.insert(seed, hits.iter().map(|h| h.frame_id.0).collect());
    }
    drop(baseline_reader);

    let baselines = Arc::new(baselines);
    let mut handles = Vec::with_capacity(N_THREADS);
    for _ in 0..N_THREADS {
        let db = Arc::clone(&db);
        let baselines = Arc::clone(&baselines);
        handles.push(thread::spawn(move || {
            let reader = db.reader();
            for seed in 0..QUERIES_PER_THREAD as u64 {
                let q = unit_vec(seed);
                let hits = reader
                    .vector_search(VectorSearchQuery {
                        embedding_space_id: space,
                        query: q,
                        k: 10,
                        channel_k: None,
                        collection_filter: None,
                        fidelity: VectorFidelity::Full,
                    })
                    .expect("worker search");
                let got: Vec<u64> = hits.iter().map(|h| h.frame_id.0).collect();
                let want = baselines.get(&seed).expect("baseline present");
                assert_eq!(&got, want, "thread saw different top-k for seed {seed}");
            }
        }));
    }
    for h in handles {
        h.join().expect("worker panic");
    }
}

fn build_text_file(path: &std::path::Path, n_docs: usize) -> (TextSpaceId, RetrievalProfileId) {
    let mut valise = ValiseFile::create(path).expect("create");
    let collection = valise.create_collection("c").expect("collection");
    let analyzer_id = valise
        .register_analyzer(AnalyzerDesc {
            analyzer_id: AnalyzerId(0),
            unicode_normalization: UnicodeNormalization::Nfkc,
            case_fold: true,
            accent_fold: true,
            tokenizer: Tokenizer::UnicodeWords,
            stemming: Stemming::Porter2English,
            stopword_set_ref: None,
            stopword_query_only: true,
            stopwords: StopwordsPolicy::None,
            english_possessive_strip: true,
            min_token_len: 2,
            max_token_len: 64,
            shingle_size: 0,
            ngram_min: None,
            ngram_max: None,
            punctuation_policy: PunctuationPolicy::Drop,
        })
        .expect("analyzer");
    let fs_id = valise
        .register_field_schema(FieldSchemaDesc {
            field_schema_id: FieldSchemaId(0),
            fields: vec![FieldDesc {
                field_id: 1,
                name: "search_text".into(),
                source: FieldSource::SearchText,
                store_positions: false,
                store_term_freq: true,
                store_set_membership: false,
                weight: 1.0,
            }],
        })
        .expect("field_schema");
    let profile_id = valise
        .register_retrieval_profile(RetrievalProfileDesc {
            profile_id: RetrievalProfileId(0),
            profile_type: RetrievalProfileType::Bm25,
            params: RetrievalProfileParams::Bm25 {
                k1: 1.2,
                b: 0.75,
                idf_variant: IdfVariant::RobertsonSparckJones,
            },
        })
        .expect("profile");
    let text_space_id = valise
        .register_text_space(TextSpaceDesc {
            text_space_id: TextSpaceId(0),
            name: "bm25".into(),
            analyzer_id,
            field_schema_id: fs_id,
            default_profile_id: profile_id,
            flags: 0,
            enabled_retrievers: 0,
        })
        .expect("space");
    let words = [
        "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
    ];
    for i in 0..n_docs {
        let mut body = String::new();
        for j in 0..6 {
            body.push_str(words[(i + j) % words.len()]);
            body.push(' ');
        }
        let frame = valise
            .put_frame(PutFrame::document(collection, body.as_bytes()))
            .expect("put_frame");
        valise
            .index_frame_text(frame, text_space_id)
            .expect("index_frame_text");
    }
    valise.commit().expect("commit");
    drop(valise);
    (text_space_id, profile_id)
}

#[test]
fn concurrent_text_readers_see_identical_top_k() {
    const N_DOCS: usize = 1_000;
    const N_THREADS: usize = 8;

    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("concurrent_text.vls");
    let (text_space_id, profile_id) = build_text_file(&path, N_DOCS);

    let db = Database::open(&path, OpenMode::ReadOnly).expect("open db");
    let queries = [
        "alpha beta",
        "gamma delta epsilon",
        "eta theta",
        "alpha gamma eta",
    ];

    let baseline_reader = db.reader();
    let mut baselines: HashMap<String, Vec<u64>> = HashMap::new();
    for q in &queries {
        let hits = baseline_reader
            .query_text(TextQuery {
                text_space_id,
                query: (*q).into(),
                algorithm: QueryAlgorithm::Profile(profile_id),
                top_k: Some(20),
                channel_k: None,
            })
            .expect("baseline text search");
        baselines.insert((*q).into(), hits.iter().map(|h| h.frame_id.0).collect());
    }
    drop(baseline_reader);

    let baselines = Arc::new(baselines);
    let queries_arc: Arc<[&'static str]> = Arc::from(queries.as_slice());
    let mut handles = Vec::with_capacity(N_THREADS);
    for _ in 0..N_THREADS {
        let db = Arc::clone(&db);
        let baselines = Arc::clone(&baselines);
        let queries = Arc::clone(&queries_arc);
        handles.push(thread::spawn(move || {
            let reader = db.reader();
            for _ in 0..32 {
                for q in queries.iter() {
                    let hits = reader
                        .query_text(TextQuery {
                            text_space_id,
                            query: (*q).into(),
                            algorithm: QueryAlgorithm::Profile(profile_id),
                            top_k: Some(20),
                            channel_k: None,
                        })
                        .expect("worker text search");
                    let got: Vec<u64> = hits.iter().map(|h| h.frame_id.0).collect();
                    let want = baselines.get(*q).expect("baseline present");
                    assert_eq!(&got, want, "thread saw different top-k for query {q:?}");
                }
            }
        }));
    }
    for h in handles {
        h.join().expect("worker panic");
    }
}

#[test]
fn readers_under_concurrent_writer_remain_consistent() {
    // A background writer doing put_frame + commit cycles must NOT
    // change the catalog view seen through reader connections pinned
    // before the writer started. Each `ReadConnection` snapshots at
    // acquisition; the snapshot pins the catalog generation so reads
    // are stable.
    const N_INITIAL: usize = 500;
    const N_THREADS: usize = 4;
    const N_COMMITS: u64 = 5;
    const QUERIES_PER_READER: usize = 100;

    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("rw_concurrent.vls");
    let (text_space_id, profile_id) = build_text_file(&path, N_INITIAL);

    let db = Database::open(&path, OpenMode::ReadWrite).expect("open db");
    let writer_done = Arc::new(AtomicU64::new(0));
    let queries = ["alpha beta", "gamma delta"];

    // Record the initial frame count + baseline query results.
    let baseline_reader = db.reader();
    let initial_frame_count = baseline_reader.frame_stubs().len();
    let mut baselines: HashMap<String, Vec<u64>> = HashMap::new();
    for q in &queries {
        let hits = baseline_reader
            .query_text(TextQuery {
                text_space_id,
                query: (*q).into(),
                algorithm: QueryAlgorithm::Profile(profile_id),
                top_k: Some(20),
                channel_k: None,
            })
            .expect("baseline");
        baselines.insert((*q).into(), hits.iter().map(|h| h.frame_id.0).collect());
    }
    drop(baseline_reader);
    let baselines = Arc::new(baselines);

    // Spawn writer thread that does put_frame + commit in a loop.
    let writer_db = Arc::clone(&db);
    let writer_done_clone = Arc::clone(&writer_done);
    let writer = thread::spawn(move || {
        let mut writer = writer_db.writer();
        let cid = writer.collections()[0].collection_id;
        for i in 0..N_COMMITS {
            writer
                .put_frame(PutFrame::document(cid, format!("late-{i}").as_bytes()))
                .expect("late put_frame");
            writer.commit().expect("late commit");
            writer_done_clone.store(i + 1, Ordering::SeqCst);
            thread::sleep(Duration::from_millis(2));
        }
    });

    // Reader threads — pin a snapshot, then assert per-query top-k +
    // catalog frame count remain frozen across writer commits.
    let mut readers = Vec::with_capacity(N_THREADS);
    for _ in 0..N_THREADS {
        let db = Arc::clone(&db);
        let baselines = Arc::clone(&baselines);
        readers.push(thread::spawn(move || {
            let reader = db.reader();
            let pinned_count = reader.frame_stubs().len();
            for _ in 0..QUERIES_PER_READER {
                // Frame count seen through this snapshot must NOT
                // advance even though the writer is committing.
                assert_eq!(
                    reader.frame_stubs().len(),
                    pinned_count,
                    "pinned snapshot frame count drifted under concurrent writer"
                );
                for q in &queries {
                    let hits = reader
                        .query_text(TextQuery {
                            text_space_id,
                            query: (*q).into(),
                            algorithm: QueryAlgorithm::Profile(profile_id),
                            top_k: Some(20),
                            channel_k: None,
                        })
                        .expect("reader-under-writer search");
                    let got: Vec<u64> = hits.iter().map(|h| h.frame_id.0).collect();
                    let want = baselines.get(*q).expect("baseline present");
                    assert_eq!(
                        &got, want,
                        "pinned snapshot drifted under concurrent writer for {q:?}"
                    );
                }
            }
        }));
    }

    writer.join().expect("writer panic");
    for r in readers {
        r.join().expect("reader panic");
    }
    assert_eq!(writer_done.load(Ordering::SeqCst), N_COMMITS);

    // A freshly-acquired reader must see the newly-committed frames.
    let fresh = db.reader();
    let fresh_count = fresh.frame_stubs().len();
    assert_eq!(
        fresh_count,
        initial_frame_count + N_COMMITS as usize,
        "post-writer snapshot should see N_COMMITS extra frames"
    );
}
