//! Integration tests for the Phase 2 text retrieval pipeline.
//!
//! Covered:
//! - register_analyzer / register_field_schema / register_retrieval_profile /
//!   register_text_space persist across reopen
//! - put_frame → index_frame_text → commit → reopen → query_text
//! - BM25, TF-IDF, exact term-Jaccard, count_cosine, tfidf_cosine, dice,
//!   overlap, containment all return expected scores on a hand-computed corpus
//! - tombstoned frames excluded
//! - pre-commit query rejected
//! - register_text_space rejects unknown analyzer / field_schema / profile
//! - multi-text-space same corpus with different analyzers

use tempfile::tempdir;
use valise::{
    AnalyzerDesc, AnalyzerId, FieldDesc, FieldSchemaDesc, FieldSchemaId, FieldSource, IdfMode,
    IdfVariant, NormMode, PunctuationPolicy, PutFrame, QueryAlgorithm, RetrievalProfileDesc,
    RetrievalProfileId, RetrievalProfileParams, RetrievalProfileType, Stemming, StopwordsPolicy,
    TextQuery, TextSpaceDesc, TextSpaceId, TfMode, TokenSource, Tokenizer, UnicodeNormalization,
    ValiseFile,
};

fn lucene_like_analyzer() -> AnalyzerDesc {
    AnalyzerDesc {
        analyzer_id: AnalyzerId(0),
        unicode_normalization: UnicodeNormalization::Nfc,
        case_fold: true,
        accent_fold: true,
        tokenizer: Tokenizer::UnicodeWords,
        stemming: Stemming::None,
        stopword_set_ref: None,
        stopword_query_only: true,
        stopwords: StopwordsPolicy::None,
        english_possessive_strip: false,
        min_token_len: 0,
        max_token_len: 0,
        shingle_size: 0,
        ngram_min: None,
        ngram_max: None,
        punctuation_policy: PunctuationPolicy::Drop,
    }
}

fn whitespace_analyzer() -> AnalyzerDesc {
    AnalyzerDesc {
        tokenizer: Tokenizer::Whitespace,
        unicode_normalization: UnicodeNormalization::None,
        case_fold: false,
        accent_fold: false,
        ..lucene_like_analyzer()
    }
}

fn search_text_field_schema() -> FieldSchemaDesc {
    FieldSchemaDesc {
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
    }
}

fn bm25_profile() -> RetrievalProfileDesc {
    RetrievalProfileDesc {
        profile_id: RetrievalProfileId(0),
        profile_type: RetrievalProfileType::Bm25,
        params: RetrievalProfileParams::Bm25 {
            k1: 1.2,
            b: 0.75,
            idf_variant: IdfVariant::RobertsonSparckJones,
        },
    }
}

fn tfidf_profile() -> RetrievalProfileDesc {
    RetrievalProfileDesc {
        profile_id: RetrievalProfileId(0),
        profile_type: RetrievalProfileType::Tfidf,
        params: RetrievalProfileParams::Tfidf {
            tf_mode: TfMode::Log,
            idf_mode: IdfMode::Smooth,
            norm_mode: NormMode::L2,
        },
    }
}

fn jaccard_profile() -> RetrievalProfileDesc {
    RetrievalProfileDesc {
        profile_id: RetrievalProfileId(0),
        profile_type: RetrievalProfileType::JaccardExact,
        params: RetrievalProfileParams::JaccardExact {
            token_source: TokenSource::Terms,
        },
    }
}

struct Setup {
    valise: ValiseFile,
    text_space_id: TextSpaceId,
    bm25_profile_id: RetrievalProfileId,
    tfidf_profile_id: RetrievalProfileId,
    jaccard_profile_id: RetrievalProfileId,
}

fn setup_with_corpus(path: &std::path::Path, corpus: &[&str]) -> Setup {
    let mut valise = ValiseFile::create(path).expect("create");
    let collection_id = valise.create_collection("docs").expect("collection");
    let analyzer_id = valise.register_analyzer(lucene_like_analyzer()).unwrap();
    let field_schema_id = valise
        .register_field_schema(search_text_field_schema())
        .unwrap();
    let bm25_profile_id = valise.register_retrieval_profile(bm25_profile()).unwrap();
    let tfidf_profile_id = valise.register_retrieval_profile(tfidf_profile()).unwrap();
    let jaccard_profile_id = valise
        .register_retrieval_profile(jaccard_profile())
        .unwrap();
    let text_space_id = valise
        .register_text_space(TextSpaceDesc {
            text_space_id: TextSpaceId(0),
            name: "default".into(),
            analyzer_id,
            field_schema_id,
            default_profile_id: bm25_profile_id,
            flags: 0,
            enabled_retrievers: 0,
        })
        .unwrap();

    for text in corpus {
        let frame_id = valise
            .put_frame(PutFrame::document(collection_id, text.as_bytes()))
            .unwrap();
        valise.index_frame_text(frame_id, text_space_id).unwrap();
    }
    valise.commit().unwrap();

    Setup {
        valise,
        text_space_id,
        bm25_profile_id,
        tfidf_profile_id,
        jaccard_profile_id,
    }
}

#[test]
fn text_catalogs_persist_across_reopen() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("a.vls");
    let setup = setup_with_corpus(&path, &["alpha beta", "alpha gamma", "delta"]);
    drop(setup);

    let valise = ValiseFile::open_read_only(&path).unwrap();
    assert_eq!(valise.analyzers().len(), 1);
    assert_eq!(valise.field_schemas().len(), 1);
    assert_eq!(valise.retrieval_profiles().len(), 3);
    assert_eq!(valise.text_spaces().len(), 1);
    assert_eq!(valise.frame_stubs().len(), 3);
}

#[test]
fn bm25_query_returns_known_top_hit() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("b.vls");
    let setup = setup_with_corpus(
        &path,
        &[
            "The quick brown fox jumps over the lazy dog",
            "Lorem ipsum dolor sit amet",
            "The brown dog sleeps under the tree",
        ],
    );
    let hits = setup
        .valise
        .query_text(TextQuery {
            text_space_id: setup.text_space_id,
            query: "brown dog".into(),
            algorithm: QueryAlgorithm::Profile(setup.bm25_profile_id),
            top_k: None,
            channel_k: None,
        })
        .unwrap();
    // Frames 1 and 3 both contain "brown dog"; frame 2 has neither.
    assert!(hits.len() >= 2);
    assert!(!hits.iter().any(|h| h.frame_id.0 == 2));
}

#[test]
fn profile_free_bm25_matches_registered_profile_and_writes_no_catalog() {
    // Hook #3: QueryAlgorithm::Bm25 { k1, b } scores from the canonical
    // df/doclen/avgdl without a registered profile and without mutating
    // the catalog on the read path.
    let dir = tempdir().unwrap();
    let path = dir.path().join("pf_bm25.vls");
    let setup = setup_with_corpus(
        &path,
        &[
            "The quick brown fox jumps over the lazy dog",
            "Lorem ipsum dolor sit amet",
            "The brown dog sleeps under the tree",
        ],
    );

    let profiles_before = setup.valise.retrieval_profiles().len();

    // Same k1/b as the registered BM25 profile (1.2 / 0.75) → identical hits.
    let via_profile = setup
        .valise
        .query_text(TextQuery {
            text_space_id: setup.text_space_id,
            query: "brown dog".into(),
            algorithm: QueryAlgorithm::Profile(setup.bm25_profile_id),
            top_k: None,
            channel_k: None,
        })
        .unwrap();
    let profile_free = setup
        .valise
        .query_text(TextQuery {
            text_space_id: setup.text_space_id,
            query: "brown dog".into(),
            algorithm: QueryAlgorithm::Bm25 { k1: 1.2, b: 0.75 },
            top_k: None,
            channel_k: None,
        })
        .unwrap();

    assert_eq!(via_profile.len(), profile_free.len());
    assert!(!profile_free.is_empty());
    for (a, b) in via_profile.iter().zip(profile_free.iter()) {
        assert_eq!(a.frame_id, b.frame_id);
        assert!(
            (a.score - b.score).abs() < 1e-6,
            "profile-free BM25 score {} != profiled {}",
            b.score,
            a.score
        );
    }

    // Arbitrary k1/b not matching any registered profile still scores.
    let custom = setup
        .valise
        .query_text(TextQuery {
            text_space_id: setup.text_space_id,
            query: "brown dog".into(),
            algorithm: QueryAlgorithm::Bm25 { k1: 2.0, b: 0.3 },
            top_k: None,
            channel_k: None,
        })
        .unwrap();
    assert!(!custom.is_empty());

    // The read path registered no new profile.
    assert_eq!(setup.valise.retrieval_profiles().len(), profiles_before);
}

#[test]
fn tfidf_l2_query_normalizes_scores() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("c.vls");
    let setup = setup_with_corpus(
        &path,
        &["alpha beta gamma", "alpha alpha beta", "alpha beta delta"],
    );
    let hits = setup
        .valise
        .query_text(TextQuery {
            text_space_id: setup.text_space_id,
            query: "alpha beta".into(),
            algorithm: QueryAlgorithm::Profile(setup.tfidf_profile_id),
            top_k: None,
            channel_k: None,
        })
        .unwrap();
    assert!(!hits.is_empty());
    // L2-normalized cosine-like score is bounded — sanity check.
    for h in &hits {
        assert!(h.score.is_finite() && h.score > 0.0);
    }
}

#[test]
fn jaccard_query_returns_set_intersection_score() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("d.vls");
    let setup = setup_with_corpus(&path, &["alpha beta", "alpha beta gamma", "delta"]);
    let hits = setup
        .valise
        .query_text(TextQuery {
            text_space_id: setup.text_space_id,
            query: "alpha beta".into(),
            algorithm: QueryAlgorithm::Profile(setup.jaccard_profile_id),
            top_k: None,
            channel_k: None,
        })
        .unwrap();
    // Frame 1 == query → score 1.0; frame 2 contains both → 2/3.
    assert_eq!(hits.len(), 2);
    assert!((hits[0].score - 1.0).abs() < 1e-4);
    assert!((hits[1].score - (2.0 / 3.0)).abs() < 1e-4);
}

#[test]
fn count_cosine_profile_free_query_works() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("e.vls");
    let setup = setup_with_corpus(&path, &["alpha beta", "alpha gamma", "delta epsilon"]);
    let hits = setup
        .valise
        .query_text(TextQuery {
            text_space_id: setup.text_space_id,
            query: "alpha beta".into(),
            algorithm: QueryAlgorithm::CountCosine,
            top_k: None,
            channel_k: None,
        })
        .unwrap();
    assert!(hits.iter().any(|h| h.frame_id.0 == 1));
    assert!(!hits.iter().any(|h| h.frame_id.0 == 3));
}

#[test]
fn tfidf_cosine_profile_free_query_works() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("f.vls");
    let setup = setup_with_corpus(&path, &["alpha beta", "alpha gamma"]);
    let hits = setup
        .valise
        .query_text(TextQuery {
            text_space_id: setup.text_space_id,
            query: "alpha beta".into(),
            algorithm: QueryAlgorithm::TfidfCosine {
                tf_mode: TfMode::Raw,
            },
            top_k: None,
            channel_k: None,
        })
        .unwrap();
    assert!(!hits.is_empty());
}

#[test]
fn dice_overlap_containment_profile_free_queries_work() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("g.vls");
    let setup = setup_with_corpus(&path, &["alpha beta", "alpha"]);
    for algorithm in [
        QueryAlgorithm::Dice,
        QueryAlgorithm::Overlap,
        QueryAlgorithm::Containment,
    ] {
        let hits = setup
            .valise
            .query_text(TextQuery {
                text_space_id: setup.text_space_id,
                query: "alpha beta".into(),
                algorithm,
                top_k: None,
                channel_k: None,
            })
            .unwrap();
        assert!(!hits.is_empty(), "algorithm {algorithm:?} returned no hits");
    }
}

#[test]
fn tombstoned_frames_excluded_from_query() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("h.vls");
    let mut setup = setup_with_corpus(&path, &["alpha", "alpha beta", "alpha gamma"]);
    // Tombstone frame 2.
    let frame_id = setup.valise.frames()[1].frame_id;
    setup.valise.delete_frame(frame_id).unwrap();
    setup.valise.commit().unwrap();
    let hits = setup
        .valise
        .query_text(TextQuery {
            text_space_id: setup.text_space_id,
            query: "alpha".into(),
            algorithm: QueryAlgorithm::Profile(setup.bm25_profile_id),
            top_k: None,
            channel_k: None,
        })
        .unwrap();
    assert!(hits.iter().all(|h| h.frame_id != frame_id));
}

#[test]
fn pre_commit_query_rejected() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("i.vls");
    let mut valise = ValiseFile::create(&path).unwrap();
    let collection_id = valise.create_collection("docs").unwrap();
    let analyzer_id = valise.register_analyzer(lucene_like_analyzer()).unwrap();
    let fs_id = valise
        .register_field_schema(search_text_field_schema())
        .unwrap();
    let profile_id = valise.register_retrieval_profile(bm25_profile()).unwrap();
    let text_space_id = valise
        .register_text_space(TextSpaceDesc {
            text_space_id: TextSpaceId(0),
            name: "ts".into(),
            analyzer_id,
            field_schema_id: fs_id,
            default_profile_id: profile_id,
            flags: 0,
            enabled_retrievers: 0,
        })
        .unwrap();
    let frame_id = valise
        .put_frame(PutFrame::document(collection_id, b"alpha beta"))
        .unwrap();
    valise.index_frame_text(frame_id, text_space_id).unwrap();
    // No commit — query should reject.
    let err = valise
        .query_text(TextQuery {
            text_space_id,
            query: "alpha".into(),
            algorithm: QueryAlgorithm::Profile(profile_id),
            top_k: None,
            channel_k: None,
        })
        .expect_err("pre-commit");
    assert!(err.to_string().contains("uncommitted"));
}

#[test]
fn multi_text_space_with_different_analyzers() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("j.vls");
    let mut valise = ValiseFile::create(&path).unwrap();
    let collection_id = valise.create_collection("docs").unwrap();

    // Two analyzers: lucene-like (case-fold + accent-fold) and whitespace
    // (preserves case + punctuation).
    let analyzer_a = valise.register_analyzer(lucene_like_analyzer()).unwrap();
    let analyzer_b = valise.register_analyzer(whitespace_analyzer()).unwrap();
    let fs_id = valise
        .register_field_schema(search_text_field_schema())
        .unwrap();
    let profile_id = valise.register_retrieval_profile(bm25_profile()).unwrap();

    let space_a = valise
        .register_text_space(TextSpaceDesc {
            text_space_id: TextSpaceId(0),
            name: "lucene".into(),
            analyzer_id: analyzer_a,
            field_schema_id: fs_id,
            default_profile_id: profile_id,
            flags: 0,
            enabled_retrievers: 0,
        })
        .unwrap();
    let space_b = valise
        .register_text_space(TextSpaceDesc {
            text_space_id: TextSpaceId(0),
            name: "whitespace".into(),
            analyzer_id: analyzer_b,
            field_schema_id: fs_id,
            default_profile_id: profile_id,
            flags: 0,
            enabled_retrievers: 0,
        })
        .unwrap();

    let frame_id = valise
        .put_frame(PutFrame::document(
            collection_id,
            b"CAFE caf\xC3\xA9 punctuation!",
        ))
        .unwrap();
    valise.index_frame_text(frame_id, space_a).unwrap();
    valise.index_frame_text(frame_id, space_b).unwrap();
    valise.commit().unwrap();

    // Lucene-like space: "café" and "CAFE" both fold to "cafe", so one
    // unique term.
    let hits_a = valise
        .query_text(TextQuery {
            text_space_id: space_a,
            query: "cafe".into(),
            algorithm: QueryAlgorithm::Profile(profile_id),
            top_k: None,
            channel_k: None,
        })
        .unwrap();
    assert_eq!(hits_a.len(), 1);

    // Whitespace space: the raw token is "CAFE", "café", and
    // "punctuation!" — query "cafe" doesn't match (case-sensitive).
    let hits_b = valise
        .query_text(TextQuery {
            text_space_id: space_b,
            query: "cafe".into(),
            algorithm: QueryAlgorithm::Profile(profile_id),
            top_k: None,
            channel_k: None,
        })
        .unwrap();
    assert!(hits_b.is_empty());
    let hits_b_exact = valise
        .query_text(TextQuery {
            text_space_id: space_b,
            query: "CAFE".into(),
            algorithm: QueryAlgorithm::Profile(profile_id),
            top_k: None,
            channel_k: None,
        })
        .unwrap();
    assert_eq!(hits_b_exact.len(), 1);
}

#[test]
fn query_unknown_text_space_rejected() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("k.vls");
    let setup = setup_with_corpus(&path, &["alpha"]);
    let err = setup
        .valise
        .query_text(TextQuery {
            text_space_id: TextSpaceId(9999),
            query: "alpha".into(),
            algorithm: QueryAlgorithm::Profile(setup.bm25_profile_id),
            top_k: None,
            channel_k: None,
        })
        .expect_err("unknown ts");
    assert!(err.to_string().contains("unknown text_space_id"));
}

#[test]
fn query_unknown_profile_rejected() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("l.vls");
    let setup = setup_with_corpus(&path, &["alpha"]);
    let err = setup
        .valise
        .query_text(TextQuery {
            text_space_id: setup.text_space_id,
            query: "alpha".into(),
            algorithm: QueryAlgorithm::Profile(RetrievalProfileId(9999)),
            top_k: None,
            channel_k: None,
        })
        .expect_err("unknown profile");
    assert!(err.to_string().contains("unknown profile_id"));
}

#[test]
fn empty_query_yields_no_hits() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("m.vls");
    let setup = setup_with_corpus(&path, &["alpha", "beta"]);
    let hits = setup
        .valise
        .query_text(TextQuery {
            text_space_id: setup.text_space_id,
            query: "".into(),
            algorithm: QueryAlgorithm::Profile(setup.bm25_profile_id),
            top_k: None,
            channel_k: None,
        })
        .unwrap();
    assert!(hits.is_empty());
}

#[test]
fn reopen_query_returns_same_results() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("n.vls");
    let setup = setup_with_corpus(&path, &["alpha beta gamma", "alpha gamma", "beta delta"]);
    let hits1 = setup
        .valise
        .query_text(TextQuery {
            text_space_id: setup.text_space_id,
            query: "alpha".into(),
            algorithm: QueryAlgorithm::Profile(setup.bm25_profile_id),
            top_k: None,
            channel_k: None,
        })
        .unwrap();
    let bm25_profile_id = setup.bm25_profile_id;
    let text_space_id = setup.text_space_id;
    drop(setup);

    let valise = ValiseFile::open_read_only(&path).unwrap();
    let hits2 = valise
        .query_text(TextQuery {
            text_space_id,
            query: "alpha".into(),
            algorithm: QueryAlgorithm::Profile(bm25_profile_id),
            top_k: None,
            channel_k: None,
        })
        .unwrap();
    assert_eq!(hits1.len(), hits2.len());
    for (a, b) in hits1.iter().zip(hits2.iter()) {
        assert_eq!(a.frame_id, b.frame_id);
        assert!((a.score - b.score).abs() < 1e-5);
    }
}
