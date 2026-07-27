// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Phase 1: `TextMode::Disabled` runtime gating (consolidation plan §12).
//!
//! A file created with text disabled MUST refuse every text register/query
//! entry point with `Error::Unsupported`, and MUST mention the offending
//! operation name in the error so a misconfigured caller can tell what to
//! change. The vector-only paths stay open.

use valise::{
    AnalyzerDesc, AnalyzerId, CreateOptions, FieldDesc, FieldSchemaDesc, FieldSchemaId,
    FieldSource, IdfVariant, PunctuationPolicy, QueryAlgorithm, RetrievalProfileDesc,
    RetrievalProfileId, RetrievalProfileParams, RetrievalProfileType, Stemming, StopwordsPolicy,
    TextMode, TextQuery, TextSpaceDesc, TextSpaceId, Tokenizer, UnicodeNormalization, ValiseFile,
};

fn tmpfile(name: &str) -> std::path::PathBuf {
    let dir = tempfile::tempdir().expect("tempdir").keep();
    dir.join(name)
}

fn make_disabled_file() -> ValiseFile {
    let path = tmpfile("text_disabled.vls");
    let opts = CreateOptions {
        text: TextMode::Disabled,
        ..Default::default()
    };
    ValiseFile::create_with_options(&path, opts).expect("create with text disabled")
}

fn lucene_analyzer() -> AnalyzerDesc {
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

fn search_field_schema() -> FieldSchemaDesc {
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

#[test]
fn register_analyzer_rejected() {
    let mut file = make_disabled_file();
    let err = file
        .register_analyzer(lucene_analyzer())
        .expect_err("must reject");
    let s = err.to_string();
    assert!(s.contains("register_analyzer"), "got: {s}");
    assert!(s.contains("Disabled"), "got: {s}");
}

#[test]
fn register_field_schema_rejected() {
    let mut file = make_disabled_file();
    let err = file
        .register_field_schema(search_field_schema())
        .expect_err("must reject");
    assert!(err.to_string().contains("register_field_schema"));
}

#[test]
fn register_retrieval_profile_rejected() {
    let mut file = make_disabled_file();
    let err = file
        .register_retrieval_profile(bm25_profile())
        .expect_err("must reject");
    assert!(err.to_string().contains("register_retrieval_profile"));
}

#[test]
fn register_text_space_rejected() {
    let mut file = make_disabled_file();
    let desc = TextSpaceDesc {
        text_space_id: TextSpaceId(0),
        name: "ts".into(),
        analyzer_id: AnalyzerId(0),
        field_schema_id: FieldSchemaId(0),
        default_profile_id: RetrievalProfileId(0),
        flags: 0,
        enabled_retrievers: 0,
    };
    let err = file.register_text_space(desc).expect_err("must reject");
    assert!(err.to_string().contains("register_text_space"));
}

#[test]
fn query_text_rejected() {
    let file = make_disabled_file();
    let err = file
        .query_text(TextQuery {
            text_space_id: TextSpaceId(0),
            query: "hello".into(),
            algorithm: QueryAlgorithm::CountCosine,
            top_k: None,
            channel_k: None,
        })
        .expect_err("must reject");
    assert!(err.to_string().contains("query_text"));
}

#[test]
fn enabled_default_passes_text_gate() {
    // Sanity check: with TextMode::Enabled (the default), the gate
    // doesn't fire — the call fails for a different reason. We only
    // require the message NOT to mention the contract violation.
    let path = tmpfile("text_enabled.vls");
    let file = ValiseFile::create(&path).expect("create");
    let err = file
        .query_text(TextQuery {
            text_space_id: TextSpaceId(0),
            query: "hello".into(),
            algorithm: QueryAlgorithm::CountCosine,
            top_k: None,
            channel_k: None,
        })
        .expect_err("query against unknown text space still errors");
    assert!(
        !err.to_string().contains("Disabled"),
        "default TextMode::Enabled should not surface Disabled error: got {err}"
    );
}

#[test]
fn introspection_reports_disabled() {
    let file = make_disabled_file();
    assert!(!file.text_enabled());
    assert!(!file.create_contract().text_enabled);
}
