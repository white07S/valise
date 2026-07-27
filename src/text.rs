// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Text retrieval layer (spec §10–§13).
//!
//! Phase 2 introduces:
//!
//! - analyzer primitives (`normalize`, `tokenize`) — Task #25
//! - the composable analyzer driven by [`AnalyzerDesc`] — Task #26
//! - canonical text segments (term dictionary, postings, doc stats) — Group C
//! - retrieval algorithms (BM25, TF-IDF, Jaccard) — Group E
//!
//! v1 is English-calibrated per spec §10.4.1; multilingual analyzers and
//! language-specific stemmers are v2 amendments.
//!
//! [`AnalyzerDesc`]: crate::format::catalog::AnalyzerDesc

pub(crate) mod analyzer;
pub(crate) mod inline_token;
pub(crate) mod normalize;
pub(crate) mod stem_cache;
pub(crate) mod stopwords;
pub(crate) mod tokenize;
