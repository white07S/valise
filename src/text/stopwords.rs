// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Built-in stopword sets selected by `AnalyzerDesc::stopwords`.
//!
//! See [`crate::format::catalog::StopwordsPolicy`] for the policy enum.
//! The sets are stored as `&'static [&'static str]` and resolved to a
//! `HashSet<&'static [u8]>` once per process via `OnceLock`. Lookup
//! happens after case-folding, so the entries here are all lowercase.

use std::collections::HashSet;
use std::sync::OnceLock;

use crate::format::catalog::StopwordsPolicy;

/// Lucene's `EnglishAnalyzer` stopword list — 33 entries. Anserini and
/// pyserini use this set unmodified for their BM25 baselines, which is
/// what BEIR papers cite. Source: Lucene
/// `analysis.en.EnglishAnalyzer#ENGLISH_STOP_WORDS_SET`.
const ENGLISH_LUCENE: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into", "is", "it",
    "no", "not", "of", "on", "or", "such", "that", "the", "their", "then", "there", "these",
    "they", "this", "to", "was", "will", "with",
];

/// Resolve `policy` to a set of stopword **byte slices** (already
/// lowercased). Returns `None` for [`StopwordsPolicy::None`] so callers
/// can elide the lookup branch entirely. The set is computed once per
/// policy via `OnceLock` — every analyzer instance shares the same
/// table.
pub(crate) fn resolve(policy: StopwordsPolicy) -> Option<&'static HashSet<&'static [u8]>> {
    match policy {
        StopwordsPolicy::None => None,
        StopwordsPolicy::EnglishLucene => Some(english_lucene_set()),
    }
}

fn english_lucene_set() -> &'static HashSet<&'static [u8]> {
    static SET: OnceLock<HashSet<&'static [u8]>> = OnceLock::new();
    SET.get_or_init(|| ENGLISH_LUCENE.iter().map(|s| s.as_bytes()).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn english_lucene_has_33_entries() {
        let s = english_lucene_set();
        assert_eq!(s.len(), 33);
    }

    #[test]
    fn english_lucene_contains_canonical_members() {
        let s = english_lucene_set();
        for word in ["the", "and", "of", "into", "with"] {
            assert!(s.contains(word.as_bytes()), "missing {word}");
        }
    }

    #[test]
    fn english_lucene_does_not_contain_non_stopwords() {
        let s = english_lucene_set();
        for word in ["python", "covid", "diabetes", "statin"] {
            assert!(!s.contains(word.as_bytes()), "unexpected stopword {word}");
        }
    }

    #[test]
    fn resolve_none_returns_none() {
        assert!(resolve(StopwordsPolicy::None).is_none());
    }
}
