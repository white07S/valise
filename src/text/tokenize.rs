// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Tokenizer and filter primitives, spec §10.4.

use std::collections::HashSet;

/// Unicode-alphanumeric-run tokenizer. Emits maximal runs of
/// `char::is_alphanumeric()` codepoints from `text`, splitting on
/// any non-alphanumeric Unicode codepoint (ASCII or otherwise).
///
/// **This is the production tokenizer for the BEIR-comparable
/// pipeline**: matches Tantivy's `SimpleTokenizer` exactly. Earlier
/// designs (a) gated on `text.is_ascii()` and (b) used byte-level
/// boundary detection — both diverged from Tantivy on real corpora
/// (Touche-2020, TREC-COVID) and dropped NDCG@10 by 10–17 pp. This
/// version iterates by codepoint via `char_indices`, splits on
/// anything that isn't `char::is_alphanumeric` (so em-dashes,
/// punctuation, whitespace, dashes, periods, apostrophes all split;
/// letters with diacritics — `é`, `ï`, `ü` — and CJK ideographs flow
/// through inside tokens).
///
/// Examples:
/// - `"U.S.A"` → `["U", "S", "A"]`
/// - `"don't"` → `["don", "t"]`
/// - `"café résumé—naïve"` → `["café", "résumé", "naïve"]` (em-dash splits)
/// - `"中文 test"` → `["中文", "test"]`
///
/// The downstream `accent_fold` step (when enabled) strips combining
/// marks from these tokens, so `["café"]` → `["cafe"]` after the
/// per-token fold pipeline.
pub(crate) fn tokenize_ascii_alphanumeric(text: &str) -> Vec<&str> {
    let mut out: Vec<&str> = Vec::with_capacity(text.len() / 6 + 1);
    let mut start: Option<usize> = None;
    for (i, c) in text.char_indices() {
        if c.is_alphanumeric() {
            if start.is_none() {
                start = Some(i);
            }
        } else if let Some(s) = start.take() {
            out.push(&text[s..i]);
        }
    }
    if let Some(s) = start {
        out.push(&text[s..]);
    }
    out
}

/// Whitespace-separated tokenization. Splits on any Unicode whitespace; keeps
/// adjacent punctuation attached to the surrounding token (unlike
/// `tokenize_unicode_words`).
pub(crate) fn tokenize_whitespace(text: &str) -> Vec<&str> {
    text.split_whitespace().collect()
}

/// Drop tokens whose **character count** falls outside `[min, max]`. A bound
/// of `0` means unlimited (per `AnalyzerDesc.min_token_len` /
/// `AnalyzerDesc.max_token_len` semantics, spec §10.4). Character count is
/// `chars().count()` (Unicode scalar values), not graphemes — multi-codepoint
/// emoji count by their constituent USVs. v1 simplification.
#[cfg(test)]
pub(crate) fn filter_token_length(tokens: Vec<&str>, min: u16, max: u16) -> Vec<&str> {
    tokens
        .into_iter()
        .filter(|token| {
            let len = token.chars().count();
            let too_short = min > 0 && len < min as usize;
            let too_long = max > 0 && len > max as usize;
            !too_short && !too_long
        })
        .collect()
}

/// Drop tokens whose UTF-8 bytes appear in `stopwords`. The set is borrowed;
/// `Vec<u8>: Borrow<[u8]>` lets `contains` look up `&[u8]` without allocating.
///
/// v1 indexing/query pipelines do not yet wire this in (the metadata-ref
/// resolver for `AnalyzerDesc.stopword_set_ref` is v2). The primitive ships
/// now so callers with their own stopword sets can apply it directly.
#[allow(dead_code)]
pub(crate) fn filter_stopwords<'a>(
    tokens: Vec<&'a str>,
    stopwords: &'_ HashSet<Vec<u8>>,
) -> Vec<&'a str> {
    tokens
        .into_iter()
        .filter(|token| !stopwords.contains(token.as_bytes()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn whitespace_splits_on_any_whitespace() {
        let words = tokenize_whitespace("  one\ttwo\nthree  ");
        assert_eq!(words, vec!["one", "two", "three"]);
    }

    #[test]
    fn whitespace_preserves_punctuation_in_tokens() {
        let words = tokenize_whitespace("hello, world!");
        assert_eq!(words, vec!["hello,", "world!"]);
    }

    #[test]
    fn length_filter_min_only() {
        let tokens = vec!["a", "ab", "abc", "abcd"];
        assert_eq!(filter_token_length(tokens, 2, 0), vec!["ab", "abc", "abcd"]);
    }

    #[test]
    fn length_filter_max_only() {
        let tokens = vec!["a", "ab", "abc", "abcd"];
        assert_eq!(filter_token_length(tokens, 0, 3), vec!["a", "ab", "abc"]);
    }

    #[test]
    fn length_filter_both_bounds() {
        let tokens = vec!["a", "ab", "abc", "abcd", "abcde"];
        assert_eq!(filter_token_length(tokens, 2, 4), vec!["ab", "abc", "abcd"]);
    }

    #[test]
    fn length_filter_zero_zero_passes_all() {
        let tokens = vec!["a", "ab", "abc"];
        assert_eq!(filter_token_length(tokens, 0, 0), vec!["a", "ab", "abc"]);
    }

    #[test]
    fn length_filter_uses_char_count_not_byte_count() {
        // "café" is 4 chars but 5 UTF-8 bytes.
        let tokens = vec!["café"];
        assert_eq!(filter_token_length(tokens.clone(), 0, 4), tokens);
        assert_eq!(filter_token_length(tokens, 5, 0), Vec::<&str>::new());
    }

    fn stopword_set(words: &[&str]) -> HashSet<Vec<u8>> {
        words.iter().map(|w| w.as_bytes().to_vec()).collect()
    }

    #[test]
    fn stopword_filter_drops_matches() {
        let stopwords = stopword_set(&["the", "a", "of"]);
        let tokens = vec!["the", "quick", "fox", "of", "logic"];
        assert_eq!(
            filter_stopwords(tokens, &stopwords),
            vec!["quick", "fox", "logic"]
        );
    }

    #[test]
    fn stopword_filter_empty_set_passes_all() {
        let stopwords = HashSet::new();
        let tokens = vec!["the", "quick", "fox"];
        assert_eq!(
            filter_stopwords(tokens.clone(), &stopwords),
            vec!["the", "quick", "fox"]
        );
    }

    #[test]
    fn stopword_filter_is_case_sensitive() {
        // The caller is responsible for case-folding before stopword filter;
        // this primitive does byte-exact match.
        let stopwords = stopword_set(&["the"]);
        let tokens = vec!["The", "the"];
        assert_eq!(filter_stopwords(tokens, &stopwords), vec!["The"]);
    }
}
