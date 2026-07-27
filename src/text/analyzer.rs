//! Composable analyzer driven by [`AnalyzerDesc`], spec §10.4.
//!
//! Pipeline:
//!
//! 1. Whole-text Unicode normalization (NFC or NFKC, when configured).
//! 2. Tokenization (UnicodeWords or Whitespace).
//! 3. Per-token case_fold and/or accent_fold.
//! 4. Built-in stopword filter (`StopwordsPolicy`, applied to the
//!    case-folded byte form — matches Lucene/Anserini semantics).
//! 5. Stemming (`Stemming::Porter2English`, via the `rust-stemmers`
//!    Snowball implementation).
//! 6. Character-length filter (`min_token_len` / `max_token_len`, `0 = unlimited`).
//!
//! Stopword filtering at steps 4 and stemming at step 5 are *opt-in*:
//! the legacy Valise defaults leave `StopwordsPolicy::None` /
//! `Stemming::None` and behave exactly as before. The BEIR-comparable
//! configuration switches them on for apples-to-apples runs against
//! Anserini / pyserini / bm25s.
//!
//! The pre-existing `stopword_set_ref` / `stopword_query_only` fields
//! on `AnalyzerDesc` are independent: they cover a future
//! reference-resolved stopword set carried as a Metadata blob and are
//! not wired into this analyzer.
//!
//! v1 limits at construction: `Tokenizer::CharNgram` / `TokenShingle` are
//! rejected as v2-reserved; only `UnicodeWords` and `Whitespace` are allowed.
//! `PunctuationPolicy` is recognized but ignored — UnicodeWords drops
//! punctuation inherently and Whitespace keeps it.

use std::collections::HashSet;

use rust_stemmers::{Algorithm as StemAlg, Stemmer};

use crate::{
    error::{Error, Result},
    format::catalog::{AnalyzerDesc, PunctuationPolicy, Stemming, Tokenizer, UnicodeNormalization},
    text::{
        normalize::{accent_fold, case_fold, normalize_nfkc},
        stem_cache::StemCache,
        stopwords,
        tokenize::{tokenize_ascii_alphanumeric, tokenize_whitespace},
    },
};

#[derive(Clone, Debug)]
pub(crate) struct Analyzer {
    unicode_normalization: UnicodeNormalization,
    case_fold: bool,
    accent_fold: bool,
    tokenizer: Tokenizer,
    min_token_len: u16,
    max_token_len: u16,
    stopwords: Option<&'static HashSet<&'static [u8]>>,
    /// Strip a trailing `'s` / `'S` (ASCII apostrophe) or `'s` / `'S`
    /// (curly U+2019) — Lucene's `EnglishPossessiveFilter`. Applied
    /// after fold, before stopword + stem.
    english_possessive_strip: bool,
    /// Snowball algorithm selector. We rebuild a `Stemmer` once per
    /// `analyze()` call rather than storing the constructed stemmer,
    /// because `rust_stemmers::Stemmer` is neither `Clone` nor `Debug`
    /// — and its construction is a single enum dispatch, cheap enough
    /// not to bother caching.
    stemmer_algorithm: Option<StemAlg>,
}

impl Analyzer {
    pub(crate) fn from_desc(desc: &AnalyzerDesc) -> Result<Self> {
        match desc.tokenizer {
            Tokenizer::UnicodeWords | Tokenizer::Whitespace => {}
            Tokenizer::CharNgram | Tokenizer::TokenShingle => {
                return Err(Error::Unsupported(format!(
                    "Tokenizer::{:?} is reserved for v2; v1 supports only UnicodeWords and Whitespace",
                    desc.tokenizer
                )));
            }
        }
        let stemmer_algorithm = match desc.stemming {
            Stemming::None => None,
            Stemming::Porter2English => Some(StemAlg::English),
        };
        // PunctuationPolicy is recognized but does not change v1 behavior.
        // UnicodeWords drops punctuation inherently; Whitespace preserves it.
        // Future tokenizers may consult the field.
        match desc.punctuation_policy {
            PunctuationPolicy::Drop | PunctuationPolicy::Keep => {}
        }

        Ok(Self {
            unicode_normalization: desc.unicode_normalization,
            case_fold: desc.case_fold,
            accent_fold: desc.accent_fold,
            tokenizer: desc.tokenizer,
            min_token_len: desc.min_token_len,
            max_token_len: desc.max_token_len,
            stopwords: stopwords::resolve(desc.stopwords),
            english_possessive_strip: desc.english_possessive_strip,
            stemmer_algorithm,
        })
    }

    /// Profiled variant of [`Self::analyze`]. Captures per-phase
    /// timings into the supplied `breakdown` (normalize / tokenize /
    /// per-token loop). The non-profiled `analyze` calls into this with
    /// a throwaway breakdown — the only overhead is three extra
    /// `Instant::now()` reads, which is well below the ns floor of the
    /// hot path on the BEIR pilot.
    pub(crate) fn analyze_with_breakdown(
        &self,
        text: &str,
        breakdown: &mut AnalyzeBreakdown,
    ) -> Vec<Vec<u8>> {
        use std::time::Instant;
        let t_norm = Instant::now();
        let normalized: std::borrow::Cow<'_, str> = match self.unicode_normalization {
            UnicodeNormalization::None => std::borrow::Cow::Borrowed(text),
            UnicodeNormalization::Nfc => {
                use unicode_normalization::IsNormalized;
                use unicode_normalization::UnicodeNormalization as _;
                match unicode_normalization::is_nfc_quick(text.chars()) {
                    IsNormalized::Yes => std::borrow::Cow::Borrowed(text),
                    _ => std::borrow::Cow::Owned(text.nfc().collect()),
                }
            }
            UnicodeNormalization::Nfkc => std::borrow::Cow::Owned(normalize_nfkc(text)),
        };
        breakdown.normalize_ns += t_norm.elapsed().as_nanos() as u64;

        let t_tok = Instant::now();
        let raw_tokens: Vec<&str> = match self.tokenizer {
            // Always byte-level alnum-run tokenization for the
            // `UnicodeWords` setting. Matches Tantivy's
            // `SimpleTokenizer` and bm25s's regex `\w+` byte-for-byte
            // on ASCII content; non-ASCII bytes are treated as token
            // boundaries (so they get stripped, like punctuation).
            // The previous `if normalized.is_ascii()` gate caused
            // tokenization to silently switch to UAX#29 for any
            // document containing a single non-ASCII byte, diverging
            // from the field on Touche/TREC-COVID and dropping
            // NDCG@10 by 10-17 pp. The `tokenize_unicode_words`
            // helper stays around for callers that explicitly want
            // UAX#29 word segmentation but is no longer wired into
            // this analyzer.
            Tokenizer::UnicodeWords => tokenize_ascii_alphanumeric(&normalized),
            Tokenizer::Whitespace => tokenize_whitespace(&normalized),
            Tokenizer::CharNgram | Tokenizer::TokenShingle => {
                breakdown.tokenize_ns += t_tok.elapsed().as_nanos() as u64;
                return Vec::new();
            }
        };
        breakdown.n_raw_tokens += raw_tokens.len() as u32;
        breakdown.tokenize_ns += t_tok.elapsed().as_nanos() as u64;

        let t_loop = Instant::now();
        let out = self.run_token_loop(&raw_tokens);
        breakdown.n_output_tokens += out.len() as u32;
        breakdown.token_loop_ns += t_loop.elapsed().as_nanos() as u64;
        out
    }

    pub(crate) fn analyze(&self, text: &str) -> Vec<Vec<u8>> {
        let mut throwaway = AnalyzeBreakdown::default();
        self.analyze_with_breakdown(text, &mut throwaway)
    }

    /// Opt #6 — analyze with a per-worker stem memoization cache.
    /// Same output shape as [`Self::analyze_with_breakdown`] (one
    /// owned `Vec<u8>` per emitted token), but the Porter2 stem step
    /// goes through [`StemCache`] which holds a direct-mapped LUT of
    /// post-fold byte slice → stemmed bytes. On the BEIR pilot's
    /// Zipf-distributed English vocabulary, cache hit rates measure
    /// >90% after the first ~50 documents — eliminating most stem
    /// > state-machine work and most `Cow::Owned(String)` allocations
    /// > the stemmer would otherwise produce.
    ///
    /// The cache is per-call only in signature; the caller threads
    /// the same `&mut StemCache` across all docs in a rayon worker
    /// via `map_init`. Cache state survives across calls.
    pub(crate) fn analyze_with_cache(
        &self,
        text: &str,
        cache: &mut StemCache,
        breakdown: &mut AnalyzeBreakdown,
    ) -> Vec<Vec<u8>> {
        use std::time::Instant;
        let t_norm = Instant::now();
        let normalized: std::borrow::Cow<'_, str> = match self.unicode_normalization {
            UnicodeNormalization::None => std::borrow::Cow::Borrowed(text),
            UnicodeNormalization::Nfc => {
                use unicode_normalization::IsNormalized;
                use unicode_normalization::UnicodeNormalization as _;
                match unicode_normalization::is_nfc_quick(text.chars()) {
                    IsNormalized::Yes => std::borrow::Cow::Borrowed(text),
                    _ => std::borrow::Cow::Owned(text.nfc().collect()),
                }
            }
            UnicodeNormalization::Nfkc => std::borrow::Cow::Owned(normalize_nfkc(text)),
        };
        breakdown.normalize_ns += t_norm.elapsed().as_nanos() as u64;

        let t_tok = Instant::now();
        let raw_tokens: Vec<&str> = match self.tokenizer {
            // See companion comment in `analyze_with_breakdown`.
            Tokenizer::UnicodeWords => tokenize_ascii_alphanumeric(&normalized),
            Tokenizer::Whitespace => tokenize_whitespace(&normalized),
            Tokenizer::CharNgram | Tokenizer::TokenShingle => {
                breakdown.tokenize_ns += t_tok.elapsed().as_nanos() as u64;
                return Vec::new();
            }
        };
        breakdown.n_raw_tokens += raw_tokens.len() as u32;
        breakdown.tokenize_ns += t_tok.elapsed().as_nanos() as u64;

        let t_loop = Instant::now();
        let out = self.run_token_loop_cached(&raw_tokens, cache);
        breakdown.n_output_tokens += out.len() as u32;
        breakdown.token_loop_ns += t_loop.elapsed().as_nanos() as u64;
        out
    }

    /// Per-token loop variant that consults `cache` for the stem step.
    /// Otherwise identical to [`Self::run_token_loop`].
    #[inline]
    fn run_token_loop_cached(&self, raw_tokens: &[&str], cache: &mut StemCache) -> Vec<Vec<u8>> {
        // Construct a stemmer once per document for the bypass path
        // (cache uses the same stemmer instance; cheap enum dispatch).
        let stemmer = self.stemmer_algorithm.map(Stemmer::create);
        let min = self.min_token_len as usize;
        let max = self.max_token_len as usize;
        let length_filter_enabled = min != 0 || max != 0;

        let mut out: Vec<Vec<u8>> = Vec::with_capacity(raw_tokens.len());
        // Scratch buffer reused across tokens in this doc. The cache
        // appends into it on each call; we move it out into `out`
        // after each token.
        let mut stem_buf: Vec<u8> = Vec::with_capacity(32);

        for token in raw_tokens {
            let bytes = token.as_bytes();
            let ascii_lowercase = !bytes.is_empty()
                && bytes
                    .iter()
                    .all(|&b| b.is_ascii() && !b.is_ascii_uppercase());

            let slow: Vec<u8>;
            let folded_bytes_raw: &[u8] = if ascii_lowercase {
                bytes
            } else {
                let mut t = if self.case_fold {
                    case_fold(token)
                } else {
                    (*token).to_owned()
                };
                if self.accent_fold {
                    t = accent_fold(&t);
                }
                slow = t.into_bytes();
                slow.as_slice()
            };

            let folded_bytes: &[u8] = if self.english_possessive_strip {
                strip_english_possessive(folded_bytes_raw)
            } else {
                folded_bytes_raw
            };
            if folded_bytes.is_empty() {
                continue;
            }

            if let Some(set) = self.stopwords
                && set.contains(folded_bytes)
            {
                continue;
            }

            // Stem via cache. `stem_into` clears `stem_buf` and writes
            // the stemmed bytes; on a hit there's no allocator traffic
            // beyond the buffer extension (amortized via reuse).
            cache.stem_into(folded_bytes, stemmer.as_ref(), &mut stem_buf);

            if length_filter_enabled {
                let char_len = if stem_buf.is_ascii() {
                    stem_buf.len()
                } else {
                    std::str::from_utf8(&stem_buf)
                        .map(|s| s.chars().count())
                        .unwrap_or(stem_buf.len())
                };
                if min != 0 && char_len < min {
                    continue;
                }
                if max != 0 && char_len > max {
                    continue;
                }
            }
            // Move `stem_buf` into the output, replacing with an
            // empty buffer so the next iteration writes from len 0.
            // This avoids an `as_slice().to_vec()` copy on the
            // cache-hit path.
            let owned = std::mem::take(&mut stem_buf);
            stem_buf.reserve(32);
            out.push(owned);
        }
        out
    }

    /// Per-token loop. Inlined into both `analyze` paths via a thin
    /// dispatch so we don't pay double-instrumentation overhead on the
    /// production path.
    ///
    /// Optimization notes (Opt #1 from the BEIR profile work):
    /// - **Stemmer hoisted**: `Stemmer::create(alg)` is invoked exactly
    ///   once per `analyze()` call rather than once per token. The
    ///   Snowball state machine is reused across the whole document.
    /// - **Single-allocation stem path**: when the stemmer returns
    ///   `Cow::Owned`, the produced `String` is converted directly to
    ///   `Vec<u8>` via `into_bytes()`, skipping the previous
    ///   `into_owned() + to_vec()` double copy. When `Cow::Borrowed`
    ///   (the token was already in stemmed form), one `to_vec` is
    ///   unavoidable to satisfy the owned-output contract.
    /// - **Skip length-filter for unbounded config**: when both bounds
    ///   are 0 (the BEIR config), the `chars().count()` walk and the
    ///   comparison branches are skipped entirely.
    #[inline]
    fn run_token_loop(&self, raw_tokens: &[&str]) -> Vec<Vec<u8>> {
        let stemmer = self.stemmer_algorithm.map(Stemmer::create);
        let min = self.min_token_len as usize;
        let max = self.max_token_len as usize;
        let length_filter_enabled = min != 0 || max != 0;

        let mut out: Vec<Vec<u8>> = Vec::with_capacity(raw_tokens.len());
        for token in raw_tokens {
            let bytes = token.as_bytes();
            let ascii_lowercase = !bytes.is_empty()
                && bytes
                    .iter()
                    .all(|&b| b.is_ascii() && !b.is_ascii_uppercase());

            let slow: Vec<u8>;
            let folded_bytes_raw: &[u8] = if ascii_lowercase {
                bytes
            } else {
                let mut t = if self.case_fold {
                    case_fold(token)
                } else {
                    (*token).to_owned()
                };
                if self.accent_fold {
                    t = accent_fold(&t);
                }
                slow = t.into_bytes();
                slow.as_slice()
            };

            // English possessive strip (Lucene `EnglishPossessiveFilter`).
            // Matches a trailing `'s` / `'S` with either ASCII apostrophe
            // (U+0027) or curly right-single-quote (U+2019, UTF-8 bytes
            // `0xE2 0x80 0x99`). Slice-narrows in place — no allocation.
            let folded_bytes: &[u8] = if self.english_possessive_strip {
                strip_english_possessive(folded_bytes_raw)
            } else {
                folded_bytes_raw
            };
            if folded_bytes.is_empty() {
                continue;
            }

            // Stopword filter (after fold so the lookup key matches the
            // post-fold byte form, which is what we'd be writing to the
            // term dict).
            if let Some(set) = self.stopwords
                && set.contains(folded_bytes)
            {
                continue;
            }

            // Stemmer. The shared `Stemmer` is reused across the entire
            // document (Opt #1). When it returns `Cow::Owned`, we move
            // the produced String directly into `Vec<u8>` — no
            // intermediate copies.
            let final_owned: Vec<u8> = match (stemmer.as_ref(), std::str::from_utf8(folded_bytes)) {
                (Some(stem), Ok(s)) => match stem.stem(s) {
                    std::borrow::Cow::Owned(string) => string.into_bytes(),
                    std::borrow::Cow::Borrowed(_) => folded_bytes.to_vec(),
                },
                _ => folded_bytes.to_vec(),
            };

            // Length filter on the *final* bytes (post-stem), measured
            // in characters. ASCII fast-path: byte_len == char_count.
            // Skipped entirely when both bounds are zero (BEIR config).
            if length_filter_enabled {
                let char_len = if final_owned.is_ascii() {
                    final_owned.len()
                } else {
                    std::str::from_utf8(&final_owned)
                        .map(|s| s.chars().count())
                        .unwrap_or(final_owned.len())
                };
                if min != 0 && char_len < min {
                    continue;
                }
                if max != 0 && char_len > max {
                    continue;
                }
            }
            out.push(final_owned);
        }
        out
    }
}

/// Per-phase breakdown for [`Analyzer::analyze_with_breakdown`]. All
/// times in nanoseconds; counters are exact. Accumulator-shaped so the
/// caller can sum across many calls.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct AnalyzeBreakdown {
    pub normalize_ns: u64,
    pub tokenize_ns: u64,
    pub token_loop_ns: u64,
    pub n_raw_tokens: u32,
    pub n_output_tokens: u32,
}

/// Strip a trailing `'s` or `'S` from `s`. Matches both ASCII
/// apostrophe (U+0027) and curly right-single-quote (U+2019, UTF-8
/// bytes `0xE2 0x80 0x99`). Returns the unchanged slice when no
/// possessive suffix is present.
fn strip_english_possessive(s: &[u8]) -> &[u8] {
    let n = s.len();
    if n < 2 {
        return s;
    }
    let last = s[n - 1];
    if last != b's' && last != b'S' {
        return s;
    }
    if s[n - 2] == b'\'' {
        return &s[..n - 2];
    }
    if n >= 4 && s[n - 4..n - 1] == [0xE2, 0x80, 0x99] {
        return &s[..n - 4];
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::catalog::StopwordsPolicy;

    fn lucene_like_desc() -> AnalyzerDesc {
        AnalyzerDesc {
            analyzer_id: crate::format::AnalyzerId(1),
            unicode_normalization: UnicodeNormalization::Nfc,
            case_fold: true,
            accent_fold: true,
            tokenizer: Tokenizer::UnicodeWords,
            stemming: Stemming::None,
            stopword_set_ref: None,
            stopword_query_only: true,
            stopwords: StopwordsPolicy::None,
            english_possessive_strip: false,
            min_token_len: 2,
            max_token_len: 64,
            shingle_size: 0,
            ngram_min: None,
            ngram_max: None,
            punctuation_policy: PunctuationPolicy::Drop,
        }
    }

    fn beir_english_desc() -> AnalyzerDesc {
        AnalyzerDesc {
            stemming: Stemming::Porter2English,
            stopwords: StopwordsPolicy::EnglishLucene,
            english_possessive_strip: true,
            min_token_len: 0,
            max_token_len: 0,
            ..lucene_like_desc()
        }
    }

    fn whitespace_desc() -> AnalyzerDesc {
        AnalyzerDesc {
            tokenizer: Tokenizer::Whitespace,
            unicode_normalization: UnicodeNormalization::None,
            case_fold: false,
            accent_fold: false,
            min_token_len: 0,
            max_token_len: 0,
            ..lucene_like_desc()
        }
    }

    fn bytes(s: &str) -> Vec<u8> {
        s.as_bytes().to_vec()
    }

    #[test]
    fn lucene_like_config_pipeline() {
        let analyzer = Analyzer::from_desc(&lucene_like_desc()).unwrap();
        // "The CAFÉ has 1 piece of résumé—naïve!" → after NFC+UnicodeWords
        // tokenization, case_fold, accent_fold, drop tokens of length < 2.
        let tokens = analyzer.analyze("The CAFÉ has 1 piece of résumé—naïve!");
        assert_eq!(
            tokens,
            vec![
                bytes("the"),
                bytes("cafe"),
                bytes("has"),
                bytes("piece"),
                bytes("of"),
                bytes("resume"),
                bytes("naive"),
            ]
        );
    }

    #[test]
    fn whitespace_keeps_punctuation_and_case() {
        let analyzer = Analyzer::from_desc(&whitespace_desc()).unwrap();
        let tokens = analyzer.analyze("Hello, world! It's 2026.");
        assert_eq!(
            tokens,
            vec![
                bytes("Hello,"),
                bytes("world!"),
                bytes("It's"),
                bytes("2026."),
            ]
        );
    }

    #[test]
    fn case_fold_only_preserves_accents() {
        let mut desc = lucene_like_desc();
        desc.accent_fold = false;
        let analyzer = Analyzer::from_desc(&desc).unwrap();
        let tokens = analyzer.analyze("CAFÉ résumé");
        assert_eq!(tokens, vec![bytes("café"), bytes("résumé")]);
    }

    #[test]
    fn accent_fold_only_preserves_case() {
        let mut desc = lucene_like_desc();
        desc.case_fold = false;
        let analyzer = Analyzer::from_desc(&desc).unwrap();
        let tokens = analyzer.analyze("CAFÉ résumé");
        assert_eq!(tokens, vec![bytes("CAFE"), bytes("resume")]);
    }

    #[test]
    fn length_filter_drops_short_and_long_tokens() {
        let mut desc = lucene_like_desc();
        desc.min_token_len = 4;
        desc.max_token_len = 6;
        let analyzer = Analyzer::from_desc(&desc).unwrap();
        let tokens = analyzer.analyze("a is the longest abc abcd");
        // After case_fold: ["a", "is", "the", "longest", "abc", "abcd"]
        // length filter [4, 6] keeps "abcd" only ("longest" is 7 chars).
        assert_eq!(tokens, vec![bytes("abcd")]);
    }

    #[test]
    fn empty_input_yields_empty_output() {
        let analyzer = Analyzer::from_desc(&lucene_like_desc()).unwrap();
        assert!(analyzer.analyze("").is_empty());
    }

    #[test]
    fn whitespace_only_input_yields_empty_output() {
        let analyzer = Analyzer::from_desc(&lucene_like_desc()).unwrap();
        assert!(analyzer.analyze("   \t\n  ").is_empty());
    }

    #[test]
    fn nfkc_normalizes_compatibility_forms() {
        let mut desc = lucene_like_desc();
        desc.unicode_normalization = UnicodeNormalization::Nfkc;
        let analyzer = Analyzer::from_desc(&desc).unwrap();
        // "ﬁnal x²" → NFKC: "final x2" → tokenize → ["final", "x2"]
        // (after case_fold both already lowercase, accent_fold no-op).
        let tokens = analyzer.analyze("ﬁnal x²");
        assert_eq!(tokens, vec![bytes("final"), bytes("x2")]);
    }

    #[test]
    fn rejects_char_ngram_tokenizer_as_v2_reserved() {
        let mut desc = lucene_like_desc();
        desc.tokenizer = Tokenizer::CharNgram;
        let err = Analyzer::from_desc(&desc).expect_err("CharNgram is v2 reserved");
        assert!(err.to_string().contains("CharNgram"));
    }

    #[test]
    fn rejects_token_shingle_tokenizer_as_v2_reserved() {
        let mut desc = lucene_like_desc();
        desc.tokenizer = Tokenizer::TokenShingle;
        let err = Analyzer::from_desc(&desc).expect_err("TokenShingle is v2 reserved");
        assert!(err.to_string().contains("TokenShingle"));
    }

    #[test]
    fn analyzer_is_cloneable_for_thread_sharing() {
        let analyzer = Analyzer::from_desc(&lucene_like_desc()).unwrap();
        let cloned = analyzer.clone();
        assert_eq!(
            analyzer.analyze("Hello world"),
            cloned.analyze("Hello world")
        );
    }

    // ---- BEIR-pipeline (Porter2 + Lucene English stopwords) tests -------

    #[test]
    fn beir_pipeline_drops_stopwords_and_stems() {
        let analyzer = Analyzer::from_desc(&beir_english_desc()).unwrap();
        // "The runner is running runs into the cafe" →
        //   lowercase: ["the", "runner", "is", "running", "runs", "into", "the", "cafe"]
        //   stopwords drop: ["runner", "running", "runs", "cafe"]
        //   stem (Porter2): ["runner", "run", "run", "cafe"]
        // ("runner" stems to "runner", not "run" — Porter2 keeps the
        // agent noun. Confirmed against Snowball test vectors.)
        let tokens = analyzer.analyze("The runner is running runs into the cafe");
        assert_eq!(
            tokens,
            vec![bytes("runner"), bytes("run"), bytes("run"), bytes("cafe"),]
        );
    }

    #[test]
    fn beir_pipeline_porter2_canonical_vectors() {
        // Direct check against well-known Porter2 stems. Any drift here
        // means rust-stemmers shipped a behavior change or we wired the
        // wrong algorithm.
        let analyzer = Analyzer::from_desc(&beir_english_desc()).unwrap();
        let tokens = analyzer.analyze("studies studying generously generation argument arguments");
        // Expected stems (Porter2):
        //   studies     -> studi
        //   studying    -> studi
        //   generously  -> generous
        //   generation  -> generat
        //   argument    -> argument
        //   arguments   -> argument
        assert_eq!(
            tokens,
            vec![
                bytes("studi"),
                bytes("studi"),
                bytes("generous"),
                bytes("generat"),
                bytes("argument"),
                bytes("argument"),
            ]
        );
    }

    #[test]
    fn beir_pipeline_keeps_accents_after_fold_when_present() {
        // BEIR analyzer has accent_fold = true (inherited from
        // lucene_like_desc). "café" → "cafe" → stem -> "cafe".
        let analyzer = Analyzer::from_desc(&beir_english_desc()).unwrap();
        assert_eq!(analyzer.analyze("CAFÉ"), vec![bytes("cafe")]);
    }

    #[test]
    fn beir_pipeline_stopwords_only_no_stemming() {
        let desc = AnalyzerDesc {
            stemming: Stemming::None,
            stopwords: StopwordsPolicy::EnglishLucene,
            min_token_len: 0,
            max_token_len: 0,
            ..lucene_like_desc()
        };
        let analyzer = Analyzer::from_desc(&desc).unwrap();
        // Stopwords dropped, no stemming → "running" stays as-is.
        assert_eq!(
            analyzer.analyze("The cats are running"),
            vec![bytes("cats"), bytes("running")]
        );
    }

    #[test]
    fn beir_pipeline_ascii_apostrophe_splits_at_tokenize() {
        let analyzer = Analyzer::from_desc(&beir_english_desc()).unwrap();
        // Opt #3 path: "John's house" is pure-ASCII → the fast-path
        // tokenizer splits on the apostrophe (non-alnum byte) →
        //   ["John", "s", "house"]
        //   case-fold      → ["john", "s", "house"]
        //   possessive-strip is a no-op (no `'s` suffix on any token)
        //   stopwords      (no matches in the 33-word Lucene set)
        //   Porter2 stem   → ["john", "s", "hous"]
        // The standalone "s" survives because it's not a Lucene
        // stopword. This matches `bm25s`'s `\w+` tokenizer behavior and
        // diverges from Lucene's StandardTokenizer (which keeps
        // "John's" intact and lets EnglishPossessiveFilter strip the
        // "'s" → "john"). Documented sub-1% NDCG@10 impact on BEIR.
        let tokens = analyzer.analyze("John's house");
        assert_eq!(tokens, vec![bytes("john"), bytes("s"), bytes("hous")]);
    }

    #[test]
    fn beir_pipeline_curly_apostrophe_splits_at_tokenize() {
        let analyzer = Analyzer::from_desc(&beir_english_desc()).unwrap();
        // U+2019 RIGHT SINGLE QUOTATION MARK is a Unicode punctuation
        // codepoint (general category Pf). The Tantivy-parity
        // tokenizer splits on every non-`is_alphanumeric` codepoint,
        // ASCII or otherwise. So `John\u{2019}s` → ["John", "s"]
        // before possessive-strip ever runs. Same end result as
        // `John's` (ASCII apostrophe) — both engines (Valise and
        // Tantivy) treat the suffix `s` as its own token.
        let tokens = analyzer.analyze("John\u{2019}s house");
        assert_eq!(tokens, vec![bytes("john"), bytes("s"), bytes("hous")]);
    }

    #[test]
    fn beir_pipeline_strip_is_noop_when_no_possessive_suffix() {
        let analyzer = Analyzer::from_desc(&beir_english_desc()).unwrap();
        // UnicodeWords trims the leading apostrophe so `"'s alone"` →
        // ["s", "alone"]. Possessive-strip then finds no trailing `'s`
        // on either token and is a no-op. `s` survives as a single-char
        // token (it is not in Lucene's 33-word stopword set, so it is
        // not dropped) and `alone` stems to `alon`.
        let tokens = analyzer.analyze("'s alone");
        assert_eq!(tokens, vec![bytes("s"), bytes("alon")]);
    }

    #[test]
    fn beir_pipeline_stemming_only_no_stopwords() {
        let desc = AnalyzerDesc {
            stemming: Stemming::Porter2English,
            stopwords: StopwordsPolicy::None,
            min_token_len: 0,
            max_token_len: 0,
            ..lucene_like_desc()
        };
        let analyzer = Analyzer::from_desc(&desc).unwrap();
        // Stems applied, stopwords kept → "the" survives.
        assert_eq!(
            analyzer.analyze("the cats are running"),
            vec![bytes("the"), bytes("cat"), bytes("are"), bytes("run")]
        );
    }
}
