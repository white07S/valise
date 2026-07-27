//! Unicode normalization primitives, spec §10.4.
//!
//! All functions are pure `&str → String`. v1 is English-calibrated
//! (§10.4.1); `case_fold` is `to_lowercase` rather than full Unicode
//! CaseFolding — sufficient for English and reasonable for most Latin
//! scripts. v2 may upgrade to `unicode-case-mapping` for full CaseFolding.

use unicode_normalization::UnicodeNormalization;
use unicode_normalization::char::is_combining_mark;

#[cfg(test)]
pub(crate) fn normalize_nfc(text: &str) -> String {
    text.nfc().collect()
}

pub(crate) fn normalize_nfkc(text: &str) -> String {
    text.nfkc().collect()
}

/// Unicode-aware lowercase. Note: `to_lowercase`, not full Unicode
/// CaseFolding (which would handle e.g. `ß → ss`). Sufficient for v1
/// English scope (§10.4.1).
pub(crate) fn case_fold(text: &str) -> String {
    text.chars().flat_map(|c| c.to_lowercase()).collect()
}

/// Strip combining marks (e.g., `café → cafe`). Decomposes to NFD, drops
/// combining marks, recomposes via NFC. Suitable for ASCII-folding Latin
/// text; non-Latin scripts that rely on combining marks for meaning may
/// degrade — this is the v1 "accent fold" defined in spec §10.4.
pub(crate) fn accent_fold(text: &str) -> String {
    text.nfd()
        .filter(|c| !is_combining_mark(*c))
        .nfc()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nfc_recomposes_decomposed_acute() {
        // 'e' + combining acute U+0301 → precomposed é U+00E9
        let decomposed = "cafe\u{0301}";
        assert_eq!(normalize_nfc(decomposed), "café");
    }

    #[test]
    fn nfc_passes_already_composed() {
        assert_eq!(normalize_nfc("café"), "café");
    }

    #[test]
    fn nfkc_compatibility_decomposes_superscript() {
        // Superscript 2 (U+00B2) compatibility-decomposes to ASCII '2'.
        assert_eq!(normalize_nfkc("x²"), "x2");
    }

    #[test]
    fn nfkc_compatibility_decomposes_ligature() {
        // Latin small ligature 'ﬁ' (U+FB01) → 'fi'.
        assert_eq!(normalize_nfkc("ﬁnal"), "final");
    }

    #[test]
    fn case_fold_lowercases_ascii_and_latin() {
        assert_eq!(case_fold("Hello WORLD"), "hello world");
        assert_eq!(case_fold("Café"), "café");
    }

    #[test]
    fn case_fold_handles_greek() {
        // Greek capital omega (U+03A9) → lowercase omega (U+03C9).
        assert_eq!(case_fold("Ω"), "ω");
    }

    #[test]
    fn accent_fold_strips_latin_diacritics() {
        assert_eq!(accent_fold("café"), "cafe");
        assert_eq!(accent_fold("naïve"), "naive");
        assert_eq!(accent_fold("résumé"), "resume");
    }

    #[test]
    fn accent_fold_preserves_unaccented_text() {
        assert_eq!(accent_fold("hello world"), "hello world");
    }

    #[test]
    fn accent_fold_handles_multiple_combining_marks() {
        // 'a' + acute + diaeresis should drop both marks.
        let multi = "a\u{0301}\u{0308}";
        assert_eq!(accent_fold(multi), "a");
    }

    #[test]
    fn accent_fold_handles_decomposed_input() {
        let decomposed = "cafe\u{0301}";
        assert_eq!(accent_fold(decomposed), "cafe");
    }
}
