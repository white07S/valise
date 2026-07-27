//! 32-byte fixed-size token used as the key/value type of
//! [`crate::text::stem_cache::StemCache`].
//!
//! `InlineToken` is `Copy + Eq + Hash` so the direct-mapped stem cache
//! can pass it by value without aliasing concerns. Equality and hashing
//! consider only the `bytes[..len]` prefix, so two `InlineToken`s
//! constructed with the same byte slice compare equal regardless of
//! what's in the unused tail.
//!
//! Length cap is 31 bytes. The stem cache silently bypasses tokens
//! that don't fit and falls through to the direct stem path; the
//! analyzer's main output is still heap `Vec<u8>` and is unaffected by
//! the cap.

use std::hash::{Hash, Hasher};

pub(crate) const INLINE_TOKEN_CAP: usize = 31;

#[derive(Clone, Copy)]
pub(crate) struct InlineToken {
    len: u8,
    bytes: [u8; INLINE_TOKEN_CAP],
}

impl InlineToken {
    /// Construct from a byte slice if it fits. Returns `None` for
    /// slices longer than [`INLINE_TOKEN_CAP`].
    #[inline]
    pub(crate) fn new(slice: &[u8]) -> Option<Self> {
        if slice.len() > INLINE_TOKEN_CAP {
            return None;
        }
        let mut bytes = [0u8; INLINE_TOKEN_CAP];
        bytes[..slice.len()].copy_from_slice(slice);
        Some(Self {
            len: slice.len() as u8,
            bytes,
        })
    }

    /// All-zero token — used by [`StemCache`] as the initial value for
    /// every slot. An InlineToken with `len = 0` will never compare
    /// equal to any non-empty input (which is what reaches the cache).
    #[inline]
    pub(crate) const fn empty() -> Self {
        Self {
            len: 0,
            bytes: [0u8; INLINE_TOKEN_CAP],
        }
    }

    #[inline]
    pub(crate) fn as_slice(&self) -> &[u8] {
        &self.bytes[..self.len as usize]
    }
}

impl PartialEq for InlineToken {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.len == other.len
            && self.bytes[..self.len as usize] == other.bytes[..other.len as usize]
    }
}

impl Eq for InlineToken {}

impl Hash for InlineToken {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write(self.as_slice());
    }
}

impl std::fmt::Debug for InlineToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match std::str::from_utf8(self.as_slice()) {
            Ok(s) => write!(f, "InlineToken({s:?})"),
            Err(_) => write!(f, "InlineToken({:?})", self.as_slice()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn size_and_align() {
        assert_eq!(std::mem::size_of::<InlineToken>(), 32);
        assert_eq!(std::mem::align_of::<InlineToken>(), 1);
    }

    #[test]
    fn empty_never_equals_real_token() {
        let empty = InlineToken::empty();
        let real = InlineToken::new(b"the").unwrap();
        assert_ne!(empty, real);
        assert!(empty.as_slice().is_empty());
        assert!(!real.as_slice().is_empty());
    }

    #[test]
    fn round_trip() {
        let t = InlineToken::new(b"running").unwrap();
        assert_eq!(t.as_slice(), b"running");
    }

    #[test]
    fn overlong_returns_none() {
        let s = b"abcdefghijklmnopqrstuvwxyz0123456789"; // 36 bytes
        assert!(InlineToken::new(s).is_none());
    }
}
