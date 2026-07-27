//! Vector input dtypes and dtype sets.
//!
//! Phase 1 of the consolidation plan (see `docs/VALISE_CONSOLIDATION_PLAN.md` §5).
//! `Dtype` enumerates the five supported input precisions; `DtypeSet` is a
//! transparent `u8` bitmask used by the file-level create-time contract
//! (`CreateContractV1`) to whitelist which dtypes embedding spaces in a
//! given file may declare.
//!
//! The enum is `#[repr(u8)]` with dense, gap-free discriminants 1..=5 so
//! `match` lowers to a small immediate. The bitmask is hand-rolled rather
//! than pulled in via the `bitflags` crate — for one type the saved
//! dependency is worth the line count.

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Deserialize, Serialize)]
#[repr(u8)]
pub enum Dtype {
    F32 = 1,
    F16 = 2,
    BF16 = 3,
    F8E4M3 = 4,
    F8E5M2 = 5,
}

impl Dtype {
    /// Bytes occupied by one element in the storage layout. f8 spaces
    /// store 1 byte/elem; f16/bf16 store 2; f32 stores 4. This is the
    /// raw-layout width, not the post-codec width — the QAM codec has its
    /// own bytes-per-vector accounting.
    #[inline]
    #[must_use]
    pub const fn byte_width(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::F8E4M3 | Self::F8E5M2 => 1,
        }
    }

    #[inline]
    #[must_use]
    pub const fn is_f8(self) -> bool {
        matches!(self, Self::F8E4M3 | Self::F8E5M2)
    }

    #[inline]
    #[must_use]
    pub const fn as_set_bit(self) -> DtypeSet {
        match self {
            Self::F32 => DtypeSet::F32,
            Self::F16 => DtypeSet::F16,
            Self::BF16 => DtypeSet::BF16,
            Self::F8E4M3 => DtypeSet::F8E4M3,
            Self::F8E5M2 => DtypeSet::F8E5M2,
        }
    }
}

/// Bitmask over `Dtype` values.
///
/// Layout: 5 bits, dense; the upper 3 bits are reserved and MUST be zero
/// in any persisted form. Decoders use [`DtypeSet::from_bits`] to reject
/// unknown high bits or [`DtypeSet::from_bits_truncate`] to silently
/// ignore them, depending on the policy the caller wants.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Deserialize, Serialize)]
#[repr(transparent)]
pub struct DtypeSet(u8);

impl DtypeSet {
    pub const F32: Self = Self(0b0000_0001);
    pub const F16: Self = Self(0b0000_0010);
    pub const BF16: Self = Self(0b0000_0100);
    pub const F8E4M3: Self = Self(0b0000_1000);
    pub const F8E5M2: Self = Self(0b0001_0000);

    /// All five supported dtypes. Default value of `VectorContract.allowed_dtypes`.
    pub const ALL: Self =
        Self(Self::F32.0 | Self::F16.0 | Self::BF16.0 | Self::F8E4M3.0 | Self::F8E5M2.0);

    /// Empty set. Validation rejects empty contracts — every file must
    /// allow at least one dtype.
    pub const EMPTY: Self = Self(0);

    const VALID_MASK: u8 = Self::ALL.0;

    #[inline]
    #[must_use]
    pub const fn empty() -> Self {
        Self::EMPTY
    }

    #[inline]
    #[must_use]
    pub const fn bits(self) -> u8 {
        self.0
    }

    /// Drop any bits outside the valid-dtype mask. Use this on inputs
    /// that may carry forward-compat unknown bits a future minor version
    /// added.
    #[inline]
    #[must_use]
    pub const fn from_bits_truncate(bits: u8) -> Self {
        Self(bits & Self::VALID_MASK)
    }

    /// Strict version: returns `None` if any bit outside the valid mask
    /// is set. Persisted records use this so unexpected bits surface as
    /// `Error::Format` instead of being silently ignored.
    #[inline]
    #[must_use]
    pub const fn from_bits(bits: u8) -> Option<Self> {
        if bits & !Self::VALID_MASK != 0 {
            None
        } else {
            Some(Self(bits))
        }
    }

    #[inline]
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    #[inline]
    #[must_use]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    #[inline]
    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    #[inline]
    #[must_use]
    pub const fn intersection(self, other: Self) -> Self {
        Self(self.0 & other.0)
    }
}

impl Default for DtypeSet {
    fn default() -> Self {
        Self::ALL
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_widths_match_spec() {
        assert_eq!(Dtype::F32.byte_width(), 4);
        assert_eq!(Dtype::F16.byte_width(), 2);
        assert_eq!(Dtype::BF16.byte_width(), 2);
        assert_eq!(Dtype::F8E4M3.byte_width(), 1);
        assert_eq!(Dtype::F8E5M2.byte_width(), 1);
    }

    #[test]
    fn is_f8_partitions_dtypes_correctly() {
        for dt in [Dtype::F32, Dtype::F16, Dtype::BF16] {
            assert!(!dt.is_f8(), "{dt:?} should not be f8");
        }
        for dt in [Dtype::F8E4M3, Dtype::F8E5M2] {
            assert!(dt.is_f8(), "{dt:?} should be f8");
        }
    }

    #[test]
    fn as_set_bit_round_trips_through_all() {
        for dt in [
            Dtype::F32,
            Dtype::F16,
            Dtype::BF16,
            Dtype::F8E4M3,
            Dtype::F8E5M2,
        ] {
            let bit = dt.as_set_bit();
            assert!(bit.contains(bit));
            assert!(DtypeSet::ALL.contains(bit));
            assert!(!DtypeSet::EMPTY.contains(bit));
        }
    }

    #[test]
    fn dtype_set_all_packs_five_bits() {
        assert_eq!(DtypeSet::ALL.bits(), 0b0001_1111);
    }

    #[test]
    fn dtype_set_from_bits_truncate_drops_unknown() {
        let s = DtypeSet::from_bits_truncate(0b1110_0001);
        assert_eq!(s.bits(), 0b0000_0001);
    }

    #[test]
    fn dtype_set_strict_from_bits_rejects_unknown() {
        assert_eq!(DtypeSet::from_bits(0b0001_1111), Some(DtypeSet::ALL));
        assert_eq!(DtypeSet::from_bits(0b0000_0000), Some(DtypeSet::EMPTY));
        assert_eq!(DtypeSet::from_bits(0b0010_0000), None);
        assert_eq!(DtypeSet::from_bits(0b1000_0001), None);
    }

    #[test]
    fn dtype_set_union_and_intersection() {
        let a = DtypeSet::F32.union(DtypeSet::F16);
        assert!(a.contains(DtypeSet::F32));
        assert!(a.contains(DtypeSet::F16));
        assert!(!a.contains(DtypeSet::F8E4M3));
        assert_eq!(a.intersection(DtypeSet::ALL), a);
        assert_eq!(a.intersection(DtypeSet::F8E4M3), DtypeSet::EMPTY);
    }

    #[test]
    fn dtype_set_default_is_all() {
        assert_eq!(DtypeSet::default(), DtypeSet::ALL);
    }

    #[test]
    fn dtype_serde_round_trips() {
        for dt in [
            Dtype::F32,
            Dtype::F16,
            Dtype::BF16,
            Dtype::F8E4M3,
            Dtype::F8E5M2,
        ] {
            let json = serde_json::to_string(&dt).expect("serialize");
            let back: Dtype = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(back, dt);
        }
    }
}
