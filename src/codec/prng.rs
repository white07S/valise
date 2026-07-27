// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! SplitMix64 PRNG, spec §14.4.4 / docs/vector.md §10.
//!
//! The reference scalar implementation defines canonical bytes. Sign masks and
//! Fisher-Yates permutations consume `next_u64()` outputs directly; any
//! deviation (vectorized variants, alternative seedings) must produce
//! byte-identical sequences.

#[derive(Clone, Copy, Debug)]
pub(crate) struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub(crate) fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub(crate) fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Golden sequence for `seed = 0` from Vigna's reference implementation.
    #[test]
    fn split_mix_64_seed_zero_golden_sequence() {
        let mut prng = SplitMix64::new(0);
        let golden: [u64; 5] = [
            0xE220_A839_7B1D_CDAF,
            0x6E78_9E6A_A1B9_65F4,
            0x06C4_5D18_8009_454F,
            0xF88B_B8A8_724C_81EC,
            0x1B39_896A_51A8_749B,
        ];
        for (i, expected) in golden.iter().enumerate() {
            let actual = prng.next_u64();
            assert_eq!(
                actual, *expected,
                "index {i}: expected {expected:#018x}, got {actual:#018x}"
            );
        }
    }

    #[test]
    fn split_mix_64_is_deterministic() {
        let mut a = SplitMix64::new(0xDEAD_BEEF);
        let mut b = SplitMix64::new(0xDEAD_BEEF);
        for _ in 0..32 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }
}
