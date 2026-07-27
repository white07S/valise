// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Create-time, file-level contract (plan §4).
//!
//! `CreateContractV1` captures the small set of invariants that are fixed
//! at file creation and never change for the life of the file:
//!
//! - whether text retrieval is enabled,
//! - the maximum embedding dimension,
//! - the allowed input dtypes,
//! - the auto-promotion thresholds (post-Phase-4).
//!
//! The contract lives in two places, both written at `create_with_options`:
//!
//! 1. **`TocFooterBody.create_contract`** — the canonical, verifiable copy.
//!    Re-stamped unchanged on every commit so the TOC body remains
//!    self-consistent.
//! 2. **A 32-byte BLAKE3 digest** stamped into the 4 KB header reserved
//!    area. The digest is the immutability anchor: a tampered footer
//!    body would mismatch the digest and `open()` would refuse the file.
//!
//! Pre-bump files (no `FEATURE_CREATE_CONTRACT` bit, no
//! `TocFooterBody.create_contract`) get a synthesized in-memory contract
//! at `open()` and continue to work without an in-place upgrade.

use serde::{Deserialize, Serialize};

use crate::{
    error::{Error, Result},
    format::{Checksum, bincode_config, blake3_checksum},
};

/// Wire schema version for the contract record.
pub const SCHEMA_VERSION: u16 = 1;

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct CreateContractV1 {
    /// Always `1` for this version. Reserved for future schema evolution.
    pub schema_version: u16,
    /// True when this file allows text retrieval. `register_text_space`
    /// and friends fail with `Error::Unsupported` when this is false.
    pub text_enabled: bool,
    /// Hard cap on `EmbeddingSpaceSpec.dimension`. 1..=65_535.
    pub max_dim: u32,
    /// Whitelisted dtypes, encoded as `DtypeSet::bits()`. Stored as a
    /// bare `u8` rather than the typed `DtypeSet` so the wire format
    /// stays stable if we ever extend `DtypeSet` with helpers that
    /// affect serde shape.
    pub allowed_dtypes: u8,
    /// VESTIGIAL since v2.3: gated the removed vote-index auto-build.
    /// Stored for format stability; no runtime path consumes it.
    pub auto_promote_non_f8: u64,
    /// VESTIGIAL since v2.3 (see `auto_promote_non_f8`). Stored for format
    /// stability; no runtime path consumes it.
    pub auto_promote_f8: u64,
    /// Zero-filled. Reserved for forward-compat additions that don't
    /// warrant a schema bump. Decoders MUST ignore the contents.
    pub reserved: [u8; 16],
}

impl Default for CreateContractV1 {
    fn default() -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            text_enabled: true,
            max_dim: 4096,
            allowed_dtypes: 0b0001_1111, // DtypeSet::ALL
            auto_promote_non_f8: 200_000,
            auto_promote_f8: 500_000,
            reserved: [0u8; 16],
        }
    }
}

impl CreateContractV1 {
    /// Bincode-encode the contract using the shared file-level config
    /// (little-endian, variable-int). The digest in the header is
    /// computed over this byte stream, so the encoding MUST be
    /// deterministic — `bincode_config()` already is.
    pub fn encode(&self) -> Result<Vec<u8>> {
        bincode::serde::encode_to_vec(self, bincode_config())
            .map_err(|err| Error::Serde(err.to_string()))
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let (contract, consumed) =
            bincode::serde::decode_from_slice::<Self, _>(bytes, bincode_config())
                .map_err(|err| Error::Serde(err.to_string()))?;
        if consumed != bytes.len() {
            return Err(Error::Format(
                "CreateContractV1 has trailing bytes after decode".into(),
            ));
        }
        contract.validate_decoded()?;
        Ok(contract)
    }

    /// BLAKE3-256 over the canonical bincode encoding. Stamped into the
    /// 4 KB header at file creation. `open()` recomputes this from the
    /// `TocFooterBody.create_contract` and rejects the file on mismatch.
    pub fn digest(&self) -> Result<Checksum> {
        Ok(blake3_checksum(&self.encode()?))
    }

    /// Validation rules from plan §4.4. Run at create time, before the
    /// header is written.
    pub fn validate_options(&self) -> Result<()> {
        if self.schema_version != SCHEMA_VERSION {
            return Err(Error::Format(format!(
                "CreateContractV1: unsupported schema_version {}",
                self.schema_version
            )));
        }
        if self.max_dim == 0 || self.max_dim > 65_535 {
            return Err(Error::Format(format!(
                "CreateContractV1: max_dim {} out of range (1..=65_535)",
                self.max_dim
            )));
        }
        // Reject unknown dtype bits; these would silently allow future
        // dtypes a Phase 1 binary cannot honor.
        if crate::format::dtype::DtypeSet::from_bits(self.allowed_dtypes).is_none() {
            return Err(Error::Format(format!(
                "CreateContractV1: allowed_dtypes 0x{:02X} contains unknown bits",
                self.allowed_dtypes
            )));
        }
        if self.allowed_dtypes == 0 {
            return Err(Error::Format(
                "CreateContractV1: allowed_dtypes is empty".into(),
            ));
        }
        if self.auto_promote_non_f8 < 1024 {
            return Err(Error::Format(format!(
                "CreateContractV1: auto_promote_non_f8 ({}) must be ≥ 1024",
                self.auto_promote_non_f8
            )));
        }
        if self.auto_promote_f8 < self.auto_promote_non_f8 {
            return Err(Error::Format(format!(
                "CreateContractV1: auto_promote_f8 ({}) must be ≥ auto_promote_non_f8 ({})",
                self.auto_promote_f8, self.auto_promote_non_f8
            )));
        }
        Ok(())
    }

    /// Lighter-weight validation applied after `decode`. Decoded
    /// contracts are presumed to have come from a trusted writer that
    /// already passed `validate_options`; we only re-check the schema
    /// version and dtype bits to defend against forged or corrupted
    /// records. The digest in the header is the primary integrity
    /// guard.
    fn validate_decoded(&self) -> Result<()> {
        if self.schema_version != SCHEMA_VERSION {
            return Err(Error::Format(format!(
                "CreateContractV1: unsupported schema_version {}",
                self.schema_version
            )));
        }
        if crate::format::dtype::DtypeSet::from_bits(self.allowed_dtypes).is_none() {
            return Err(Error::Format(format!(
                "CreateContractV1: allowed_dtypes 0x{:02X} contains unknown bits",
                self.allowed_dtypes
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::dtype::DtypeSet;

    #[test]
    fn defaults_pass_validation() {
        CreateContractV1::default()
            .validate_options()
            .expect("default contract validates");
    }

    #[test]
    fn round_trip_through_bincode() {
        let original = CreateContractV1 {
            schema_version: SCHEMA_VERSION,
            text_enabled: false,
            max_dim: 1024,
            allowed_dtypes: DtypeSet::F32.union(DtypeSet::F16).bits(),
            auto_promote_non_f8: 50_000,
            auto_promote_f8: 100_000,
            reserved: [0u8; 16],
        };
        // Note: this contract has auto_promote_non_f8 < 1024 disabled, so
        // it would fail validate_options. We still want to round-trip the
        // bytes — validate is the *creator-side* guard, not a decode
        // invariant. (The decoder uses validate_decoded.)
        let bytes = original.encode().expect("encode");
        let decoded = CreateContractV1::decode(&bytes).expect("decode");
        assert_eq!(decoded, original);
    }

    #[test]
    fn digest_is_stable_across_calls() {
        let c = CreateContractV1::default();
        assert_eq!(c.digest().unwrap(), c.digest().unwrap());
    }

    #[test]
    fn digest_changes_when_contract_changes() {
        let a = CreateContractV1::default();
        let b = CreateContractV1 {
            text_enabled: false,
            ..a.clone()
        };
        assert_ne!(a.digest().unwrap(), b.digest().unwrap());
    }

    #[test]
    fn rejects_zero_max_dim() {
        let c = CreateContractV1 {
            max_dim: 0,
            ..Default::default()
        };
        c.validate_options().unwrap_err();
    }

    #[test]
    fn rejects_oversized_max_dim() {
        let c = CreateContractV1 {
            max_dim: 1 << 17,
            ..Default::default()
        };
        c.validate_options().unwrap_err();
    }

    #[test]
    fn rejects_empty_allowed_dtypes() {
        let c = CreateContractV1 {
            allowed_dtypes: 0,
            ..Default::default()
        };
        c.validate_options().unwrap_err();
    }

    #[test]
    fn rejects_unknown_dtype_bits() {
        let c = CreateContractV1 {
            allowed_dtypes: 0b1110_0000,
            ..Default::default()
        };
        c.validate_options().unwrap_err();
    }

    #[test]
    fn rejects_low_non_f8_threshold() {
        let c = CreateContractV1 {
            auto_promote_non_f8: 100,
            ..Default::default()
        };
        c.validate_options().unwrap_err();
    }

    #[test]
    fn rejects_f8_threshold_below_non_f8() {
        let c = CreateContractV1 {
            auto_promote_non_f8: 200_000,
            auto_promote_f8: 100_000,
            ..Default::default()
        };
        c.validate_options().unwrap_err();
    }

    #[test]
    fn decode_rejects_unknown_dtype_bits() {
        let c = CreateContractV1 {
            allowed_dtypes: 0b1110_0000,
            ..Default::default()
        };
        let bytes = c.encode().expect("encode (no validate_options on encode)");
        CreateContractV1::decode(&bytes).expect_err("decode validate rejects");
    }

    #[test]
    fn decode_rejects_trailing_bytes() {
        let c = CreateContractV1::default();
        let mut bytes = c.encode().expect("encode");
        bytes.push(0);
        CreateContractV1::decode(&bytes).expect_err("trailing bytes rejected");
    }
}
