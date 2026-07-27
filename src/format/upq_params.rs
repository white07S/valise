//! Persisted parameter block for the UPQ (unrestricted polar) codec —
//! the `CodecFamily::Upq` sibling of `qam_lloyd_max_params`.
//!
//! UPQ stores one joint cell index per complex pair: `M` amplitude
//! rings with power-of-two per-ring phase counts, `Σ phases ≤ cells`.
//! The persisted parameters fully describe the encoder/decoder:
//!
//! - the block-Hadamard sign mask derives from `rotation_seed`
//!   (SplitMix64, spec §14.4.4) — never stored;
//! - the per-cell decode LUT derives from `(ring_radii, ring_phases)`
//!   at open — never stored;
//! - `sigma_per_pair` is the only per-pair data-dependent table, and
//!   `ring_radii` the only data-dependent shape table (empty for a
//!   `Rayleigh`-designed codec only in the sense that it is then a
//!   pure function of `cells`; it is persisted either way so decode
//!   never re-runs the fit).
//!
//! Wire layout — magic `NXUP`, version 1, little-endian throughout:
//!
//! ```text
//! [magic NXUP                : 4 B            ]
//! [version u16 = 1           : 2 B            ]
//! [dimension u32             : 4 B            ]
//! [num_pairs u32             : 4 B            ] (= dimension / 2)
//! [block_size u32            : 4 B            ]
//! [rotation_seed u64         : 8 B            ]
//! [cells u32                 : 4 B            ] (power of two; bits/pair = log2)
//! [num_rings u16             : 2 B            ]
//! [design_source u8          : 1 B            ] (0 = empirical, 1 = rayleigh)
//! [renormalize_at_decode u8  : 1 B            ] (0 = false, 1 = true)
//! [ring_radii: f32 × M       : 4·num_rings B  ] (ascending, σ-normalized units)
//! [ring_phases: u16 × M      : 2·num_rings B  ] (each a power of two ≥ 4)
//! [sigma_count u32           : 4 B            ] (= num_pairs)
//! [sigma_per_pair: f32 × P   : 4·num_pairs B  ]
//! [calibration_id            : 32 B           ]
//! ```

use crate::error::{Error, Result};

pub const UPQ_PARAMS_MAGIC: [u8; 4] = *b"VLUP";
pub const UPQ_PARAMS_VERSION: u16 = 1;

/// Cell-count bounds: 8 cells = 3 bits/pair floor, 65 536 = 16 bits/pair
/// ceiling (a u16 code per pair).
pub const UPQ_CELLS_MIN: u32 = 8;
pub const UPQ_CELLS_MAX: u32 = 65_536;
/// Minimum per-ring phase count; ≥ 4 keeps quadrant information so the
/// sign sketch derived from codes stays faithful.
pub const UPQ_MIN_RING_PHASES: u32 = 4;

const HEAD_BYTES: usize = 4 + 2 + 4 + 4 + 4 + 8 + 4 + 2 + 1 + 1;
const CALIBRATION_ID_BYTES: usize = 32;

/// Where the ring design came from. Persisted for provenance; decode
/// never re-runs the fit either way.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpqDesignSource {
    /// Rings + allocation fitted on the calibration sample's empirical
    /// σ-normalized amplitude distribution (data-adaptive; default).
    Empirical,
    /// Rings + allocation designed on the analytic unit-Rayleigh
    /// density — zero data dependence beyond the per-pair σ fit.
    Rayleigh,
}

impl UpqDesignSource {
    fn to_wire(self) -> u8 {
        match self {
            UpqDesignSource::Empirical => 0,
            UpqDesignSource::Rayleigh => 1,
        }
    }

    fn from_wire(b: u8) -> Result<Self> {
        match b {
            0 => Ok(UpqDesignSource::Empirical),
            1 => Ok(UpqDesignSource::Rayleigh),
            other => Err(Error::Format(format!(
                "UpqParams: invalid design_source byte {other} (expected 0 or 1)"
            ))),
        }
    }
}

/// Decoded UPQ codec parameters (see module docs for the wire layout).
#[derive(Debug, Clone, PartialEq)]
pub struct UpqParams {
    pub dimension: u32,
    pub num_pairs: u32,
    pub block_size: u32,
    pub rotation_seed: u64,
    pub cells: u32,
    pub design_source: UpqDesignSource,
    pub renormalize_at_decode: bool,
    /// Ring radii in σ-normalized units, strictly ascending.
    pub ring_radii: Vec<f32>,
    /// Power-of-two phase count per ring; `Σ ≤ cells`.
    pub ring_phases: Vec<u32>,
    pub sigma_per_pair: Vec<f32>,
    pub calibration_id: [u8; 32],
}

pub(crate) fn encode_upq_params(params: &UpqParams) -> Result<Vec<u8>> {
    params.validate()?;
    let m = params.ring_radii.len();
    let p = params.sigma_per_pair.len();
    let mut out = Vec::with_capacity(HEAD_BYTES + 4 * m + 2 * m + 4 + 4 * p + CALIBRATION_ID_BYTES);
    out.extend_from_slice(&UPQ_PARAMS_MAGIC);
    out.extend_from_slice(&UPQ_PARAMS_VERSION.to_le_bytes());
    out.extend_from_slice(&params.dimension.to_le_bytes());
    out.extend_from_slice(&params.num_pairs.to_le_bytes());
    out.extend_from_slice(&params.block_size.to_le_bytes());
    out.extend_from_slice(&params.rotation_seed.to_le_bytes());
    out.extend_from_slice(&params.cells.to_le_bytes());
    out.extend_from_slice(&(m as u16).to_le_bytes());
    out.push(params.design_source.to_wire());
    out.push(u8::from(params.renormalize_at_decode));
    for &r in &params.ring_radii {
        out.extend_from_slice(&r.to_le_bytes());
    }
    for &ph in &params.ring_phases {
        out.extend_from_slice(&(ph as u16).to_le_bytes());
    }
    out.extend_from_slice(&(p as u32).to_le_bytes());
    for &s in &params.sigma_per_pair {
        out.extend_from_slice(&s.to_le_bytes());
    }
    out.extend_from_slice(&params.calibration_id);
    Ok(out)
}

pub(crate) fn decode_upq_params(bytes: &[u8]) -> Result<UpqParams> {
    if bytes.len() < HEAD_BYTES {
        return Err(Error::Format(format!(
            "UpqParams: truncated header ({} < {HEAD_BYTES} bytes)",
            bytes.len()
        )));
    }
    if bytes[0..4] != UPQ_PARAMS_MAGIC {
        return Err(Error::Format("UpqParams: bad magic (expected NXUP)".into()));
    }
    let version = u16::from_le_bytes([bytes[4], bytes[5]]);
    if version != UPQ_PARAMS_VERSION {
        return Err(Error::Format(format!(
            "UpqParams: unsupported version {version} (expected {UPQ_PARAMS_VERSION})"
        )));
    }
    let rd_u32 =
        |o: usize| u32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]);
    let dimension = rd_u32(6);
    let num_pairs = rd_u32(10);
    let block_size = rd_u32(14);
    let rotation_seed = u64::from_le_bytes([
        bytes[18], bytes[19], bytes[20], bytes[21], bytes[22], bytes[23], bytes[24], bytes[25],
    ]);
    let cells = rd_u32(26);
    let num_rings = u16::from_le_bytes([bytes[30], bytes[31]]) as usize;
    let design_source = UpqDesignSource::from_wire(bytes[32])?;
    let renorm_byte = bytes[33];
    if renorm_byte > 1 {
        return Err(Error::Format(format!(
            "UpqParams: invalid renormalize_at_decode byte {renorm_byte}"
        )));
    }

    let mut off = HEAD_BYTES;
    let need = |off: usize, n: usize| -> Result<()> {
        if bytes.len() < off + n {
            Err(Error::Format(format!(
                "UpqParams: truncated body (need {} bytes at offset {off}, have {})",
                n,
                bytes.len().saturating_sub(off)
            )))
        } else {
            Ok(())
        }
    };

    need(off, 4 * num_rings)?;
    let mut ring_radii = Vec::with_capacity(num_rings);
    for _ in 0..num_rings {
        ring_radii.push(f32::from_le_bytes([
            bytes[off],
            bytes[off + 1],
            bytes[off + 2],
            bytes[off + 3],
        ]));
        off += 4;
    }
    need(off, 2 * num_rings)?;
    let mut ring_phases = Vec::with_capacity(num_rings);
    for _ in 0..num_rings {
        ring_phases.push(u16::from_le_bytes([bytes[off], bytes[off + 1]]) as u32);
        off += 2;
    }
    need(off, 4)?;
    let sigma_count = rd_u32(off) as usize;
    off += 4;
    if sigma_count != num_pairs as usize {
        return Err(Error::Format(format!(
            "UpqParams: sigma_count {sigma_count} != num_pairs {num_pairs}"
        )));
    }
    need(off, 4 * sigma_count)?;
    let mut sigma_per_pair = Vec::with_capacity(sigma_count);
    for _ in 0..sigma_count {
        sigma_per_pair.push(f32::from_le_bytes([
            bytes[off],
            bytes[off + 1],
            bytes[off + 2],
            bytes[off + 3],
        ]));
        off += 4;
    }
    need(off, CALIBRATION_ID_BYTES)?;
    let mut calibration_id = [0u8; 32];
    calibration_id.copy_from_slice(&bytes[off..off + CALIBRATION_ID_BYTES]);
    off += CALIBRATION_ID_BYTES;
    if off != bytes.len() {
        return Err(Error::Format(format!(
            "UpqParams: {} trailing bytes after params",
            bytes.len() - off
        )));
    }

    let params = UpqParams {
        dimension,
        num_pairs,
        block_size,
        rotation_seed,
        cells,
        design_source,
        renormalize_at_decode: renorm_byte == 1,
        ring_radii,
        ring_phases,
        sigma_per_pair,
        calibration_id,
    };
    params.validate()?;
    Ok(params)
}

impl UpqParams {
    pub fn validate(&self) -> Result<()> {
        if self.dimension == 0 || !self.dimension.is_multiple_of(2) {
            return Err(Error::Format(format!(
                "UpqParams: dimension {} must be positive and even",
                self.dimension
            )));
        }
        if self.num_pairs != self.dimension / 2 {
            return Err(Error::Format(format!(
                "UpqParams: num_pairs {} != dimension/2 ({})",
                self.num_pairs,
                self.dimension / 2
            )));
        }
        if self.block_size == 0
            || !self.block_size.is_power_of_two()
            || !self.dimension.is_multiple_of(self.block_size)
        {
            return Err(Error::Format(format!(
                "UpqParams: block_size {} must be a power of two dividing dimension {}",
                self.block_size, self.dimension
            )));
        }
        if !self.cells.is_power_of_two() || self.cells < UPQ_CELLS_MIN || self.cells > UPQ_CELLS_MAX
        {
            return Err(Error::Format(format!(
                "UpqParams: cells {} must be a power of two in [{UPQ_CELLS_MIN}, {UPQ_CELLS_MAX}]",
                self.cells
            )));
        }
        let m = self.ring_radii.len();
        if m < 2 || m > u16::MAX as usize {
            return Err(Error::Format(format!(
                "UpqParams: num_rings {m} out of range (2..=65535)"
            )));
        }
        if self.ring_phases.len() != m {
            return Err(Error::Format(format!(
                "UpqParams: ring_phases len {} != ring_radii len {m}",
                self.ring_phases.len()
            )));
        }
        let mut prev = 0.0_f32;
        for (i, &r) in self.ring_radii.iter().enumerate() {
            if !r.is_finite() || r <= 0.0 {
                return Err(Error::Format(format!(
                    "UpqParams: ring_radii[{i}] = {r} must be finite and positive"
                )));
            }
            if r <= prev {
                return Err(Error::Format(format!(
                    "UpqParams: ring_radii must be strictly ascending (radii[{i}] = {r} <= {prev})"
                )));
            }
            prev = r;
        }
        let mut total: u64 = 0;
        for (i, &p) in self.ring_phases.iter().enumerate() {
            if !p.is_power_of_two() || p < UPQ_MIN_RING_PHASES || p > self.cells {
                return Err(Error::Format(format!(
                    "UpqParams: ring_phases[{i}] = {p} must be a power of two in \
                     [{UPQ_MIN_RING_PHASES}, cells]"
                )));
            }
            total += p as u64;
        }
        if total > self.cells as u64 {
            return Err(Error::Format(format!(
                "UpqParams: sum of ring_phases {total} exceeds cells {}",
                self.cells
            )));
        }
        if self.sigma_per_pair.len() != self.num_pairs as usize {
            return Err(Error::Format(format!(
                "UpqParams: sigma_per_pair len {} != num_pairs {}",
                self.sigma_per_pair.len(),
                self.num_pairs
            )));
        }
        for (i, &s) in self.sigma_per_pair.iter().enumerate() {
            if !s.is_finite() || s <= 0.0 {
                return Err(Error::Format(format!(
                    "UpqParams: sigma_per_pair[{i}] = {s} must be finite and positive"
                )));
            }
        }
        if self.calibration_id == [0u8; 32] {
            return Err(Error::Format(
                "UpqParams: calibration_id must be non-zero".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_params() -> UpqParams {
        UpqParams {
            dimension: 64,
            num_pairs: 32,
            block_size: 16,
            rotation_seed: 0xC0DE,
            cells: 256,
            design_source: UpqDesignSource::Empirical,
            renormalize_at_decode: true,
            ring_radii: vec![0.4, 0.9, 1.4, 2.1],
            ring_phases: vec![16, 64, 128, 32],
            sigma_per_pair: vec![0.125; 32],
            calibration_id: [7u8; 32],
        }
    }

    #[test]
    fn roundtrip() {
        let p = sample_params();
        let bytes = encode_upq_params(&p).expect("encode");
        let q = decode_upq_params(&bytes).expect("decode");
        assert_eq!(p, q);
    }

    #[test]
    fn deterministic_encode_is_byte_identical() {
        let p = sample_params();
        assert_eq!(
            encode_upq_params(&p).expect("a"),
            encode_upq_params(&p).expect("b")
        );
    }

    #[test]
    fn rejects_bad_magic_version_and_trailing() {
        let p = sample_params();
        let bytes = encode_upq_params(&p).expect("encode");
        let mut bad = bytes.clone();
        bad[0] = b'X';
        assert!(decode_upq_params(&bad).is_err());
        let mut bad = bytes.clone();
        bad[4] = 9;
        assert!(decode_upq_params(&bad).is_err());
        let mut bad = bytes.clone();
        bad.push(0);
        assert!(decode_upq_params(&bad).is_err());
        assert!(decode_upq_params(&bytes[..HEAD_BYTES - 1]).is_err());
    }

    #[test]
    fn rejects_invalid_fields() {
        let mut p = sample_params();
        p.ring_phases[1] = 48; // not a power of two
        assert!(p.validate().is_err());

        let mut p = sample_params();
        p.ring_phases = vec![128, 128, 128, 128]; // sum 512 > cells 256
        assert!(p.validate().is_err());

        let mut p = sample_params();
        p.ring_radii[2] = 0.5; // not ascending
        assert!(p.validate().is_err());

        let mut p = sample_params();
        p.cells = 100; // not a power of two
        assert!(p.validate().is_err());

        let mut p = sample_params();
        p.sigma_per_pair[3] = 0.0;
        assert!(p.validate().is_err());

        let mut p = sample_params();
        p.calibration_id = [0u8; 32];
        assert!(p.validate().is_err());

        let mut p = sample_params();
        p.block_size = 24;
        assert!(p.validate().is_err());
    }
}
