//! UPQ — unrestricted polar quantization (measurement prototype).
//!
//! One 11-bit **joint cell index per complex pair** instead of the
//! production split (5-bit amp × 6-bit phase): `M` amplitude rings with
//! power-of-two per-ring phase counts `P_i`, `Σ P_i ≤ cells` (2048 =
//! 11 bits). Ring radii and the phase allocation are fitted on the
//! per-pair σ-normalized amplitude distribution of a calibration
//! sample, minimizing the polar distortion
//! `(a − r_i)² + a·r_i·(2π/P_i)²/12` (radial + tangential).
//!
//! Provenance: `experiments/PARETO_RESEARCH_2026-06.md` Part 6 — the
//! Python prototype measured **+1.07 dB / +0.64 recall pts at iso-bits
//! (5.5 b/dim)** over QAM(5,6) on Cohere d=768. This module is the
//! minimal Rust counterpart used by `examples/upq_768_bench.rs` to
//! validate recall / latency / storage / memory against the production
//! codec. It deliberately reuses the production rotation + σ
//! conventions (block-Hadamard, `σ = mean|z|/√(π/2)`), so a future
//! codec integration can lift the design/encode/decode parts wholesale.
//!
//! Scoring kernels (per the 2026-06 Apple-Silicon kernel review):
//!
//! - **K-B `score_i8_cache`** (default): vectors are decoded once at
//!   file-open into an interleaved-i8 cache row (`dim` bytes, unit-norm
//!   with a per-row dequant scale). Stage-2 scoring is a contiguous
//!   i8·i8 dot in the exact indexed-loop shape LLVM auto-vectorizes to
//!   a 4-accumulator SDOT kernel on aarch64, plus `prfm pldl1strm`
//!   software prefetch over the next candidate's cache lines (the
//!   candidate rows are scattered; without prefetch the kernel is
//!   memory-latency-bound, not ALU-bound).
//! - **K-C `score_packed11`** (zero extra RAM): scores the bit-packed
//!   11-bit codes directly — per pair: extract field, look up the
//!   normalized (re,im) i8 cell, MAC against a σ-fused i8 query.
//!   Scalar; ~3-4× slower than K-B but streams only the packed codes.
//!
//! There is intentionally **no gather-u16 kernel**: it needs the same
//! RAM as the i8 cache (2 B/pair × 384 = 768 B/vec) and is strictly
//! slower (NEON has no gather; ≥2 dependent loads/pair).

use crate::error::{Error, Result};
use crate::format::catalog::VectorMetric;
use crate::format::upq_params::UpqParams;

pub use crate::format::upq_params::UpqDesignSource;

use super::prng::SplitMix64;
use super::qam_lloyd_max::{block_hadamard_forward, block_hadamard_inverse};

const TAU: f32 = std::f32::consts::TAU;
/// Default joint-cell budget: 2048 = 11 bits/pair = 5.5 bits/dim, the
/// production operating point (same wire bits as QAM(5,6)).
pub const DEFAULT_UPQ_CELLS: u32 = 2048;
/// Default ring-count sweep for the design fit.
pub const DEFAULT_UPQ_RING_SWEEP: &[usize] = &[24, 32, 40, 48];
/// Default rotation seed — same value as the QAM codec's default
/// ("lama_qam"), so a UPQ codec calibrated on the same sample sees the
/// identical rotated space.
pub const DEFAULT_UPQ_ROTATION_SEED: u64 = 0x6C61_6D61_5F71_616D;
/// The v1 runtime packs codes as fixed 11-bit fields; `cells` may be
/// smaller (wasting wire bits) but never larger.
const UPQ_WIRE_BITS: u32 = 11;
const UPQ_WIRE_MAX_CELLS: u32 = 1 << UPQ_WIRE_BITS;
/// Design-time subsample cap (deterministic stride subsample).
const DESIGN_SAMPLE_CAP: usize = 400_000;
/// Minimum per-ring phase count (power of two ≥ 4 keeps quadrant info
/// so the sign sketch stays faithful).
const MIN_RING_PHASES: u32 = 4;
/// Alternating-optimization iterations for the ring fit.
const FIT_ITERS: usize = 30;

// ============================================================
// Design: rings + power-of-two phase allocation
// ============================================================

/// Fitted UPQ cell layout, in σ-normalized amplitude units (shared
/// across pairs; per-pair scale enters via `sigma_per_pair`).
#[derive(Debug, Clone)]
pub struct UpqDesign {
    /// Total joint cells per pair (2048 ⇒ 11 bits/pair = 5.5 bits/dim).
    pub cells: u32,
    /// Ring radii, ascending, normalized units.
    pub ring_radii: Vec<f32>,
    /// Power-of-two phase count per ring, `Σ ≤ cells`.
    pub ring_phases: Vec<u32>,
    /// Cumulative first cell index per ring.
    pub ring_bases: Vec<u32>,
    /// `M−1` amplitude decision boundaries (normalized units).
    pub ring_boundaries: Vec<f32>,
    /// Mean fitted distortion per pair on the design sample.
    pub design_cost: f64,
}

/// Deterministic ±1 sign mask from a u64 seed — the canonical
/// SplitMix64 derivation of spec §14.4.4 (64 sign bits per
/// `next_u64()`, LSB first; bit 0 ⇒ +1). Byte-identical to the QAM
/// codec's mask for the same seed.
fn signs_from_seed(seed: u64, dim: usize) -> Vec<f32> {
    let mut prng = SplitMix64::new(seed);
    let mut signs = Vec::with_capacity(dim);
    let mut bits: u64 = 0;
    let mut bits_left: u32 = 0;
    for _ in 0..dim {
        if bits_left == 0 {
            bits = prng.next_u64();
            bits_left = 64;
        }
        signs.push(if bits & 1 == 0 { 1.0_f32 } else { -1.0_f32 });
        bits >>= 1;
        bits_left -= 1;
    }
    signs
}

/// Deterministic quantile sample of the unit Rayleigh density
/// (inverse-CDF quadrature): `a_i = sqrt(-2 ln(1 - (i+0.5)/n))`.
/// Feeding this to the empirical fitter is numerically the same as
/// fitting on the analytic density.
fn rayleigh_quantile_sample(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let u = (i as f64 + 0.5) / n as f64;
            (-2.0 * (1.0 - u).ln()).sqrt() as f32
        })
        .collect()
}

/// Tangential coefficient `(2π/P)²/12` of a ring.
fn tangential_coeff(phases: u32) -> f64 {
    let step = std::f64::consts::TAU / phases as f64;
    step * step / 12.0
}

/// Cell cost of normalized amplitude `a` against ring `(r, c)`.
/// Test-only reference; the fitter uses `ring_cut_boundaries`.
#[cfg(test)]
fn ring_cost(a: f64, r: f64, c: f64) -> f64 {
    (a - r) * (a - r) + a * r * c
}

/// Closed-form decision boundaries between adjacent (sorted) rings:
/// ring i beats ring i+1 for `a ≤ (r₁²−r₀²)/(2(r₁−r₀)+r₀c₀−r₁c₁)`.
/// Monotone-clamped so the partition is always a valid interval split —
/// this is what makes each fit iteration O(M log n) on a sorted sample
/// instead of an O(n·M) brute argmin (the former 1.1 s of calibration).
fn ring_cut_boundaries(radii: &[f64], coeffs: &[f64]) -> Vec<f64> {
    let m = radii.len();
    let mut bounds = vec![0.0_f64; m.saturating_sub(1)];
    let mut prev = 0.0_f64;
    for i in 0..m.saturating_sub(1) {
        let (r0, c0, r1, c1) = (radii[i], coeffs[i], radii[i + 1], coeffs[i + 1]);
        let den = 2.0 * (r1 - r0) + r0 * c0 - r1 * c1;
        let mut b = if den > 1e-12 {
            (r1 * r1 - r0 * r0) / den
        } else {
            f64::INFINITY
        };
        if b < prev {
            b = prev;
        }
        bounds[i] = b;
        prev = b;
    }
    bounds
}

/// Power-of-two phase allocation: start every ring at `MIN_RING_PHASES`
/// and greedily double the ring with the largest marginal distortion
/// reduction (`0.75·w/P²`) that still fits the budget.
fn alloc_pow2(weights: &[f64], cells: u32) -> Vec<u32> {
    let mut phases = vec![MIN_RING_PHASES; weights.len()];
    loop {
        let used: u32 = phases.iter().sum();
        let budget = cells.saturating_sub(used);
        if budget == 0 {
            return phases;
        }
        let mut best: Option<(usize, f64)> = None;
        for (i, (&w, &p)) in weights.iter().zip(&phases).enumerate() {
            if p > budget {
                continue;
            }
            let gain = 0.75 * w.max(1e-12) / (p as f64 * p as f64);
            if best.is_none_or(|(_, g)| gain > g) {
                best = Some((i, gain));
            }
        }
        match best {
            Some((i, _)) => phases[i] *= 2,
            None => return phases, // nothing fits; leave slack unused
        }
    }
}

impl UpqDesign {
    /// Fit `num_rings` rings on a pooled sample of σ-normalized
    /// amplitudes via alternating optimization (assign → radii →
    /// allocation), keeping the best iterate by mean cell cost.
    pub fn fit(amps_norm: &[f32], num_rings: usize, cells: u32) -> Result<Self> {
        if amps_norm.is_empty() || num_rings < 2 {
            return Err(Error::Format(
                "UpqDesign::fit: need a non-empty sample and ≥2 rings".into(),
            ));
        }
        let stride = (amps_norm.len() / DESIGN_SAMPLE_CAP).max(1);
        let mut sample: Vec<f64> = amps_norm
            .iter()
            .step_by(stride)
            .map(|&a| a as f64)
            .collect();
        // Sorted sample + prefix sums of a and a²: every iteration's
        // assignment, statistics, and exact cost become segment lookups.
        sample.sort_unstable_by(|a, b| a.total_cmp(b));
        let nsamp = sample.len();
        let mut pa = vec![0.0_f64; nsamp + 1];
        let mut paa = vec![0.0_f64; nsamp + 1];
        for (i, &a) in sample.iter().enumerate() {
            pa[i + 1] = pa[i] + a;
            paa[i + 1] = paa[i] + a * a;
        }
        let a_max = sample.last().copied().unwrap_or(0.0).max(8.0);

        // Init radii: cube-root-density (Panter–Dité) histogram quantiles.
        let bins = 2048usize;
        let mut hist = vec![0.0_f64; bins];
        for &a in &sample {
            let b = ((a / a_max) * bins as f64) as usize;
            hist[b.min(bins - 1)] += 1.0;
        }
        let cum: Vec<f64> = hist
            .iter()
            .scan(0.0, |acc, &h| {
                *acc += h.cbrt();
                Some(*acc)
            })
            .collect();
        let cum_max = *cum.last().expect("BUG: bins > 0");
        let mut radii = vec![0.0_f64; num_rings];
        for (i, r) in radii.iter_mut().enumerate() {
            let target = (i as f64 + 0.5) / num_rings as f64 * cum_max;
            let b = cum.partition_point(|&c| c < target);
            *r = (b.min(bins - 1) as f64 + 0.5) / bins as f64 * a_max;
        }
        let mut phases = vec![cells / num_rings as u32; num_rings];
        let mut best: Option<(f64, Vec<f64>, Vec<u32>)> = None;

        let mut cut = vec![0usize; num_rings + 1];
        for _ in 0..FIT_ITERS {
            // -- assign via closed-form interval boundaries (O(M log n))
            let coeffs: Vec<f64> = phases.iter().map(|&p| tangential_coeff(p)).collect();
            let bounds = ring_cut_boundaries(&radii, &coeffs);
            cut[0] = 0;
            cut[num_rings] = nsamp;
            for (i, &b) in bounds.iter().enumerate() {
                cut[i + 1] = sample.partition_point(|&a| a <= b);
            }
            for i in 1..num_rings {
                if cut[i] < cut[i - 1] {
                    cut[i] = cut[i - 1];
                }
            }

            // -- exact segment statistics + cost from prefix sums
            let mut total = 0.0_f64;
            let mut sums = vec![0.0_f64; num_rings];
            let mut counts = vec![0u64; num_rings];
            for i in 0..num_rings {
                let (lo, hi) = (cut[i], cut[i + 1]);
                let n_i = (hi - lo) as f64;
                let sa = pa[hi] - pa[lo];
                let saa = paa[hi] - paa[lo];
                let (r, c) = (radii[i], coeffs[i]);
                total += saa - 2.0 * r * sa + n_i * r * r + r * c * sa;
                sums[i] = sa;
                counts[i] = (hi - lo) as u64;
            }
            let mean_cost = total / nsamp as f64;
            if best.as_ref().is_none_or(|(c, _, _)| mean_cost < *c) {
                best = Some((mean_cost, radii.clone(), phases.clone()));
            }

            // -- radii update: r_i = E[a|i]·(1 − c_i/2); respawn empty rings
            let densest = counts
                .iter()
                .enumerate()
                .max_by_key(|(_, c)| **c)
                .map(|(i, _)| i)
                .unwrap_or(0);
            for i in 0..num_rings {
                if counts[i] > 0 {
                    radii[i] = sums[i] / counts[i] as f64 * (1.0 - coeffs[i] / 2.0);
                } else {
                    radii[i] = radii[densest] * (1.0 + 0.013 * (i as f64 + 1.0));
                }
            }
            // -- keep rings sorted (allocation weights follow the sort)
            let mut order: Vec<usize> = (0..num_rings).collect();
            order.sort_by(|&x, &y| radii[x].total_cmp(&radii[y]));
            radii = order.iter().map(|&i| radii[i]).collect();
            let sums: Vec<f64> = order.iter().map(|&i| sums[i]).collect();
            let counts: Vec<u64> = order.iter().map(|&i| counts[i]).collect();

            // -- allocation: w_i = n_i·E[a|i]·r_i·(2π)²/12, P ∝ w^{1/3} (pow2 greedy)
            let weights: Vec<f64> = (0..num_rings)
                .map(|i| {
                    let mean_a = if counts[i] > 0 {
                        sums[i] / counts[i] as f64
                    } else {
                        0.0
                    };
                    counts[i] as f64 * mean_a * radii[i] * std::f64::consts::TAU.powi(2) / 12.0
                })
                .collect();
            phases = alloc_pow2(&weights, cells);
        }

        let (design_cost, radii, phases) = best.expect("BUG: ≥1 fit iteration");
        let mut bases = vec![0u32; num_rings];
        let mut acc = 0u32;
        for (b, &p) in bases.iter_mut().zip(&phases) {
            *b = acc;
            acc += p;
        }
        if acc > cells {
            return Err(Error::Format(format!(
                "UpqDesign::fit: allocation overflow ({acc} > {cells})"
            )));
        }

        // Encode boundaries: same closed form the fit assigned with.
        let coeffs: Vec<f64> = phases.iter().map(|&p| tangential_coeff(p)).collect();
        let boundaries: Vec<f32> = ring_cut_boundaries(&radii, &coeffs)
            .into_iter()
            .map(|b| b as f32)
            .collect();

        Ok(Self {
            cells,
            ring_radii: radii.iter().map(|&r| r as f32).collect(),
            ring_phases: phases,
            ring_bases: bases,
            ring_boundaries: boundaries,
            design_cost,
        })
    }

    /// Fit over several ring counts; keep the lowest design cost.
    pub fn fit_sweep(amps_norm: &[f32], ring_choices: &[usize], cells: u32) -> Result<Self> {
        let mut best: Option<Self> = None;
        for &m in ring_choices {
            let d = Self::fit(amps_norm, m, cells)?;
            if best.as_ref().is_none_or(|b| d.design_cost < b.design_cost) {
                best = Some(d);
            }
        }
        best.ok_or_else(|| Error::Format("UpqDesign::fit_sweep: empty ring_choices".into()))
    }

    /// Data-free design: rings + allocation on the analytic unit
    /// Rayleigh density (deterministic; no calibration sample needed
    /// beyond the σ fit, mirroring the production QAM codebook).
    pub fn fit_sweep_rayleigh(ring_choices: &[usize], cells: u32) -> Result<Self> {
        let sample = rayleigh_quantile_sample(DESIGN_SAMPLE_CAP);
        Self::fit_sweep(&sample, ring_choices, cells)
    }

    /// Mean per-pair cell cost of THIS design evaluated on a given
    /// σ-normalized amplitude sample (argmin assignment, same cost as
    /// the fitter). Test-only: used to compare designs on common data.
    #[cfg(test)]
    pub fn mean_cost_on(&self, amps_norm: &[f32]) -> f64 {
        let coeffs: Vec<f64> = self
            .ring_phases
            .iter()
            .map(|&p| tangential_coeff(p))
            .collect();
        let radii: Vec<f64> = self.ring_radii.iter().map(|&r| r as f64).collect();
        let mut total = 0.0_f64;
        for &a in amps_norm {
            let a = a as f64;
            let mut bc = f64::INFINITY;
            for (&r, &c) in radii.iter().zip(&coeffs) {
                let cost = ring_cost(a, r, c);
                if cost < bc {
                    bc = cost;
                }
            }
            total += bc;
        }
        total / amps_norm.len().max(1) as f64
    }

    /// Encode one pair (normalized amplitude, phase) → joint cell index.
    #[inline]
    pub fn encode_pair(&self, a_norm: f32, theta: f32) -> u16 {
        let ring = self.ring_boundaries.partition_point(|&b| a_norm > b);
        let p = self.ring_phases[ring];
        let m = (theta * (p as f32 / TAU)).round() as i64;
        let m = m.rem_euclid(p as i64) as u32;
        (self.ring_bases[ring] + m) as u16
    }
}

// ============================================================
// Codec: per-pair σ + decode tables + sketch + caches
// ============================================================

/// UPQ codec: block-Hadamard rotation + one joint polar cell index per
/// complex pair, packed as fixed 11-bit fields. Promoted to
/// `CodecFamily::Upq` in v2.4; persisted parameters are
/// [`crate::format::upq_params::UpqParams`].
pub struct UpqCodec {
    pub dim: usize,
    pub num_pairs: usize,
    pub block_size: usize,
    pub rotation_seed: u64,
    pub renormalize_at_decode: bool,
    pub design_source: UpqDesignSource,
    pub design: UpqDesign,
    pub sigma_per_pair: Vec<f32>,
    /// Seed-derived ±1 mask for the block-Hadamard rotation.
    signs: Vec<f32>,
    /// On-disk row stride (`packed11_stride(num_pairs)`).
    base_bytes: usize,
    /// Normalized-unit decode table: cell → (re, im), len = cells.
    lut_f32: Vec<[f32; 2]>,
    /// Sign bits per cell: bit0 = re ≥ 0, bit1 = im ≥ 0 (sketch).
    lut_sign: Vec<u8>,
}

/// Prepared-query context for the trait scoring path: the rotated
/// query and its L2 norm.
pub(crate) struct UpqPreparedQuery {
    pub(crate) q_rot: Vec<f32>,
    pub(crate) q_norm: f32,
}

/// Decode tables from a fitted/restored design: per-cell `(re, im)` in
/// σ-normalized units, the i8 mirror, and the per-cell sign bits.
#[allow(clippy::type_complexity)]
fn build_tables(design: &UpqDesign) -> (Vec<[f32; 2]>, Vec<u8>) {
    let cells = design.cells as usize;
    let mut lut_f32 = vec![[0.0_f32; 2]; cells];
    let mut lut_sign = vec![0u8; cells];
    for (&r, (&p, &b)) in design
        .ring_radii
        .iter()
        .zip(design.ring_phases.iter().zip(&design.ring_bases))
    {
        for m in 0..p {
            let theta = m as f32 * (TAU / p as f32);
            let (sin, cos) = theta.sin_cos();
            let cell = (b + m) as usize;
            lut_f32[cell] = [r * cos, r * sin];
            lut_sign[cell] = u8::from(r * cos >= 0.0) | (u8::from(r * sin >= 0.0) << 1);
        }
    }
    (lut_f32, lut_sign)
}

/// Production default `block_size` rule (mirrors the QAM codec):
/// `min(1024, largest power of two dividing dim)`.
pub(crate) fn default_block_size_for(dim: usize) -> usize {
    let mut block = 1usize;
    while block * 2 <= 1024 && dim.is_multiple_of(block * 2) {
        block *= 2;
    }
    block
}

impl UpqCodec {
    fn validate_dims(dim: usize, block_size: usize, cells: u32) -> Result<()> {
        if dim == 0 || !dim.is_multiple_of(2) {
            return Err(Error::Format(format!(
                "UpqCodec: dim {dim} must be positive and even"
            )));
        }
        if block_size == 0 || !block_size.is_power_of_two() || !dim.is_multiple_of(block_size) {
            return Err(Error::Format(format!(
                "UpqCodec: block_size {block_size} must be a power of two dividing dim {dim}"
            )));
        }
        if cells > UPQ_WIRE_MAX_CELLS {
            return Err(Error::Unsupported(format!(
                "UpqCodec: cells {cells} exceeds the {UPQ_WIRE_BITS}-bit wire packing \
                 (max {UPQ_WIRE_MAX_CELLS})"
            )));
        }
        Ok(())
    }

    /// Production calibration entry: rotate `sample`, fit σ per pair,
    /// design the rings per `source`, and assemble a registrable codec.
    pub fn fit_from_sample(
        dim: usize,
        block_size: usize,
        rotation_seed: u64,
        cells: u32,
        ring_choices: &[usize],
        source: UpqDesignSource,
        sample: &[Vec<f32>],
    ) -> Result<Self> {
        Self::validate_dims(dim, block_size, cells)?;
        if sample.is_empty() {
            return Err(Error::Format("UpqCodec: empty calibration sample".into()));
        }
        let signs = signs_from_seed(rotation_seed, dim);
        let mut rotated = Vec::with_capacity(sample.len());
        for (i, v) in sample.iter().enumerate() {
            if v.len() != dim {
                return Err(Error::Format(format!(
                    "UpqCodec: sample[{i}] dim mismatch ({} vs {dim})",
                    v.len()
                )));
            }
            let mut row = v.clone();
            for (x, s) in row.iter_mut().zip(&signs) {
                if !x.is_finite() {
                    return Err(Error::Format(format!(
                        "UpqCodec: sample[{i}] contains a non-finite value"
                    )));
                }
                let _ = s;
            }
            block_hadamard_forward(&mut row, &signs, block_size)?;
            rotated.push(row);
        }
        let mut codec = Self::fit_on_rotated(&rotated, dim, ring_choices, cells, source)?;
        codec.block_size = block_size;
        codec.rotation_seed = rotation_seed;
        codec.signs = signs;
        Ok(codec)
    }

    /// Restore a codec from persisted parameters (file-open path).
    pub fn from_params(params: &UpqParams) -> Result<Self> {
        params.validate()?;
        Self::validate_dims(
            params.dimension as usize,
            params.block_size as usize,
            params.cells,
        )?;
        let num_rings = params.ring_radii.len();
        let mut bases = vec![0u32; num_rings];
        let mut acc = 0u32;
        for (b, &p) in bases.iter_mut().zip(&params.ring_phases) {
            *b = acc;
            acc += p;
        }
        let radii_f64: Vec<f64> = params.ring_radii.iter().map(|&r| r as f64).collect();
        let coeffs: Vec<f64> = params
            .ring_phases
            .iter()
            .map(|&p| tangential_coeff(p))
            .collect();
        let boundaries: Vec<f32> = ring_cut_boundaries(&radii_f64, &coeffs)
            .into_iter()
            .map(|b| b as f32)
            .collect();
        let design = UpqDesign {
            cells: params.cells,
            ring_radii: params.ring_radii.clone(),
            ring_phases: params.ring_phases.clone(),
            ring_bases: bases,
            ring_boundaries: boundaries,
            // Not persisted: the fit-time sample cost has no decode-side
            // meaning. 0.0 marks "restored from params".
            design_cost: 0.0,
        };
        let (lut_f32, lut_sign) = build_tables(&design);
        let dim = params.dimension as usize;
        Ok(Self {
            dim,
            num_pairs: params.num_pairs as usize,
            block_size: params.block_size as usize,
            rotation_seed: params.rotation_seed,
            renormalize_at_decode: params.renormalize_at_decode,
            design_source: params.design_source,
            design,
            sigma_per_pair: params.sigma_per_pair.clone(),
            signs: signs_from_seed(params.rotation_seed, dim),
            base_bytes: packed11_stride(params.num_pairs as usize),
            lut_f32,
            lut_sign,
        })
    }

    /// Bake the codec into persistable parameters.
    pub fn to_params(&self, calibration_id: [u8; 32]) -> UpqParams {
        UpqParams {
            dimension: self.dim as u32,
            num_pairs: self.num_pairs as u32,
            block_size: self.block_size as u32,
            rotation_seed: self.rotation_seed,
            cells: self.design.cells,
            design_source: self.design_source,
            renormalize_at_decode: self.renormalize_at_decode,
            ring_radii: self.design.ring_radii.clone(),
            ring_phases: self.design.ring_phases.clone(),
            sigma_per_pair: self.sigma_per_pair.clone(),
            calibration_id,
        }
    }

    /// Forward-rotate a query into the codec's space; returns the
    /// rotated vector and its L2 norm.
    pub fn rotate_query(&self, query: &[f32]) -> Result<(Vec<f32>, f32)> {
        if query.len() != self.dim {
            return Err(Error::Format(format!(
                "UpqCodec: query has {} dims, codec expects {}",
                query.len(),
                self.dim
            )));
        }
        let mut rot = query.to_vec();
        block_hadamard_forward(&mut rot, &self.signs, self.block_size)?;
        let norm = rot.iter().map(|x| x * x).sum::<f32>().sqrt();
        Ok((rot, norm))
    }

    /// Fit σ on a sample of **rotated** vectors; design rings per
    /// `source` (empirical sample vs analytic Rayleigh — the latter
    /// needs the sample only for σ, like the production QAM codec).
    /// Rotation fields default to the production rule; use
    /// [`Self::fit_from_sample`] for the unrotated production entry.
    pub fn fit_on_rotated(
        rotated_sample: &[Vec<f32>],
        dim: usize,
        ring_choices: &[usize],
        cells: u32,
        source: UpqDesignSource,
    ) -> Result<Self> {
        let block_size = default_block_size_for(dim);
        Self::validate_dims(dim, block_size, cells)?;
        if rotated_sample.is_empty() {
            return Err(Error::Format("UpqCodec: empty calibration sample".into()));
        }
        let num_pairs = dim / 2;
        let mut sums = vec![0.0_f64; num_pairs];
        for v in rotated_sample {
            if v.len() != dim {
                return Err(Error::Format("UpqCodec: sample dim mismatch".into()));
            }
            for (k, s) in sums.iter_mut().enumerate() {
                *s += (v[2 * k] as f64).hypot(v[2 * k + 1] as f64);
            }
        }
        let inv = 1.0 / (rotated_sample.len() as f64 * (std::f64::consts::PI / 2.0).sqrt());
        let sigma_per_pair: Vec<f32> = sums
            .iter()
            .map(|&s| ((s * inv) as f32).max(1e-12))
            .collect();

        let design = match source {
            UpqDesignSource::Rayleigh => UpqDesign::fit_sweep_rayleigh(ring_choices, cells)?,
            UpqDesignSource::Empirical => {
                let mut amps_norm = Vec::with_capacity(rotated_sample.len() * num_pairs);
                for v in rotated_sample {
                    for k in 0..num_pairs {
                        let a = v[2 * k].hypot(v[2 * k + 1]);
                        amps_norm.push(a / sigma_per_pair[k]);
                    }
                }
                UpqDesign::fit_sweep(&amps_norm, ring_choices, cells)?
            }
        };

        let (lut_f32, lut_sign) = build_tables(&design);
        Ok(Self {
            dim,
            num_pairs,
            block_size,
            rotation_seed: DEFAULT_UPQ_ROTATION_SEED,
            renormalize_at_decode: true,
            design_source: source,
            design,
            sigma_per_pair,
            signs: signs_from_seed(DEFAULT_UPQ_ROTATION_SEED, dim),
            base_bytes: packed11_stride(num_pairs),
            lut_f32,
            lut_sign,
        })
    }

    /// Encode a rotated vector into one u16 cell index per pair.
    pub fn encode_rotated(&self, rotated: &[f32], out: &mut [u16]) {
        debug_assert_eq!(rotated.len(), self.dim);
        debug_assert_eq!(out.len(), self.num_pairs);
        for (k, o) in out.iter_mut().enumerate() {
            let re = rotated[2 * k];
            let im = rotated[2 * k + 1];
            let a_norm = re.hypot(im) / self.sigma_per_pair[k];
            *o = self.design.encode_pair(a_norm, im.atan2(re));
        }
    }

    /// Decode codes into the σ-scaled, unit-renormalized rotated-domain
    /// reconstruction. Test-only reference (production decode is
    /// `decode_lossy` / `decode_dot_from_base`).
    #[cfg(test)]
    pub fn decode_codes(&self, codes: &[u16], out: &mut [f32]) {
        debug_assert_eq!(codes.len(), self.num_pairs);
        debug_assert_eq!(out.len(), self.dim);
        let mut norm_sq = 0.0_f32;
        for (k, &c) in codes.iter().enumerate() {
            let [re, im] = self.lut_f32[c as usize];
            let s = self.sigma_per_pair[k];
            out[2 * k] = re * s;
            out[2 * k + 1] = im * s;
            norm_sq += (re * s).mul_add(re * s, (im * s) * (im * s));
        }
        let inv = norm_sq.sqrt().max(1e-12).recip();
        for v in out.iter_mut() {
            *v *= inv;
        }
    }

    /// 1-bit/dim sign sketch derived from codes. Test-only reference
    /// (production derives sketches from packed bytes via `sign_sketch`).
    #[cfg(test)]
    pub fn sketch_from_codes(&self, codes: &[u16], out: &mut [u64]) {
        debug_assert_eq!(out.len(), self.dim.div_ceil(64));
        out.fill(0);
        for (k, &c) in codes.iter().enumerate() {
            let s = self.lut_sign[c as usize] as u64;
            let bit = 2 * k;
            out[bit / 64] |= (s & 1) << (bit % 64);
            let bit = 2 * k + 1;
            out[bit / 64] |= ((s >> 1) & 1) << (bit % 64);
        }
    }

    /// 1-bit/dim sign sketch of a rotated f32 vector. Test-only
    /// reference (the query side uses `retrieval::sketch::pack_query_sketch`).
    #[cfg(test)]
    pub fn sketch_from_rotated(rotated: &[f32], out: &mut [u64]) {
        out.fill(0);
        for (i, &v) in rotated.iter().enumerate() {
            if v >= 0.0 {
                out[i / 64] |= 1 << (i % 64);
            }
        }
    }

    /// K-B query prep: plain i8 with one global max-abs scale (rows
    /// already carry σ and the row norm).
    pub fn prepare_query_i8(rotated_q: &[f32], out: &mut [i8]) {
        let max_abs = rotated_q.iter().fold(0.0_f32, |m, &v| m.max(v.abs()));
        let s = 127.0 / max_abs.max(1e-12);
        for (o, &v) in out.iter_mut().zip(rotated_q.iter()) {
            *o = (v * s).round().clamp(-127.0, 127.0) as i8;
        }
    }

    /// Extract + bounds-check the `k`-th cell index from a packed row.
    #[inline]
    fn cell_at(&self, base: &[u8], k: usize) -> Result<u16> {
        let cell = extract11(base, k);
        if (cell as u32) < self.design.cells {
            Ok(cell)
        } else {
            Err(Error::Integrity(format!(
                "UpqCodec: pair {k} decodes to cell {cell} >= cells {}",
                self.design.cells
            )))
        }
    }

    fn check_base_len(&self, base: &[u8]) -> Result<()> {
        if base.len() < self.base_bytes {
            return Err(Error::Format(format!(
                "UpqCodec: base record is {} bytes, expected {}",
                base.len(),
                self.base_bytes
            )));
        }
        Ok(())
    }

    /// 1-bit/dim sign sketch derived from a stored packed row
    /// (file-open path; nothing persisted).
    pub fn sign_sketch(&self, base: &[u8]) -> Result<Vec<u64>> {
        self.check_base_len(base)?;
        let mut out = vec![0u64; self.dim.div_ceil(64)];
        for k in 0..self.num_pairs {
            let s = self.lut_sign[self.cell_at(base, k)? as usize] as u64;
            let bit = 2 * k;
            out[bit / 64] |= (s & 1) << (bit % 64);
            let bit = 2 * k + 1;
            out[bit / 64] |= ((s >> 1) & 1) << (bit % 64);
        }
        Ok(out)
    }

    /// Build a decoded-i8 cache row from a stored packed row: the
    /// unit-norm rotated-domain reconstruction quantized to i8 with a
    /// per-row max-abs scale. Returns the dequant factor; `score =
    /// dot_i8(q_i8, row) × factor` ranks like cosine.
    pub fn i8_row_from_base(
        &self,
        base: &[u8],
        scratch: &mut [f32],
        out: &mut [i8],
    ) -> Result<f32> {
        self.check_base_len(base)?;
        debug_assert_eq!(scratch.len(), self.dim);
        debug_assert_eq!(out.len(), self.dim);
        let mut norm_sq = 0.0_f32;
        for k in 0..self.num_pairs {
            let [re, im] = self.lut_f32[self.cell_at(base, k)? as usize];
            let s = self.sigma_per_pair[k];
            scratch[2 * k] = re * s;
            scratch[2 * k + 1] = im * s;
            norm_sq += (re * s).mul_add(re * s, (im * s) * (im * s));
        }
        let inv = norm_sq.sqrt().max(1e-12).recip();
        let mut max_abs = 0.0_f32;
        for v in scratch.iter_mut() {
            *v *= inv;
            max_abs = max_abs.max(v.abs());
        }
        let s = 127.0 / max_abs.max(1e-12);
        for (o, &v) in out.iter_mut().zip(scratch.iter()) {
            *o = (v * s).round().clamp(-127.0, 127.0) as i8;
        }
        Ok(1.0 / s)
    }

    /// Stage-3 exact scoring straight off a packed row: cosine
    /// **distance** (1 − cos) of the rotated query vs the σ-scaled
    /// reconstruction. `q_norm` is the rotated query's L2 norm.
    pub fn decode_dot_from_base(&self, base: &[u8], q_rot: &[f32], q_norm: f32) -> Result<f32> {
        self.check_base_len(base)?;
        debug_assert_eq!(q_rot.len(), self.dim);
        let mut dot = 0.0_f32;
        let mut norm_sq = 0.0_f32;
        for k in 0..self.num_pairs {
            let [re, im] = self.lut_f32[self.cell_at(base, k)? as usize];
            let s = self.sigma_per_pair[k];
            let (re, im) = (re * s, im * s);
            dot += re.mul_add(q_rot[2 * k], im * q_rot[2 * k + 1]);
            norm_sq += re.mul_add(re, im * im);
        }
        let denom = (norm_sq.sqrt() * q_norm).max(1e-12);
        Ok(1.0 - dot / denom)
    }
}

impl super::VectorCodec for UpqCodec {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn family(&self) -> crate::format::catalog::CodecFamily {
        crate::format::catalog::CodecFamily::Upq
    }

    fn dimension(&self) -> u32 {
        self.dim as u32
    }

    fn sign_sketch(&self, base: &[u8]) -> Result<Vec<u64>> {
        UpqCodec::sign_sketch(self, base)
    }

    fn base_bytes_per_vector(&self) -> usize {
        self.base_bytes
    }

    fn encode(&self, values: &[f32]) -> Result<Vec<u8>> {
        let (rot, _) = self.rotate_query(values)?;
        let mut codes = vec![0u16; self.num_pairs];
        self.encode_rotated(&rot, &mut codes);
        let mut out = vec![0u8; self.base_bytes];
        pack11(&codes, &mut out[..packed11_len(self.num_pairs)]);
        Ok(out)
    }

    fn decode_lossy(&self, base: &[u8]) -> Result<Vec<f32>> {
        self.check_base_len(base)?;
        let mut x = vec![0.0_f32; self.dim];
        for k in 0..self.num_pairs {
            let [re, im] = self.lut_f32[self.cell_at(base, k)? as usize];
            let s = self.sigma_per_pair[k];
            x[2 * k] = re * s;
            x[2 * k + 1] = im * s;
        }
        block_hadamard_inverse(&mut x, &self.signs, self.block_size)?;
        if self.renormalize_at_decode {
            let norm = x.iter().map(|v| v * v).sum::<f32>().sqrt();
            if norm > 1e-12 {
                let inv = 1.0 / norm;
                for v in x.iter_mut() {
                    *v *= inv;
                }
            }
        }
        Ok(x)
    }

    fn asymmetric_distance(&self, query: &[f32], base: &[u8], metric: VectorMetric) -> Result<f32> {
        let ctx = <Self as super::VectorCodec>::prepare_query(self, query)?;
        self.asymmetric_distance_prepared(ctx.as_ref(), base, metric)
    }

    fn prepare_query(&self, query: &[f32]) -> Result<Box<dyn std::any::Any + Send + Sync>> {
        let (q_rot, q_norm) = self.rotate_query(query)?;
        Ok(Box::new(UpqPreparedQuery { q_rot, q_norm }))
    }

    fn asymmetric_distance_prepared(
        &self,
        ctx: &(dyn std::any::Any + Send + Sync),
        base: &[u8],
        metric: VectorMetric,
    ) -> Result<f32> {
        let prep = ctx.downcast_ref::<UpqPreparedQuery>().ok_or_else(|| {
            Error::Integrity("UpqCodec: prepared ctx is not UpqPreparedQuery".into())
        })?;
        self.check_base_len(base)?;
        let mut dot = 0.0_f32;
        let mut norm_sq = 0.0_f32;
        for k in 0..self.num_pairs {
            let [re, im] = self.lut_f32[self.cell_at(base, k)? as usize];
            let s = self.sigma_per_pair[k];
            let (re, im) = (re * s, im * s);
            dot += re.mul_add(prep.q_rot[2 * k], im * prep.q_rot[2 * k + 1]);
            norm_sq += re.mul_add(re, im * im);
        }
        distance_from_parts(dot, norm_sq.sqrt(), prep.q_norm, metric)
    }
}

/// Distance ("smaller = more similar") from a dot product and the two
/// operand norms. Rotation is orthogonal, so rotated-domain values
/// equal original-domain values for all three metrics.
fn distance_from_parts(dot: f32, norm_a: f32, norm_b: f32, metric: VectorMetric) -> Result<f32> {
    match metric {
        VectorMetric::Cosine => Ok(1.0 - dot / (norm_a * norm_b).max(1e-12)),
        VectorMetric::InnerProduct => Ok(-dot),
        VectorMetric::L2 => Ok(norm_a.mul_add(norm_a, norm_b * norm_b) - 2.0 * dot),
    }
}

// ============================================================
// 11-bit packing (wire format / K-C input)
// ============================================================

/// Pack one u16-per-pair code row into contiguous 11-bit fields.
pub fn pack11(codes: &[u16], out: &mut [u8]) {
    debug_assert_eq!(out.len(), packed11_len(codes.len()));
    out.fill(0);
    for (k, &c) in codes.iter().enumerate() {
        let bit = k * 11;
        let (byte, off) = (bit / 8, bit % 8);
        let v = (c as u32) << off;
        out[byte] |= v as u8;
        out[byte + 1] |= (v >> 8) as u8;
        if off > 5 {
            out[byte + 2] |= (v >> 16) as u8;
        }
    }
}

/// Packed length in bytes for `num_pairs` 11-bit fields.
pub fn packed11_len(num_pairs: usize) -> usize {
    (num_pairs * 11).div_ceil(8)
}

/// Extract the `k`-th 11-bit field.
#[inline(always)]
pub fn extract11(packed: &[u8], k: usize) -> u16 {
    let bit = k * 11;
    let (byte, off) = (bit / 8, bit % 8);
    // Rows are padded so a u32 load never crosses the row end.
    let w = u32::from_le_bytes([
        packed[byte],
        packed[byte + 1],
        packed[byte + 2],
        packed[byte + 3],
    ]);
    ((w >> off) & 0x7FF) as u16
}

/// Row stride for packed storage: packed length padded so `extract11`'s
/// u32 load stays in-row, rounded to 16 B.
pub fn packed11_stride(num_pairs: usize) -> usize {
    (packed11_len(num_pairs) + 4).next_multiple_of(16)
}

// ============================================================
// Kernels
// ============================================================

/// Contiguous i8·i8 dot. The indexed-loop shape compiles to a
/// 4-accumulator SDOT kernel on aarch64 (verified rustc 1.95); do not
/// rewrite as iterator chains — that shape does not vectorize.
#[inline]
pub fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let mut acc = 0i32;
    for i in 0..n {
        acc += a[i] as i32 * b[i] as i32;
    }
    acc
}

// ============================================================
// tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic unit-Rayleigh sample via inverse CDF.
    fn rayleigh_sample(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let u = (i as f64 + 0.5) / n as f64;
                (-2.0 * (1.0 - u).ln()).sqrt() as f32
            })
            .collect()
    }

    #[test]
    fn fit_beats_restricted_polar_on_rayleigh() {
        let sample = rayleigh_sample(200_000);
        let d = UpqDesign::fit_sweep(&sample, &[24, 32], 2048).expect("fit");
        // Restricted (5,6) sits at ~2.48e-3 per pair on unit Rayleigh;
        // UPQ must land clearly below (theory: ~2.0e-3, +0.8-1.0 dB).
        assert!(
            d.design_cost < 2.3e-3,
            "design cost {} not better than restricted polar",
            d.design_cost
        );
        let used: u32 = d.ring_phases.iter().sum();
        assert!(used <= 2048);
        for &p in &d.ring_phases {
            assert!(p.is_power_of_two() && p >= MIN_RING_PHASES);
        }
        for w in d.ring_radii.windows(2) {
            assert!(w[0] < w[1], "radii not ascending");
        }
        assert_eq!(d.ring_boundaries.len(), d.ring_radii.len() - 1);
    }

    fn test_codec() -> UpqCodec {
        // Synthetic rotated sample: deterministic pseudo-Gaussian pairs
        // with per-pair scale spread (mimics the 4.5x sigma spread).
        let dim = 32;
        let rows = 512;
        let mut sample = Vec::with_capacity(rows);
        let mut state = 0x12345678_u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f64 / (1u64 << 31) as f64 - 1.0) as f32
        };
        for _ in 0..rows {
            let mut v = Vec::with_capacity(dim);
            for k in 0..dim {
                let scale = 1.0 + (k / 2) as f32 * 0.2;
                // sum of 4 uniforms ≈ gaussian-ish; adequate for tests
                v.push(scale * (next() + next() + next() + next()) * 0.5);
            }
            sample.push(v);
        }
        UpqCodec::fit_on_rotated(&sample, dim, &[8, 12], 256, UpqDesignSource::Empirical)
            .expect("codec fit")
    }

    #[test]
    fn boundary_assignment_matches_brute_argmin() {
        let sample = rayleigh_sample(100_000);
        let d = UpqDesign::fit(&sample, 24, 2048).expect("fit");
        let coeffs: Vec<f64> = d.ring_phases.iter().map(|&p| tangential_coeff(p)).collect();
        let radii: Vec<f64> = d.ring_radii.iter().map(|&r| r as f64).collect();
        let mut mismatches = 0usize;
        for i in (0..sample.len()).step_by(7) {
            let a = sample[i];
            let by_bounds = d.ring_boundaries.partition_point(|&b| a > b);
            let mut bi = 0usize;
            let mut bc = f64::INFINITY;
            for (j, (&r, &c)) in radii.iter().zip(&coeffs).enumerate() {
                let cost = ring_cost(a as f64, r, c);
                if cost < bc {
                    bc = cost;
                    bi = j;
                }
            }
            if by_bounds != bi {
                mismatches += 1;
            }
        }
        let checked = sample.len().div_ceil(7);
        assert!(
            (mismatches as f64) < 0.001 * checked as f64,
            "boundary vs argmin mismatch on {mismatches}/{checked} samples"
        );
    }

    #[test]
    fn rayleigh_design_matches_empirical_fit_on_rayleigh_data() {
        // On Rayleigh-distributed data, the data-free analytic design
        // must be as good as fitting on the sample itself.
        let sample = rayleigh_sample(200_000);
        let emp = UpqDesign::fit_sweep(&sample, &[24, 32], 2048).expect("emp fit");
        let ray = UpqDesign::fit_sweep_rayleigh(&[24, 32], 2048).expect("ray fit");
        let c_emp = emp.mean_cost_on(&sample);
        let c_ray = ray.mean_cost_on(&sample);
        let gap_db = 10.0 * (c_ray / c_emp).log10();
        assert!(
            gap_db.abs() < 0.05,
            "Rayleigh design off empirical by {gap_db:.3} dB (emp {c_emp:.3e}, ray {c_ray:.3e})"
        );
        let used: u32 = ray.ring_phases.iter().sum();
        assert!(used <= 2048);
        for &p in &ray.ring_phases {
            assert!(p.is_power_of_two() && p >= MIN_RING_PHASES);
        }
    }

    #[test]
    fn encode_decode_roundtrip_and_sketch() {
        let codec = test_codec();
        let v: Vec<f32> = (0..codec.dim)
            .map(|i| ((i * 37 + 11) % 17) as f32 / 17.0 - 0.45)
            .collect();
        let mut codes = vec![0u16; codec.num_pairs];
        codec.encode_rotated(&v, &mut codes);
        for &c in &codes {
            assert!((c as u32) < codec.design.cells);
        }
        let mut recon = vec![0.0f32; codec.dim];
        codec.decode_codes(&codes, &mut recon);
        let norm: f32 = recon.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4, "decode not unit-norm: {norm}");

        // Sketch from codes must equal sketch of the reconstruction.
        let words = codec.dim.div_ceil(64);
        let mut s_codes = vec![0u64; words];
        let mut s_recon = vec![0u64; words];
        codec.sketch_from_codes(&codes, &mut s_codes);
        UpqCodec::sketch_from_rotated(&recon, &mut s_recon);
        assert_eq!(s_codes, s_recon);
    }

    #[test]
    fn pack11_extract11_roundtrip() {
        let codes: Vec<u16> = (0..384u16).map(|k| (k * 7 + 3) % 2048).collect();
        let mut packed = vec![0u8; packed11_stride(codes.len())];
        pack11(&codes, &mut packed[..packed11_len(codes.len())]);
        for (k, &c) in codes.iter().enumerate() {
            assert_eq!(extract11(&packed, k), c, "pair {k}");
        }
    }

    #[test]
    fn dot_i8_matches_reference() {
        let a: Vec<i8> = (0..768).map(|i| ((i * 31 + 7) % 255) as i8).collect();
        let b: Vec<i8> = (0..768).map(|i| ((i * 17 + 3) % 251) as i8).collect();
        let reference: i64 = a.iter().zip(&b).map(|(&x, &y)| x as i64 * y as i64).sum();
        assert_eq!(dot_i8(&a, &b) as i64, reference);
    }
}
