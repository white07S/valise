// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Harness-only UPQ progressive-cache experiment.
//!
//! Compares the current decoded-i8 stage-2 cache against a one-byte-per-
//! complex-pair coarse cache while keeping the original packed 11-bit cells
//! for exact stage 3. This does not change the Valise wire format or production
//! search path.

#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::collections::HashSet;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, ensure};
use clap::{Parser, ValueEnum};
use memmap2::MmapOptions;
use rayon::prelude::*;
use serde::Serialize;
use valise::{
    UpqCodec, UpqDesignSource, dot_i8, extract11, pack11, packed11_len, packed11_stride,
    sketch_bench,
};

const CELLS: u32 = 2048;
const CALIB_ROWS: usize = 4096;
const RING_SWEEP: &[usize] = &[24, 32, 40, 48];
const ROTATION_SEED: u64 = 0x6C61_6D61_5F71_616D;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Variant {
    Before,
    CoarseInt4,
    CoarseLut,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Mapping {
    Kd,
    Capacity,
}

impl Mapping {
    fn label(self) -> &'static str {
        match self {
            Self::Kd => "kd",
            Self::Capacity => "capacity",
        }
    }
}

impl Variant {
    fn label(self) -> &'static str {
        match self {
            Self::Before => "before-decoded-i8",
            Self::CoarseInt4 => "after-coarse-int4",
            Self::CoarseLut => "after-coarse-lut",
        }
    }
}

#[derive(Debug, Parser)]
#[command(about = "UPQ 8-bit coarse + 3-bit residual cache experiment")]
struct Cli {
    #[arg(long, default_value = "bench/datasets/cohere-medium-1m-f32")]
    vector_dir: PathBuf,
    #[arg(long, value_enum)]
    variant: Variant,
    #[arg(long, value_enum, default_value = "kd")]
    mapping: Mapping,
    #[arg(long, default_value_t = 768)]
    dim: usize,
    #[arg(long, default_value_t = 100_000)]
    n: usize,
    #[arg(long, default_value_t = 200)]
    nq: usize,
    #[arg(long, default_value_t = 100)]
    k: usize,
    #[arg(long, default_value_t = 2048)]
    channel_k: usize,
    #[arg(long, default_value_t = 3)]
    trials: usize,
    #[arg(long, default_value_t = 20)]
    warmup: usize,
    #[arg(long, value_delimiter = ',', default_value = "300")]
    pools: Vec<usize>,
    #[arg(long)]
    validate_only: bool,
    #[arg(long)]
    json: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug)]
struct Cell {
    id: u16,
    xy: [f32; 2],
    sign: u8,
    weight: f64,
}

#[derive(Clone, Debug)]
struct CoarseMap {
    cell_to_byte: Vec<u8>,
    cell_to_residual: Vec<u8>,
    byte_residual_to_cell: Vec<[u16; 8]>,
    lut_i8: Vec<[i8; 2]>,
    direct_xy: Vec<[i8; 2]>,
    exact_xy: Vec<[f32; 2]>,
    assignment_scale: f64,
    weighted_mse: f64,
}

#[derive(Serialize)]
struct QualityRow {
    pool: usize,
    oracle_coverage_at_10: f64,
    oracle_coverage_at_100: f64,
    final_oracle_overlap_at_10: f64,
    final_oracle_overlap_at_100: f64,
    recall_at_10: f64,
    recall_at_100: f64,
}

#[derive(Default)]
struct QualitySum {
    oracle_coverage_at_10: f64,
    oracle_coverage_at_100: f64,
    final_oracle_overlap_at_10: f64,
    final_oracle_overlap_at_100: f64,
    recall_at_10: f64,
    recall_at_100: f64,
}

#[derive(Serialize)]
struct TimingRow {
    pool: usize,
    cpu_us_per_query: f64,
    stage1_p50_us: f64,
    stage1_p95_us: f64,
    stage2_p50_us: f64,
    stage2_p95_us: f64,
    stage3_p50_us: f64,
    stage3_p95_us: f64,
    harness_total_p50_us: f64,
    harness_total_p95_us: f64,
}

#[derive(Serialize)]
struct Report {
    variant: String,
    mapping: String,
    dataset: String,
    n: usize,
    nq: usize,
    dim: usize,
    k: usize,
    channel_k: usize,
    trials: usize,
    exact_roundtrip_cells: usize,
    exact_sign_cells: usize,
    assignment_scale: f64,
    grouping_weighted_mse: f64,
    packed_logical_bytes_per_vector: usize,
    packed_stride_bytes_per_vector: usize,
    stage2_cache_bytes_per_vector: usize,
    sketch_bytes_per_vector: usize,
    norm_bytes_per_vector: usize,
    derived_cache_bytes_per_vector: usize,
    derived_cache_plus_packed_stride_bytes_per_vector: usize,
    steady_rss_bytes: u64,
    process_lifetime_high_water_bytes: u64,
    cpu_us_per_query: f64,
    quality: Vec<QualityRow>,
    timing: Vec<TimingRow>,
    note: String,
}

enum Cache {
    Before {
        rows: Vec<i8>,
        dequant: Vec<f32>,
    },
    Coarse {
        rows: Vec<u8>,
        inv_norm: Vec<f32>,
        xy: Vec<[i8; 2]>,
        direct_int4: bool,
    },
}

fn block_size_for_dim(dim: usize) -> Result<usize> {
    ensure!(
        dim > 0 && dim.is_multiple_of(2),
        "dim must be positive and even"
    );
    Ok((1usize << dim.trailing_zeros()).min(1024))
}

fn map_f32_file(path: &Path, required_floats: usize) -> Result<memmap2::Mmap> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let need = required_floats
        .checked_mul(4)
        .ok_or_else(|| anyhow!("mapped byte length overflow"))?;
    ensure!(
        file.metadata()?.len() >= need as u64,
        "{} is too short",
        path.display()
    );
    // SAFETY: the file remains open for the duration of this call and the
    // returned read-only mapping owns the mapping lifetime.
    let mmap = unsafe { MmapOptions::new().len(need).map(&file)? };
    Ok(mmap)
}

fn as_f32(mmap: &[u8]) -> Result<&[f32]> {
    bytemuck::try_cast_slice(mmap).map_err(|e| anyhow!("unaligned f32 dataset: {e}"))
}

fn load_queries(path: &Path, floats: usize) -> Result<Vec<f32>> {
    let mmap = map_f32_file(path, floats)?;
    Ok(as_f32(&mmap)?.to_vec())
}

fn gt_path(vector_dir: &Path, n: usize, nq: usize) -> Result<PathBuf> {
    let name = vector_dir
        .file_name()
        .and_then(|v| v.to_str())
        .ok_or_else(|| anyhow!("dataset directory has no UTF-8 basename"))?;
    let bench_dir = vector_dir
        .parent()
        .and_then(Path::parent)
        .ok_or_else(|| anyhow!("dataset must be under bench/datasets/<name>"))?;
    let exact = bench_dir
        .join("cache")
        .join(format!("gt_{name}_n{n}_nq{nq}_cosine_k100.u32"));
    if exact.exists() {
        return Ok(exact);
    }
    let superset = bench_dir
        .join("cache")
        .join(format!("gt_{name}_n{n}_nq1000_cosine_k100.u32"));
    ensure!(
        nq <= 1000 && superset.exists(),
        "no exact GT cache at {} and no nq=1000 superset",
        exact.display()
    );
    Ok(superset)
}

fn load_gt(path: &Path, nq: usize) -> Result<Vec<u32>> {
    let bytes = fs::read(path).with_context(|| format!("read GT {}", path.display()))?;
    ensure!(
        bytes.len() >= nq * 100 * 4,
        "GT must contain at least nq x 100 u32 values"
    );
    Ok(bytes[..nq * 100 * 4]
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

fn cell_table(codec: &UpqCodec, counts: &[u64]) -> Result<Vec<Cell>> {
    let mut cells = Vec::with_capacity(CELLS as usize);
    for (&radius, (&phases, &base)) in codec.design.ring_radii.iter().zip(
        codec
            .design
            .ring_phases
            .iter()
            .zip(&codec.design.ring_bases),
    ) {
        for m in 0..phases {
            let theta = m as f32 * (std::f32::consts::TAU / phases as f32);
            let (sin, cos) = theta.sin_cos();
            let xy = [radius * cos, radius * sin];
            let sign = u8::from(xy[0] >= 0.0) | (u8::from(xy[1] >= 0.0) << 1);
            let id = (base + m) as u16;
            cells.push(Cell {
                id,
                xy,
                sign,
                weight: counts[id as usize] as f64 + 0.5,
            });
        }
    }
    cells.sort_unstable_by_key(|c| c.id);
    ensure!(
        cells.len() == CELLS as usize,
        "ring phases use {} cells, not {CELLS}",
        cells.len()
    );
    for (id, cell) in cells.iter().enumerate() {
        ensure!(cell.id as usize == id, "UPQ cell ids are not dense at {id}");
    }
    Ok(cells)
}

fn weighted_centroid(ids: &[usize], cells: &[Cell]) -> ([f64; 2], f64) {
    let mut sum = [0.0; 2];
    let mut weight = 0.0;
    for &id in ids {
        let w = cells[id].weight;
        sum[0] += w * cells[id].xy[0] as f64;
        sum[1] += w * cells[id].xy[1] as f64;
        weight += w;
    }
    ([sum[0] / weight, sum[1] / weight], weight)
}

fn kd_groups(ids: &mut [usize], cells: &[Cell], output: &mut Vec<Vec<usize>>) {
    if ids.len() == 8 {
        ids.sort_unstable();
        output.push(ids.to_vec());
        return;
    }

    let mut low = [f32::INFINITY; 2];
    let mut high = [f32::NEG_INFINITY; 2];
    for &id in ids.iter() {
        for axis in 0..2 {
            low[axis] = low[axis].min(cells[id].xy[axis]);
            high[axis] = high[axis].max(cells[id].xy[axis]);
        }
    }
    let axis = usize::from(high[1] - low[1] > high[0] - low[0]);
    let other = 1 - axis;
    ids.sort_unstable_by(|&left, &right| {
        cells[left].xy[axis]
            .total_cmp(&cells[right].xy[axis])
            .then_with(|| cells[left].xy[other].total_cmp(&cells[right].xy[other]))
            .then_with(|| left.cmp(&right))
    });
    let middle = ids.len() / 2;
    let (left, right) = ids.split_at_mut(middle);
    kd_groups(left, cells, output);
    kd_groups(right, cells, output);
}

/// Deterministic square Hungarian assignment. `assignment[row] = column`.
fn hungarian(cost: &[Vec<f64>]) -> Vec<usize> {
    let n = cost.len();
    debug_assert!(n > 0 && cost.iter().all(|row| row.len() == n));
    let mut u = vec![0.0f64; n + 1];
    let mut v = vec![0.0f64; n + 1];
    let mut p = vec![0usize; n + 1];
    let mut way = vec![0usize; n + 1];

    for i in 1..=n {
        p[0] = i;
        let mut j0 = 0usize;
        let mut minv = vec![f64::INFINITY; n + 1];
        let mut used = vec![false; n + 1];
        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = f64::INFINITY;
            let mut j1 = 0usize;
            for j in 1..=n {
                if used[j] {
                    continue;
                }
                let current = cost[i0 - 1][j - 1] - u[i0] - v[j];
                if current < minv[j] {
                    minv[j] = current;
                    way[j] = j0;
                }
                if minv[j] < delta || (minv[j] == delta && j < j1) {
                    delta = minv[j];
                    j1 = j;
                }
            }
            for j in 0..=n {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }

    let mut assignment = vec![usize::MAX; n];
    for j in 1..=n {
        assignment[p[j] - 1] = j - 1;
    }
    assignment
}

fn int4_coordinate(sign: u8, axis: usize, slot: usize) -> i8 {
    let positive = sign & (1 << axis) != 0;
    if positive { slot as i8 } else { slot as i8 - 8 }
}

fn grid_xy(sign: u8, slot: usize) -> [i8; 2] {
    [
        int4_coordinate(sign, 0, slot / 8),
        int4_coordinate(sign, 1, slot % 8),
    ]
}

fn grid_byte(sign: u8, slot: usize) -> u8 {
    let [re, im] = grid_xy(sign, slot);
    (((re as u8) & 0x0f) << 4) | ((im as u8) & 0x0f)
}

fn byte_sign(byte: u8) -> u8 {
    u8::from(byte & 0x80 == 0) | (u8::from(byte & 0x08 == 0) << 1)
}

fn byte_xy(byte: u8) -> [i8; 2] {
    [(byte as i8) >> 4, ((byte << 4) as i8) >> 4]
}

/// Capacity-constrained nearest-grid assignment: every one of the 64
/// quadrant-local int4 points receives exactly eight exact UPQ cells.
/// Assign the most constrained cell first (largest nearest-vs-second-
/// nearest regret), which avoids the spatial cuts imposed by the earlier
/// kd-tree prototype while remaining deterministic and inexpensive.
fn balanced_grid_assign(ids: &[usize], cells: &[Cell], sign: u8, scale: f64) -> Vec<Vec<usize>> {
    let mut groups: Vec<Vec<usize>> = (0..64).map(|_| Vec::with_capacity(8)).collect();
    let mut remaining = ids.to_vec();
    while !remaining.is_empty() {
        let mut chosen_index = 0usize;
        let mut chosen_slot = 0usize;
        let mut chosen_regret = f64::NEG_INFINITY;
        for (index, &cell_id) in remaining.iter().enumerate() {
            let cell = cells[cell_id];
            let mut best = (f64::INFINITY, 0usize);
            let mut second = f64::INFINITY;
            for (slot, group) in groups.iter().enumerate() {
                if group.len() == 8 {
                    continue;
                }
                let xy = grid_xy(sign, slot);
                let dx = cell.xy[0] as f64 - scale * xy[0] as f64;
                let dy = cell.xy[1] as f64 - scale * xy[1] as f64;
                let cost = cell.weight * dx.mul_add(dx, dy * dy);
                if cost < best.0 {
                    second = best.0;
                    best = (cost, slot);
                } else if cost < second {
                    second = cost;
                }
            }
            let regret = second - best.0;
            if regret > chosen_regret
                || (regret == chosen_regret && cell_id < remaining[chosen_index])
            {
                chosen_index = index;
                chosen_slot = best.1;
                chosen_regret = regret;
            }
        }
        let cell_id = remaining.swap_remove(chosen_index);
        groups[chosen_slot].push(cell_id);
    }
    for group in &mut groups {
        group.sort_unstable();
        debug_assert_eq!(group.len(), 8);
    }
    groups
}

fn kd_grid_groups(
    ids_by_sign: &[Vec<usize>],
    cells: &[Cell],
    initial_scale: f64,
) -> (Vec<Vec<Vec<usize>>>, f64) {
    let mut raw_groups: Vec<Vec<Vec<usize>>> = Vec::with_capacity(4);
    for ids in ids_by_sign {
        let mut ids = ids.clone();
        let mut groups = Vec::with_capacity(64);
        kd_groups(&mut ids, cells, &mut groups);
        debug_assert_eq!(groups.len(), 64);
        raw_groups.push(groups);
    }

    let mut scale = initial_scale;
    let mut assignments = vec![vec![0usize; 64]; 4];
    for _ in 0..12 {
        for sign in 0..4u8 {
            let cost: Vec<Vec<f64>> = raw_groups[sign as usize]
                .iter()
                .map(|group| {
                    let (centroid, weight) = weighted_centroid(group, cells);
                    (0..64)
                        .map(|slot| {
                            let xy = grid_xy(sign, slot);
                            let dx = centroid[0] - scale * xy[0] as f64;
                            let dy = centroid[1] - scale * xy[1] as f64;
                            weight * dx.mul_add(dx, dy * dy)
                        })
                        .collect()
                })
                .collect();
            assignments[sign as usize] = hungarian(&cost);
        }

        let mut numerator = 0.0;
        let mut denominator = 0.0;
        for sign in 0..4u8 {
            for (group, &slot) in raw_groups[sign as usize]
                .iter()
                .zip(&assignments[sign as usize])
            {
                let (centroid, weight) = weighted_centroid(group, cells);
                let xy = grid_xy(sign, slot);
                numerator += weight * (centroid[0] * xy[0] as f64 + centroid[1] * xy[1] as f64);
                denominator += weight * (i32::from(xy[0]).pow(2) + i32::from(xy[1]).pow(2)) as f64;
            }
        }
        let next = numerator / denominator.max(1e-12);
        if (next - scale).abs() < 1e-10 {
            scale = next;
            break;
        }
        scale = next;
    }

    // Make the returned assignment correspond to the final fitted scale.
    for sign in 0..4u8 {
        let cost: Vec<Vec<f64>> = raw_groups[sign as usize]
            .iter()
            .map(|group| {
                let (centroid, weight) = weighted_centroid(group, cells);
                (0..64)
                    .map(|slot| {
                        let xy = grid_xy(sign, slot);
                        let dx = centroid[0] - scale * xy[0] as f64;
                        let dy = centroid[1] - scale * xy[1] as f64;
                        weight * dx.mul_add(dx, dy * dy)
                    })
                    .collect()
            })
            .collect();
        assignments[sign as usize] = hungarian(&cost);
    }

    let mut groups_by_sign: Vec<Vec<Vec<usize>>> = (0..4)
        .map(|_| (0..64).map(|_| Vec::new()).collect())
        .collect();
    for sign in 0..4 {
        for (group, &slot) in raw_groups[sign].iter().zip(&assignments[sign]) {
            groups_by_sign[sign][slot] = group.clone();
        }
    }
    (groups_by_sign, scale)
}

fn capacity_grid_groups(
    ids_by_sign: &[Vec<usize>],
    cells: &[Cell],
    initial_scale: f64,
) -> (Vec<Vec<Vec<usize>>>, f64) {
    let mut scale = initial_scale;
    let mut groups_by_sign: Vec<Vec<Vec<usize>>> = vec![Vec::new(); 4];
    for _ in 0..10 {
        for sign in 0..4u8 {
            groups_by_sign[sign as usize] =
                balanced_grid_assign(&ids_by_sign[sign as usize], cells, sign, scale);
        }
        let mut numerator = 0.0;
        let mut denominator = 0.0;
        for sign in 0..4u8 {
            for (slot, group) in groups_by_sign[sign as usize].iter().enumerate() {
                let xy = grid_xy(sign, slot);
                for &cell_id in group {
                    let cell = cells[cell_id];
                    numerator += cell.weight
                        * (cell.xy[0] as f64 * xy[0] as f64 + cell.xy[1] as f64 * xy[1] as f64);
                    denominator +=
                        cell.weight * (i32::from(xy[0]).pow(2) + i32::from(xy[1]).pow(2)) as f64;
                }
            }
        }
        let next = numerator / denominator.max(1e-12);
        if (next - scale).abs() < 1e-10 {
            scale = next;
            break;
        }
        scale = next;
    }
    (groups_by_sign, scale)
}

fn validate_production_sign(codec: &UpqCodec, cells: &[Cell]) -> Result<usize> {
    let logical = packed11_len(codec.num_pairs);
    let stride = packed11_stride(codec.num_pairs);
    let mut codes = vec![0u16; codec.num_pairs];
    let mut row = vec![0u8; stride];
    let mut exact = 0usize;
    for cell in cells {
        codes.fill(cell.id);
        row.fill(0);
        pack11(&codes, &mut row[..logical]);
        let sketch = codec.sign_sketch(&row)?;
        for k in 0..codec.num_pairs {
            let re_bit = 2 * k;
            let im_bit = re_bit + 1;
            let observed = ((sketch[re_bit / 64] >> (re_bit % 64)) & 1) as u8
                | (((sketch[im_bit / 64] >> (im_bit % 64)) & 1) as u8) << 1;
            ensure!(
                observed == cell.sign,
                "production sign_sketch mismatch for cell {} pair {k}",
                cell.id
            );
        }
        exact += 1;
    }
    println!("[validate] production codec.sign_sketch exact sign {exact}/{CELLS}");
    Ok(exact)
}

fn build_coarse_map(
    codec: &UpqCodec,
    packed: &[u8],
    stride: usize,
    mapping: Mapping,
) -> Result<CoarseMap> {
    let mut counts = vec![0u64; CELLS as usize];
    for row in packed.chunks_exact(stride).take(CALIB_ROWS) {
        for k in 0..codec.num_pairs {
            counts[extract11(row, k) as usize] += 1;
        }
    }
    let cells = cell_table(codec, &counts)?;
    let mut ids_by_sign: Vec<Vec<usize>> = vec![Vec::new(); 4];
    for sign in 0..4u8 {
        ids_by_sign[sign as usize] = cells
            .iter()
            .filter(|c| c.sign == sign)
            .map(|c| c.id as usize)
            .collect();
        ensure!(
            ids_by_sign[sign as usize].len() == 512,
            "sign quadrant {sign} has {} cells, not 512",
            ids_by_sign[sign as usize].len()
        );
    }

    let production_sign_cells = validate_production_sign(codec, &cells)?;
    ensure!(
        production_sign_cells == CELLS as usize,
        "production sign validation was incomplete"
    );

    let max_radius = codec
        .design
        .ring_radii
        .iter()
        .copied()
        .fold(0.0f32, f32::max) as f64;
    let initial_scale = max_radius / 7.0;
    let (groups_by_sign, scale) = match mapping {
        Mapping::Kd => kd_grid_groups(&ids_by_sign, &cells, initial_scale),
        Mapping::Capacity => capacity_grid_groups(&ids_by_sign, &cells, initial_scale),
    };

    let mut cell_to_byte = vec![0u8; CELLS as usize];
    let mut cell_to_residual = vec![0u8; CELLS as usize];
    let mut inverse = vec![[u16::MAX; 8]; 256];
    let mut byte_centroid = vec![[0.0f64; 2]; 256];
    let mut weighted_error = 0.0;
    let mut total_weight = 0.0;
    for sign in 0..4u8 {
        for (slot, group) in groups_by_sign[sign as usize].iter().enumerate() {
            let byte = grid_byte(sign, slot);
            byte_centroid[byte as usize] = weighted_centroid(group, &cells).0;
            let xy = grid_xy(sign, slot);
            for (residual, &cell_id) in group.iter().enumerate() {
                let cell = cells[cell_id];
                cell_to_byte[cell_id] = byte;
                cell_to_residual[cell_id] = residual as u8;
                inverse[byte as usize][residual] = cell_id as u16;
                let dx = cell.xy[0] as f64 - scale * xy[0] as f64;
                let dy = cell.xy[1] as f64 - scale * xy[1] as f64;
                weighted_error += cell.weight * dx.mul_add(dx, dy * dy);
                total_weight += cell.weight;
            }
        }
    }

    let mut exact_roundtrip = 0usize;
    let mut exact_sign = 0usize;
    for cell in &cells {
        let id = cell.id as usize;
        let byte = cell_to_byte[id];
        let residual = cell_to_residual[id] as usize;
        ensure!(
            inverse[byte as usize][residual] == cell.id,
            "coarse+residual roundtrip failed for cell {id}"
        );
        ensure!(
            byte_sign(byte) == cell.sign,
            "coarse sign mismatch for cell {id}"
        );
        exact_roundtrip += 1;
        exact_sign += 1;
    }
    ensure!(
        inverse.iter().flatten().all(|&id| id != u16::MAX),
        "coarse inverse has holes"
    );
    println!(
        "[validate] exact coarse+residual roundtrip {exact_roundtrip}/{CELLS}; exact sign {exact_sign}/{CELLS}"
    );

    let max_abs = byte_centroid
        .iter()
        .flat_map(|xy| xy.iter())
        .map(|v| v.abs())
        .fold(0.0f64, f64::max);
    let quant = 127.0 / max_abs.max(1e-12);
    let lut_i8: Vec<[i8; 2]> = byte_centroid
        .iter()
        .map(|xy| {
            [
                (xy[0] * quant).round().clamp(-127.0, 127.0) as i8,
                (xy[1] * quant).round().clamp(-127.0, 127.0) as i8,
            ]
        })
        .collect();
    let direct_xy = (0..=u8::MAX).map(byte_xy).collect();

    Ok(CoarseMap {
        cell_to_byte,
        cell_to_residual,
        byte_residual_to_cell: inverse,
        lut_i8,
        direct_xy,
        exact_xy: cells.iter().map(|cell| cell.xy).collect(),
        assignment_scale: scale,
        weighted_mse: weighted_error / total_weight,
    })
}

fn build_sketches(
    codec: &UpqCodec,
    packed: &[u8],
    stride: usize,
    words: usize,
) -> Result<Vec<u64>> {
    let mut sketches = vec![0u64; packed.len() / stride * words];
    sketches.par_chunks_mut(words).enumerate().try_for_each(
        |(row, out)| -> valise::Result<()> {
            let base = &packed[row * stride..(row + 1) * stride];
            out.copy_from_slice(&codec.sign_sketch(base)?);
            Ok(())
        },
    )?;
    Ok(sketches)
}

fn build_before_cache(codec: &UpqCodec, packed: &[u8], stride: usize) -> Result<Cache> {
    let n = packed.len() / stride;
    let mut rows = vec![0i8; n * codec.dim];
    let mut dequant = vec![0f32; n];
    rows.par_chunks_mut(codec.dim)
        .zip(dequant.par_iter_mut())
        .enumerate()
        .try_for_each_init(
            || vec![0f32; codec.dim],
            |scratch, (row, (out, factor))| -> valise::Result<()> {
                let base = &packed[row * stride..(row + 1) * stride];
                *factor = codec.i8_row_from_base(base, scratch, out)?;
                Ok(())
            },
        )?;
    Ok(Cache::Before { rows, dequant })
}

fn build_coarse_cache(
    codec: &UpqCodec,
    packed: &[u8],
    stride: usize,
    map: &CoarseMap,
    variant: Variant,
) -> Result<Cache> {
    let n = packed.len() / stride;
    let mut rows = vec![0u8; n * codec.num_pairs];
    let mut inv_norm = vec![0f32; n];
    let xy = match variant {
        Variant::CoarseInt4 => map.direct_xy.clone(),
        Variant::CoarseLut => map.lut_i8.clone(),
        Variant::Before => return Err(anyhow!("before requested from coarse cache builder")),
    };
    rows.par_chunks_mut(codec.num_pairs)
        .zip(inv_norm.par_iter_mut())
        .enumerate()
        .for_each(|(row, (out, norm))| {
            let base = &packed[row * stride..(row + 1) * stride];
            let mut norm_sq = 0.0f32;
            for (k, target) in out.iter_mut().enumerate() {
                let cell = extract11(base, k) as usize;
                let byte = map.cell_to_byte[cell];
                *target = byte;
                // Keep the exact UPQ row norm. The coarse coordinate is
                // only a numerator approximation; replacing the known
                // denominator with a second approximation needlessly
                // compounds ranking error.
                let [re, im] = map.exact_xy[cell];
                let sigma = codec.sigma_per_pair[k];
                let re = re * sigma;
                let im = im * sigma;
                norm_sq += re.mul_add(re, im * im);
            }
            *norm = norm_sq.sqrt().max(1e-12).recip();
        });
    Ok(Cache::Coarse {
        rows,
        inv_norm,
        xy,
        direct_int4: matches!(variant, Variant::CoarseInt4),
    })
}

fn pack_query_sketch(rotated: &[f32], out: &mut [u64]) {
    out.fill(0);
    for (i, &value) in rotated.iter().enumerate() {
        if value >= 0.0 {
            out[i / 64] |= 1u64 << (i % 64);
        }
    }
}

fn scan_candidates(
    q: &[u64],
    sketches: &[u64],
    words: usize,
    ck: usize,
    dim: usize,
    hbuf: &mut Vec<u16>,
    hist: &mut Vec<u32>,
) -> Vec<u32> {
    let n = sketches.len() / words;
    hbuf.clear();
    hbuf.resize(n, 0);
    hist.clear();
    hist.resize(dim + 2, 0);
    for (row, h) in hbuf.iter_mut().enumerate() {
        *h = sketch_bench::hamming(q, &sketches[row * words..(row + 1) * words]) as u16;
        hist[*h as usize] += 1;
    }
    let ck = ck.min(n);
    let mut below = 0usize;
    let mut threshold = dim + 1;
    for (distance, &count) in hist.iter().enumerate() {
        if below + count as usize >= ck {
            threshold = distance;
            break;
        }
        below += count as usize;
    }
    let mut ties = ck.saturating_sub(below);
    let mut out = Vec::with_capacity(ck);
    for (row, &distance) in hbuf.iter().enumerate() {
        let distance = distance as usize;
        if distance < threshold || (distance == threshold && ties > 0) {
            out.push(row as u32);
            if distance == threshold {
                ties -= 1;
            }
        }
    }
    out
}

fn prepare_sigma_query(codec: &UpqCodec, q_rot: &[f32], out: &mut [i8]) {
    let mut max_abs = 0.0f32;
    for k in 0..codec.num_pairs {
        max_abs = max_abs.max((q_rot[2 * k] * codec.sigma_per_pair[k]).abs());
        max_abs = max_abs.max((q_rot[2 * k + 1] * codec.sigma_per_pair[k]).abs());
    }
    let scale = 127.0 / max_abs.max(1e-12);
    for k in 0..codec.num_pairs {
        out[2 * k] = (q_rot[2 * k] * codec.sigma_per_pair[k] * scale)
            .round()
            .clamp(-127.0, 127.0) as i8;
        out[2 * k + 1] = (q_rot[2 * k + 1] * codec.sigma_per_pair[k] * scale)
            .round()
            .clamp(-127.0, 127.0) as i8;
    }
}

#[inline]
fn dot_i4_bytes_scalar(query: &[i8], row: &[u8]) -> i32 {
    let mut dot = 0i32;
    for (k, &byte) in row.iter().enumerate() {
        let [re, im] = byte_xy(byte);
        dot += re as i32 * query[2 * k] as i32 + im as i32 * query[2 * k + 1] as i32;
    }
    dot
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "dotprod")]
unsafe fn sdot_16(
    acc: std::arch::aarch64::int32x4_t,
    a: std::arch::aarch64::int8x16_t,
    b: std::arch::aarch64::int8x16_t,
) -> std::arch::aarch64::int32x4_t {
    let mut out = acc;
    // Stable Rust does not yet expose vdotq_s32, so use the same
    // inline-assembly pattern as the production QAM sliding kernel.
    unsafe {
        std::arch::asm!(
            "sdot {0:v}.4s, {1:v}.16b, {2:v}.16b",
            inout(vreg) out,
            in(vreg) a,
            in(vreg) b,
            options(pure, nomem, nostack),
        );
    }
    out
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
unsafe fn dot_i4_bytes_neon(query: &[i8], row: &[u8]) -> i32 {
    use std::arch::aarch64::*;

    debug_assert_eq!(query.len(), row.len() * 2);
    debug_assert!(row.len().is_multiple_of(16));
    let mut acc = vdupq_n_s32(0);
    for k in (0..row.len()).step_by(16) {
        // SAFETY: the loop bounds guarantee 16 row bytes and 32 query
        // bytes are available at each iteration.
        let packed = unsafe { vld1q_u8(row.as_ptr().add(k)) };
        let packed_i8 = vreinterpretq_s8_u8(packed);
        // Native two's-complement int4: arithmetic shifts sign-extend
        // each nibble. This uses all 16 levels (no +/- zero duplicate).
        let re = vshrq_n_s8::<4>(packed_i8);
        let im = vshrq_n_s8::<4>(vshlq_n_s8::<4>(packed_i8));
        let db0 = vzip1q_s8(re, im);
        let db1 = vzip2q_s8(re, im);
        let q0 = unsafe { vld1q_s8(query.as_ptr().add(2 * k)) };
        let q1 = unsafe { vld1q_s8(query.as_ptr().add(2 * k + 16)) };
        acc = unsafe { sdot_16(acc, db0, q0) };
        acc = unsafe { sdot_16(acc, db1, q1) };
    }
    vaddvq_s32(acc)
}

#[inline]
fn dot_i4_bytes(query: &[i8], row: &[u8]) -> i32 {
    #[cfg(target_arch = "aarch64")]
    {
        // Apple Silicon supports the Armv8.2 dot-product extension.
        // SAFETY: the cache has 384-byte rows and the query has 768 bytes.
        return unsafe { dot_i4_bytes_neon(query, row) };
    }
    #[allow(unreachable_code)]
    dot_i4_bytes_scalar(query, row)
}

fn validate_i4_kernel(num_pairs: usize) -> Result<()> {
    const CASES: usize = 1024;
    let mut state = 0xD1B5_4A32_D192_ED03u64;
    let mut row = vec![0u8; num_pairs];
    let mut query = vec![0i8; 2 * num_pairs];
    let mut checksum = 0i64;
    for case in 0..CASES {
        for value in &mut row {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            *value = (state >> 24) as u8;
        }
        for value in &mut query {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            *value = (state >> 32) as i8;
        }
        let scalar = dot_i4_bytes_scalar(&query, &row);
        let measured = dot_i4_bytes(&query, &row);
        ensure!(
            measured == scalar,
            "int4 SIMD dot {measured} != scalar {scalar} in differential case {case}"
        );
        checksum = checksum.wrapping_add(i64::from(measured));
    }
    std::hint::black_box(checksum);
    println!("[validate] int4 scalar/SIMD differential {CASES}/{CASES} rows+queries");
    Ok(())
}

fn score_stage2(
    codec: &UpqCodec,
    cache: &Cache,
    q_rot: &[f32],
    candidates: &[u32],
    q_i8: &mut [i8],
) -> Vec<(f32, u32)> {
    match cache {
        Cache::Before { rows, dequant } => {
            UpqCodec::prepare_query_i8(q_rot, q_i8);
            candidates
                .iter()
                .map(|&id| {
                    let row = &rows[id as usize * codec.dim..(id as usize + 1) * codec.dim];
                    (dot_i8(q_i8, row) as f32 * dequant[id as usize], id)
                })
                .collect()
        }
        Cache::Coarse {
            rows,
            inv_norm,
            xy,
            direct_int4,
        } => {
            prepare_sigma_query(codec, q_rot, q_i8);
            candidates
                .iter()
                .map(|&id| {
                    let row =
                        &rows[id as usize * codec.num_pairs..(id as usize + 1) * codec.num_pairs];
                    let dot = if *direct_int4 {
                        dot_i4_bytes(q_i8, row)
                    } else {
                        row.iter().enumerate().fold(0i32, |dot, (k, &byte)| {
                            let [re, im] = xy[byte as usize];
                            dot + re as i32 * q_i8[2 * k] as i32
                                + im as i32 * q_i8[2 * k + 1] as i32
                        })
                    };
                    (dot as f32 * inv_norm[id as usize], id)
                })
                .collect()
        }
    }
}

fn top_desc(mut scored: Vec<(f32, u32)>, count: usize) -> Vec<u32> {
    let count = count.min(scored.len());
    let cmp = |a: &(f32, u32), b: &(f32, u32)| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1));
    if count > 0 && scored.len() > count {
        scored.select_nth_unstable_by(count - 1, cmp);
        scored.truncate(count);
    }
    scored.sort_unstable_by(cmp);
    scored.into_iter().map(|(_, id)| id).collect()
}

fn exact_top(
    codec: &UpqCodec,
    packed: &[u8],
    stride: usize,
    q_rot: &[f32],
    q_norm: f32,
    ids: &[u32],
    count: usize,
) -> Result<Vec<u32>> {
    let scored: Result<Vec<(f32, u32)>> = ids
        .iter()
        .map(|&id| {
            let base = &packed[id as usize * stride..(id as usize + 1) * stride];
            Ok((-codec.decode_dot_from_base(base, q_rot, q_norm)?, id))
        })
        .collect();
    Ok(top_desc(scored?, count))
}

fn overlap(a: &[u32], b: &[u32], k: usize) -> f64 {
    let truth: HashSet<u32> = a.iter().take(k).copied().collect();
    b.iter().take(k).filter(|id| truth.contains(id)).count() as f64 / k as f64
}

fn coverage(oracle: &[u32], pool: &[u32], k: usize) -> f64 {
    let present: HashSet<u32> = pool.iter().copied().collect();
    oracle
        .iter()
        .take(k)
        .filter(|id| present.contains(id))
        .count() as f64
        / k as f64
}

#[allow(clippy::too_many_arguments)]
fn validate_quality(
    codec: &UpqCodec,
    cache: &Cache,
    packed: &[u8],
    stride: usize,
    sketches: &[u64],
    queries: &[f32],
    gt: &[u32],
    cli: &Cli,
    pools: &[usize],
) -> Result<Vec<QualityRow>> {
    let words = cli.dim.div_ceil(64);
    let mut sums: Vec<QualitySum> = pools.iter().map(|_| QualitySum::default()).collect();
    let mut hbuf = Vec::new();
    let mut hist = Vec::new();
    let mut q_sketch = vec![0u64; words];
    let mut q_i8 = vec![0i8; cli.dim];
    for qi in 0..cli.nq {
        let query = &queries[qi * cli.dim..(qi + 1) * cli.dim];
        let (q_rot, q_norm) = codec.rotate_query(query)?;
        pack_query_sketch(&q_rot, &mut q_sketch);
        let candidates = scan_candidates(
            &q_sketch,
            sketches,
            words,
            cli.channel_k,
            cli.dim,
            &mut hbuf,
            &mut hist,
        );
        let oracle = exact_top(codec, packed, stride, &q_rot, q_norm, &candidates, cli.k)?;
        let stage2 = score_stage2(codec, cache, &q_rot, &candidates, &mut q_i8);
        let max_pool = *pools.iter().max().unwrap_or(&cli.k);
        let ordered = top_desc(stage2, max_pool);
        let gt_row = &gt[qi * 100..(qi + 1) * 100];
        for (index, &pool_size) in pools.iter().enumerate() {
            let pool = &ordered[..pool_size.min(ordered.len())];
            let final_ids = exact_top(codec, packed, stride, &q_rot, q_norm, pool, cli.k)?;
            sums[index].oracle_coverage_at_10 += coverage(&oracle, pool, 10);
            sums[index].oracle_coverage_at_100 += coverage(&oracle, pool, 100);
            sums[index].final_oracle_overlap_at_10 += overlap(&oracle, &final_ids, 10);
            sums[index].final_oracle_overlap_at_100 += overlap(&oracle, &final_ids, 100);
            sums[index].recall_at_10 += overlap(gt_row, &final_ids, 10);
            sums[index].recall_at_100 += overlap(gt_row, &final_ids, 100);
        }
    }
    let denom = cli.nq as f64;
    Ok(pools
        .iter()
        .zip(sums)
        .map(|(&pool, sum)| QualityRow {
            pool,
            oracle_coverage_at_10: sum.oracle_coverage_at_10 / denom,
            oracle_coverage_at_100: sum.oracle_coverage_at_100 / denom,
            final_oracle_overlap_at_10: sum.final_oracle_overlap_at_10 / denom,
            final_oracle_overlap_at_100: sum.final_oracle_overlap_at_100 / denom,
            recall_at_10: sum.recall_at_10 / denom,
            recall_at_100: sum.recall_at_100 / denom,
        })
        .collect())
}

fn percentile_us(values: &[Duration], percentile: f64) -> f64 {
    let mut micros: Vec<f64> = values.iter().map(|v| v.as_secs_f64() * 1e6).collect();
    micros.sort_by(f64::total_cmp);
    let index = ((micros.len().saturating_sub(1)) as f64 * percentile).round() as usize;
    micros.get(index).copied().unwrap_or(0.0)
}

fn rusage() -> (f64, u64) {
    // SAFETY: getrusage initializes the supplied plain-old-data structure.
    unsafe {
        let mut usage: libc::rusage = std::mem::zeroed();
        libc::getrusage(libc::RUSAGE_SELF, &mut usage);
        let cpu = usage.ru_utime.tv_sec as f64
            + usage.ru_utime.tv_usec as f64 * 1e-6
            + usage.ru_stime.tv_sec as f64
            + usage.ru_stime.tv_usec as f64 * 1e-6;
        let peak = if cfg!(target_os = "linux") {
            usage.ru_maxrss as u64 * 1024
        } else {
            usage.ru_maxrss as u64
        };
        (cpu, peak)
    }
}

#[cfg(target_os = "macos")]
#[allow(deprecated)]
fn steady_rss() -> u64 {
    // SAFETY: task_info fills a correctly-sized mach_task_basic_info.
    unsafe {
        let mut info: libc::mach_task_basic_info = std::mem::zeroed();
        let mut count = libc::MACH_TASK_BASIC_INFO_COUNT;
        let result = libc::task_info(
            libc::mach_task_self(),
            libc::MACH_TASK_BASIC_INFO,
            &mut info as *mut _ as libc::task_info_t,
            &mut count,
        );
        if result == libc::KERN_SUCCESS {
            info.resident_size
        } else {
            0
        }
    }
}

#[cfg(not(target_os = "macos"))]
fn steady_rss() -> u64 {
    0
}

#[allow(clippy::too_many_arguments)]
fn time_pipeline(
    codec: &UpqCodec,
    cache: &Cache,
    packed: &[u8],
    stride: usize,
    sketches: &[u64],
    queries: &[f32],
    cli: &Cli,
    pools: &[usize],
) -> Result<Vec<TimingRow>> {
    let words = cli.dim.div_ceil(64);
    let mut output = Vec::with_capacity(pools.len());
    let mut checksum = 0u64;

    for &pool_size in pools {
        let mut stage1 = Vec::with_capacity(cli.trials * cli.nq);
        let mut stage2 = Vec::with_capacity(cli.trials * cli.nq);
        let mut stage3 = Vec::with_capacity(cli.trials * cli.nq);
        let mut total = Vec::with_capacity(cli.trials * cli.nq);
        let mut hbuf = Vec::new();
        let mut hist = Vec::new();
        let mut q_sketch = vec![0u64; words];
        let mut q_i8 = vec![0i8; cli.dim];
        let run_one = |qi: usize,
                       record: bool,
                       stage1: &mut Vec<Duration>,
                       stage2: &mut Vec<Duration>,
                       stage3: &mut Vec<Duration>,
                       total: &mut Vec<Duration>,
                       hbuf: &mut Vec<u16>,
                       hist: &mut Vec<u32>,
                       q_sketch: &mut Vec<u64>,
                       q_i8: &mut Vec<i8>,
                       checksum: &mut u64|
         -> Result<()> {
            let query = &queries[qi * cli.dim..(qi + 1) * cli.dim];
            let start1 = Instant::now();
            let (q_rot, q_norm) = codec.rotate_query(query)?;
            pack_query_sketch(&q_rot, q_sketch);
            let candidates = scan_candidates(
                q_sketch,
                sketches,
                words,
                cli.channel_k,
                cli.dim,
                hbuf,
                hist,
            );
            let elapsed1 = start1.elapsed();
            let start2 = Instant::now();
            let scored = score_stage2(codec, cache, &q_rot, &candidates, q_i8);
            let ordered = top_desc(scored, pool_size);
            let elapsed2 = start2.elapsed();
            let start3 = Instant::now();
            let final_ids = exact_top(codec, packed, stride, &q_rot, q_norm, &ordered, cli.k)?;
            let elapsed3 = start3.elapsed();
            *checksum = checksum.wrapping_add(final_ids.first().copied().unwrap_or(0) as u64);
            if record {
                stage1.push(elapsed1);
                stage2.push(elapsed2);
                stage3.push(elapsed3);
                total.push(elapsed1 + elapsed2 + elapsed3);
            }
            Ok(())
        };
        for i in 0..cli.warmup {
            run_one(
                i % cli.nq,
                false,
                &mut stage1,
                &mut stage2,
                &mut stage3,
                &mut total,
                &mut hbuf,
                &mut hist,
                &mut q_sketch,
                &mut q_i8,
                &mut checksum,
            )?;
        }
        let (cpu0, _) = rusage();
        for _ in 0..cli.trials {
            for qi in 0..cli.nq {
                run_one(
                    qi,
                    true,
                    &mut stage1,
                    &mut stage2,
                    &mut stage3,
                    &mut total,
                    &mut hbuf,
                    &mut hist,
                    &mut q_sketch,
                    &mut q_i8,
                    &mut checksum,
                )?;
            }
        }
        let (cpu1, _) = rusage();
        output.push(TimingRow {
            pool: pool_size,
            cpu_us_per_query: (cpu1 - cpu0) * 1e6 / (cli.trials * cli.nq) as f64,
            stage1_p50_us: percentile_us(&stage1, 0.50),
            stage1_p95_us: percentile_us(&stage1, 0.95),
            stage2_p50_us: percentile_us(&stage2, 0.50),
            stage2_p95_us: percentile_us(&stage2, 0.95),
            stage3_p50_us: percentile_us(&stage3, 0.50),
            stage3_p95_us: percentile_us(&stage3, 0.95),
            harness_total_p50_us: percentile_us(&total, 0.50),
            harness_total_p95_us: percentile_us(&total, 0.95),
        });
    }
    std::hint::black_box(checksum);
    Ok(output)
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    ensure!(
        cli.k == 100,
        "this experiment requires k=100 for paired r@10/r@100"
    );
    ensure!(!cli.pools.is_empty(), "at least one pool is required");
    ensure!(
        cli.pools
            .iter()
            .all(|&pool| pool >= cli.k && pool <= cli.channel_k),
        "every pool must be in k..=channel_k"
    );
    let pools = cli.pools.clone();
    let dataset = cli
        .vector_dir
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("?")
        .to_string();
    let block_size = block_size_for_dim(cli.dim)?;
    println!(
        "[config] {} mapping={} pools={pools:?} dataset={} N={} nq={} d={} k={} ck={} block={}",
        cli.variant.label(),
        cli.mapping.label(),
        dataset,
        cli.n,
        cli.nq,
        cli.dim,
        cli.k,
        cli.channel_k,
        block_size
    );

    let corpus_map = map_f32_file(&cli.vector_dir.join("corpus.f32"), cli.n * cli.dim)?;
    let corpus = as_f32(&corpus_map)?;
    let queries = load_queries(&cli.vector_dir.join("queries.f32"), cli.nq * cli.dim)?;
    let calibration: Vec<Vec<f32>> = corpus
        .chunks_exact(cli.dim)
        .take(CALIB_ROWS.min(cli.n))
        .map(<[f32]>::to_vec)
        .collect();
    let fit_start = Instant::now();
    let codec = UpqCodec::fit_from_sample(
        cli.dim,
        block_size,
        ROTATION_SEED,
        CELLS,
        RING_SWEEP,
        UpqDesignSource::Empirical,
        &calibration,
    )?;
    println!(
        "[fit] {:?}; rings={} sumP={} cost={:.6e}",
        fit_start.elapsed(),
        codec.design.ring_radii.len(),
        codec.design.ring_phases.iter().sum::<u32>(),
        codec.design.design_cost
    );
    drop(calibration);

    let stride = packed11_stride(codec.num_pairs);
    let logical = packed11_len(codec.num_pairs);
    let mut packed = vec![0u8; cli.n * stride];
    let encode_start = Instant::now();
    packed
        .par_chunks_mut(stride)
        .enumerate()
        .try_for_each(|(row, out)| -> valise::Result<()> {
            let input = &corpus[row * cli.dim..(row + 1) * cli.dim];
            let (rotated, _) = codec.rotate_query(input)?;
            let mut codes = vec![0u16; codec.num_pairs];
            codec.encode_rotated(&rotated, &mut codes);
            pack11(&codes, &mut out[..logical]);
            Ok(())
        })?;
    println!("[encode] {:?}", encode_start.elapsed());
    drop(corpus_map);

    let map = build_coarse_map(&codec, &packed, stride, cli.mapping)?;
    validate_i4_kernel(codec.num_pairs)?;
    // Keep the exact tables live and black-boxed: this catches accidental
    // optimizer removal of the exhaustive proof in release builds.
    std::hint::black_box(&map.cell_to_residual);
    std::hint::black_box(&map.byte_residual_to_cell);
    if cli.validate_only {
        println!(
            "[validate-only] scale={:.8} weighted_mse={:.8e}",
            map.assignment_scale, map.weighted_mse
        );
        return Ok(());
    }

    let words = cli.dim.div_ceil(64);
    let open_start = Instant::now();
    let sketches = build_sketches(&codec, &packed, stride, words)?;
    let cache = match cli.variant {
        Variant::Before => build_before_cache(&codec, &packed, stride)?,
        Variant::CoarseInt4 | Variant::CoarseLut => {
            build_coarse_cache(&codec, &packed, stride, &map, cli.variant)?
        }
    };
    println!("[open-cache] {:?}", open_start.elapsed());
    let steady_rss_bytes = steady_rss();
    println!(
        "[memory-sample] process current RSS immediately after cache+sketch build: {:.1} MiB",
        steady_rss_bytes as f64 / 1_048_576.0
    );

    let gt_file = gt_path(&cli.vector_dir, cli.n, cli.nq)?;
    let gt = load_gt(&gt_file, cli.nq)?;
    println!("[gt] {}", gt_file.display());
    let quality = validate_quality(
        &codec, &cache, &packed, stride, &sketches, &queries, &gt, &cli, &pools,
    )?;
    for row in &quality {
        println!(
            "[quality pool={}] oracle-cover @10={:.6} @100={:.6}; final-oracle @10={:.6} @100={:.6}; recall @10={:.6} @100={:.6}",
            row.pool,
            row.oracle_coverage_at_10,
            row.oracle_coverage_at_100,
            row.final_oracle_overlap_at_10,
            row.final_oracle_overlap_at_100,
            row.recall_at_10,
            row.recall_at_100
        );
    }

    let timing = time_pipeline(
        &codec, &cache, &packed, stride, &sketches, &queries, &cli, &pools,
    )?;
    let cpu_us_per_query = timing.first().map_or(0.0, |row| row.cpu_us_per_query);
    let (_, process_lifetime_high_water_bytes) = rusage();
    for row in &timing {
        println!(
            "[timing pool={}] s1 {:.1}/{:.1} us; s2 {:.1}/{:.1}; s3 {:.1}/{:.1}; harness-total {:.1}/{:.1} (p50/p95); CPU {:.1} us/q",
            row.pool,
            row.stage1_p50_us,
            row.stage1_p95_us,
            row.stage2_p50_us,
            row.stage2_p95_us,
            row.stage3_p50_us,
            row.stage3_p95_us,
            row.harness_total_p50_us,
            row.harness_total_p95_us,
            row.cpu_us_per_query
        );
    }

    let stage2_cache_bytes = match cli.variant {
        Variant::Before => cli.dim,
        Variant::CoarseInt4 | Variant::CoarseLut => codec.num_pairs,
    };
    let sketch_bytes = words * 8;
    let norm_bytes = 4;
    let derived_cache_bytes = stage2_cache_bytes + sketch_bytes + norm_bytes;
    let derived_cache_plus_packed_stride_bytes = derived_cache_bytes + stride;
    println!(
        "[memory] cache={} sketch={} factor={} => cache+sketch+factor={} B/vec; plus packed stride={} B/vec; current steady={:.1} MiB; lifetime high-water={:.1} MiB (encoding-contaminated); CPU(first pool)={:.1} us/query",
        stage2_cache_bytes,
        sketch_bytes,
        norm_bytes,
        derived_cache_bytes,
        derived_cache_plus_packed_stride_bytes,
        steady_rss_bytes as f64 / 1_048_576.0,
        process_lifetime_high_water_bytes as f64 / 1_048_576.0,
        cpu_us_per_query
    );
    println!(
        "[storage] unchanged: exact packed11 logical={} B/vec, current stride={} B/vec",
        logical, stride
    );

    let report = Report {
        variant: cli.variant.label().into(),
        mapping: cli.mapping.label().into(),
        dataset,
        n: cli.n,
        nq: cli.nq,
        dim: cli.dim,
        k: cli.k,
        channel_k: cli.channel_k,
        trials: cli.trials,
        exact_roundtrip_cells: CELLS as usize,
        exact_sign_cells: CELLS as usize,
        assignment_scale: map.assignment_scale,
        grouping_weighted_mse: map.weighted_mse,
        packed_logical_bytes_per_vector: logical,
        packed_stride_bytes_per_vector: stride,
        stage2_cache_bytes_per_vector: stage2_cache_bytes,
        sketch_bytes_per_vector: sketch_bytes,
        norm_bytes_per_vector: norm_bytes,
        derived_cache_bytes_per_vector: derived_cache_bytes,
        derived_cache_plus_packed_stride_bytes_per_vector:
            derived_cache_plus_packed_stride_bytes,
        steady_rss_bytes,
        process_lifetime_high_water_bytes,
        cpu_us_per_query,
        quality,
        timing,
        note: "Harness-only serial production-shaped pipeline; not end-to-end Valise latency. Channel oracle is exact only within ck=2048, not a global exact oracle. Exact packed11 remains stage-3 authority; the coarse+residual proof is logical and does not implement or test a physical 8+3-bit residual layout.".into(),
    };
    if let Some(path) = &cli.json {
        fs::write(path, serde_json::to_vec_pretty(&report)?)?;
        println!("[json] {}", path.display());
    }
    Ok(())
}
