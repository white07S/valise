#!/usr/bin/env python3
"""Codec-level statistical rigor study for the FAST paper: UPQ vs QAM.

Self-contained numpy reimplementation (vendored, no cross-repo imports) of

  * the production rotation: SplitMix64 sign mask ("lama_qam" seed) +
    per-block normalized FWHT with the production block-size rule
    (src/codec/qam_lloyd_max.rs, src/codec/upq.rs::default_block_size_for);
  * QAM(a,p): per-pair sigma = mean|z| / sqrt(pi/2) on the calibration
    sample, 2^a-level Lloyd-Max amplitude codebook on the analytic unit
    Rayleigh (R_MAX = 8, PD-compander init), 2^p uniform phases
    (src/codec/qam_lloyd_max.rs; numpy port cross-checked at 0.9647 R@10
    on cohere-768 in valise_experiments/pareto/upq_tcq_proto.py);
  * UPQ(cells): ring sweep {24,32,40,48}, objective
    (a - r_i)^2 + a*r_i*(2pi/P_i)^2/12, power-of-two P_i >= 4,
    sum P_i <= cells, 30 alternating-fit iterations with the sorted-sample
    prefix-sum inner loop and closed-form ring-cut boundaries — a faithful
    port of src/codec/upq.rs::UpqDesign::fit (the ground truth).

Protocol (per dataset): corpus = first 100k rows, queries = first 1000,
L2-normalized (sift/gist are L2 datasets; the normalized-cosine surrogate
is used, consistent with the paper). Search mirrors production: 1-bit/dim
sign sketch derived from the stored codes -> Hamming top-2048 candidates
-> exact-on-reconstruction rerank (renormalized decode) -> top-10/100
graded against exact f32 ground truth.

Statistical design (headline): 3 rotation seeds x 5 disjoint 4096-row
calibration samples drawn from corpus rows 100_000..120_480 (all seven
corpora have >= 120_480 rows, so calibration NEVER overlaps the search
corpus) = 15 paired fits per codec. The primary claim statistic is the
run-level paired UPQ-QAM difference with a 95% t CI (df = 14).

Run (bench venv):
  cd bench/python
  .venv/bin/python codec_study.py --phase sanity
  .venv/bin/python codec_study.py --phase headline
  .venv/bin/python codec_study.py --phase frontier
  .venv/bin/python codec_study.py --phase comparators

Output: bench/results/paper/codec_study.json (merged incrementally, so
phases can run in separate invocations and resume).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from valise_data import load_vectors

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
OUT_DEFAULT = HERE.parent / "results" / "paper" / "codec_study.json"

TAU = 2.0 * math.pi
MASK64 = (1 << 64) - 1
PROD_SEED = 0x6C61_6D61_5F71_616D  # "lama_qam"

N_CORPUS = 100_000
N_QUERIES = 1_000
CAL_LO = 100_000  # calibration pool start row (disjoint from corpus)
CAL_ROWS = 4_096  # rows per calibration sample
N_CALIB = 5  # disjoint calibration samples
CK = 2_048  # sketch candidate budget (production-style stage 1)
RING_SWEEP = (24, 32, 40, 48)
FIT_ITERS = 30
MIN_RING_PHASES = 4
DESIGN_SAMPLE_CAP = 400_000
KMEANS_NITER = 30

ALL_DATASETS = [
    "glove-100",
    "sift-1m",
    "cohere-medium-1m-f32",
    "mpnet-768",
    "gist-1m",
    "cohere-v3-1024",
    "openai-medium-500k",
]
FRONTIER_DATASETS = ["cohere-medium-1m-f32", "mpnet-768", "glove-100"]

# two-sided 95% Student-t critical values by df
TCRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
         7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
         13: 2.160, 14: 2.145}


# ----------------------------------------------------------------------------
# rotation (vendored: SplitMix64 sign mask + normalized block FWHT)
# ----------------------------------------------------------------------------
def splitmix64_next(state: int) -> tuple[int, int]:
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, z ^ (z >> 31)


def rotation_seeds(k: int = 3) -> list[int]:
    """Seed 1 = the production rotation seed; the rest are its SplitMix64
    successors (deterministic, documented)."""
    seeds, state = [PROD_SEED], PROD_SEED
    while len(seeds) < k:
        state, z = splitmix64_next(state)
        seeds.append(z)
    return seeds


def splitmix64_signs(seed: int, dim: int) -> np.ndarray:
    state = seed & MASK64
    signs = np.empty(dim, dtype=np.float32)
    bits, left = 0, 0
    for i in range(dim):
        if left == 0:
            state, bits = splitmix64_next(state)
            left = 64
        signs[i] = 1.0 if (bits & 1) == 0 else -1.0
        bits >>= 1
        left -= 1
    return signs


def block_size_for(dim: int) -> int:
    """Production rule: min(1024, largest power of two dividing dim)."""
    block = 1
    while block * 2 <= 1024 and dim % (block * 2) == 0:
        block *= 2
    return block


def fwht_rows(X: np.ndarray, bs: int) -> np.ndarray:
    n, dim = X.shape
    Y = X.reshape(-1, bs).copy()
    h = 1
    while h < bs:
        Y = Y.reshape(-1, 2, h)
        a = Y[:, 0, :] + Y[:, 1, :]
        b = Y[:, 0, :] - Y[:, 1, :]
        Y[:, 0, :] = a
        Y[:, 1, :] = b
        Y = Y.reshape(-1, bs)
        h *= 2
    return (Y * np.float32(1.0 / math.sqrt(bs))).reshape(n, dim)


def rotate(X: np.ndarray, signs: np.ndarray, bs: int) -> np.ndarray:
    return fwht_rows(X * signs[None, :], bs)


# ----------------------------------------------------------------------------
# QAM: Lloyd-Max amplitude codebook on the analytic unit Rayleigh
# ----------------------------------------------------------------------------
def _rayleigh_partial_moments(a: float, b: float) -> tuple[float, float]:
    ea, eb = math.exp(-a * a / 2.0), math.exp(-b * b / 2.0)
    p = ea - eb
    m = math.sqrt(math.pi / 2.0) * (math.erf(b / math.sqrt(2)) - math.erf(a / math.sqrt(2)))
    m += a * ea - b * eb
    return p, m


_LM_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def lloyd_max_rayleigh(n_levels: int, r_max=8.0, grid=8192, iters=200, tol=1e-10):
    if n_levels in _LM_CACHE:
        return _LM_CACHE[n_levels]
    rs = np.linspace(0.0, r_max, grid)
    f = rs * np.exp(-rs * rs / 2.0)
    f /= np.trapezoid(f, rs)
    f13 = np.maximum(f, 1e-30) ** (1.0 / 3.0)
    cum = np.cumsum(f13)
    cum /= cum[-1]
    levels = np.interp((np.arange(n_levels) + 0.5) / n_levels, cum, rs)
    bounds = np.zeros(n_levels + 1)
    bounds[-1] = r_max
    for _ in range(iters):
        bounds[1:-1] = 0.5 * (levels[:-1] + levels[1:])
        max_d = 0.0
        for i in range(n_levels):
            p, m = _rayleigh_partial_moments(bounds[i], bounds[i + 1])
            new = m / p if p > 1e-15 else levels[i]
            new = min(max(new, bounds[i] + 1e-9), bounds[i + 1] - 1e-9)
            max_d = max(max_d, abs(new - levels[i]))
            levels[i] = new
        if max_d < tol:
            break
    _LM_CACHE[n_levels] = (levels.astype(np.float64), bounds.astype(np.float64))
    return _LM_CACHE[n_levels]


def qam_quantize(AN, TH, amp_bits: int, phase_bits: int):
    """(normalized amp, angle) -> (a_hat_norm f32, theta_hat f32)."""
    levels, bounds = lloyd_max_rayleigh(1 << amp_bits)
    ai = np.clip(np.searchsorted(bounds, AN, side="right") - 1, 0, len(levels) - 1)
    n_phase = 1 << phase_bits
    pi_ = np.rint(TH * np.float32(n_phase / TAU)).astype(np.int32) % n_phase
    a_hat = levels.astype(np.float32)[ai]
    th_hat = pi_.astype(np.float32) * np.float32(TAU / n_phase)
    return a_hat, th_hat


# ----------------------------------------------------------------------------
# UPQ fit — faithful numpy port of src/codec/upq.rs::UpqDesign::fit
# ----------------------------------------------------------------------------
def ring_cut_boundaries(radii: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    m = len(radii)
    bounds = np.zeros(max(m - 1, 0))
    prev = 0.0
    for i in range(m - 1):
        r0, c0, r1, c1 = radii[i], coeffs[i], radii[i + 1], coeffs[i + 1]
        den = 2.0 * (r1 - r0) + r0 * c0 - r1 * c1
        b = (r1 * r1 - r0 * r0) / den if den > 1e-12 else np.inf
        b = max(b, prev)
        bounds[i] = b
        prev = b
    return bounds


def alloc_pow2(weights: np.ndarray, cells: int, min_p: int = MIN_RING_PHASES) -> np.ndarray:
    """Start every ring at min_p; greedily double the ring with the largest
    marginal gain 0.75*w/P^2 that still fits the budget (upq.rs::alloc_pow2)."""
    phases = np.full(len(weights), min_p, dtype=np.int64)
    w = np.maximum(np.asarray(weights, dtype=np.float64), 1e-12)
    while True:
        budget = cells - int(phases.sum())
        if budget <= 0:
            return phases
        gains = 0.75 * w / phases.astype(np.float64) ** 2
        gains = np.where(phases <= budget, gains, -np.inf)
        i = int(np.argmax(gains))
        if not np.isfinite(gains[i]):
            return phases  # nothing fits; leave slack unused
        phases[i] *= 2


def upq_fit(amps_norm: np.ndarray, num_rings: int, cells: int):
    """One (M, cells) fit; returns (mean_cost, radii, phases)."""
    stride = max(len(amps_norm) // DESIGN_SAMPLE_CAP, 1)
    a = np.sort(np.asarray(amps_norm[::stride], dtype=np.float64))
    n = len(a)
    pa = np.concatenate([[0.0], np.cumsum(a)])
    paa = np.concatenate([[0.0], np.cumsum(a * a)])
    a_max = max(float(a[-1]), 8.0)

    # init radii: cube-root-density (Panter-Dite) histogram quantiles
    bins = 2048
    hist, _ = np.histogram(a, bins=bins, range=(0.0, a_max))
    cum = np.cumsum(np.cbrt(hist.astype(np.float64)))
    radii = np.empty(num_rings, dtype=np.float64)
    for i in range(num_rings):
        target = (i + 0.5) / num_rings * cum[-1]
        b = int(np.searchsorted(cum, target, side="left"))
        radii[i] = (min(b, bins - 1) + 0.5) / bins * a_max
    phases = np.full(num_rings, cells // num_rings, dtype=np.int64)

    best = None
    for _ in range(FIT_ITERS):
        coeffs = (TAU / phases.astype(np.float64)) ** 2 / 12.0
        bounds = ring_cut_boundaries(radii, coeffs)
        cut = np.empty(num_rings + 1, dtype=np.int64)
        cut[0], cut[num_rings] = 0, n
        cut[1:num_rings] = np.searchsorted(a, bounds, side="right")
        np.maximum.accumulate(cut, out=cut)
        lo, hi = cut[:-1], cut[1:]
        counts = hi - lo
        n_i = counts.astype(np.float64)
        sa = pa[hi] - pa[lo]
        saa = paa[hi] - paa[lo]
        total = float(np.sum(saa - 2.0 * radii * sa + n_i * radii**2 + radii * coeffs * sa))
        mean_cost = total / n
        if best is None or mean_cost < best[0]:
            best = (mean_cost, radii.copy(), phases.copy())

        # radii update r_i = E[a|i]*(1 - c_i/2); in-place, Rust order
        # (empty rings respawn off the possibly-already-updated densest)
        densest = int(np.argmax(counts))
        for i in range(num_rings):
            if counts[i] > 0:
                radii[i] = sa[i] / n_i[i] * (1.0 - coeffs[i] / 2.0)
            else:
                radii[i] = radii[densest] * (1.0 + 0.013 * (i + 1.0))
        order = np.argsort(radii, kind="stable")
        radii = radii[order]
        sa_s, n_s = sa[order], n_i[order]
        mean_a = np.where(n_s > 0, sa_s / np.maximum(n_s, 1.0), 0.0)
        weights = n_s * mean_a * radii * TAU**2 / 12.0
        phases = alloc_pow2(weights, cells)

    return best


def upq_fit_sweep(amps_norm: np.ndarray, cells: int, ring_sweep=RING_SWEEP):
    """Sweep ring counts, keep the lowest design cost. Returns a dict."""
    best = None
    for m in ring_sweep:
        cost, radii, phases = upq_fit(amps_norm, m, cells)
        if best is None or cost < best["cost"]:
            best = {"cost": cost, "radii": radii, "phases": phases, "M": m}
    coeffs = (TAU / best["phases"].astype(np.float64)) ** 2 / 12.0
    best["boundaries"] = ring_cut_boundaries(best["radii"], coeffs)
    assert int(best["phases"].sum()) <= cells
    return best


def upq_quantize(AN, TH, design):
    """Encode with the closed-form ring boundaries (production encode path)."""
    ring = np.searchsorted(design["boundaries"], AN, side="left")
    radii32 = design["radii"].astype(np.float32)
    phases = design["phases"]
    a_hat = radii32[ring]
    Pp = phases[ring].astype(np.float32)
    m = np.rint(TH * (Pp / np.float32(TAU))).astype(np.int64) % phases[ring]
    th_hat = m.astype(np.float32) * (np.float32(TAU) / Pp)
    return a_hat, th_hat


# ----------------------------------------------------------------------------
# reconstruction + evaluation (production-mirroring search)
# ----------------------------------------------------------------------------
def recon_polar(a_hat_norm, th_hat, sigma):
    """Renormalized decode from (normalized amp, phase) x per-pair sigma."""
    n = a_hat_norm.shape[0]
    dim = a_hat_norm.shape[1] * 2
    XH = np.empty((n, dim), dtype=np.float32)
    amp = a_hat_norm * sigma[None, :]
    XH[:, 0::2] = amp * np.cos(th_hat)
    XH[:, 1::2] = amp * np.sin(th_hat)
    return _renorm(XH)


def recon_xy(cx, cy, sigma):
    n = cx.shape[0]
    dim = cx.shape[1] * 2
    XH = np.empty((n, dim), dtype=np.float32)
    XH[:, 0::2] = cx * sigma[None, :]
    XH[:, 1::2] = cy * sigma[None, :]
    return _renorm(XH)


def _renorm(XH):
    nrm = np.linalg.norm(XH, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    XH /= nrm
    return XH


def search_eval(XH, Qrot, gt_ids, ck=CK):
    """Sign-sketch (signs of the stored reconstruction, exactly what the
    production sketch derives from the codes at file-open) -> Hamming
    top-ck -> exact rerank on the renormalized decode -> recall@10/@100.

    Hamming distance on +-1 signs is an affine function of the sign dot
    product, so stage 1 is a GEMM (bit-exact same ranking)."""
    nq, dim = Qrot.shape
    Xs = np.where(XH >= 0, np.float32(1.0), np.float32(-1.0))
    Qs = np.where(Qrot >= 0, np.float32(1.0), np.float32(-1.0))
    r10 = np.empty(nq)
    r100 = np.empty(nq)
    B = int(np.clip((800 << 20) // (ck * dim * 4), 8, 256))
    for i in range(0, nq, B):
        sl = slice(i, min(i + B, nq))
        S = Qs[sl] @ Xs.T
        cand = np.argpartition(-S, ck, axis=1)[:, :ck]
        sub = XH[cand.ravel()].reshape(cand.shape[0], ck, dim)
        sc = np.matmul(sub, Qrot[sl][:, :, None])[:, :, 0]
        top = np.argpartition(-sc, 100, axis=1)[:, :100]
        ids = np.take_along_axis(cand, top, axis=1)
        sc100 = np.take_along_axis(sc, top, axis=1)
        order = np.argsort(-sc100, axis=1, kind="stable")
        ids = np.take_along_axis(ids, order, axis=1)
        g = gt_ids[sl]
        r10[sl] = (ids[:, :10, None] == g[:, None, :10]).any(axis=1).mean(axis=1)
        r100[sl] = (ids[:, :, None] == g[:, None, :100]).any(axis=1).mean(axis=1)
    return float(r10.mean()), float(r100.mean())


def eval_reconstruction(XH, Xref, Qref, gt_ids):
    """Distortion (relative, renormalized decode) + recall via the
    production-mirroring search. Xref/Qref are the same-domain references
    (rotated for rotated codecs, raw for the no-rotation comparator)."""
    dot = np.einsum("ij,ij->i", Xref, XH)
    nx2 = np.einsum("ij,ij->i", Xref, Xref)
    err2 = np.maximum(nx2 + 1.0 - 2.0 * dot, 0.0)  # ||xhat|| = 1
    rel = float(err2.mean() / nx2.mean())
    eps_p95 = float(np.percentile(np.sqrt(err2), 95))
    r10, r100 = search_eval(XH, Qref, gt_ids)
    return {"r10": r10, "r100": r100, "distortion": rel,
            "eps_rms": math.sqrt(rel), "eps_p95": eps_p95}


# ----------------------------------------------------------------------------
# data + exact ground truth (+ margins)
# ----------------------------------------------------------------------------
def _norm_rows(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n


def load_norm(name, n=N_CORPUS, nq=N_QUERIES):
    ds = load_vectors(name)
    need = CAL_LO + N_CALIB * CAL_ROWS
    if ds.corpus_len < need:
        raise RuntimeError(
            f"{name}: corpus_len {ds.corpus_len} < {need}; calibration rows "
            "would overlap the search corpus — adjust CAL_LO")
    X = _norm_rows(np.array(ds.corpus[:n], dtype=np.float32))
    Q = _norm_rows(np.array(ds.queries[:nq], dtype=np.float32))
    C = _norm_rows(np.array(ds.corpus[CAL_LO:need], dtype=np.float32))
    return X, Q, C


def gt_scores(name, X, Q, k=101):
    """Exact cosine top-k ids + scores on the L2-normalized f32 data
    (unrotated domain; the rotation is orthogonal so scores agree).
    Cached; cross-checked against the bench FAISS GT cache when present."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"codec_study_gt_{name}_n{len(X)}_nq{len(Q)}_cos_k{k}.npz"
    if path.exists():
        z = np.load(path)
        return z["ids"], z["scores"]
    nq = len(Q)
    ids = np.empty((nq, k), dtype=np.int32)
    sc = np.empty((nq, k), dtype=np.float32)
    for i in range(0, nq, 128):
        sl = slice(i, min(i + 128, nq))
        S = Q[sl] @ X.T
        part = np.argpartition(-S, k, axis=1)[:, :k]
        ps = np.take_along_axis(S, part, axis=1)
        order = np.argsort(-ps, axis=1, kind="stable")
        ids[sl] = np.take_along_axis(part, order, axis=1)
        sc[sl] = np.take_along_axis(ps, order, axis=1)
    np.savez_compressed(path, ids=ids, scores=sc)
    # cross-check vs the bench cache (cosine datasets only; sift/gist bench
    # GT is raw-L2, ours is the normalized-cosine surrogate)
    bench_gt = CACHE / f"gt_{name}_n{len(X)}_nq{len(Q)}_cosine_k100.npy"
    if bench_gt.exists():
        ref = np.load(bench_gt)
        ov = np.mean([len(set(a[:10]) & set(b[:10])) / 10.0
                      for a, b in zip(ids, ref)])
        print(f"  [{name}] GT cross-check vs bench cache: top-10 overlap {ov:.4f}")
    return ids, sc


def margins_from_scores(sc):
    m = (sc[:, 9] - sc[:, 10]).astype(np.float64)
    return {"margin_p10": float(np.percentile(m, 10)),
            "margin_p50": float(np.percentile(m, 50)),
            "margin_p90": float(np.percentile(m, 90))}


# ----------------------------------------------------------------------------
# comparators
# ----------------------------------------------------------------------------
def upq_init_centroids(design, cells):
    """Cell centers of a fitted UPQ design as a k-means init: k-means started
    here can only lower the (in-sample) distortion below UPQ's, so the
    resulting codebook is a tight 2-D-quantizer family bound."""
    xs, ys = [], []
    for r, p in zip(design["radii"], design["phases"]):
        th = np.arange(p) * (TAU / p)
        xs.append(np.full(p, r) * np.cos(th))
        ys.append(np.full(p, r) * np.sin(th))
    C = np.stack([np.concatenate(xs), np.concatenate(ys)], axis=1).astype(np.float32)
    if len(C) < cells:  # pad unused budget with tiny jitters of the origin ring
        extra = C[: cells - len(C)] * 1.001
        C = np.concatenate([C, extra])
    return np.ascontiguousarray(C[:cells])


def kmeans2d_fit(pairs, cells, seed, init=None):
    """Unconstrained 2-D k-means codebook on sigma-normalized pairs, shared
    across pairs (mirrors UPQ's ring sharing). This IS PQ with 2-d
    subspaces + a learned codebook = the 2-D-quantizer family bound."""
    import faiss

    stride = max(len(pairs) // DESIGN_SAMPLE_CAP, 1)
    pts = np.ascontiguousarray(pairs[::stride], dtype=np.float32)
    km = faiss.Kmeans(2, cells, niter=KMEANS_NITER, verbose=False,
                      seed=int(seed % (1 << 31)), max_points_per_centroid=10**7)
    km.train(pts, init_centroids=init)
    C = km.centroids.reshape(cells, 2).astype(np.float32)
    # grid LUT encode (O(1)/pair, grid cell << quantizer cell)
    lim = float(max(4.5, np.abs(C).max() * 1.05))
    G = 1024
    gx = ((np.arange(G) + 0.5) / G * 2 * lim - lim).astype(np.float32)
    grid = np.stack(np.meshgrid(gx, gx, indexing="ij"), axis=-1).reshape(-1, 2)
    index = faiss.IndexFlatL2(2)
    index.add(C)
    _, lut = index.search(np.ascontiguousarray(grid), 1)
    return C, lut.reshape(G, G).astype(np.int32), lim


def kmeans2d_encode(zre_n, zim_n, C, lut, lim):
    G = lut.shape[0]
    ix = np.clip(((zre_n + lim) * (G / (2 * lim))).astype(np.int64), 0, G - 1)
    iy = np.clip(((zim_n + lim) * (G / (2 * lim))).astype(np.int64), 0, G - 1)
    cells_idx = lut[ix, iy]
    return C[cells_idx, 0], C[cells_idx, 1]


def sq_quantize_dim(xn, bits, clip=4.0):
    """Uniform mid-rise quantizer on sigma-normalized coords, [-clip, clip]."""
    L = 1 << bits
    idx = np.clip(np.floor((xn + clip) * (L / (2 * clip))), 0, L - 1)
    return ((idx + 0.5) * (2 * clip / L) - clip).astype(np.float32)


# ----------------------------------------------------------------------------
# per-run drivers
# ----------------------------------------------------------------------------
def pair_polar(Xrot):
    A = np.hypot(Xrot[:, 0::2], Xrot[:, 1::2]).astype(np.float32)
    TH = np.arctan2(Xrot[:, 1::2], Xrot[:, 0::2]).astype(np.float32)
    return A, TH


def sigma_from_amps(A_cal):
    s = (A_cal.mean(axis=0, dtype=np.float64) / math.sqrt(math.pi / 2.0)).astype(np.float32)
    return np.maximum(s, 1e-12)


def run_qam(A, TH, sigma, amp_bits, phase_bits, Xrot, Qrot, gt_ids):
    AN = A / sigma[None, :]
    a_hat, th_hat = qam_quantize(AN, TH, amp_bits, phase_bits)
    XH = recon_polar(a_hat, th_hat, sigma)
    return eval_reconstruction(XH, Xrot, Qrot, gt_ids)


def run_upq(A, TH, sigma, A_cal, cells, Xrot, Qrot, gt_ids):
    an_cal = (A_cal / sigma[None, :]).astype(np.float64).ravel()
    design = upq_fit_sweep(an_cal, cells)
    AN = A / sigma[None, :]
    a_hat, th_hat = upq_quantize(AN, TH, design)
    r = eval_reconstruction(recon_polar(a_hat, th_hat, sigma), Xrot, Qrot, gt_ids)
    r["fit"] = {"M": design["M"], "sumP": int(design["phases"].sum()),
                "design_cost": design["cost"]}
    return r


def run_binary(Xrot, Qrot, gt_ids):
    dim = Xrot.shape[1]
    XH = np.where(Xrot >= 0, np.float32(1.0), np.float32(-1.0)) / np.float32(math.sqrt(dim))
    return eval_reconstruction(XH, Xrot, Qrot, gt_ids)


def run_sq55(Xrot, sigma, Qrot, gt_ids):
    sig_dim = np.repeat(sigma, 2)
    XN = Xrot / sig_dim[None, :]
    XH = np.empty_like(Xrot)
    XH[:, 0::2] = sq_quantize_dim(XN[:, 0::2], 5)  # even dims: 5 bits
    XH[:, 1::2] = sq_quantize_dim(XN[:, 1::2], 6)  # odd dims: 6 bits
    XH *= sig_dim[None, :]
    return eval_reconstruction(_renorm(XH), Xrot, Qrot, gt_ids)


def run_kmeans2d(Xd, Qd, cal_rows, gt_ids, seed, cells=2048):
    """Xd/Qd/cal_rows in the codec domain (rotated or raw). k-means is
    seeded from the UPQ codebook fitted on the same calibration sample
    (prototype convention), so it starts at UPQ's distortion and descends."""
    zre, zim = Xd[:, 0::2], Xd[:, 1::2]
    A_cal = np.hypot(cal_rows[:, 0::2], cal_rows[:, 1::2]).astype(np.float32)
    sigma = sigma_from_amps(A_cal)
    pairs = np.stack([(cal_rows[:, 0::2] / sigma[None, :]).ravel(),
                      (cal_rows[:, 1::2] / sigma[None, :]).ravel()], axis=1)
    design = upq_fit_sweep((A_cal / sigma[None, :]).astype(np.float64).ravel(), cells)
    C, lut, lim = kmeans2d_fit(pairs, cells, seed, init=upq_init_centroids(design, cells))
    cx, cy = kmeans2d_encode(zre / sigma[None, :], zim / sigma[None, :], C, lut, lim)
    return eval_reconstruction(recon_xy(cx, cy, sigma), Xd, Qd, gt_ids)


# ----------------------------------------------------------------------------
# statistics
# ----------------------------------------------------------------------------
def mean_ci(vals):
    a = np.asarray(vals, dtype=np.float64)
    m = float(a.mean())
    if len(a) < 2:
        return m, [m, m]
    t = TCRIT.get(len(a) - 1, 1.96)
    h = t * float(a.std(ddof=1)) / math.sqrt(len(a))
    return m, [m - h, m + h]


def summarize_runs(runs):
    out = {}
    for key in ("r10", "r100", "distortion"):
        m, ci = mean_ci([r[key] for r in runs])
        out[f"{key}_mean"] = m
        out[f"{key}_ci95"] = ci
    out["eps_rms_mean"] = float(np.mean([r["eps_rms"] for r in runs]))
    out["eps_p95_mean"] = float(np.mean([r["eps_p95"] for r in runs]))
    out["n_runs"] = len(runs)
    return out


# ----------------------------------------------------------------------------
# phases
# ----------------------------------------------------------------------------
def phase_sanity(res):
    """Vendored QAM(5,6)+sketch on cohere-768 at full seed/calibration
    parity with the prototype (calib = corpus rows 0..4096, production
    rotation seed) must land within ~±0.01 of 0.9647 (prototype) /
    0.965 (production e2e)."""
    t0 = time.time()
    name = "cohere-medium-1m-f32"
    X, Q, _ = load_norm(name)
    gt_ids, _ = gt_scores(name, X, Q)
    signs = splitmix64_signs(PROD_SEED, X.shape[1])
    bs = block_size_for(X.shape[1])
    Xrot = rotate(X, signs, bs)
    Qrot = rotate(Q, signs, bs)
    A, TH = pair_polar(Xrot)
    sigma = sigma_from_amps(A[:4096])  # prototype parity: calib = first 4096 corpus rows
    q = run_qam(A, TH, sigma, 5, 6, Xrot, Qrot, gt_ids)
    u = run_upq(A, TH, sigma, A[:4096], 2048, Xrot, Qrot, gt_ids)
    ok = abs(q["r10"] - 0.9647) <= 0.011
    res["meta"]["sanity"] = {
        "dataset": name, "seed": "production", "calib": "corpus[:4096] (prototype parity)",
        "qam56_r10": q["r10"], "qam56_r100": q["r100"], "qam56_distortion": q["distortion"],
        "upq2048_r10": u["r10"], "upq2048_r100": u["r100"],
        "reference_prototype_r10": 0.9647, "reference_e2e_r10": 0.9616,
        "within_tolerance": bool(ok), "seconds": time.time() - t0,
    }
    print(f"[sanity] QAM(5,6)+sketch r@10 = {q['r10']:.4f} "
          f"(prototype 0.9647, e2e 0.9616) -> {'OK' if ok else 'DEVIATION'}; "
          f"UPQ(2048) r@10 = {u['r10']:.4f}  [{time.time()-t0:.0f}s]")
    if not ok:
        raise SystemExit("sanity check failed — investigate before the full design")


def phase_headline(res, seeds, datasets):
    res.setdefault("headline", {})
    res.setdefault("margins", {})
    for name in datasets:
        if name in res["headline"]:
            print(f"[headline] {name}: already done, skipping")
            continue
        t0 = time.time()
        X, Q, Cpool = load_norm(name)
        gt_ids, gt_sc = gt_scores(name, X, Q)
        qam_runs, upq_runs, run_tags = [], [], []
        for seed in seeds:
            signs = splitmix64_signs(seed, X.shape[1])
            bs = block_size_for(X.shape[1])
            Xrot = rotate(X, signs, bs)
            Qrot = rotate(Q, signs, bs)
            Crot = rotate(Cpool, signs, bs)
            A, TH = pair_polar(Xrot)
            for ci in range(N_CALIB):
                cal = Crot[ci * CAL_ROWS:(ci + 1) * CAL_ROWS]
                A_cal = np.hypot(cal[:, 0::2], cal[:, 1::2]).astype(np.float32)
                sigma = sigma_from_amps(A_cal)
                qam_runs.append(run_qam(A, TH, sigma, 5, 6, Xrot, Qrot, gt_ids))
                upq_runs.append(run_upq(A, TH, sigma, A_cal, 2048, Xrot, Qrot, gt_ids))
                run_tags.append({"seed": f"{seed:#018x}", "calib": ci})
            del Xrot, Qrot, Crot, A, TH
        d10 = [u["r10"] - q["r10"] for q, u in zip(qam_runs, upq_runs)]
        d100 = [u["r100"] - q["r100"] for q, u in zip(qam_runs, upq_runs)]
        m10, ci10 = mean_ci(d10)
        m100, ci100 = mean_ci(d100)
        res["headline"][name] = {
            "qam": summarize_runs(qam_runs),
            "upq": summarize_runs(upq_runs),
            "paired_diff": {"r10_mean": m10, "r10_ci95": ci10,
                            "r100_mean": m100, "r100_ci95": ci100},
            "runs": {"tags": run_tags,
                     "qam_r10": [r["r10"] for r in qam_runs],
                     "upq_r10": [r["r10"] for r in upq_runs],
                     "qam_r100": [r["r100"] for r in qam_runs],
                     "upq_r100": [r["r100"] for r in upq_runs],
                     "qam_distortion": [r["distortion"] for r in qam_runs],
                     "upq_distortion": [r["distortion"] for r in upq_runs],
                     "upq_fit": [r["fit"] for r in upq_runs]},
            "seconds": time.time() - t0,
        }
        res["margins"][name] = margins_from_scores(gt_sc) | {
            "epsilon_bound": {
                "qam_eps_rms": res["headline"][name]["qam"]["eps_rms_mean"],
                "qam_eps_p95": res["headline"][name]["qam"]["eps_p95_mean"],
                "upq_eps_rms": res["headline"][name]["upq"]["eps_rms_mean"],
                "upq_eps_p95": res["headline"][name]["upq"]["eps_p95_mean"],
            }}
        h = res["headline"][name]
        print(f"[headline] {name}: QAM r@10 {h['qam']['r10_mean']:.4f}  "
              f"UPQ r@10 {h['upq']['r10_mean']:.4f}  "
              f"paired d {m10:+.4f} [{ci10[0]:+.4f},{ci10[1]:+.4f}]  "
              f"[{time.time()-t0:.0f}s]")
        save_results(res)


FRONTIER_LADDER = [
    (2, [("binary", None)]),
    (10, [("qam", (4, 6)), ("qam", (5, 5)), ("upq", 1024)]),
    (11, [("qam", (5, 6)), ("upq", 2048)]),
    (12, [("qam", (5, 7)), ("qam", (6, 6)), ("upq", 4096)]),
    (13, [("qam", (6, 7)), ("upq", 8192)]),
]


def phase_frontier(res, seeds, datasets=FRONTIER_DATASETS):
    res.setdefault("frontier", {})
    for name in datasets:
        if name in res["frontier"]:
            print(f"[frontier] {name}: already done, skipping")
            continue
        t0 = time.time()
        X, Q, Cpool = load_norm(name)
        gt_ids, _ = gt_scores(name, X, Q)
        acc: dict[tuple, list] = {}
        for seed in seeds:
            signs = splitmix64_signs(seed, X.shape[1])
            bs = block_size_for(X.shape[1])
            Xrot = rotate(X, signs, bs)
            Qrot = rotate(Q, signs, bs)
            cal = rotate(Cpool[:CAL_ROWS], signs, bs)  # calibration sample 0
            A, TH = pair_polar(Xrot)
            A_cal = np.hypot(cal[:, 0::2], cal[:, 1::2]).astype(np.float32)
            sigma = sigma_from_amps(A_cal)
            for bits, configs in FRONTIER_LADDER:
                for codec, cfg in configs:
                    if codec == "qam":
                        r = run_qam(A, TH, sigma, cfg[0], cfg[1], Xrot, Qrot, gt_ids)
                    elif codec == "upq":
                        r = run_upq(A, TH, sigma, A_cal, cfg, Xrot, Qrot, gt_ids)
                    else:
                        r = run_binary(Xrot, Qrot, gt_ids)
                    acc.setdefault((bits, codec, str(cfg)), []).append(r)
            del Xrot, Qrot, A, TH
        rows = []
        for (bits, codec, cfg), runs in sorted(acc.items()):
            s = summarize_runs(runs)
            rows.append({"bits_per_pair": bits, "codec": codec, "config": cfg,
                         "r10_mean": s["r10_mean"], "r10_ci95": s["r10_ci95"],
                         "r100_mean": s["r100_mean"], "r100_ci95": s["r100_ci95"],
                         "distortion_mean": s["distortion_mean"],
                         "runs_r10": [r["r10"] for r in runs]})
        res["frontier"][name] = rows
        print(f"[frontier] {name}: {len(rows)} configs x {len(seeds)} seeds "
              f"[{time.time()-t0:.0f}s]")
        save_results(res)


def phase_comparators(res, seeds, datasets=FRONTIER_DATASETS):
    res.setdefault("comparators", {})
    for name in datasets:
        if name in res["comparators"]:
            print(f"[comparators] {name}: already done, skipping")
            continue
        t0 = time.time()
        X, Q, Cpool = load_norm(name)
        gt_ids, _ = gt_scores(name, X, Q)
        acc: dict[str, list] = {}
        for seed in seeds:
            signs = splitmix64_signs(seed, X.shape[1])
            bs = block_size_for(X.shape[1])
            Xrot = rotate(X, signs, bs)
            Qrot = rotate(Q, signs, bs)
            cal_rot = rotate(Cpool[:CAL_ROWS], signs, bs)
            A_cal = np.hypot(cal_rot[:, 0::2], cal_rot[:, 1::2]).astype(np.float32)
            sigma = sigma_from_amps(A_cal)
            # (i) 2-D k-means on rotated sigma-normalized pairs (family bound)
            acc.setdefault("kmeans2d_rot", []).append(
                run_kmeans2d(Xrot, Qrot, cal_rot, gt_ids, seed))
            # (ii) same codebook family on RAW pairs — isolates rotation
            acc.setdefault("kmeans2d_raw", []).append(
                run_kmeans2d(X, Q, Cpool[:CAL_ROWS], gt_ids, seed ^ 0xA5A5))
            # (iii) scalar quantization 5.5 b/dim (5/6 alternating, rotated)
            acc.setdefault("sq_5p5", []).append(
                run_sq55(Xrot, sigma, Qrot, gt_ids))
            # (iv) 1-bit sign binary (2 b/pair; also on the frontier ladder)
            acc.setdefault("binary_sketch", []).append(
                run_binary(Xrot, Qrot, gt_ids))
            del Xrot, Qrot
        res["comparators"][name] = {
            k: summarize_runs(v) | {"runs_r10": [r["r10"] for r in v]}
            for k, v in acc.items()}
        res["comparators"][name]["notes"] = (
            "all at 11 b/pair except binary_sketch (2 b/pair). kmeans2d_* = "
            "faiss k-means, 2048 shared centroids on sigma-normalized pairs, "
            "seeded from the same-sample UPQ codebook so it descends from "
            f"UPQ's distortion (niter={KMEANS_NITER}, grid-LUT encode); "
            "_raw has no rotation "
            "(and its per-pair sigma/calibration is deterministic, so its 3 "
            "runs differ only in the k-means seed). sq_5p5 = uniform mid-rise "
            "on [-4,4] sigma-normalized, 5/6 bits alternating per dim.")
        print(f"[comparators] {name}: done [{time.time()-t0:.0f}s]")
        save_results(res)


# ----------------------------------------------------------------------------
# report
# ----------------------------------------------------------------------------
def print_summary(res):
    print("\n" + "=" * 88)
    print("HEADLINE — QAM(5,6) vs UPQ(2048) at 11 b/pair, 15 runs "
          "(3 seeds x 5 calibrations), paired 95% t CI")
    print("=" * 88)
    hdr = f"{'dataset':<24}{'QAM r@10':>10}{'UPQ r@10':>10}{'paired d(r@10) [95% CI]':>28}{'d(r@100)':>10}"
    print(hdr)
    for name, h in res.get("headline", {}).items():
        d = h["paired_diff"]
        lo, hi = d["r10_ci95"]
        strict = "+" if lo > 0 else ("=" if hi >= 0 else "-")
        print(f"{name:<24}{h['qam']['r10_mean']:>10.4f}{h['upq']['r10_mean']:>10.4f}"
              f"{d['r10_mean']:>+11.4f} [{lo:+.4f},{hi:+.4f}] {strict}"
              f"{d['r100_mean']:>+10.4f}")
    if res.get("frontier"):
        print("\n" + "=" * 88)
        print("FRONTIER — matched-bits ladder (3 seeds x 1 calibration, mean r@10)")
        print("=" * 88)
        for name, rows in res["frontier"].items():
            print(f"-- {name}")
            for r in rows:
                print(f"   {r['bits_per_pair']:>3} b/pair  {r['codec']:<7}{r['config']:<10}"
                      f"r@10 {r['r10_mean']:.4f} [{r['r10_ci95'][0]:.4f},{r['r10_ci95'][1]:.4f}]"
                      f"  dist {r['distortion_mean']:.3e}")
    if res.get("comparators"):
        print("\n" + "=" * 88)
        print("COMPARATORS at 11 b/pair (binary at 2 b/pair)")
        print("=" * 88)
        for name, comp in res["comparators"].items():
            print(f"-- {name}")
            for k in ("kmeans2d_rot", "kmeans2d_raw", "sq_5p5", "binary_sketch"):
                if k in comp:
                    c = comp[k]
                    print(f"   {k:<15} r@10 {c['r10_mean']:.4f} "
                          f"[{c['r10_ci95'][0]:.4f},{c['r10_ci95'][1]:.4f}]  "
                          f"dist {c['distortion_mean']:.3e}")
    if res.get("margins"):
        print("\n" + "=" * 88)
        print("MARGINS — exact-GT #10 vs #11 cosine gap vs reconstruction eps")
        print("=" * 88)
        print(f"{'dataset':<24}{'p10':>10}{'p50':>10}{'p90':>10}"
              f"{'eps_rms qam':>13}{'eps_rms upq':>13}")
        for name, m in res["margins"].items():
            e = m["epsilon_bound"]
            print(f"{name:<24}{m['margin_p10']:>10.2e}{m['margin_p50']:>10.2e}"
                  f"{m['margin_p90']:>10.2e}{e['qam_eps_rms']:>13.4f}{e['upq_eps_rms']:>13.4f}")


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
_OUT_PATH = OUT_DEFAULT


def save_results(res):
    _OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_OUT_PATH, "w") as f:
        json.dump(res, f, indent=1, default=float)


def main():
    global _OUT_PATH
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--phase", default="all",
                    choices=["sanity", "headline", "frontier", "comparators", "all", "report"])
    ap.add_argument("--datasets", nargs="+", default=None,
                    help="restrict headline datasets (default: all 7)")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    args = ap.parse_args()
    _OUT_PATH = Path(args.out)

    res = {}
    if _OUT_PATH.exists():
        res = json.loads(_OUT_PATH.read_text())
    res.setdefault("meta", {})
    res["meta"].setdefault("protocol", {
        "corpus": N_CORPUS, "queries": N_QUERIES,
        "calibration": f"{args.seeds} rotation seeds x {N_CALIB} disjoint "
                       f"{CAL_ROWS}-row samples from corpus rows "
                       f"{CAL_LO}..{CAL_LO + N_CALIB * CAL_ROWS} (never overlap corpus)",
        "seeds": [f"{s:#018x}" for s in rotation_seeds(args.seeds)],
        "search": f"sign-sketch (from codes) top-{CK} -> exact rerank on "
                  "renormalized decode -> top-10/100 vs exact f32 cosine GT",
        "sift_gist": "L2-normalized cosine surrogate (paper convention)",
        "upq_fit": f"ring sweep {list(RING_SWEEP)}, pow2 phases >= {MIN_RING_PHASES}, "
                   f"{FIT_ITERS} alternating iters (port of src/codec/upq.rs)",
        "qam": "Lloyd-Max on unit Rayleigh (R_MAX=8), uniform phases "
               "(port of src/codec/qam_lloyd_max.rs)",
    })
    seeds = rotation_seeds(args.seeds)
    datasets = args.datasets or ALL_DATASETS

    t0 = time.time()
    if args.phase in ("sanity", "all"):
        phase_sanity(res)
        save_results(res)
    if args.phase in ("headline", "all"):
        phase_headline(res, seeds, datasets)
    if args.phase in ("frontier", "all"):
        phase_frontier(res, seeds)
    if args.phase in ("comparators", "all"):
        phase_comparators(res, seeds)

    res["meta"].setdefault("wall_seconds", {})
    res["meta"]["wall_seconds"][args.phase] = res["meta"]["wall_seconds"].get(args.phase, 0.0) + (time.time() - t0)
    save_results(res)
    print_summary(res)
    print(f"\nwrote {_OUT_PATH}  (this invocation: {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
