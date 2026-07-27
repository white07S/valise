# Baseline snapshots

Committed numeric snapshots of the SIMD microbench results, captured per
arch at Phase-1 close-out. Future phases regress against these.

## Files

- `neon_m_series.json` — M-series (aarch64-apple-darwin) baseline. NEON
  dispatcher path AND scalar reference, captured side-by-side. The
  `simd` variant in each kernel is the production code today (NEON);
  `scalar` is the always-available reference.
- `scalar_i7_8550u.json` — lab i7-8550U baseline. Captured BEFORE any
  AVX2 kernel lands. `simd` and `scalar` here are identical (both use
  the scalar fallback since no AVX2 path exists in Phase 1); they
  diverge as Phases 2–6 land.

## Capture procedure

```sh
cargo bench -p valise-bench
python3 bench/baseline/harvest.py > bench/baseline/<arch>.json
```

The harvester reads `target/criterion/<group>/<variant>/<size>/new/
estimates.json` and produces a flat JSON with `kernels.<group>.<variant>
.<size>.{mean_ns, median_ns}`. Bench output is deterministic enough
that mean drift > 5% between consecutive captures on the same box
indicates noise (thermal, background CPU); take the median across 3
runs in that case.
