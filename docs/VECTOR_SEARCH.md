# Valise Vector Search

This document is the source of truth for how Valise does vector search, what the
production configuration is and why, what the recall/latency/storage envelope
actually is, and — most importantly — **what has already been tried and failed**,
so the same dead ends are not re-explored.

If a number here disagrees with a benchmark, re-run the benchmark; if the design
here disagrees with the code, the code is a bug (file an issue). All claims below
are backed by committed experiments — commit hashes and `bench/results/sweep/*`
files are cited inline.

---

## 1. Production configuration: QAM Lloyd-Max (5, 6)

**The production codec is QAM Lloyd-Max with 5 amplitude bits and 6 phase bits
per complex pair — written `(5,6)`.** It is the default chosen by
`register_codec_qam_from_sample` (`DEFAULT_AMP_BITS = 5`, `DEFAULT_PHASE_BITS = 6`
in `src/codec/qam_lloyd_max.rs`) and the only config with a hand-tuned SIMD
rerank kernel. Other budgets are supported — `(5,6)` is the *recommended*
default, not the only option: `register_codec_qam_from_sample_with_bits(dim,
amp, phase, sample)` registers any `(amp, phase)`, and all of them take the
sketch search path (see §6). `(8,8)` trades ~30 % more storage for ~+2.5 pts
recall (0.965 → 0.990 on Cohere d=768).

At dim 768 a `(5,6)` vector occupies **576 B** on the codec (256 B amplitude
stream + 320 B phase stream, each padded to a 64-byte cache line), **~617 B/vec
on disk** including frame/segment overhead — a **~5.3× shrink** versus 3072 B
of f32.

---

## 2. How it works (end to end)

```
        ingest (write)                              search (read)
  ┌───────────────────────────┐          ┌────────────────────────────────────┐
  │ x (f32, dim d)            │          │ q (f32, dim d)                      │
  │  │ normalize → unit dir   │          │  │ rotate (same block-Hadamard)     │
  │  │ block-Hadamard rotate  │          │  │ sign(rot(q)) → query sketch      │
  │  │ pair → polar (a, θ)    │          │  ▼                                  │
  │  │ Lloyd-Max quantize:    │          │ STAGE 1 — candidate selection       │
  │  │   amp 5b, phase 6b     │          │  Hamming(query sketch, db sketch)   │
  │  ▼                        │          │  over ALL active vectors (popcount, │
  │ store amp+phase codes     │          │  SIMD), keep the channel_k closest  │
  │ (576 B/vec @ d=768)       │          │  ▼                                  │
  └───────────────────────────┘          │ STAGE 2 — rerank (QAM-sliding i8)   │
                                          │  score candidates with the i8       │
   at file-open (free):                   │  asymmetric kernel, keep top 3·k    │
   derive 1-bit/dim SIGN SKETCH           │  ▼                                  │
   from the stored phase codes            │ STAGE 3 — optional f32 rerank       │
   (sign(cos θ), sign(sin θ)),            │  (VectorFidelity::Full) decode the  │
   96 B/vec, in RAM only                  │  survivors, recompute the metric    │
                                          │  ▼ top-k                            │
                                          └────────────────────────────────────┘
```

**Key properties:**

- **No persisted index.** No HNSW graph, no IVF lists, no CSR vote segment. The
  only search structure is the in-memory **sign sketch** (1 bit/dim), which is
  *derived for free* from the stored phase codes when the file is opened
  (`build_vector_base_ptrs` in `src/file/vector_search.rs`, `QamSlidingEngine::sign_sketch` in
  `src/codec/qam_sliding.rs`). Nothing is written at commit beyond the codes.

- **Selection is a dense, full-coverage scan.** Every active vector is compared
  every query (popcount-Hamming, NEON / AVX2). It is `O(N)` but cache-friendly
  and ~free per candidate (96 B/vec). It is *not* an approximate graph walk —
  there is no recall loss from "unreachable" nodes.

- **The candidate budget `channel_k` is the recall/latency lever**, not a graph
  parameter. See §4.

- **Recall is bounded by the codec, not the search.** See §3.

The dispatch is implicit per embedding space: a QAM(5,6) space takes the
sketch→rerank path; any other space falls back to a full brute-force decode scan
(`brute_force_vector`).

---

## 3. Recall is codec-bound — the most important thing to understand

The end-to-end recall ceiling on a given dataset is set by **how faithfully the
codec reconstructs the vectors**, *not* by the candidate selection or the rerank
pool. Valise stores only the lossy codes — it keeps **no f32** — so the best any
search can ever do is rank by cosine on the *reconstruction*.

Measured on real Cohere `text-embedding` d=768 (100k), recall@10 vs exact f32
(measured with the exploratory harnesses that produced the sweep files
in §7; those were not kept in the repository):

| reference truth / rerank target                    | recall@10 |
|----------------------------------------------------|-----------|
| selection coverage at ck≥4000 (true nbrs in pool)  | **1.000** |
| rerank on **reconstruction** (what Valise does)       | **0.965–0.968** ← the ceiling |
| rerank on **codec-achievable** GT                  | →1.000    |
| rerank on **original f32** (Valise does NOT store f32) | →1.000    |

Read this carefully, because it is the single most common source of confusion:

- **Selection is not the bottleneck.** At ck≈2000 the sketch already covers
  ~99.9% of the true top-10; at ck≈4000 it covers 100%. (the d=768 sweep
  stage decomposition; `bench/results/sweep/06_simhash_sketch.txt`.)
- **"Coverage 1.0 but recall 0.965" is not a contradiction.** The dropped
  neighbors *are* in the candidate set, but they are borderline (#9/#10 true
  neighbors) whose lossy reconstruction ranks them at ~#10–11, just outside the
  top-k. Finding a neighbor (coverage) ≠ ranking it correctly (recall); correct
  ranking needs an accurate score, and the score is computed on a lossy
  reconstruction.
- **Why Cohere caps lower than OpenAI.** Cohere's block-Hadamard-rotated pair
  distribution is far less Rayleigh-like than OpenAI-Large's, so the codec's
  rate–distortion assumption is looser: relative codec MSE ≈ **0.86** (Cohere)
  vs **0.005** (OpenAI-Large). OpenAI-Large reaches ~0.99 at (5,6); Cohere is
  ~0.965. (`ccfb6aa` valise-codec-ablation.)

**Corollary — the only levers that raise recall above the ceiling are (a) more
codec bits or (b) storing full-precision vectors for an exact rerank stage.**
Candidate budget and rerank-pool size cannot exceed it. See §5 and §6.

---

## 4. Candidate budget (`channel_k`) — and why `N/4` was removed

`channel_k` is the number of sketch candidates fed into rerank. When the caller
passes `None`, the engine uses a **fixed, corpus-size-INDEPENDENT default**:

```
channel_k = max(4 * k, DEFAULT_SKETCH_CANDIDATE_BUDGET)   // 2048, clamped to N
```

(`DEFAULT_SKETCH_CANDIDATE_BUDGET` in `src/file/query_types.rs`.)

### Do not reintroduce the `N/4` rule

A previous default scaled the budget with the corpus: `channel_k = N/4`. **It was
wrong and has been removed.** Two reasons:

1. **It is `O(N)` — it defeats the entire point of a candidate stage.** At 1M
   vectors `N/4` reranks 250k candidates per query; at 10M, 2.5M. The candidate
   stage exists precisely to make rerank cost *independent* of N.

2. **Its high recall was an artifact of the law of large numbers, not selection
   quality.** Reranking 25% of the corpus recovers the true top-k with near
   certainty *regardless of how good the sketch is* — a random selector would
   also score ~0.99 at ck=N/4. It measured coverage *volume*, not selection.

The measured truth: selection coverage **saturates by ck≈2000–4000** at d=768
(`06_simhash_sketch.txt` in the §7 sweep set). Everything past that is pure
rerank tax for zero recall — at 100k, ck=2000 and ck=25000 give the identical
0.965. The fixed default sits at the knee.

### Tuning beyond the default

For a different operating point, pass `channel_k` explicitly. The principled way
to choose it is the **binomial 2σ calibration** already used for the text channel
(`bench/src/bin/valise-e2e-bench.rs`): sweep ck on a query sample, pick the smallest
tier whose sample recall clears `target − 2σ`. This yields a fixed, N-independent
budget chosen by *measured* selection quality. (Per-space calibrated `channel_k`
storage is a documented future hook, not yet implemented.)

### `accurate()` vs `fast()`

`VectorSearchQuery::accurate()` and `::fast()` both leave `channel_k = None`
(default budget) and differ only in rerank fidelity:

- `accurate()` → `VectorFidelity::Full` (f32 rerank pass) → reaches the codec
  ceiling (~0.965 on Cohere).
- `fast()` → `VectorFidelity::Lossy` (i8 QAM-sliding scores only) → ~3 points
  lower, because the i8 elide-norm scorer mis-orders borderline neighbors. Cheaper.

> Historical note: `accurate()` used to pin `channel_k = max(4k, 100)` ("footgun
> guard"). That value *undershot* recall (≈0.87) — it was the very budget the old
> code comment called "collapsed recall." Removed.

---

## 5. Current best numbers (QAM (5,6), Cohere d=768, 100k, k=10)

| metric | value | notes |
|---|---|---|
| Recall@10 (vs exact f32) | **0.965–0.97** | codec-reconstruction-bound (§3) |
| Storage | **576 B/vec** codec (5.3× vs f32) · **617 B/vec** on disk (~5.0×) | f32 = 3072 B |
| Latency, single query | ~470 µs @ ck=2000 (all-core) · ~1061 µs p50 single-thread | at the ceiling operating point |
| Latency, concurrent | ~13–15k q/s @ ck=1000–2000 | Valise concurrent read |
| Ingest | ~700k vec/s | incremental encode, no graph build |

Against USearch HNSW on the same data (`0b6b529` usearch-bpv-sweep): QAM (5,6) is
**~3× more storage-efficient at the same recall band**. USearch only reaches
0.994 at ~1685 B/vec (F16/M=16) and exceeds it (0.998) at ~1941 B/vec
(F16/M=48) — a band QAM (5,6) cannot reach (see §6).

---

## 6. Known limitations

1. **The SIMD *sliding* kernel is (5,6)-only — but the *sketch path* is not.**
   Any QAM `(amp_bits, phase_bits)` now derives a sign-sketch at open
   (`QamLloydMaxCodec::sign_sketch`) and takes the sketch-then-rerank path:
   `(5,6)` reranks with the register-resident i8 sliding kernel
   (`QamSlidingEngine`, whose 32/64-entry NEON/AVX2 TBL tables can't hold larger
   codebooks), every other config reranks with the general asymmetric kernel —
   slower per candidate, but still `O(channel_k)`, not the old `O(N)` brute
   force. Register a non-(5,6) codec with
   `register_codec_qam_from_sample_with_bits(dim, amp, phase, sample)`.
   Measured on Cohere d=768, 100k (sketch path, default budget):
   **(5,6) → recall 0.965 @ 617 B/vec; (8,8) → recall 0.990 @ 809 B/vec.** So
   the higher-recall config is now reachable at sketch speed; the remaining gap
   is that its rerank uses the general (non-SIMD-sliding) kernel.

2. **No full-precision rerank stage.** Valise stores only the codec bytes. Reranking
   on the reconstruction caps recall at codec fidelity (§3). Systems that reach
   ~1.0 (FAISS `IVFPQ+RFlat`, ScaNN, DiskANN) keep the originals (or a residual)
   for a final exact rerank. Valise does not — by design, for storage. To exceed the
   ceiling you must add such a stage (and pay the storage).

3. **Vestigial format fields.** `AutoPromote` (in the create contract) and
   `EmbeddingSpaceDesc.secondary_codec_id` are dead since the v2.2/v2.3 ANN/vote
   burial. They are retained as persisted fields only for format stability; no
   runtime path reads them, and `register_embedding_space` now *rejects* a
   non-`None` `secondary_codec_id`.

---

## 7. What we tried and why it failed (do not re-explore without new evidence)

Every row is a committed experiment. The recurring lesson: **on a tuned codec,
recall is bounded by reconstruction fidelity; selection is already near-optimal;
only more codec bits (or stored f32) move the ceiling.**

| Idea | Verdict | Why | Source |
|---|---|---|---|
| **Residual VQ refine** (1 byte = index into K=256 residual codebook, added at rerank) | ✗ no effect | Residual `x − ŷ` lives in d=768; one global codebook can't cover it. ±0.001 (noise). | `6a18ba0` |
| **Residual PQ refine** (M-subspace product-quantized residual, 1–32 B/vec) | ✗ no effect | Residuals after (5,6) are near-isotropic; per-subspace entropy exceeds 8 bits. All within noise. Conclusion: *raise codec bits, not side-info.* | `2b352de` |
| **Pre-rotate mixer** (random orthogonal matrix before block-Hadamard) | ✗ don't ship | +0.07 pts at (5,6) (noise); costs ~100× encode time. The ceiling is codebook coarseness, not uneven pair energy. | `c4df5c0` |
| **Vote-bucket phase radius sweep** (4 → 32) | ✗ no effect | Held ~0.47 on Cohere — codec MSE is the wall, not the walk window. | `ccfb6aa` |
| **Shorter sketch** (512/384/256/128 bits instead of 768) | ✗ recall craters | Latency is cache-bound, not bandwidth-bound, so fewer bits don't speed it up; they only lose resolution. | `14_shorter_sketch.txt` |
| **Two-tier scan** (64-bit prefix → top-M → full sketch) | ✗ fails twice | Low-res prefix can't preserve neighborhoods (keeps only 39–70% of true nbrs); and the single-core bottleneck is rerank, not scan. | `15_two_tier.txt` |
| **LSH prune / PCA-coarse prune** | ✗ fails to prune | Same lesson as prefix/shorter-sketch: nothing prunes below full-resolution sign Hamming without dropping neighbors. | `09_lsh_prune.txt`, `10_pca_coarse.txt` |
| **`N/4` candidate budget** | ✗ removed | `O(N)`; its high recall was LLN coverage volume, not selection. Replaced by a fixed default. | this doc §4 |
| **i8 rerank store** (8-bit) | ✗ too coarse | 0.876 vs the 0.933 perfect-rerank ceiling; the top-100 boundary is razor-close. int16 hits the ceiling but costs 1632 B/vec. QAM (5.5 b) gets 0.924 at 672 B/vec — the storage sweet spot. | `16`, `17_bits_vs_recall.txt` |
| **CSR vote index** (persisted) | ✗ replaced (v2.3) | Scattered posting walk cost ~L·T, cancelling the rerank savings of a small ck. The dense sign-sketch reaches iso-recall at ~16× fewer candidates, ~free, zero tuning. | `04_coverage_TL.txt`, `06`, `4be1455` |
| **HNSW / IVF / ANN graphs** | ✗ buried (v2.2) | Persisted graph + build cost + worse update story; QAM is ~3× more storage-efficient at the same recall band. | `26f3564`, `0b6b529` |

### What *worked* and is in production

| Decision | Why | Source |
|---|---|---|
| **Sign-sketch (SimHash) candidate gen** | 1 bit/dim, full coverage, ~free (derived from phase codes), ~16× fewer rerank candidates than the vote index at iso-recall. | `06_simhash_sketch.txt`, `4be1455` |
| **QAM (5,6) production codec** | Best recall/byte balance: 0.97 @ 576 B/vec. (Note: on the *old vote index*, (5,6) collapsed to 0.59 because the phase-match radius got too tight — that was a vote-index artifact; on the sketch path (5,6) wins. Don't confuse the two.) | `70f68cf`, `0b6b529` |
| **Per-row `‖ŷ‖` in the cosine asymmetric distance** | Omitting it collapsed Cohere recall 0.966 → 0.30 (per-row norm swings dominate on low-fidelity data). Now computed/cached (4 B/vec, 2.3–2.9× rerank speedup). | `2c2793b`, `5401793`, `e0f9344` |
| **"More codec bits → more recall"** | The only clean lever. **(5,6)→(8,8): +2.4 pts (0.97→0.993) for +192 B/vec** (768 B, measured). A single extra phase bit (5,6)→(5,7) is ~+0.7 pts for +64 B/vec (one 64-byte-aligned cache line). | `0b6b529`, `2b352de`, `e0f9344` |

---

## 8. Reproducing

Everything below runs through the end-to-end bench. Get the data first:

```bash
python3 bench/prep_data.py --list        # options and sizes
python3 bench/prep_data.py all-small     # scifact + sift-1m, ~504 MB
```

- Engine-level (latency / recall / storage / concurrent throughput, stage
  decomposition):

  ```bash
  cargo build --release -p valise-bench --bin valise-e2e-bench
  target/release/valise-e2e-bench \
      --beir-dir bench/beir-data/scifact \
      --vector-dir bench/datasets/sift-1m \
      --vector-n 100000 --out bench/results/e2e.json
  ```

- Codec comparison: rerun with `--codec upq` and diff the reports.
- Full end-to-end vs peers (Tantivy / USearch / hnsw_rs) is part of the same
  binary; see `bench/REPRODUCE.md` for the knobs and the reference numbers.

The numbers quoted in this document were measured on the Cohere d=768 corpus
(`bench/datasets/cohere-medium-1m-f32/`), which is gated on HuggingFace and
must be built by hand — `bench/prep_data.py --list` explains how. `sift-1m`
is the substitute that reproduces without credentials; absolute figures will
differ with dimension and distribution, the shape of the curves should not.
