# QAM Lloyd-Max Codec — Specification

Status: **shipping codec wire spec**. The QAM params wire layout is `VLQL`
version 1 and remains part of the current v2.4 line (`FORMAT_MAJOR = 2`,
`FORMAT_MINOR = 3`). Any QAM params on-disk change requires a wire-version bump.
For the current vector search path, see [`VECTOR_SEARCH.md`](VECTOR_SEARCH.md):
queries use an in-memory sign-sketch scan derived from stored codes.

QAM Lloyd-Max is the default vector codec family. The previous primary codec
(`Int4PcaZero`) was deleted in the consolidation cleanup; its enum discriminant
survives in `CodecFamily` as the reserved `_LegacyInt4PcaZero` placeholder so
existing-discriminant bincode positions don't shift. The current implementation
also ships UPQ as an opt-in codec family.

## Mental model

Apply a fixed orthogonal rotation, pair adjacent coordinates into
complex numbers, then quantize amplitude and phase separately. Each
complex pair becomes an `(amp_idx, phase_idx)` pair stored as
`amp_bits + phase_bits` bits.

```
input v ∈ ℝᴰ
    │
    ▼   block-Hadamard with deterministic ±1 mask
H · v ∈ ℝᴰ
    │   pair (rot_v[2k], rot_v[2k+1]) → complex z_k
    ▼
z_k = a_k · exp(i · θ_k)      for k ∈ [0, num_pairs)
    │
    ▼   per-pair Lloyd-Max amp + uniform phase
amp_idx_k ∈ [0, 2^amp_bits)
phase_idx_k ∈ [0, 2^phase_bits)
```

With `(amp_bits, phase_bits) = (5, 6)` this is **11 bits per complex
pair = 5.5 bits per scalar = 0.6875 B/dim**.

| dim   | num_pairs | bytes/vec @ (5, 6), unpadded |
|------:|----------:|-------------------:|
|   128 |        64 |                88  |
|   768 |       384 |               528  |
|  1024 |       512 |               704  |
|  1536 |       768 |             1,056  |
|  3072 |     1,536 |             2,112  |

These are the bit-packed payload sizes. On disk the amplitude and phase
streams are each padded to a 64-byte cache line, so the figure quoted in
`VECTOR_SEARCH.md` is larger — 576 B at dim 768 rather than 528 B.

## Wire layout — `QamLloydMaxParams`

Source of truth: `src/format/qam_lloyd_max_params.rs`. Magic `VLQL`,
version 1, little-endian throughout:

```text
[magic VLQL                : 4 B            ]
[version u16 = 1           : 2 B            ]
[dimension u32             : 4 B            ]
[num_pairs u32             : 4 B            ] (= dimension / 2)
[block_size u32            : 4 B            ]
[rotation_seed u64         : 8 B            ]
[amp_bits u8               : 1 B            ]
[phase_bits u8             : 1 B            ]
[renormalize_at_decode u8  : 1 B            ] (0 = false, 1 = true)
[reserved u8 = 0           : 1 B            ]
[sigma_count u32           : 4 B            ] (= num_pairs)
[sigma_per_pair: f32 × P   : 4·num_pairs B  ]
[calibration_id            : 32 B           ]
```

`calibration_id` is a 32-byte BLAKE3 digest of the calibration corpus
fingerprint — recorded so two codecs built from the same sample produce
byte-identical params.

The persisted parameters fully describe the encoder/decoder:

- The block-Hadamard sign mask is derived deterministically from
  `rotation_seed` via SplitMix64 — never stored.
- The Lloyd-Max amplitude codebook is a closed-form function of
  `amp_bits` — never stored.
- `sigma_per_pair` is the only data-dependent table.

## Parameter constraints

| Field                   | Constraint                                                   |
|-------------------------|--------------------------------------------------------------|
| `dimension`             | even, positive, ≤ `CreateContractV1.max_dim`                 |
| `num_pairs`             | `= dimension / 2`                                            |
| `block_size`            | power of two, divides `dimension`                            |
| `rotation_seed`         | any `u64`                                                    |
| `amp_bits`              | `1 ≤ amp_bits ≤ 8` (`QAM_BITS_MIN..=QAM_BITS_MAX`)            |
| `phase_bits`            | `1 ≤ phase_bits ≤ 8`                                         |
| `renormalize_at_decode` | `bool`. `true` for cosine spaces; `false` otherwise          |
| `sigma_per_pair`        | length `= num_pairs`; each value finite, positive            |
| `calibration_id`        | non-zero (zero rejected at validate)                         |

`(amp_bits, phase_bits) = (5, 6)` is the production configuration. The
bit counts can be varied for experimentation but the SIMD asymmetric
kernel is hand-tuned for (5, 6).

## Block sizes per dimension

`block_size` must divide `dimension`. Recommended choices:

| dim  | recommended `block_size` |
|-----:|-------------------------:|
|  128 |                      128 |
|  768 |                      256 |
| 1024 |                     1024 |
| 1536 |                      512 |
| 3072 |                     1024 |

Larger blocks mix more coordinates per Hadamard hop and give better
recall when `dimension % block_size == 0`. The bench at
`bench/src/bin/valise-e2e-bench.rs` uses `block_size = 256` for the
Cohere d=768 corpus.

## Calibration procedure

1. Construct the codec with `QamLloydMaxCodec::with_config(dim,
   block_size, amp_bits, phase_bits, renormalize_at_decode)`.
2. Feed in a `&[Vec<f32>]` of representative vectors (a few thousand
   rows is enough — the bench uses the first 4 096). The codec rotates
   each vector, computes per-pair amplitude variances, and fits the
   Lloyd-Max codebooks per `sigma_per_pair[k]`.
3. Bake the codec into `QamLloydMaxParams` via
   `QamLloydMaxCodec::to_params(calibration_id)`.
4. Register with `ValiseFile::register_codec_qam(params)`.

The bench-only convenience wrapper `QamLloydMaxBench` (under the
`bench` feature) shortcuts steps 1-3 for benchmark harnesses.
Production callers go through `register_codec_qam` directly.

## Asymmetric distance — search hot path

The search path encodes the query once at the high-precision side, selects
candidates through the in-memory sign sketch, then scores against the quantized
DB side. The QAM codec exposes the rerank kernel via the `prepare_query` /
`asymmetric_distance` pair; the precomputed query blob carries the per-pair
`(cos θ_q, sin θ_q, a_q · cos θ_q, a_q · sin θ_q)` quadruple per pair, packed
for the SIMD kernel.

The old vote-index prototype relied on three architectural invariants of the
codec; the first two still explain why QAM can derive a useful sign sketch from
phase codes:

1. `amp_idx` is a monotonic measure of pair energy → it can be used as
   the inverted-list key directly.
2. `phase_idx` is a uniform modulo ring → neighborhood scans
   `[phase − r, phase + r] mod N_p` are O(r) and topologically clean.
3. amp and phase are stored as **independent** indices, never a joint
   codebook — a joint codebook would destroy CSR transpose and ruin
   O(1) phase-bucket routing.

## Codec family discriminant

`CodecFamily` is a bincode-tagged enum. The wire representation is the
variant ordinal:

```rust
pub enum CodecFamily {
    _LegacyInt4PcaZero, // ordinal 0 — reserved, never constructed
    QamLloydMax,        // ordinal 1
    Upq,                // ordinal 2
}
```

The `_LegacyInt4PcaZero` placeholder keeps `QamLloydMax`'s ordinal stable at 1.
Registering or matching the placeholder errors out at runtime — it exists only
so existing-discriminant decoders don't have to shift indices. Enum variants
remain append-only.

## Reproduction

The single benchmark binary at `bench/src/bin/valise-e2e-bench.rs`
exercises the QAM codec end-to-end: ingest → commit → sign-sketch search, with
peer comparison against `usearch` and `hnsw_rs`. See
[`bench/REPRODUCE.md`](../bench/REPRODUCE.md) for the command line and expected
numbers.

The `VLQL` wire-format version is independent of the top-level Valise format
minor. A change to the QAM params byte layout must bump the `VLQL` wire version
and add codec-level tests for old/new rejection behavior.
