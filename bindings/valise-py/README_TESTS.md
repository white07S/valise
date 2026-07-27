# valise Python bindings — tests & bench

## Build the extension

```bash
source .venv/bin/activate
cd bindings/valise-py
maturin develop          # first build compiles pyo3 + numpy + valise (minutes)
```

## Run the parity / safety / concurrency suite

```bash
source .venv/bin/activate
cd bindings/valise-py/python
python -m pytest -q
```

Suite layout (`python/tests/`):

| file | covers |
|---|---|
| `conftest.py` | shared `unit_vec` / `matrix` helpers, `store` fixture, `make_hybrid` |
| `test_roundtrip.py` | put/get, update-by-key remap, delete, put_auto, multi-vector, persistence across reopen |
| `test_calibration.py` | deferred calibration on commit (the only Python calibration mode) |
| `test_search.py` | text, vector-nearest, hybrid RRF (summing), weighted fusion, recency range/halflife, collection isolation, get_many |
| `test_validation.py` | dim mismatch, float64/non-contiguous rejection, bad space dim, bool key, divergent redefine, search field/channel errors |
| `test_concurrency.py` | GIL release proven via `Reader.search` (`py.detach`); commit non-deadlock vs a concurrent reader |
| `test_writer_contention.py` | (pre-existing) single-writer lock release + all-or-nothing put_many |

### Test invariants

- Vectors are always `np.float32` + `np.ascontiguousarray`.
- `DIM = 64` (smallest legal — a positive multiple of 64).
- Assertions are top-1 / membership / count only. QAM is lossy, so reconstructed
  vector **values** are never asserted equal to the input — only field name,
  dimension, and finiteness.
- float64 inputs are rejected by rust-numpy's `PyReadonlyArray<f32>` as a Python
  `TypeError` before the binding body runs, so those tests accept
  `(TypeError, valise.ValidationError)`.

### Recency coverage

The Python binding exposes `recency_range`, `recency_half_life`, and
`recency_rrf_channel`. The parity suite covers range and half-life directly;
`RrfChannel` is covered through the builder/native lowering path and should get
a dedicated ranking assertion if its scoring behavior changes.

## Run the benchmark

```bash
source .venv/bin/activate
cd bindings/valise-py
python bench/bench.py            # defaults: --n 10000 --dim 64
python bench/bench.py --n 20000 --dim 768
```

The bench prints a fixed-width table (`op | n | total_ms | per_op_us | per_s`),
the **batching speedup** (per-call put vs zero-copy `put_many`), and a zero-copy
evidence block. It exits 0 even when the rejection probes fire (they are
expected). `--dim` must be a positive multiple of 64.
