# Contributing to the docs

## Build order: `maturin develop` BEFORE `mkdocs build`

mkdocstrings imports the `valise` package at build time, which imports the compiled
`valise._native` cdylib. If you run `mkdocs build` without first building the
extension, every `:::` reference directive on the API page fails to resolve and
`--strict` turns the build red.

The correct local flow (from `bindings/valise-py`):

```bash
# 1. Build the native extension into the venv.
source .venv/bin/activate
maturin develop

# 2. Install the docs + test dependency groups.
uv pip install --python .venv -e '.[docs,test]'

# 3. Run the gates.
python scripts/check_native_stub.py          # native-vs-stub name drift
python -m mypy --strict python/valise            # facade type check
python -m pytest -q python/tests              # the parity suite
python -m pytest -q --doctest-modules python/valise   # the example-rot gate
mkdocs build --strict                         # the docs-drift gate

# 4. Spot-check locally.
mkdocs serve
```

## Examples are tests

Every runnable docstring example must be hermetic and deterministic:

- Use `tempfile` for any store path — never a bare relative `"data.vls"`.
- A vector field always passes `Vector(dim=..., calibrate=Auto(sample=8))`,
  and the example always commits at least one staged vector before any vector
  search (otherwise `NotCalibratedError`).
- `dim` is a positive multiple of 64.
- Use deterministic input (`np.zeros`, fixed arrays, or
  `np.random.default_rng(0)`) — never bare `np.random.rand`.
- Any printed score/search-output line gets `# doctest: +SKIP` (lossy QAM + RNG
  is not reproducible). Structural lines (counts, keys, `len(result)`) may assert
  real values.

Keep load-bearing runnable examples as **docstrings** in `python/valise` so they are
a single source of truth: mkdocstrings renders them into the API page, and
`pytest --doctest-modules` tests them. The Markdown snippets in the concept pages
are illustrative and are not executed.
