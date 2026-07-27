# Schemas & Spaces

A **schema** declares a collection's fields in one call. Each field is a typed
spec — [`Text`](../reference.md) or [`Vector`](../reference.md) — and the
declaration is **persisted in the file**: reopen restores it.

```python
from valise import Schema, Vector

store.collection("kb", Schema()
    .text("body")                       # English BM25 (the default)
    .vector("dense", Vector(dim=64)))   # cosine, auto-calibrating QAM(5,6)
```

## Text fields

`Schema().text(name)` uses the default English analyzer (NFKC, case-fold,
Porter2 stemming, stopwords). Pass a spec for anything else:

```python
from valise import Text, Lang

Schema().text("body", Text(lang=Lang.RAW))   # pass-through whitespace tokenizer
```

At most one text field per collection in v1.

## Vector fields

A vector field needs its dimension; everything else has defaults:

```python
from valise import Vector, Metric, Qam, Upq, Auto, Now

Vector(dim=64)                                  # cosine, Qam(5, 6), Auto(50_000)
Vector(dim=768, metric=Metric.DOT)              # Metric.COSINE | DOT | L2
Vector(dim=768, codec=Upq())                    # higher-accuracy UPQ codec
Vector(dim=768, codec=Qam(8, 8))                # more bits, more recall
Vector(dim=768, calibrate=Auto(sample=1024))    # smaller calibration sample
Vector(dim=768, calibrate=Now(sample_ndarray))  # fit eagerly at declaration
```

The `dim` **must be a positive multiple of 64** — a hard constraint of the
polar codecs. `Schema().vector(name, 768)` is accepted as shorthand for
`Vector(dim=768)`.

### Calibration

Quantization codecs are fitted to your data. `Auto` (the default) defers the
fit to the **first commit that contains vectors** for the field, using up to
`sample` staged vectors (default 50 000). Until that commit, vector search on
the field raises `NotCalibratedError` (the message names the space, e.g.
`~auto/kb/dense`). `Now(ndarray)` fits eagerly at declaration from a
representative sample you supply.

### Codec choice

Per field, opaque value objects: `Qam(amp_bits=5, phase_bits=6)` is the
production default; `Upq(cells=2048, design=Design.EMPIRICAL)` is the
higher-accuracy alternative. The choice is persisted with the schema and
survives reopen — even if the field has not been calibrated yet.

## Reopen and re-declaration

Schemas are persisted in-file, so **reopen needs no re-declaration**:

```python
store = Store.open("kb.vls")
store.get("kb", "rust")          # schema restored from the file
```

Calling `store.collection(...)` again with the **same** schema is an
idempotent no-op. A purely **additive** re-declare (new fields appended,
existing fields unchanged) is accepted and updates the persisted schema.
Anything else — a changed field, codec, metric, or calibration — raises
`SchemaMismatchError` with field-level detail.

## Shared spaces (advanced)

By default every inline field gets its own private space, named
`~auto/{collection}/{field}` — you never see it unless you list
`store.spaces()` or read a `NotCalibratedError`. Names starting with `~` are
reserved (rejected for user collections and spaces).

When two collections must share one embedding space (their vectors mutually
searchable, one calibration), define the space once and bind it by handle:

```python
from valise import Vector, Text

emb = store.define_space("emb", Vector(dim=768))
en  = store.define_space("en", Text())

store.collection("articles", Schema().text("body", Text(space=en))
                                     .vector("dense", Vector(space=emb)))
store.collection("comments", Schema().vector("dense", Vector(space=emb)))
```

Codec, metric, and calibration belong to the **space** — passing them on a
shared binding (`Vector(space=..., codec=...)`) is a declaration-time error;
configure them in `define_space` instead. `define_space` is idempotent by
name; `store.space(name)` looks a space up, `store.spaces()` lists all of
them (auto spaces flagged).

!!! note "One caveat on shared spaces"
    A shared vector space that has **never been calibrated** (no vector commit
    yet) is not reconstructable from the persisted schema docs; after a reopen
    its collections come back shape-less until you `define_space` + re-declare
    once. Calibrated shared spaces, all text spaces, and all inline/auto fields
    (the default path) restore fine.
