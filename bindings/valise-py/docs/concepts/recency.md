# Recency

A [`Search`](../reference.md) can be biased or filtered by record age via
`Search.recency(...)`, using one of two typed strategies, plus an optional
reference "now".

## Hard window filter — `Range`

`Range(from_, to)` restricts results to the **inclusive** `[from_, to]` window
of unix-second timestamps. `from_` is named with a trailing underscore because
`from` is a Python keyword. Candidates are pruned before fusion; a very narrow
window over a large corpus can return fewer than `top_k` hits.

```python
from valise import Search, Range

q = Search().text("body", "fox").recency(Range(from_=1_700_000_000,
                                                to=1_700_086_400))
```

## Exponential decay — `HalfLife`

`HalfLife(days)` applies an exponential recency decay with the given half-life
(in days). Combine with `.now(unix_secs)` to set the reference time the decay is
measured from (otherwise the current wall clock is used).

```python
from valise import Search, HalfLife

q = (
    Search()
    .text("body", "fox")
    .recency(HalfLife(days=7.0))
    .now(1_700_000_000)
)
```

A third strategy treats recency as one more RRF channel (newest-first,
contributing `weight/(k + rank + 1)`; requires RRF fusion):

```python
from valise import Rrf, RrfChannel, Search

hits = store.search("notes", Search()
    .text("body", "release notes")
    .recency(RrfChannel(half_life_days=14, weight=0.5))
    .fuse(Rrf(k=60))
    .top_k(10))
```
