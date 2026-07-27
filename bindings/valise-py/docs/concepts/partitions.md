# Partitions

A **partitioned collection** routes records by time (or a custom rule) into
physical collections named `{base}:{period}`, all sharing one schema and one
set of auto spaces — so there is exactly one codec calibration for the whole
family, not one per partition.

```python
from valise import Store, Schema, Vector, Record, Search, Partition, Window

store = Store.open("events.vls")
events = store.partitioned("events", Schema()
    .text("body")
    .vector("dense", Vector(dim=64)),
    Partition.BY_DAY)                  # or Partition.BY_MONTH

with store.writer() as w:
    w.put("events", "e-1", Record().text("body", "deploy started")
                                   .vector("dense", vec)
                                   .at(1_750_000_000))
    w.commit()
```

Routing reads the record's `created_at` (`.at(...)`, else the wall clock):
`Partition.BY_DAY` yields `events:2026-06-11`-style collections,
`Partition.BY_MONTH` yields `events:2026-06`. Custom routing
(`Partition::Custom`, a Rust closure) is Rust-only — closures cannot be
serialized across the FFI boundary (see `docs/PARITY.md`).

## Views — search a window

A **view** resolves a window of partitions; `search_view` fuses results
globally across them (one ranked list, not per-partition lists):

```python
view = events.view(Window.last_days(7))           # partitions touching the last 7 days
view = events.view(Window.range(t0, t1))           # inclusive unix-second range
view = events.view(Window.partitions(["2026-06-10"]))  # explicit period suffixes
view = events.all()                                # every partition

hits = store.search_view(view, Search().text("body", "deploy").top_k(10))
```

## Retention — `forget_before`

`forget_before(cutoff)` soft-deletes (tombstones) every **whole partition**
older than the cutoff and returns the number of **frames** tombstoned. Bytes
are reclaimed by an explicit [`store.compact()`](../reference.md) — nothing
fires automatically:

```python
dropped = events.forget_before(cutoff_unix_secs)
if store.stats().needs_compaction(0.3):
    store.compact()
```

*(The Rust API is identical: `Partition::ByDay`, `Window::LastDays`,
`Partitioned::view` / `forget_before`.)*
