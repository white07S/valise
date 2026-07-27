# Records

A **record** carries the text and vector field values for one key, plus optional
metadata. Build one with the chained [`Record`](../reference.md) builder, then
stage it with a [`Writer`](../reference.md).

```python
import numpy as np
from valise import Record

rec = (
    Record()
    .text("body", "the quick brown fox")
    .vector("dense", np.zeros(64, dtype=np.float32))
    .at(1_700_000_000)        # created_at, unix seconds
    .child_of("parent-key")   # link as a child of another record
)

with store.writer() as w:
    w.put("kb", "a", rec)
    w.commit()
```

- `text(field, value)` — set a text field.
- `vector(field, values)` — set a vector field; `values` is a 1-D `float32`
  ndarray, borrowed zero-copy by the native layer.
- `at(unix_secs)` — set the record timestamp. The engine requires `created_at`
  to be non-decreasing in commit order; out-of-order timestamps in one commit
  error at commit time.
- `child_of(parent)` — link this record under a parent key (`str | int | bytes`).

## Keys

A key is `str`, `int`, or `bytes`. Booleans and negative integers are rejected.

## Stored records

`Reader.get` / `Reader.get_many` (and `Store.get`) return a
[`Stored`](../reference.md) dataclass:

```python
s = store.reader().get("kb", "a")
s.key          # the record key
s.collection   # owning collection
s.created_at   # unix seconds
s.text         # the text field value, or None
s.vectors      # list[(field_name, np.ndarray)] — the moved arrays, not copied
```

!!! warning "Lossy vector codec"
    Vectors are stored through a quantization codec (QAM or UPQ, chosen per
    field in the [schema](spaces.md)), which is **lossy**. A round-tripped
    vector preserves its **dimension and finiteness**, but not its exact values.
    Do not assert that a fetched vector equals the one you stored.
