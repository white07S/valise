# The copy-under-write demo

The recording in the project README is real. These are the scripts it runs,
and you can run them yourself.

```bash
brew install vhs          # only needed to re-record the GIF
demo/record.sh            # stages /tmp/valise-demo, records, writes the gif
```

`record.sh` records against the **published wheel** by default, so what you
see is what `pip install valise` gives you. Pass `VALISE_DEMO_LOCAL=1` to
record against the working tree instead.

## What it actually does

| File | Role |
|---|---|
| `writer.py` | Holds one writer open and commits in a tight loop, so the file is never quiescent. Any copy taken while it runs is genuinely mid-write. |
| `copy_storm.sh` | Takes N copies with plain `cp`. No lock, no quiesce, no coordination — the naive thing everybody already does. |
| `verify_copies.py` | Opens every copy and checks it hard (below). |
| `copy-under-write.tape` | The vhs script that drives the recording. |
| `record.sh` | Stages a clean environment and calls vhs. |

## What counts as passing

A copy passes only if **all** of these hold:

- it opens at all;
- its record count is an exact multiple of the writer's commit batch — a
  count that isn't is proof we caught a half-applied commit;
- a search against it returns results;
- a record read back from it is byte-exact.

Anything else is a failure and is printed as one. The verifier exits non-zero
if a single copy fails, so this is usable as a test rather than a
demonstration.

## Running it without recording

```bash
python3 -m venv .venv && .venv/bin/pip install valise numpy
cargo build --release --bin valise

.venv/bin/python demo/writer.py --out corpus.vls --stop-file .stop &
sleep 4
demo/copy_storm.sh corpus.vls copies 50
touch .stop
.venv/bin/python demo/verify_copies.py --glob 'copies/*.vls'
```

The `--delay` flag on `writer.py` controls the pause between commits. The
recording uses `0.1` to keep each snapshot small enough that checking fifty
of them stays quick; the default `0.01` writes considerably faster.

## Why this is the demo

Every retrieval stack can answer a query. The thing that decides whether a
corpus is a *file* or a *deployment* is what happens when you copy it while
something is writing to it — and that is where a directory of index shards,
or a database with sidecar files, stops being copyable.

The measured comparison against Tantivy + USearch and SQLite + FTS5 +
sqlite-vec is in the project [README](../README.md#the-number-we-care-about-most),
and the durability work behind it is in
[bench/CRASH_CAMPAIGN.md](../bench/CRASH_CAMPAIGN.md).
