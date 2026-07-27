"""Continuously ingest and commit into a capsule, until told to stop.

This is the "live corpus" half of the copy-under-write demo: it keeps a
writer open and commits in a tight loop, so that any copy taken while it
runs is genuinely taken mid-write rather than during a quiet moment.

It stops when the file named by --stop-file appears, and prints its final
record count to --count-file so the verifier knows what it is checking
against.

    python3 demo/writer.py --out corpus.vls --stop-file .stop
"""

import argparse
import pathlib
import sys
import time

import numpy as np
from valise import Record, Schema, Store, Vector

DIM = 128
BATCH = 250

TERMS = [
    "neural network", "retrieval augmented generation", "vector quantization",
    "inverted index", "approximate nearest neighbour", "language model",
    "semantic search", "embedding space", "relevance feedback",
    "document ranking", "query expansion", "transformer attention",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--stop-file", required=True)
    ap.add_argument("--count-file", default=None)
    ap.add_argument("--max-records", type=int, default=200_000)
    # Pause between commits. Larger values keep the snapshots small
    # enough that the demo's deep verify stays quick, without changing
    # what is being demonstrated.
    ap.add_argument("--delay", type=float, default=0.01)
    args = ap.parse_args()

    stop = pathlib.Path(args.stop_file)
    rng = np.random.default_rng(20260727)

    store = Store.open(args.out)
    store.collection(
        "docs",
        Schema().text("body").vector("dense", Vector(dim=DIM)),
    )

    written = 0
    # One writer held open across every commit — the file is never
    # quiescent between batches, which is the whole point.
    with store.writer() as w:
        while written < args.max_records and not stop.exists():
            for i in range(BATCH):
                n = written + i
                body = f"{TERMS[n % len(TERMS)]} study {n} corpus entry"
                w.put(
                    "docs",
                    f"doc-{n:07d}",
                    Record()
                    .text("body", body)
                    .vector("dense", rng.standard_normal(DIM, dtype=np.float32)),
                )
            w.commit()
            written += BATCH
            # Newline-terminated: this is normally redirected to a log
            # that the demo tails, and a partial last line makes the
            # shell print a stray `%`.
            print(f"  writer: {written:,} records committed", flush=True)
            # A brief pause so the copies land across many distinct
            # generations rather than all inside one long commit.
            time.sleep(args.delay)

    print(f"  writer: {written:,} records committed — stopped", flush=True)
    if args.count_file:
        pathlib.Path(args.count_file).write_text(str(written))
    return 0


if __name__ == "__main__":
    sys.exit(main())
