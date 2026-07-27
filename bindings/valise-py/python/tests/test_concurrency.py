# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""FFI GIL-release proof.

The load-bearing test targets Reader.search(), which genuinely releases the GIL
(`py.detach`) around the engine scan. A background thread incrementing a counter
in a tight loop MUST advance substantially while the main thread runs many
searches — impossible if search held the GIL for its whole native duration.

commit() does NOT detach (OwnedWriter is !Send, so the binding keeps commit
GIL-attached). So commit gets a SEPARATE, non-flaky test that only proves a
concurrent reader is not deadlocked by a committing writer (ArcSwap readers
never block on the writer lock) and that generations advance monotonically.
"""

import threading
import time

import numpy as np

import valise

from conftest import DIM, make_hybrid, matrix
from valise import Bm25, Rerank


def _ingest_corpus(store, n):
    keys = [f"k{i}" for i in range(n)]
    texts = ["quantum retrieval document benchmark"] * n
    mat = matrix(n, DIM, seed=1)
    with store.writer() as w:
        w.put_many("kb", keys, mat, texts=texts)
        w.commit()
    return mat


def test_search_releases_gil(store):
    """A background counter advances substantially while many searches run."""
    make_hybrid(store, deferred_sample=2000)
    n = 20_000
    mat = _ingest_corpus(store, n)
    reader = store.reader()

    counter = [0]
    stop = threading.Event()

    def spin():
        while not stop.is_set():
            counter[0] += 1

    t = threading.Thread(target=spin, daemon=True)
    t.start()
    try:
        # Let the spinner warm up, then measure advancement during searches.
        time.sleep(0.05)
        c0 = counter[0]
        wall0 = time.perf_counter()
        for i in range(200):
            q = np.ascontiguousarray(mat[i % n])
            reader.search("kb", valise.Search().vector("dense", q, Rerank.FAST).top_k(10))
        wall = time.perf_counter() - wall0
        c1 = counter[0]
    finally:
        stop.set()
        t.join(timeout=2)

    # The searches must take measurable wall time so the test is meaningful.
    assert wall > 1e-3, f"searches too fast to be meaningful ({wall}s)"
    # If search held the GIL, the spinner could not advance at all.
    assert c1 - c0 > 1000, f"counter barely moved: {c1 - c0} (wall={wall}s)"


def test_commit_does_not_deadlock_concurrent_reader(store):
    """A reader thread doing get()/search() while the writer commits must not
    deadlock, and commit generations must increase monotonically."""
    make_hybrid(store, deferred_sample=8)
    # Seed + calibrate so reads have something to do.
    _ingest_corpus(store, 64)
    reader = store.reader()

    stop = threading.Event()
    errors = []

    def reader_loop():
        try:
            while not stop.is_set():
                reader.get("kb", "k0")
                reader.search("kb", valise.Search().text("body", "quantum", Bm25()).top_k(5))
        except Exception as e:  # pragma: no cover - surfaced via assert below
            errors.append(e)

    t = threading.Thread(target=reader_loop, daemon=True)
    t.start()
    try:
        gens = []
        for batch in range(5):
            with store.writer() as w:
                keys = [f"b{batch}_{i}" for i in range(50)]
                w.put_many("kb", keys, matrix(50, DIM, seed=batch + 100))
                gen = w.commit()
                gens.append(gen)
    finally:
        stop.set()
        t.join(timeout=5)

    assert not t.is_alive(), "reader thread deadlocked against committing writer"
    assert not errors, f"concurrent reader errored: {errors}"
    # Generations are monotonically non-decreasing across successive commits.
    assert gens == sorted(gens), gens
    assert gens[-1] > gens[0], gens
