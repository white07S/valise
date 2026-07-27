"""Single-writer lock contention across Python threads.

Regression guard for the GIL-deadlock bug: `Store.writer()` must acquire the
single-writer lock by *releasing the GIL while it waits* (poll + `py.detach`
sleep), not by parking the calling thread with the GIL held. A blocking
GIL-held acquire deadlocks the whole interpreter, because the thread already
holding the writer can never reacquire the GIL to run `close()`/Drop and
release the lock.
"""

import os
import threading
import time

import numpy as np
import pytest

import valise
from valise import TfidfCosine, TfMode


def _make_store(path: str) -> valise.Store:
    if os.path.exists(path):
        os.remove(path)
    s = valise.Store.open(path)  # open-or-create
    s.collection("kb", valise.Schema().text("body").vector("dense", valise.Vector(dim=64)))
    return s


def test_second_writer_blocks_then_succeeds(tmp_path):
    """Two threads contend for the writer; the second must block, then acquire
    only after the first closes — never deadlock."""
    s = _make_store(str(tmp_path / "contention.vls"))

    order = []
    order_lock = threading.Lock()
    hold = 0.5  # seconds the first writer holds the lock

    def hold_writer():
        w = s.writer()
        with order_lock:
            order.append("t1-acquired")
        time.sleep(hold)
        with order_lock:
            order.append("t1-closing")
        w.close()

    def wait_for_writer():
        time.sleep(0.1)  # let the first thread win the lock
        with order_lock:
            order.append("t2-trying")
        w = s.writer()  # must block then succeed (no GIL deadlock)
        with order_lock:
            order.append("t2-acquired")
        w.close()

    t1 = threading.Thread(target=hold_writer)
    t2 = threading.Thread(target=wait_for_writer)
    t1.start()
    t2.start()
    # Generous timeout: a deadlock would hang forever; the real run is ~`hold`.
    t1.join(timeout=8)
    t2.join(timeout=8)

    assert not t1.is_alive(), f"deadlock: writer holder never finished ({order})"
    assert not t2.is_alive(), f"deadlock: waiting writer never acquired ({order})"
    # The waiter must acquire strictly after the holder began closing.
    assert order.index("t2-acquired") > order.index("t1-closing"), order


def test_context_manager_releases_lock(tmp_path):
    """`with store.writer()` releases the lock on exit so a later writer works."""
    s = _make_store(str(tmp_path / "ctx.vls"))
    embs = np.ascontiguousarray(np.random.rand(2, 64).astype(np.float32))
    with s.writer() as w:
        w.put_many("kb", ["a", "b"], embs, texts=["x", "y"])
        w.commit()
    # If the lock leaked, this second acquire would hang.
    with s.writer() as w:
        w.put_many("kb", ["c"], embs[:1], texts=["z"])
        w.commit()
    assert s.get("kb", "c") is not None


def test_put_many_noncontiguous_is_all_or_nothing(tmp_path):
    """A non C-contiguous 2-D vectors array is rejected up front, staging no
    rows — a subsequent commit on the same writer persists nothing."""
    s = _make_store(str(tmp_path / "partial.vls"))
    w = s.writer()
    noncontig = np.random.rand(3, 128).astype(np.float32)[:, ::2]
    assert not noncontig.flags["C_CONTIGUOUS"]
    with pytest.raises(valise.ValidationError):
        w.put_many("kb", ["x", "y", "z"], noncontig)
    w.commit()  # nothing was staged
    w.close()
    assert s.get("kb", "x") is None
    assert s.get("kb", "y") is None
    assert s.get("kb", "z") is None


def test_tfidf_cosine_scorer_reachable(tmp_path):
    """The `tfidf_cosine` text scorer (with optional `tf_mode`) is reachable."""
    s = _make_store(str(tmp_path / "tfidf.vls"))
    embs = np.ascontiguousarray(np.random.rand(2, 64).astype(np.float32))
    with s.writer() as w:
        w.put_many("kb", ["a", "b"], embs, texts=["hello world", "foo bar"])
        w.commit()
    r = s.reader()
    res = r.search(
        "kb", valise.Search().text("body", "hello world", TfidfCosine()).top_k(2)
    )
    assert "a" in res.keys
    # tf_mode carried by the typed scorer
    r.search(
        "kb",
        valise.Search().text("body", "hello", TfidfCosine(TfMode.RAW)).top_k(1),
    )
