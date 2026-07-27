#!/usr/bin/env python3
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Portability baseline: ONE SQLite database file as a hybrid retrieval store.

The strongest *single-file* alternative to Valise that can be composed from
off-the-shelf parts: a single SQLite database holding

  * ``docs``  — payload store (rowid INTEGER PRIMARY KEY, doc_id, body)
  * ``fts``   — FTS5 external-content index over ``docs.body`` (BM25)
  * ``vecs``  — sqlite-vec ``vec0`` virtual table (float[768], cosine),
                rowid-aligned with ``docs``

The database is built in ROLLBACK-JOURNAL mode (``PRAGMA
journal_mode=DELETE``) so the artifact is *genuinely* one file at rest —
WAL mode would leave ``-wal``/``-shm`` sidecars next to a live database,
which is exactly the property this experiment interrogates.

Mirrors the scenarios / probe metrics / race protocol of
``bench/src/bin/valise-portability-bench.rs`` (read that file's module docs
first). Datasets are identical: BEIR text corpora + the first N rows of
``cohere-medium-1m-f32/corpus.f32`` (d=768, f32 LE row-major), row-aligned
by document order.

Subcommands (all emit JSON):

  build --scenario {airgapped-manual|agent-corpus} --out DIR
      Build DIR/corpus.db + DIR/build.json. Asserts the artifact is one
      file (no -wal/-shm/-journal left behind).

  probe --path DB --queries QJSON --dataset-dir DIR --out JSON
      Cold-start client. Re-invokes this script as a FRESH subprocess
      (internal ``_probe-child`` flag) so the measured process is cold;
      cumulative-from-process-entry timings, same report shape as the
      Rust probe (deployment="sqlite").

  race --scenario S --db DB --copies N --mode {delete|wal} --out JSON
      Copy-under-update correctness: a writer commits one doc per
      transaction (docs+fts+vecs) while a thread copies the .db file N
      times at seeded intervals. In wal mode the -wal/-shm sidecars are
      DELIBERATELY not copied — that is precisely the torn-copy hazard a
      naive ``cp`` of "the database file" exhibits. Each copy is then
      classified in a fresh connection: {ok_consistent, ok_inconsistent,
      open_error}.

FTS5 pattern used (documented choice): standard external-content table
(``content=docs, content_rowid=rowid``). The bulk build populates ``docs``
then issues ``INSERT INTO fts(fts) VALUES('rebuild')`` + ``('optimize')``;
incremental writers (the race loop) follow the external-content contract
by inserting the (rowid, body) pair into ``fts`` in the same transaction
as the ``docs`` row. No triggers — the two writers in this file are the
only writers.
"""

import time

# t0 for the cold-probe child: as close to process entry as user code can
# get. Interpreter startup itself (~20-40 ms) is before this line and is
# not measurable from within; everything else (imports, arg parsing,
# open, queries) counts. The Rust probe's t0 is main()-entry, the same
# convention modulo runtime startup.
T0 = time.perf_counter()

import argparse
import hashlib
import json
import os
import re
import resource
import shutil
import sqlite3
import struct
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import sqlite_vec

DIM = 768
VEC_BYTES = DIM * 4
BUILD_BATCH = 5_000
RRF_K = 60
MASK64 = (1 << 64) - 1

SCENARIOS = {
    "airgapped-manual": "beir-data/trec-covid",
    "agent-corpus": "beir-data/scifact",
}

# Repo bench/ dir (this file lives in bench/python/).
DEFAULT_BENCH_DIR = Path(__file__).resolve().parent.parent
DEFAULT_VECTOR_SUBDIR = "datasets/cohere-medium-1m-f32"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def open_db(path, readonly=False, load_vec=True):
    """Open a connection; optionally load the sqlite-vec extension."""
    if readonly:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    else:
        conn = sqlite3.connect(str(path))
    if load_vec:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
    return conn


def sidecar_paths(db_path):
    p = str(db_path)
    return [p + s for s in ("-journal", "-wal", "-shm")]


def load_beir_corpus(path):
    """[(doc_id, body)] in file order; body = title + ' ' + text (same as
    the Rust bench's load_beir_corpus)."""
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            title = row.get("title", "") or ""
            body = (title + " " + row["text"]) if title else row["text"]
            docs.append((row["_id"], body))
    return docs


def read_query_vector_bytes(queries_f32, row):
    """Raw little-endian f32 bytes of one query row — exactly the wire
    format sqlite-vec expects for a float[] MATCH parameter."""
    with open(queries_f32, "rb") as f:
        f.seek(row * VEC_BYTES)
        buf = f.read(VEC_BYTES)
    if len(buf) != VEC_BYTES:
        raise SystemExit(f"short read of query row {row} from {queries_f32}")
    return buf


def fts5_query(text):
    """Tokenize a free-text query into an OR of quoted FTS5 phrases.

    FTS5 MATCH has its own query syntax ('-' etc. are operators), so raw
    user text like "COVID-19" is not valid. Quoting each \\w+ token and
    OR-ing mirrors the tantivy default-parser semantics the composed
    probe uses (OR of terms).
    """
    toks = re.findall(r"\w+", text)
    return " OR ".join(f'"{t}"' for t in toks)


def rrf_fuse(a, b, k):
    """Reciprocal Rank Fusion of two ranked id lists; same convention as
    the Rust bench (0-based rank p -> 1/(k+p+1); ties break by id)."""
    score = {}
    for p, i in enumerate(a):
        score[i] = score.get(i, 0.0) + 1.0 / (k + p + 1)
    for p, i in enumerate(b):
        score[i] = score.get(i, 0.0) + 1.0 / (k + p + 1)
    return [i for i, _ in sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))]


def peak_rss_bytes():
    """ru_maxrss: bytes on macOS, KiB on Linux — normalized to bytes
    (same convention as the Rust bench)."""
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return maxrss * 1024 if sys.platform.startswith("linux") else maxrss


def sha256_hex(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


class XorShift64:
    """Deterministic xorshift64* — same finalizer as the Rust bench."""

    def __init__(self, seed):
        self.s = seed if seed != 0 else 0x9E3779B97F4A7C15

    def next(self):
        x = self.s
        x ^= (x << 13) & MASK64
        x ^= x >> 7
        x ^= (x << 17) & MASK64
        self.s = x
        return (x * 0x2545F4914F6CDD1D) & MASK64


def embed(text, dim=DIM):
    """Deterministic pseudo-random unit vector (FNV + xorshift), ported
    from the Rust bench's race-writer embed(). Returned as raw f32 LE
    bytes ready for a vec0 insert."""
    state = 0xCBF29CE484222325
    for b in text.encode():
        state = ((state ^ b) * 0x100000001B3) & MASK64
    vals = []
    for _ in range(dim):
        state ^= (state << 13) & MASK64
        state &= MASK64
        state ^= state >> 7
        state ^= (state << 17) & MASK64
        state &= MASK64
        vals.append((state >> 40) / float(1 << 24) - 0.5)
    norm = max(sum(v * v for v in vals) ** 0.5, 1e-12)
    return struct.pack(f"<{dim}f", *(v / norm for v in vals))


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------


def cmd_build(a):
    beir_subdir = SCENARIOS[a.scenario]
    bench_dir = Path(a.bench_dir)
    corpus_jsonl = bench_dir / beir_subdir / "corpus.jsonl"
    corpus_f32 = bench_dir / a.vector_subdir / "corpus.f32"

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = out_dir / "corpus.db"

    print(f"[build] scenario={a.scenario} out={db_path}", flush=True)
    docs = load_beir_corpus(corpus_jsonl)
    n = len(docs)
    print(f"[load] {n} docs; reading first {n} vector rows (d={DIM})", flush=True)
    with open(corpus_f32, "rb") as f:
        vec_buf = f.read(n * VEC_BYTES)
    if len(vec_buf) != n * VEC_BYTES:
        raise SystemExit(f"short read: corpus.f32 has < {n} rows")
    vec_view = memoryview(vec_buf)

    # Clean slate: the db and any sidecars from a previous run.
    for p in [db_path, *sidecar_paths(db_path)]:
        if os.path.exists(p):
            os.remove(p)

    t_build = time.perf_counter()
    conn = open_db(db_path)
    # Genuinely single-file at rest: rollback journal, not WAL.
    mode = conn.execute("PRAGMA journal_mode=DELETE").fetchone()[0]
    assert mode == "delete", f"journal_mode={mode}"
    conn.execute("PRAGMA synchronous=FULL")
    # Build-side page cache only (256 MiB); does not affect the probe.
    conn.execute("PRAGMA cache_size=-262144")

    conn.execute(
        "CREATE TABLE docs(rowid INTEGER PRIMARY KEY, doc_id TEXT UNIQUE, body TEXT)"
    )
    conn.execute(
        "CREATE VIRTUAL TABLE fts USING fts5(body, content=docs, content_rowid=rowid)"
    )
    conn.execute(
        f"CREATE VIRTUAL TABLE vecs USING vec0(embedding float[{DIM}] distance_metric=cosine)"
    )
    conn.commit()

    # Batched inserts: rowids 1..n keep docs/vecs row-aligned by document
    # order (the same alignment the Rust bench uses).
    for start in range(0, n, BUILD_BATCH):
        end = min(start + BUILD_BATCH, n)
        conn.execute("BEGIN")
        conn.executemany(
            "INSERT INTO docs(rowid, doc_id, body) VALUES (?, ?, ?)",
            ((i + 1, docs[i][0], docs[i][1]) for i in range(start, end)),
        )
        conn.executemany(
            "INSERT INTO vecs(rowid, embedding) VALUES (?, ?)",
            (
                (i + 1, vec_view[i * VEC_BYTES : (i + 1) * VEC_BYTES])
                for i in range(start, end)
            ),
        )
        conn.commit()
        print(f"[build] {end}/{n} rows", flush=True)

    # External-content FTS5: index the docs table, then merge segments.
    print("[build] fts rebuild + optimize", flush=True)
    conn.execute("INSERT INTO fts(fts) VALUES('rebuild')")
    conn.execute("INSERT INTO fts(fts) VALUES('optimize')")
    conn.commit()
    conn.close()
    build_seconds = time.perf_counter() - t_build

    # Single-file assertion: nothing but corpus.db may remain.
    leftovers = [p for p in sidecar_paths(db_path) if os.path.exists(p)]
    assert not leftovers, f"sidecars left behind: {leftovers}"
    bytes_total = os.path.getsize(db_path)
    file_count = 1

    report = {
        "deployment": "sqlite",
        "scenario": a.scenario,
        "path": str(db_path),
        "doc_count": n,
        "dim": DIM,
        "bytes_total": bytes_total,
        "file_count": file_count,
        "build_seconds": build_seconds,
        # Single-file artifact — already the unit of transfer, like Valise.
        "package_tar_seconds": 0,
        "sha256": sha256_hex(db_path),
        "note": (
            "one SQLite db: docs payload + FTS5 external-content (rebuild+optimize) "
            "+ sqlite-vec vec0 float[768] cosine; journal_mode=DELETE (rollback journal)"
        ),
    }
    build_json = out_dir / "build.json"
    build_json.write_text(json.dumps(report, indent=2))
    print(f"[wrote] {build_json}", flush=True)
    print(json.dumps(report, indent=2))


# ---------------------------------------------------------------------------
# probe (parent re-invokes itself so the measured process is cold)
# ---------------------------------------------------------------------------


def cmd_probe(a):
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "_probe-child",
        "--path", str(a.path),
        "--queries", str(a.queries),
        "--dataset-dir", str(a.dataset_dir),
        "--out", str(a.out),
    ]
    proc = subprocess.run(cmd)
    sys.exit(proc.returncode)


def cmd_probe_child(a):
    # Cumulative timings from T0 (module import, ~process entry).
    with open(a.queries) as f:
        q = json.load(f)
    qvec = read_query_vector_bytes(Path(a.dataset_dir) / "queries.f32", q["vector_query_row"])

    conn = open_db(a.path, readonly=True)
    open_ms = (time.perf_counter() - T0) * 1e3

    # ---- Text channel: FTS5 MATCH, bm25() rank (ascending = best first),
    # then one payload fetch (top-1 body), as a real client would.
    match = fts5_query(q["text_query"])
    text_rows = conn.execute(
        "SELECT rowid FROM fts WHERE fts MATCH ? ORDER BY bm25(fts) LIMIT 10",
        (match,),
    ).fetchall()
    text_rowids = [r[0] for r in text_rows]
    top10_text_ids = [
        conn.execute("SELECT doc_id FROM docs WHERE rowid=?", (r,)).fetchone()[0]
        for r in text_rowids
    ]
    if text_rowids:
        conn.execute("SELECT body FROM docs WHERE rowid=?", (text_rowids[0],)).fetchone()
    first_text_hit_ms = (time.perf_counter() - T0) * 1e3

    # ---- Vector channel: vec0 KNN (cosine), map rowids -> doc ids,
    # fetch top-1 body.
    vec_rows = conn.execute(
        "SELECT rowid, distance FROM vecs WHERE embedding MATCH ? ORDER BY distance LIMIT 10",
        (qvec,),
    ).fetchall()
    vec_rowids = [r[0] for r in vec_rows]
    top10_vector_ids = [
        conn.execute("SELECT doc_id FROM docs WHERE rowid=?", (r,)).fetchone()[0]
        for r in vec_rowids
    ]
    if vec_rowids:
        conn.execute("SELECT body FROM docs WHERE rowid=?", (vec_rowids[0],)).fetchone()
    first_vector_hit_ms = (time.perf_counter() - T0) * 1e3

    # ---- Hybrid channel: RRF of the two top-10 id lists (k=60), then a
    # top-1 payload fetch.
    top10_hybrid_ids = rrf_fuse(top10_text_ids, top10_vector_ids, RRF_K)[:10]
    id_to_rowid = dict(zip(top10_text_ids, text_rowids))
    for i, r in zip(top10_vector_ids, vec_rowids):
        id_to_rowid.setdefault(i, r)
    if top10_hybrid_ids:
        conn.execute(
            "SELECT body FROM docs WHERE rowid=?", (id_to_rowid[top10_hybrid_ids[0]],)
        ).fetchone()
    first_hybrid_hit_ms = (time.perf_counter() - T0) * 1e3
    conn.close()

    report = {
        "deployment": "sqlite",
        "open_ms": open_ms,
        "first_text_hit_ms": first_text_hit_ms,
        "first_vector_hit_ms": first_vector_hit_ms,
        "first_hybrid_hit_ms": first_hybrid_hit_ms,
        "top10_text_ids": top10_text_ids,
        "top10_vector_ids": top10_vector_ids,
        "top10_hybrid_ids": top10_hybrid_ids,
        "peak_rss_bytes": peak_rss_bytes(),
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


# ---------------------------------------------------------------------------
# race
# ---------------------------------------------------------------------------


def cmd_race(a):
    built = Path(a.db)
    if not built.exists():
        raise SystemExit(f"race: built db not found at {built} (run `build` first)")

    work = built.parent / f"race_{a.mode}"
    if work.exists():
        shutil.rmtree(work)
    copies_dir = work / "copies"
    copies_dir.mkdir(parents=True)
    live = work / "live.db"
    shutil.copyfile(built, live)

    if a.mode == "wal":
        # Convert the working copy to WAL. The live db now keeps its
        # recent commits in live.db-wal until checkpoint.
        conn = open_db(live, load_vec=False)
        m = conn.execute("PRAGMA journal_mode=WAL").fetchone()[0]
        assert m == "wal", f"journal_mode={m}"
        conn.close()

    print(f"[race:sqlite:{a.mode}] writer loop + {a.copies} racing file copies", flush=True)

    # Copier thread: N shutil.copyfile snapshots of the .db ONLY, at
    # seeded 1-8 ms intervals. In wal mode the -wal/-shm sidecars are
    # deliberately NOT copied — a naive `cp corpus.db backup.db` doesn't
    # copy them either; that is the hazard being demonstrated.
    stop = threading.Event()
    made = []  # (path, t_start, ok)

    def copier():
        rng = XorShift64(a.seed)
        for i in range(a.copies):
            time.sleep((1 + rng.next() % 8) / 1e3)
            dst = copies_dir / f"copy_{i}.db"
            t_start = time.perf_counter()
            try:
                shutil.copyfile(live, dst)
                ok = True
            except OSError:
                ok = False
            made.append((dst, t_start, ok))
        stop.set()

    copier_t = threading.Thread(target=copier)

    # Writer loop (this thread): one fresh doc per transaction across all
    # three tables (docs + fts external-content pair + vecs), then COMMIT;
    # a doc is "acknowledged" once commit() returns.
    wconn = open_db(live)
    wconn.isolation_level = None  # explicit BEGIN/COMMIT
    base_docs = wconn.execute("SELECT count(*) FROM docs").fetchone()[0]
    next_rowid = wconn.execute("SELECT COALESCE(MAX(rowid),0)+1 FROM docs").fetchone()[0]
    acked = []  # (i, doc_id, rowid, body, vec_bytes, t_ack)

    copier_t.start()
    i = 0
    while not stop.is_set() and i < a.max_iters:
        doc_id = f"race-{i}"
        body = f"race payload record {i} racetok{i}"
        vec = embed(doc_id)
        rowid = next_rowid + i
        wconn.execute("BEGIN IMMEDIATE")
        wconn.execute(
            "INSERT INTO docs(rowid, doc_id, body) VALUES (?, ?, ?)", (rowid, doc_id, body)
        )
        wconn.execute("INSERT INTO fts(rowid, body) VALUES (?, ?)", (rowid, body))
        wconn.execute("INSERT INTO vecs(rowid, embedding) VALUES (?, ?)", (rowid, vec))
        wconn.execute("COMMIT")
        acked.append((i, doc_id, rowid, body, vec, time.perf_counter()))
        i += 1
    wconn.close()
    copier_t.join()
    print(f"[race:sqlite:{a.mode}] committed {len(acked)} docs over base {base_docs}", flush=True)

    # Classify every copy in a fresh connection (parallel: the sqlite3 C
    # calls release the GIL).
    with ThreadPoolExecutor(max_workers=4) as pool:
        outcomes = list(pool.map(lambda m: classify_copy(m, acked), made))

    counts = {"ok_consistent": 0, "ok_inconsistent": 0, "open_error": 0}
    for o in outcomes:
        counts[o["class"]] += 1

    report = {
        "deployment": "sqlite",
        "scenario": a.scenario,
        "mode": a.mode,
        "copies": a.copies,
        "base_docs": base_docs,
        "committed_docs": len(acked),
        "counts": counts,
        "outcomes": outcomes,
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in report if k != "outcomes"}, indent=2))
    print(f"[wrote] {out}", flush=True)

    if not a.keep_copies:
        shutil.rmtree(work)


def classify_copy(m, acked):
    """Open one copy fresh; integrity-check, cross-count the three tables,
    and verify every acknowledged doc present is FULLY intact."""
    path, t_start, copied_ok = m
    out = {
        "copy": str(path),
        "class": "open_error",
        "copy_started_at": t_start,
        "integrity_ok": False,
        "docs_count": None,
        "fts_count": None,
        "vecs_count": None,
        "acked_full": 0,
        "acked_absent": 0,
        "acked_partial": 0,
        "acked_wrong_data": 0,
        "acked_before_copy_missing": 0,
        "detail": "",
    }
    if not copied_ok:
        out["detail"] = "shutil.copyfile failed (source mid-rewrite)"
        return out
    try:
        # Read-write open: a read-only open of a WAL-mode copy with no
        # -shm/-wal sidecars depends on version-specific fallbacks, and no
        # sidecars exist next to a copy, so a rw open recovers nothing.
        conn = open_db(path)
    except sqlite3.Error as e:
        out["detail"] = f"open failed: {e}"
        return out
    try:
        rows = conn.execute("PRAGMA integrity_check").fetchall()
        integrity_ok = rows == [("ok",)]
        out["integrity_ok"] = integrity_ok
        if not integrity_ok:
            out["detail"] = f"integrity_check: {rows[:3]!r}"
            return out

        docs_n = conn.execute("SELECT count(*) FROM docs").fetchone()[0]
        # External-content FTS5: the index's own row count lives in the
        # fts_docsize shadow table (one row per indexed doc).
        fts_n = conn.execute("SELECT count(*) FROM fts_docsize").fetchone()[0]
        vecs_n = conn.execute("SELECT count(*) FROM vecs").fetchone()[0]
        out.update(docs_count=docs_n, fts_count=fts_n, vecs_count=vecs_n)

        partial = wrong = full = absent = before_missing = 0
        for i, doc_id, rowid, body, vec, t_ack in acked:
            drow = conn.execute("SELECT body FROM docs WHERE rowid=?", (rowid,)).fetchone()
            in_docs = drow is not None
            docs_intact = in_docs and drow[0] == body
            frow = conn.execute(
                "SELECT rowid FROM fts WHERE fts MATCH ?", (f'"racetok{i}"',)
            ).fetchone()
            in_fts = frow is not None and frow[0] == rowid
            vrow = conn.execute(
                "SELECT embedding FROM vecs WHERE rowid=?", (rowid,)
            ).fetchone()
            in_vecs = vrow is not None
            vec_intact = in_vecs and bytes(vrow[0]) == vec

            present_any = in_docs or in_fts or in_vecs
            if not present_any:
                absent += 1
                if t_ack < t_start:
                    # Acked before this copy even began, yet absent — the
                    # WAL staleness signal (commit lived in -wal only).
                    before_missing += 1
            elif docs_intact and in_fts and vec_intact:
                full += 1
            else:
                partial += 1
                if (in_docs and not docs_intact) or (in_vecs and not vec_intact):
                    wrong += 1

        out.update(
            acked_full=full,
            acked_absent=absent,
            acked_partial=partial,
            acked_wrong_data=wrong,
            acked_before_copy_missing=before_missing,
        )

        consistent = docs_n == fts_n == vecs_n and partial == 0 and wrong == 0
        out["class"] = "ok_consistent" if consistent else "ok_inconsistent"
        out["detail"] = (
            f"docs={docs_n} fts={fts_n} vecs={vecs_n} "
            f"acked full={full} absent={absent} partial={partial} wrong={wrong}"
        )
        return out
    except sqlite3.Error as e:
        out["detail"] = f"sql error during classification: {e}"
        out["class"] = "open_error"
        return out
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(
        description="SQLite single-file hybrid baseline (FTS5 + sqlite-vec + payloads)"
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--scenario", choices=sorted(SCENARIOS), required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--bench-dir", default=str(DEFAULT_BENCH_DIR))
    b.add_argument("--vector-subdir", default=DEFAULT_VECTOR_SUBDIR)

    for name in ("probe", "_probe-child"):
        p = sub.add_parser(name)
        p.add_argument("--path", required=True)
        p.add_argument("--queries", required=True)
        p.add_argument("--dataset-dir", required=True)
        p.add_argument("--out", required=True)

    r = sub.add_parser("race")
    r.add_argument("--scenario", choices=sorted(SCENARIOS), required=True)
    r.add_argument("--db", required=True)
    r.add_argument("--copies", type=int, default=50)
    r.add_argument("--mode", choices=("delete", "wal"), required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--seed", type=lambda s: int(s, 0), default=0x5EED1234ABCD0001)
    r.add_argument("--max-iters", type=int, default=20_000)
    r.add_argument("--keep-copies", action="store_true")

    a = ap.parse_args()
    if a.cmd == "build":
        cmd_build(a)
    elif a.cmd == "probe":
        cmd_probe(a)
    elif a.cmd == "_probe-child":
        cmd_probe_child(a)
    elif a.cmd == "race":
        cmd_race(a)


if __name__ == "__main__":
    main()
