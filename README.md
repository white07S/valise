<div align="center">

# valise

**Retrieval in one file.**

[![CI](https://github.com/white07S/valise/actions/workflows/ci.yml/badge.svg)](https://github.com/white07S/valise/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/valise.svg)](https://crates.io/crates/valise)
[![docs.rs](https://docs.rs/valise/badge.svg)](https://docs.rs/valise)
[![PyPI](https://img.shields.io/pypi/v/valise.svg)](https://pypi.org/project/valise/)
[![License](https://img.shields.io/badge/license-MPL--2.0-blue.svg)](#license)

</div>

---

Your RAG prototype works. Now ship it.

Suddenly the corpus isn't a thing you have — it's a thing you *operate*. A
vector database container. An index directory that has to travel with the
documents and stay consistent with them. A rebuild step in CI. A migration
when the schema drifts. The most valuable artifact you built, the corpus
itself, is the one piece you can't hand to anyone.

SQLite solved this for relational data thirty years ago: put the whole
database in one file and the operational problem evaporates.

**Valise does that for retrieval.** Documents, a BM25 index, compressed
vectors, and the schema describing them — one `.vls` file. Copy it, version
it, attach it to a release, `scp` it to an air-gapped box. Open it and query
it in place.

```bash
pip install valise          # or:  cargo add valise
```

```python
from valise import Store, Schema, Record, Search

store = Store.open("kb.vls")                    # one file, created if missing
store.collection("notes", Schema().text("body"))

with store.writer() as w:
    w.put("notes", "doc-1", Record().text("body", "portable retrieval capsules"))
    w.commit()

store.search("notes", Search().text("body", "retrieval").top_k(10))
```

That's the whole setup. No daemon, no connection string, no index build.

---

## The number we care about most

Take a copy of your corpus *while it is being written*, move it to a machine
with a different OS and a different instruction set, and open it there.

On a 171,000-document hybrid corpus:

| <sub>[†](#check-the-numbers-yourself)</sub> | **Valise** | Tantivy + USearch + payloads | SQLite + FTS5 + sqlite-vec |
|---|---:|---:|---:|
| **Copies taken mid-write that opened to a correct snapshot** | **50 / 50** | 4 / 50 | 0 / 50 |
| Deployment artifact | **239 MB** | 677 MB | 865 MB |
| Files to move | **1** | 62 | 1 |
| Packaging step before transfer | **none** | 23.8 s | none |

Two of SQLite's "successful" copies passed `integrity_check` and then served
bytes that differed from writes it had already acknowledged. Valise's copies
were byte-verified against the acknowledged write set every time.

And the results are the same on the far side: **identical top-10 hits for
lexical, vector, and hybrid queries** across macOS/AArch64 and Linux/x86-64,
from the same file, with no reconfiguration.

That is the product. Everything below is why it's possible.

---

## There is no index to build

Every ANN library builds a graph before it can answer anything. That cost
comes back every time the corpus changes materially.

Valise builds **nothing**. Vectors are stored only as compact quantized codes,
and the candidate-search structure is derived *from those same codes* when the
file is opened — a one-bit-per-dimension sign sketch, held in memory, scanned
with SIMD popcount, then reranked exactly through the codec.

One representation does double duty. So:

| 100k vectors × 768d, matched recall <sub>[†](#check-the-numbers-yourself)</sub> | Build time |
|---|---:|
| **Valise** | **0.50 s** |
| FAISS HNSW | 50.3 s — 101× slower |
| USearch | 91.2 s — 182× slower |
| hnswlib | 138.5 s — 277× slower |

At a million vectors it's **5.5 s versus 889 s**. Across corpora the range is
**90–310× faster to build**.

The knock-on effects matter more than the number:

- **Nothing on disk but your data.** No graph file, no sidecar, no `.idx`.
- **Writes don't invalidate an index**, because there isn't one.
- **Nothing to corrupt independently of the data.** The file *is* the index.
- **Re-indexing from scratch becomes a reasonable default** instead of an
  overnight job.

## Compression: the part that's genuinely hard

Compression claims are easy to make and hard to calibrate, so here is the
whole frontier. Cohere-v2, 768 dimensions, 100k vectors, every system tuned to
its best operating point <sub>[†](#check-the-numbers-yourself)</sub>:

| Bytes/vector | System | recall@10 | Query p50 |
|---:|---|---:|---:|
| 120 | sqlite-vec 1-bit | 0.486 | 7,821 µs |
| 151 | FAISS IVF-PQ | 0.611 | 438 µs |
| 151 | FAISS RaBitQ | 0.757 | 807 µs |
| 174 | FAISS OPQ+IVF-PQ | 0.747 | 445 µs |
| 245 | USearch binary | 0.485 | 57 µs |
| **585** | **Valise** | **0.965** | **532 µs** |
| 768 | FAISS SQ8 (exact scan) | 0.989 | 13,762 µs |
| 815 | FAISS IVF-SQ8 | 0.974 | 2,094 µs |
| 917 | USearch int8 | 0.880 | 336 µs |
| 3,221 | USearch BF16 / hnswlib | 0.966 | 544 / 743 µs |
| 3,344 | FAISS HNSW | 0.968 | 237 µs |

Read the whole table before the bold row. Two things fall out of it:

**Nothing below 585 bytes per vector gets past 0.757 recall.** Every
aggressive quantizer on the market — product quantization, RaBitQ, binary
codes, 1-bit — buys its size by giving up a quarter to a half of the
neighbors.

**Valise is the only configuration under 3 KB per vector that clears 0.96
recall in under a millisecond.** The systems in between get their recall from
exact or near-exact scans and pay 4–26× the query latency for it. The cheapest
peer that matches both our recall *and* our latency is hnswlib, at **5.5× the
bytes** — because it stores full-precision vectors *and* a graph.

### Why 5.5 bits per dimension is the interesting number

At that budget the obvious approaches break. Scalar quantization at the *same*
bits reaches 0.940. A one-bit sign code reaches 0.541. Skipping the rotation
step costs half a point.

What closes the gap is quantizing rotated coordinate *pairs* in polar form —
one joint cell index over amplitude rings with power-of-two phase counts,
fitted in closed form to the distribution the rotation produces.

Here is the result worth pausing on. Against a **trained** 2,048-centroid
k-means codebook at the same bit budget:

| | recall@10 |
|---|---:|
| Trained k-means codebook | 0.968 |
| Valise, closed-form fit | 0.967 |

Statistically indistinguishable. A formula matches a learned codebook — which
means no training pass, no codebook to ship alongside your data, and no
data-dependent artifact baked into your file.

[ANATOMY.md](docs/ANATOMY.md) shows where every byte of a real 100,000-record
capsule goes, and how the total compares against the same corpus in SQLite.

And then the part no other quantizer in that table does: **the codes are also
the index.** Everyone else pays for compression *and* a separate search
structure. Valise derives its candidate sketch from the same bytes it already
stored, so the compression and the search acceleration come out of one
representation.

## The text engine is not an afterthought

Most "hybrid" systems bolt a keyword index onto a vector store. Valise's
lexical side was built first, and it holds up against Tantivy — the Rust
Lucene — across four BEIR corpora of 2.7M to 5.4M documents
<sub>[†](#check-the-numbers-yourself)</sub>:

| Corpus | Index size | nDCG@10 | Recall@100 | Query p50 |
|---|---|---|---|---|
| NQ (2.7M docs) | **43% smaller** | **0.300** vs 0.283 | **0.766** vs 0.720 | **12.0× faster** |
| DBpedia-entity (4.6M) | **32% smaller** | **0.302** vs 0.275 | **0.444** vs 0.404 | **5.9× faster** |
| HotpotQA (5.2M) | **36% smaller** | **0.591** vs 0.586 | **0.771** vs 0.764 | **1.18× faster** |
| FEVER (5.4M) | **40% smaller** | 0.512 vs **0.515** | **0.863** vs 0.859 | 0.71× — Tantivy wins |

**Smaller on all four. Better recall@100 on all four. Better nDCG@10 on three
of four**, and the fourth is a 0.003 difference — noise.

Beating a Lucene-lineage engine on *relevance*, not just on size, is the part
we'd have expected to lose.

**On latency, the regime matters and we'd rather explain it than quote the best
number.** Short queries — entity names, keyword-shaped questions — run 6–12×
faster. Long multi-hop questions and claim-verification queries converge, and
on FEVER Tantivy is 1.4× faster. The pattern is mechanical: more query terms
means more postings to score, and our per-term advantage narrows as the term
count grows. If your queries look like NQ or DBpedia, expect the top of that
range. If they look like FEVER, expect parity.

Tantivy builds its index 3–4× faster than we do. Our commit includes a
durability barrier and full canonicalization; theirs doesn't.

### One index, any scorer, no re-indexing

This is the architectural difference, and it's the one that compounds.

Lucene and Tantivy persist an *index image* — the scoring function is baked
into the bytes on disk. Change the scorer and you rebuild.

Valise persists the **statistics**: canonical term dictionaries, frame-sorted
postings, document stats. Scoring happens at query time. So BM25, TF-IDF
cosine, count cosine, approximate cosine variants, Dice, overlap, and
containment all read the same index — chosen per query, not per build.

A new scoring function applies to files that already exist, without rewriting
a byte. Ship a corpus today, change how you rank it next quarter, and the
files your users already have keep working.

## Crash safety that went looking for trouble

Commits are footer-rooted and atomic behind a single durability barrier. After
a crash a reader sees the previous committed state or the new one, never a
mixture. If the active footer is unusable, recovery scans back to the newest
one that fully validates.

That was tested rather than asserted: **122,200 seeded fault injections** across
ten fault classes, plus **200 virtual-machine power cuts** over ext4, XFS and
btrfs covering **493,931 acknowledged commits**. Zero wrong data, zero lost
commits, zero panics.

The campaign also found three real bugs in Valise — a decompression path that
never re-hashed wire bytes, a torn commit that could render a file unopenable,
and a trusted length field that drove a 2⁵⁴-byte allocation. All fixed, all
now regression-tested. We'd rather tell you that than claim we wrote it
perfectly the first time.

---

## The trade, stated plainly

The scan is linear in corpus size. A graph's isn't. That is the bill for
having no index:

| SIFT-1M, 100k × 128d, 1,000 queries <sub>[‡](#check-the-numbers-yourself)</sub> | **Valise** | usearch (HNSW) |
|---|---:|---:|
| Build | **0.24 s** | 7.9 s |
| On disk | **16.1 MiB** | 38.6 MiB |
| Query p50 | 812 µs | **72 µs** |
| recall@10 | 0.933 | **0.991** |

**A mature HNSW answers individual queries about 11× faster, at higher recall.**
If you build once and serve a billion queries against a static corpus, build
the graph — use Qdrant, LanceDB, or Milvus, and be happy.

The rule of thumb from the measurements: prefer Valise when the corpus fits
your latency budget through a scan — roughly **a million vectors per
millisecond of budget at d=768** — *or* at any scale when the corpus is
copied, shipped, or rebuilt more often than about once per ten thousand
queries. Below ~256 dimensions, a tuned index wins outright.

**Reach for Valise when:**

- **The corpus changes often.** Rebuilding is 90–310× cheaper.
- **The corpus has to travel.** To a customer, an air-gapped site, a release
  artifact, a container image, a colleague.
- **You have many small corpora.** Per user, per agent, per tenant, per branch.
  A service per corpus is absurd. A file per corpus is obvious.
- **Reproducibility matters.** Same bytes, same results, on a different
  machine. Pin it to an eval run. Diff two of them.

**Don't, when:** you need high QPS over a large static corpus, multi-writer
concurrency across machines, or a database that makes the embeddings for you.
Valise stores vectors; it does not produce them.

---

## Beyond the quickstart

<details open>
<summary><b>Hybrid RAG — lexical catches what the embedding smooths away</b></summary>

```python
import numpy as np
from valise import Store, Schema, Search, Vector, Rrf

store = Store.open("docs.vls")
store.collection("chunks", Schema()
    .text("body")
    .vector("dense", Vector(dim=384)))

# One native call for the whole batch — no per-row Python loop.
vectors = np.ascontiguousarray(model.encode(chunks), dtype=np.float32)
with store.writer() as w:
    w.put_many("chunks", ids, vectors, texts=chunks)
    w.commit()

def retrieve(question, k=8):
    q = np.asarray(model.encode([question])[0], dtype=np.float32)
    hits = store.search("chunks", Search()
        .text("body", question)      # exact terms: error codes, SKUs, names
        .vector("dense", q)          # paraphrase and synonymy
        .fuse(Rrf(k=60))             # reciprocal-rank fusion
        .top_k(k))
    r = store.reader()
    return [r.get("chunks", key).text for key in hits.keys]
```

Pure vector search loses the error code your user pasted in. Pure lexical
loses the paraphrase. Running both over one index costs you one extra line.

</details>

<details>
<summary><b>Agent memory — one file per agent, survives restarts</b></summary>

```python
from valise import Store, Schema, Record, Search, HalfLife

store = Store.open(f"memory/{agent_id}.vls")
store.collection("events", Schema().text("body"))

def remember(text):
    with store.writer() as w:
        w.put("events", str(uuid4()), Record().text("body", text))
        w.commit()

def recall(cue, k=5):
    # Recent memories outrank stale ones at the same lexical score.
    return store.search("events", Search()
        .text("body", cue)
        .recency(HalfLife(days=7))
        .top_k(k))
```

Back it up by copying it. Inspect it with `valise info memory/agent-7.vls`.
Ship a pre-warmed memory with the agent by committing the file.

</details>

<details>
<summary><b>Time-partitioned logs — query a window, drop a range</b></summary>

```python
from valise import Store, Schema, Partition, Window, Search

store = Store.open("logs.vls")
logs = store.partitioned("logs", Schema().text("message"), Partition.BY_DAY)

view = logs.view(Window.last_days(7))
hits = store.search_view(view, Search().text("message", "timeout").top_k(20))

logs.forget_before(cutoff)        # drop whole partitions, not row-by-row
```

</details>

<details>
<summary><b>Ship a queryable artifact through CI</b></summary>

```bash
python build_index.py && ls -lh kb.vls        # 16 MiB, one file

# Attach to a GitHub release, then anywhere:
valise info   kb.vls                           # what's in here?
valise search kb.vls notes "quantization" --top-k 5
valise export kb.vls > kb.jsonl                # your data, no library required
```

The CLI ships with the crate. `export` streams JSON Lines, so **your data is
never locked in** — you can walk an entire capsule without writing code.

</details>

<details>
<summary><b>The same thing in Rust</b></summary>

```rust
use valise::prelude::*;

let store = Store::open("kb.vls")?;
store.collection("notes", Schema::new()
    .text("body")
    .vector("dense", Vector::dim(768)))?;

let mut w = store.writer();
w.put("notes", "doc-1", Record::new()
    .text("body", "portable retrieval capsules")
    .vector("dense", &embedding))?;
w.commit()?;

let hits = store.search("notes", Search::new()
    .text("body", "retrieval capsule")
    .vector("dense", &query)
    .top_k(10))?;
```

</details>

---

## Check the numbers yourself

**‡ — reproducible here, in two commands.** The SIFT-1M figures in
[the trade](#the-trade-stated-plainly) come from a benchmark in this repo. It builds the Valise index and
runs Tantivy, usearch, and hnsw_rs in the same process over identical data and
ground truth:

```bash
python3 bench/prep_data.py all-small     # fetches BEIR scifact + SIFT-1M
cargo build --release -p valise-bench --bin valise-e2e-bench
target/release/valise-e2e-bench \
    --beir-dir bench/beir-data/scifact \
    --vector-dir bench/datasets/sift-1m \
    --vector-n 100000 --out bench/results/e2e.json
```

[bench/REPRODUCE.md](bench/REPRODUCE.md) documents every knob, the x86-64 AVX2
results, and the cases where Valise loses.

**† — from a larger study, not reproducible here yet.** The portability,
build-time, storage, lexical-size and crash-campaign figures come from runs at
100k–1M vectors against recall-matched baselines, on corpora that need separate
download (some behind a gated licence). The full methodology is written up in a
systems paper currently under submission — **a preprint link will be added here
when it goes up.** We keep these visually separate from the ‡ numbers
rather than blending them, because you can check one set today and have to take
our word on the other — and you shouldn't have to guess which is which.

---

## Two API levels

| Level | Entry point | For |
|---|---|---|
| **Application** | `import valise` / `valise::prelude` | Keyed records, schemas, text/vector/hybrid search, partitions, compaction. **Start here.** |
| **Engine** | `ValiseFile` / `Database` | Explicit catalog registration, frame and vector primitives, raw format work. |

The Python package is a typed facade over the same concepts — full type hints
under `mypy --strict`, strict enums instead of magic strings, vectors crossing
the FFI boundary zero-copy.

## Status

Pre-1.0 and moving. Below 1.0, minor versions may break the API and the
on-disk format; every format change is recorded in [MIGRATION.md](MIGRATION.md)
and caught by a golden-hash test.

**Platforms:** Linux and macOS on x86-64 and aarch64. **Windows is not
supported yet** — the commit protocol relies on positional file IO and fcntl
OFD advisory locks, and the Windows equivalents are not implemented.

**Known limitation:** `ReadConnection`'s search methods are not served from the
pinned snapshot; catalog and payload reads are. A query issued after a
concurrent commit can return a frame the same connection's `frame_stubs()`
does not list. Call `refresh_snapshot()` when the two must agree.

## Documentation

| | |
|---|---|
| [Python docs](https://white07S.github.io/valise) | Quickstart, concepts, full API reference |
| [docs.rs](https://docs.rs/valise) | Rust API |
| [docs/ANATOMY.md](docs/ANATOMY.md) | Where the bytes go — a real 100k-record capsule, segment by segment, vs SQLite |
| [docs/FORMAT.md](docs/FORMAT.md) | On-disk format, commit and recovery protocol |
| [docs/VECTOR_SEARCH.md](docs/VECTOR_SEARCH.md) | Sketch design, recall/latency envelope, failed experiments |
| [docs/CONCURRENCY.md](docs/CONCURRENCY.md) | Reader/writer model and decision log |
| [docs/EXTENDING.md](docs/EXTENDING.md) | Adding scorers or codec families |
| [bench/REPRODUCE.md](bench/REPRODUCE.md) | Every published number, and how to regenerate it |

## Contributing

Contributions welcome — [CONTRIBUTING.md](CONTRIBUTING.md) covers the build,
the review bar, and the format invariants a change must not break. Security
issues go through
[private reporting](https://github.com/white07S/valise/security/advisories/new),
not the tracker; see [SECURITY.md](SECURITY.md).

## License

[Mozilla Public License 2.0](LICENSE). In practice: **you can link Valise into
closed-source commercial software and publish nothing.** MPL is file-level, not
viral — it never reaches your code the way GPL or AGPL would. The only
obligation is that modifications to *Valise's own files* get published.
