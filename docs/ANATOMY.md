# Anatomy of a `.vls` file

Where the bytes actually go, measured on a real capsule rather than described
in the abstract.

Every number here comes from `valise info --segments`, which you can run on
your own file. The corpus is reproducible: 100,000 records, each with a
128-dimensional vector from SIFT-1M and a 28-word text field drawn from a
20,000-word Zipf-distributed vocabulary — chosen because a small vocabulary
compresses unrealistically well and would flatter the payload numbers.

## The shape of the file

```text
┌────────────────────────────────┐
│ Header (4 KiB, fixed)          │  magic, format version, footer pointer,
│                                │  snapshot generation, coordination region
├────────────────────────────────┤
│ Segments (append-only)         │  payloads, vector codes, term dictionary,
│   …                            │  postings, catalogs — everything, appended
│   …                            │  and never rewritten in place
├────────────────────────────────┤
│ TOC footer                     │  the root of the active snapshot, with two
└────────────────────────────────┘  embedded BLAKE3 checksums
```

Three properties follow from that layout:

- **The header's footer pointer is the commit switch.** An 8-byte aligned
  store swings the file from one consistent snapshot to the next.
- **Nothing is modified in place**, so a copy taken mid-write either sees the
  old footer or the new one.
- **The file is self-describing.** Schemas, analyzers, codec parameters, and
  scoring profiles are all segments inside it — there is no external metadata
  a reader needs.

## A real 100,000-record capsule

Input: **48.8 MiB of raw `float32` vectors** plus **21.4 MiB of text** — 70.2 MiB
of source data.

Result: **37.8 MiB**, or **397 bytes per record**.

```console
$ valise info capsule.vls --segments

  where the bytes are (live segments only):
    segment                 count         bytes    share
    VectorData                  1      13.0 MiB    34.3%
    Postings                    1       8.8 MiB    23.3%
    Payload                     6       6.6 MiB    17.4%
    FrameCatalog                1       5.1 MiB    13.6%
    VectorCatalog               1       2.2 MiB     5.8%
    Metadata                    6     966.2 KiB     2.5%
    TermDictionary              1     512.2 KiB     1.3%
    DocStats                    1     390.8 KiB     1.0%
    TimeIndex                   1     293.1 KiB     0.8%
    CodecParams                 1         398 B     0.0%
    CollectionFilter            2         211 B     0.0%
    …seven catalog segments             ~1.0 KiB    0.0%
    TOTAL (live)                       37.8 MiB
```

### What each piece is

| Segment | Share | What it holds |
|---|---:|---|
| **VectorData** | 34.3% | The quantized vector codes — 136 B per 128-d vector, against 512 B raw. This is the compression story. |
| **Postings** | 23.3% | The inverted index: for each term, the frames containing it. Varint gaps with a tf-exception scheme, so `tf = 1` costs zero extra bytes. |
| **Payload** | 17.4% | Your documents, zstd-compressed and batched into ~4 MiB segments rather than one segment per record. |
| **FrameCatalog** | 13.6% | Per-record metadata — status, collection, timestamps, payload location. Columnar: varint deltas, RLE, delta-of-delta timestamps. ~53 B/record. |
| **VectorCatalog** | 5.8% | Vector descriptors: which embedding space, which codec, where the codes live. |
| **Metadata** | 2.5% | Your record keys, batched like payloads. |
| **TermDictionary** | 1.3% | The canonical term set. Small because it's a delta chain — only newly-seen terms per commit. |
| **DocStats** | 1.0% | Per-document length and term counts, which BM25 needs. Columnar, ~4 B/record. |
| **TimeIndex** | 0.8% | Chronological index backing time-range queries and partitions. ~3 B/record. |
| **Catalogs** | ~0.0% | Collections, analyzers, field schemas, retrieval profiles, embedding spaces, codecs. About a kilobyte total — this is the self-describing part, and it is essentially free. |

Two things worth noticing.

**The schema costs nothing.** Everything that makes the file self-describing —
analyzers, field schemas, codec parameters, retrieval profiles — is about one
kilobyte in a 37.8 MiB file. Portability is not paid for in bytes.

**Per-record overhead is real.** FrameCatalog and VectorCatalog together are
19.4% of the file, roughly 76 bytes per record. At 100k records that is
7.3 MiB. It is the price of every record being individually addressable,
tombstonable, and time-indexed, and it does not shrink with corpus size.

## Against SQLite

The nearest single-file equivalent is SQLite with FTS5 for the lexical side and
[sqlite-vec](https://github.com/asg017/sqlite-vec) for vectors. Same 100,000
records, same text, same vectors:

| | **Valise** | SQLite + FTS5 + sqlite-vec |
|---|---:|---:|
| Total | **37.8 MiB** | 87.2 MiB |
| Per record | **397 B** | 914 B |
| vs. the 70.2 MiB of raw input | **1.86× smaller** | 1.24× *larger* |

Where the difference comes from:

| Component | **Valise** | SQLite | |
|---|---:|---:|---|
| Vectors (codes + descriptors) | **15.2 MiB** | 51.2 MiB | **3.4× smaller** |
| Documents | **6.6 MiB** | 24.3 MiB | **3.7× smaller** |
| Lexical index | 9.7 MiB | 10.0 MiB | *near-identical* |

**The lexical indexes are the same size.** FTS5 is good, and we do not beat it
on raw index bytes at this scale. Our size advantage comes from two places
neither of which is the inverted index: vectors are quantized to 5.5 bits per
dimension rather than stored as raw `float32`, and documents are
zstd-compressed rather than stored as rows.

That also means the ratio moves with your data. A vector-heavy corpus widens
the gap; a text-only corpus of already-compressed documents narrows it to
almost nothing. **Measure your own corpus** — the command is one line.

## Run it on your own file

```bash
valise info yourfile.vls --segments          # human-readable
valise info yourfile.vls --segments --json   # machine-readable
```

The breakdown counts **live segments only**. After deletes, the total will be
smaller than the file on disk — the difference is tombstoned segments that
`Store::compact` has not yet reclaimed. `valise info` prints both numbers, so
the gap between them tells you what compaction would recover.

From code, the same accounting is on the read handle:

```rust
for (kind, count, bytes) in store.raw().reader().storage_breakdown() {
    println!("{kind:?}: {count} segments, {bytes} bytes");
}
```

## Related

- [FORMAT.md](FORMAT.md) — format version, commit and recovery protocol
- [VECTOR_SEARCH.md](VECTOR_SEARCH.md) — how the vector codes are produced and searched
- [text.md](text.md) — the lexical primitives in detail
- [bench/REPRODUCE.md](../bench/REPRODUCE.md) — the end-to-end benchmark
