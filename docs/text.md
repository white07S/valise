# TXC1 — Valise Text Feature Codec v1

Status: Draft, aligned with `FORMAT.md` v2.2. Section numbers below
quote the un-rewritten v0.1 prose spec; the current source-of-truth
references for each primitive live in `src/format/text.rs`,
`src/format/postings.rs`, `src/format/doc_stats.rs`, and the BM25 /
TF-IDF / Jaccard scorers under `src/retrieval/`.

`TXC1` is the reference text codec for Valise. It is **not a separate file format** — it is a set of segment-payload layouts and algorithms that Valise uses to populate the canonical text primitives defined in `FORMAT.md` §12.0.

## 0. Language scope: English only for v1

TXC1 v1 is calibrated and tested for **English text only**. The following components are tightly coupled to English assumptions:

- **Tokenizers** (`UnicodeWords`, `Whitespace`, `CharNgram`, `TokenShingle`) work well for whitespace-separated, Latin-script languages. They produce poor results for CJK (Chinese, Japanese, Korean), Thai, Khmer, and other languages that need dictionary-based or model-based segmentation.
- **Stemming** is `None` only in v1. Porter and Snowball stemmers are language-specific and arrive in v2.
- **Stopword set** is the standard English stopword reference; multilingual stopword sets are v2+.
- **Case folding and accent folding** behave correctly for Latin scripts but do not match language-aware normalization in scripts that do not use case (Arabic, Hebrew, CJK).
- **Retrieval calibration** (BM25 `k1`/`b` defaults, TF-IDF parameters, Jaccard thresholds) is set against English benchmarks (TREC, MS MARCO).

**Operational rule**: index any Unicode text you want — TXC1 v1 will not error — but the format does not promise meaningful retrieval quality on non-English input. For multilingual corpora, wait for v2 (script-aware tokenizers, BPE/WordPiece subwords, multilingual stopword sets, language-specific stemmers) or register multiple text spaces with custom analyzers.

This scope is a deliberate v1 choice. Valise's segment types are language-neutral; nothing in the on-disk format is English-specific. The scope restriction lives in the analyzer and retrieval-profile defaults, not in the file structure. v2 will widen the analyzer enum (adding `CjkBigram`, `Subword(Bpe)`, `Subword(WordPiece)`, etc.) and the stemmer enum without breaking v1 readers.

## 1. Goal and principle

Store one canonical set of text features per `text_space_id`. Compute every classical lexical algorithm from that canonical set. Never persist algorithm-specific outputs (no stored BM25 score, no stored TF-IDF vector, no stored Jaccard score).

Canonical primitives (Valise §12.0):

| # | Primitive | Valise segment |
|---|---|---|
| 1 | term dictionary | `TermDictionarySegment` (0x0010) |
| 2 | postings (inverted index) | `PostingsSegment` (0x0011) |
| 3 | document statistics | `DocStatsSegment` (0x0013) |

Optional, opt-in primitives:

| Primitive | Valise segment | Required for |
|---|---|---|
| compressed raw text | `RawTextSegment` (0x0017) | exact reconstruction, char edit distance, snippets |
| corpus zstd dictionary | `RawTextDictionarySegment` (0x0018) | better compression on small docs |
| positions | `PositionsSegment` (0x0012) | phrase / proximity queries |
| token sets (shingles, char-ngrams) | `TokenSetSegment` (0x0014) | shingle Jaccard, char n-gram Jaccard |
| MinHash signatures | `MinHashSignatureSegment` (0x0015) | candidate generation before Jaccard rerank |
| forward index cache | `ForwardIndexSegment` (0x0016) | per-doc inspection / pairwise scoring speed |

`ForwardIndexSegment` is **derived** — fully reconstructable from postings + docstats. It is never canonical truth.

## 2. Algorithms supported by the canonical layer

All of the following are computed directly from the canonical primitives. None require additional storage.

**Token-set similarity**: Jaccard, Dice, overlap, containment, equality, subset/superset.

**Bag-of-words distances**: count cosine, L1/Manhattan, L2/Euclidean, Minkowski, Canberra, Bray-Curtis, chi-square, Hellinger, Jensen-Shannon, KL divergence (with smoothing).

**TF-IDF family**: TF-IDF (raw, log, sublinear, binary, augmented), TF-IDF cosine, TF-IDF L2 distance, TF-IDF dot product. Variant frozen by `RetrievalProfileDesc.params.tfidf` (Valise §10.6).

**BM25 family**: BM25, BM25+, BM25L, BM25-Adpt. BM25F additionally requires the `field_mask` in postings and the `field_lengths` array in docstats — both already in Valise §12.2 / §12.4. Variant frozen by `RetrievalProfileDesc.params.bm25`.

**Boolean retrieval**: AND, OR, NOT, required/optional, minimum_should_match.

**Language models**: query likelihood with Dirichlet smoothing, Jelinek-Mercer smoothing, divergence-from-randomness scoring. Use `term_dict.collection_freq` for corpus-frequency smoothing.

**Phrase / proximity** (require `PositionsSegment`): exact phrase, slop-N phrase, ordered-window, unordered-window, positional language models.

**N-gram and shingle**: char 3/4/5-gram Jaccard, token bigram/trigram similarity, w-shingling. **Each tokenization mode lives in its own `text_space_id`** — char-3-grams and word tokens are not mixed into one term dictionary.

**Edit distance** (requires `RawTextSegment`): Levenshtein, Damerau-Levenshtein, Jaro, Jaro-Winkler, LCS, LCSubstring, Needleman-Wunsch, Smith-Waterman.

**Approximate near-duplicate**: MinHash, SimHash, w-shingling LSH. MinHash signatures are **derivable from the canonical token set**; they are persisted as `MinHashSignatureSegment` only as a query-time speed cache, and are invalidated whenever the analyzer or canonical primitives change.

## 3. What the canonical layer does NOT support without raw text

The canonical primitives lose:

- original case (after `case_fold`)
- exact punctuation (after `punctuation_policy = Drop`)
- whitespace, newlines, formatting
- byte offsets
- token order (without `PositionsSegment`)
- exact original string

For exact original-string reconstruction, store `RawTextSegment` (§7).

## 4. Multi-tokenizer composition

A `text_space_id` is bound to one analyzer. To support multiple tokenizations of the same corpus:

| Use case | text_space_id | Analyzer |
|---|---|---|
| word-level BM25 / TF-IDF / term-Jaccard | `text_space_word` | `tokenizer = UnicodeWords`, `case_fold = true` |
| char-3-gram fuzzy match | `text_space_char3gram` | `tokenizer = CharNgram`, `ngram_min = 3`, `ngram_max = 3` |
| shingle near-duplicate | `text_space_shingle3` | `tokenizer = TokenShingle`, `shingle_size = 3` |

Each text space owns its own canonical primitives. Frames participate in multiple text spaces by being indexed under each analyzer separately. The frame payload bytes are the source for all of them; only the analyzer differs.

A retrieval profile (Valise §10.6) lives in exactly one `text_space_id`. Cross-space queries (e.g., "BM25 over word tokens AND shingle Jaccard above threshold") are a fusion concern and are computed by querying each text space and merging results — there is no on-disk cross-space structure.

## 5. Analyzer descriptor

Valise §10.4 defines the persisted analyzer:

```text
AnalyzerDesc
- analyzer_id: u32
- unicode_normalization: enum { None, Nfc, Nfkc }
- case_fold: bool
- accent_fold: bool
- tokenizer: enum { UnicodeWords, Whitespace, CharNgram, TokenShingle }
- stemming: enum { None }                         // v1 supports None only
- stopword_set_ref: optional MetadataRef
- stopword_query_only: bool                       // v1 default: true
- min_token_len: u16                              // 0 = unlimited
- max_token_len: u16                              // 0 = unlimited
- shingle_size: u8                                // 0 if not used
- ngram_min, ngram_max: optional u8
- punctuation_policy: enum { Drop, Keep }
```

### 5.1 Stopword policy

The default `stopword_query_only = true` keeps stopword postings in the index. Reasons:

- Stopwords carry information for phrase queries ("to be or not to be" loses meaning if stopwords are dropped at index time).
- IDF naturally downweights high-frequency terms; index-time removal is redundant with this and irreversible.
- A stopword list at query time is a single-line filter; at index time it is a destructive transformation.

Setting `stopword_query_only = false` removes stopwords from the analyzed token stream before indexing. This saves a small amount of storage at the cost of phrase capability. Index-time removal is supported but discouraged.

### 5.2 Determinism

Analyzer behavior must be byte-deterministic. Given the same `AnalyzerDesc` and the same input text, two implementations must produce the same analyzed token stream including order, term bytes, and intra-field positions.

Implementation tests must include:

- Unicode normalization: NFC and NFKC golden vectors for each case-folding combination.
- Tokenizer boundary cases (CJK, mixed script, emoji, ZWJ sequences).
- Stopword set application (when `stopword_query_only = false`).
- Token-length filters at exact boundary values.

## 6. Term ID assignment

Term IDs are **first-seen monotonic** within a `text_space_id` (Valise §12.0.2):

- The first term ever indexed in this text space gets `term_id = 1`.
- New terms get the next available ID.
- Once assigned, a term ID is stable across all subsequent commits.
- Existing postings are never rewritten to renumber terms.

Within a single commit, multiple new terms are sorted by lexicographic order of `term_bytes` before assignment, so the assignment is deterministic given a deterministic input ordering.

Rationale: any other policy (e.g., frequency-descending) requires re-numbering on commit, which in an append-only file means rewriting all postings. First-seen monotonic is the only assignment that scales.

Compression cost: posting `frame_id` gaps still compress well under varint because gaps depend on document arrival order, not term ordering. Term-ID gaps in the dictionary are uniform — varint cost is approximately constant per term.

## 7. Compressed raw text

`RawTextSegment` (Valise §12.7) stores compressed original UTF-8 per `(frame_id, field_id)`. Recommended profile:

- `compression = Zstd`
- per-document independent compression for documents > 4 KB
- shared zstd dictionary for documents ≤ 4 KB

### 7.1 Shared zstd dictionary

For corpora dominated by small documents, train a zstd dictionary at first commit on a representative sample (default: up to 100 MB or 10 000 documents, whichever is smaller). Store the trained dictionary as `RawTextDictionarySegment` (type `0x0018`) and reference it from `RawTextSegment.dictionary_ref`.

Compression of a document then uses `zstd_compress_with_dict(text, dict, level)`. Decompression uses `zstd_decompress_with_dict(blob, dict)`.

Re-training the dictionary on later commits is **not** allowed — once a `RawTextSegment` references a dictionary, the dictionary is immutable. To switch dictionaries, write a new dictionary segment and a new raw-text segment; old documents continue to decode against the old dictionary.

### 7.2 Reconstruction

```
fn reconstruct_raw(frame_id, field_id) -> String:
  entry = lookup(RawTextSegment, frame_id, field_id)
  blob = read_bytes(entry.bytes_offset, entry.compressed_len)
  if entry.compression == Zstd && dictionary_ref.is_some():
    dict = read_dictionary(dictionary_ref)
    return String::from_utf8(zstd_decompress_with_dict(blob, dict))
  else if entry.compression == Zstd:
    return String::from_utf8(zstd_decompress(blob))
  else if entry.compression == None:
    return String::from_utf8(blob)
  else: error
```

Validation: decompressed length must equal `entry.uncompressed_len`. Decompressed bytes must be valid UTF-8.

### 7.3 When to skip raw text storage

If exact reconstruction, character edit distance, and snippet generation are not required, omit `RawTextSegment` entirely. The canonical primitives are sufficient for every algorithm in §2 except the edit-distance family. Saves 0.2×–0.4× of original text size.

## 8. Term dictionary

Valise §12.1:

```text
TermDictionaryEntry
- term_id: u32
- term_bytes: Vec<u8>                             // UTF-8 bytes; analyzer-normalized
- collection_freq: u64
- doc_freq: u32
```

### 8.1 Layout

A `TermDictionarySegment` is a delta segment: it carries only newly-assigned `term_id`s relative to the prior segment in the chain (Valise §19.2). Each segment payload uses the standard catalog envelope (`magic = VALISEC`, table `TermDictionary`).

Within a segment, entries are sorted ascending by `term_id`. Term bytes are stored as length-prefixed bytes, followed by varint `collection_freq` and `doc_freq`.

### 8.2 Aggregating df and cf across the chain

The `doc_freq` and `collection_freq` stored in a delta segment cover **only** the contribution of that segment's commits. The active values at query time are the sum across the chain:

```
df_active(term_id)  = Σ over chain segments s : s.doc_freq[term_id]
cf_active(term_id)  = Σ over chain segments s : s.collection_freq[term_id]
```

Readers walk the chain once at file open and materialize an in-memory `HashMap<u32, (u64, u32)>` for fast access. Memory cost: ~16 bytes per unique term.

### 8.3 Tombstones

When a frame is deleted, postings are not rewritten (Valise §12.0.3), so `df_active` and `cf_active` may overcount. Implementations have two options:

1. **Accept the overcount** as a small bias. For most retrieval workloads, deletion rates are low and the bias is < 1%.
2. **Maintain a deleted-frames bitmap** loaded at file open, and adjust `df_active` and `cf_active` by walking postings for affected terms once. Cost: O(deleted_frames × avg_terms_per_frame).

## 9. Postings

Valise §12.2:

```text
PostingList
- term_id: u32
- df: u32                                         // df contributed by this delta segment only
- postings[]:
  - frame_id: u64                                 // gap-encoded varint
  - collection_id: u32
  - field_mask: u32
  - term_freq: u32                                // varint
  - positions_ref: optional PositionsRef
```

### 9.1 Encoding

Each posting list is encoded as:

```
varint term_id
varint df (segment-local)
varint frame_id_first
[frame_id_gap, term_freq, field_mask, positions_ref_present_bit, optional positions_ref] × (df - 1)
```

Frame IDs within a posting list are sorted ascending and gap-encoded. The encoding splits into parallel streams (one for `frame_id_gap`, one for `term_freq`, one for `field_mask`) for slightly better compression and SIMD-friendly decode.

### 9.2 Field mask

`field_mask` is a bitmask over `FieldSchemaDesc.fields`. Bit `i` is set if the term occurs in field `i` for that document. Used for BM25F field-aware scoring and field-restricted Boolean queries (`title:foo`).

For documents that store only a single text field, `field_mask` is always `0b1`. The cost is minimal: varint-encoded, almost always one byte per posting.

### 9.3 Term frequency

Stored as raw count, not log-normalized. Scoring functions apply their own `tf_mode` transformation at query time per `RetrievalProfileDesc`.

### 9.4 Positions reference

`positions_ref: Option<PositionsRef>` points to a per-`(frame_id, field_id)` positions blob in a `PositionsSegment`. Stored only when the analyzer's text space has positional retrieval enabled.

```text
PositionsRef
- segment_id: u64
- ordinal: u32
```

Resolution: open the positions segment via the segment registry, locate the entry at `ordinal`, decode delta-coded positions.

### 9.5 Delta segments

Each commit appends a `PostingsSegment` carrying only postings introduced by that commit. The active state is the union across the chain, with newer entries overriding older ones for the same `(term_id, frame_id)` pair.

For a non-update workload (frames are immutable once committed), there are no `(term_id, frame_id)` duplicates across segments. For an update workload (re-indexing a frame), the newer posting wins; the older posting is logically tombstoned but physically remains.

## 10. Document statistics

Valise §12.4:

```text
DocStatsEntry
- frame_id: u64
- collection_id: u32
- text_space_id: u32
- field_lengths[]: u32                            // length per field, parallel to FieldSchemaDesc.fields
- total_terms: u32
- unique_terms: u32
```

### 10.1 Required for

- BM25 normalization: `doc_len = Σ field_lengths`, `avg_doc_len = corpus.total_terms / corpus.N`
- BM25F: per-field `field_lengths[i]`
- TF-IDF L2 normalization (when `norm_mode = L2`)
- term-Jaccard derivation: `|d| = unique_terms`

### 10.2 Corpus-level aggregates

Computed once at file open by walking the docstats chain:

```
N            = count of active doc rows (filter tombstoned frames using frame catalog)
total_terms  = Σ DocStatsEntry.total_terms over active rows
avg_doc_len  = total_terms / N
```

Materialized in memory as `CorpusStats { N: u64, total_terms: u64, avg_doc_len: f32 }` per text space.

### 10.3 Layout

`DocStatsSegment` is a delta segment: it carries only doc rows added or updated by the current commit. Stored sorted by `frame_id`. Field lengths use varint encoding; `total_terms` and `unique_terms` are varint.

## 11. Forward index (derived cache)

Optional. Stored as `ForwardIndexSegment` (type `0x0016`).

```text
ForwardIndexEntry
- frame_id: u64
- entries[]:
  - term_id: u32                                  // gap-encoded varint
  - term_freq: u32                                // varint
  - positions_ref: optional PositionsRef
```

### 11.1 Recovery from postings

A complete forward index can always be reconstructed:

```
for each PostingList p in inverted index:
  for each posting (frame_id, term_freq, field_mask, positions_ref) in p:
    forward[frame_id].push((p.term_id, term_freq, positions_ref))
sort each forward[frame_id] by term_id
```

Storage cost is O(total postings); time cost is O(total postings). For a corpus where forward access dominates queries (per-doc inspection, pairwise scoring), persisting the cache is worthwhile. For pure search workloads (BM25, TF-IDF over many terms), it adds storage with no query benefit.

`ForwardIndexSegment` may be invalidated and rebuilt at any time. It is never the source of truth for `tf` or positions; those live in postings.

### 11.2 Byte-identity rule

If persisted, `ForwardIndexSegment` payload bytes must be byte-identical to the bytes produced by reconstructing from the canonical primitives. This makes the cache verifiable: a mismatch indicates corruption or a stale segment.

## 12. Score computation

All scoring functions read from the in-memory representation built at file open: term dictionary HashMap, postings indexed by `term_id`, docstats indexed by `frame_id`, corpus stats. Tombstoned frames are filtered post-hoc using the live frame catalog.

### 12.1 BM25

```
score(query, doc) =
  Σ over query term t in postings:
    if doc is tombstoned: skip
    if t not in term_dict: skip
    df  = df_active(t)
    tf  = posting(t, doc).term_freq                 // 0 if posting absent
    dl  = docstats(doc).total_terms
    avgdl = corpus.avg_doc_len
    N   = corpus.N
    idf(t) = ln((N - df + 0.5) / (df + 0.5) + 1)    // RobertsonSparckJones
    norm   = 1 - b + b × dl / avgdl
    score += idf(t) × tf × (k1 + 1) / (tf + k1 × norm)
```

`k1`, `b`, and `idf_variant` come from `RetrievalProfileDesc.params.bm25`.

For BM25F:

```
combined_tf = Σ over fields f : weight[f] × posting(t, doc, field_mask=f).term_freq / norm_field[f]
norm_field[f] = 1 - b_f + b_f × field_length(doc, f) / avg_field_length[f]
```

### 12.2 TF-IDF

```
idf(t)    = log((N + 1) / (df + 1)) + 1            // for idf_mode = Smooth
tf_eff(t) = match tf_mode:
  Raw       => raw_tf
  Log       => 1 + ln(raw_tf)
  Sublinear => sqrt(raw_tf)
weight(d, t) = tf_eff(t) × idf(t)
```

For TF-IDF cosine, normalize by `||d||₂` where `||d||₂² = Σ_t weight(d, t)²`. Norms can be precomputed per doc and cached, or computed on demand. v1 computes on demand; norm caching is a v2 optimization.

For TF-IDF L2 distance: walk the union of query and doc terms, sum squared differences of weights. Same primitives, same cost as cosine.

### 12.3 Term-Jaccard

```
|q ∩ d| = count of distinct query term_ids whose posting list contains d
|d|     = docstats(d).unique_terms
|q|     = count of distinct query term_ids
J(q, d) = |q ∩ d| / (|q| + |d| − |q ∩ d|)
```

Document-at-a-time evaluation: for each candidate doc `d` produced by the query terms' posting lists, increment `|q ∩ d|` and emit when all relevant postings have been visited.

### 12.4 Count cosine, Dice, overlap

All derive from postings:

```
cosine(d_a, d_b) = dot(d_a, d_b) / (||d_a||₂ × ||d_b||₂)
  where dot(d_a, d_b) = Σ over t : tf(t, d_a) × tf(t, d_b)
        ||d||₂² = Σ over t : tf(t, d)²

dice(d_a, d_b)    = 2 × |A ∩ B| / (|A| + |B|)
overlap(d_a, d_b) = |A ∩ B| / min(|A|, |B|)
                    where A, B are the term sets of d_a, d_b
```

These are pairwise scoring algorithms. For pairwise workloads, `ForwardIndexSegment` materially speeds them up; for sparse query workloads, the inverted index is sufficient.

### 12.5 Phrase / proximity (positions required)

```
phrase(q, d) =
  for each query term t_i, get position list P_i in d
  return true iff there exist p_1, p_2, ..., p_k with
    p_i ∈ P_i and p_i+1 = p_i + 1 for all i

proximity(q, d, slop) =
  for each query term t_i, get position list P_i in d
  return true iff there exist p_1, ..., p_k with
    p_i ∈ P_i, max - min ≤ slop, all distinct
```

Position decoding cost: O(|positions|) per `(frame_id, term_id)` lookup, with delta-coded varint.

### 12.6 Edit distance (raw text required)

```
levenshtein(d_a, d_b) =
  text_a = reconstruct_raw(d_a, search_field)
  text_b = reconstruct_raw(d_b, search_field)
  return levenshtein_distance(text_a, text_b)
```

Cost: O(|text_a| × |text_b|) via dynamic programming. Suitable for small candidate sets after MinHash or Jaccard candidate generation.

### 12.7 MinHash candidate generation

```
candidates(q, k) =
  q_signature = compute_minhash(q.tokens, signature_len)
  for each frame f with persisted MinHash:
    f_signature = MinHashSignatureSegment.lookup(f)
    estimated_jaccard = matches(q_signature, f_signature) / signature_len
    if estimated_jaccard > threshold: emit (f, estimated_jaccard)
  return top-k by estimated_jaccard
```

Then rerank candidates with exact term-Jaccard from postings. MinHash provides O(N) candidate filtering; exact rerank is O(K) where K is the candidate count.

## 13. Query API surface

The text retrieval module exposes:

```rust
// Score a query against the active corpus under a specific retrieval profile.
fn score(
    text_space_id: TextSpaceId,
    profile_id: RetrievalProfileId,
    query: &str,
    options: ScoreOptions,
) -> Vec<(FrameId, f32)>;

// Pairwise score between two frames under a specific profile or algorithm.
fn pairwise(
    text_space_id: TextSpaceId,
    algorithm: PairwiseAlgorithm,                 // BM25, Cosine, Jaccard, Dice, Overlap, Levenshtein, ...
    a: FrameId,
    b: FrameId,
) -> Result<f32>;

// Boolean retrieval.
fn boolean_search(
    text_space_id: TextSpaceId,
    expr: BooleanExpr,
) -> Vec<FrameId>;

// Reconstruct original text (requires RawTextSegment).
fn reconstruct_text(frame_id: FrameId, field_id: FieldId) -> Result<String>;
```

Score options carry collection filters (only return results from these `collection_id`s), top-K, score threshold, and tombstone handling.

## 14. Indexing flow

The library exposes:

```rust
// One-time setup per text space:
let analyzer_id      = valise.register_analyzer(AnalyzerDesc { ... })?;
let field_schema_id  = valise.register_field_schema(FieldSchemaDesc { fields: vec![SearchText, Title, ...] })?;
let profile_id       = valise.register_retrieval_profile(RetrievalProfileDesc::Bm25 { k1: 1.2, b: 0.75 })?;
let text_space_id    = valise.register_text_space(TextSpaceDesc {
    analyzer_id, field_schema_id, default_profile_id: profile_id, ...
})?;

// Per frame:
valise.put_frame(PutFrame { collection_id, role: Document, payload: text.into_bytes(), ... })?;
valise.index_frame_text(frame_id, text_space_id, &[(SearchText, &text), (Title, &title)])?;

// Commit:
valise.commit()?;
```

`index_frame_text` runs the analyzer over the input text and queues `(frame_id, [(field_id, analyzed_tokens)])` in an in-memory builder per text space. `commit()` then:

1. For each touched text space:
   a. Generate new `term_id`s for any newly-seen `term_bytes` (first-seen monotonic, ties broken by lex order).
   b. Build a delta `TermDictionarySegment` carrying new terms with this commit's `df` and `cf` contributions.
   c. Build a delta `PostingsSegment` carrying postings for `(term_id, frame_id)` pairs introduced this commit. Term IDs that are not new still get postings — the postings segment is the source of truth for the per-segment df.
   d. Build a delta `DocStatsSegment` for newly-indexed frames.
   e. (Optional) If `store_raw_text = true`, build a delta `RawTextSegment`. For very small documents, accumulate into a shared zstd dictionary at first commit.
2. Update the active TOC's per-text-space roots to point at the new segment heads.
3. Catalog deltas link to the previous heads via `previous_root` exactly as collection/frame catalogs do.

### 14.1 Indexing without WAL persistence

The in-memory builder is lost if the process crashes before `commit()`. On reopen, the user must re-run `index_frame_text` for any frames that were buffered but uncommitted. This is acceptable because:

- The frame payload is the source of truth (durably stored as a payload segment by `put_frame`).
- The analyzer is deterministic.
- Re-running `index_frame_text` over the same payload produces identical token streams.

Implementations that prefer to persist the buffer across crashes may add an `IndexFrameText` WAL op; this is a v2 amendment to Valise §8.2.

## 15. Calibration

For BM25 specifically, the `k1` and `b` parameters are tuneable. Default `k1 = 1.2`, `b = 0.75` are the long-standing search-engine defaults.

For TF-IDF, the `tf_mode × idf_mode × norm_mode` combination must be persisted in `RetrievalProfileDesc.params.tfidf` because TF-IDF "results" are not portable without it. Two TF-IDF profiles in the same text space that differ in any of these are distinct profiles and produce different rankings.

For analyzer choice, run a small-scale calibration before committing to an analyzer for a corpus:

1. Hand-label 100–1 000 query-document relevance pairs for the target corpus.
2. For each candidate `AnalyzerDesc` (varying case folding, accent folding, tokenizer, stopword behavior), index a sample, score the labeled pairs.
3. Pick the analyzer with the best NDCG@10 or MRR.
4. Register that analyzer; do not change it for the lifetime of the text space (changing the analyzer requires a new text space and a full re-index).

## 16. Storage budget (1M short documents)

Approximate storage for a corpus of 1 million documents averaging 200 words each:

| Component | Bytes |
|---|---|
| compressed raw text (zstd dict) | ~250 MB |
| term dictionary (~500K unique terms) | ~30 MB |
| postings (no positions) | ~600 MB |
| docstats | ~30 MB |
| **canonical-only total** | **~910 MB** |
| positions (optional) | +~400 MB |
| MinHash signatures (sig_len=128) | +~125 MB |
| forward index (optional cache) | +~600 MB (duplicates posting tf) |

For comparison, raw uncompressed UTF-8 of 1M × 200 words × 6 bytes/word = ~1.2 GB. Canonical-only Valise text storage is 0.7×–0.8× of the raw text size while supporting every algorithm in §2 except the edit-distance family.

## 17. Comparison with alternatives

- **Lucene / Tantivy**: opaque engine-specific blob with high-quality retrieval. Valise's canonical-then-derive layer is more portable (any reader can reconstruct any algorithm without engine code) at the cost of giving up engine-specific optimizations like skip lists in the on-disk format. Valise postings are byte-canonical; Lucene's are not.
- **PISA**: research IR system focused on fast BM25 with block-max indexes. Valise's posting layout is simpler (no block-max) and trades query speed for portability. A v2 amendment may add block-max as an optional posting encoding.
- **DiskANN / SPFresh** (vector-only): not comparable; Valise text is the lexical layer to complement those.
- **JSONL + recompute**: the "no index" baseline. Works for offline analysis; impossible for online retrieval at scale.
- **Custom vector-only stores** (Faiss, ScaNN, hnswlib): not comparable; they handle vectors only.

The Valise text layer is positioned for: portable text+vector storage where the file is the source of truth and any reader can reconstruct any algorithm without engine-specific dependencies. It is not positioned to beat Lucene on raw query throughput; it is positioned to never have to re-index when the retrieval algorithm changes.

## 18. v1 implementation order

1. Analyzer pipeline (`analyze(&AnalyzerDesc, &str) -> Vec<AnalyzedToken>`) with golden-vector tests for determinism.
2. `TermDictionarySegment` + `PostingsSegment` + `DocStatsSegment` codecs with roundtrip tests.
3. Delta-chain reader for term dictionary (HashMap materialization) and postings (lazy per-term).
4. BM25 scoring with hand-computed reference tests on a 3-doc corpus.
5. TF-IDF scoring with mode coverage tests.
6. Term-Jaccard via the postings + docstats derivation.
7. Tombstone filtering at query time using the frame catalog.
8. `RawTextSegment` with per-doc zstd. Reconstruction round-trip test.
9. Shared zstd dictionary (`RawTextDictionarySegment`).
10. Boolean retrieval (AND/OR/NOT).
11. Pairwise: count cosine, Dice, overlap.
12. Edit distance via raw text (Levenshtein only in v1).
13. Lifecycle integration tests: register text space → index frames → commit → reopen → query.

Out of scope for v1: positions, phrase/proximity, MinHash, shingles, char n-grams, forward index cache, BM25F, language models, calibration tooling. These land as separate slices.

## 19. Test surface

Per-section test counts to land before declaring v1 complete:

| Concern | Tests |
|---|---|
| Analyzer determinism | ~10 (one per `(unicode, case, accent, tokenizer)` combination, plus boundary cases) |
| Term dict / postings / docstats codec roundtrip | ~12 (encode → decode → compare; truncated input rejection; checksum mismatch) |
| Delta chain materialization | ~6 (one new term across 3 commits; df aggregation; tombstone filtering) |
| BM25 hand-computed reference | ~3 (3-doc corpus; k1, b sweep) |
| TF-IDF mode coverage | ~6 (each tf_mode × norm_mode pairing) |
| Term-Jaccard derived-vs-stored | ~3 (build TokenSetSegment from postings, compare) |
| Tombstone behavior | ~3 (delete then query; df bias bound; bitmap correction) |
| Raw text roundtrip | ~3 (with zstd, with shared dict, without compression) |
| Boolean retrieval | ~5 (AND, OR, NOT, MSM, nested) |
| Pairwise algorithms | ~5 (cosine, Dice, overlap, Levenshtein, JSD with smoothing) |

≈55 tests for v1.

## 20. Forward-compat hooks

- New segment types reserve `0x0019`–`0x001F` for additional text-related segments (e.g., positions delta encoding v2, learned tokenizer state).
- `AnalyzerDesc.tokenizer` enum has space for `Subword`, `Bpe`, `WordPiece` variants in v2.
- `RetrievalProfileDesc.params` is a discriminated union; new profile types add new variants without breaking existing readers (readers reject unknown variants — new profiles bump `format_minor`).
- `MinHashSignatureSegment` versioning: future `signature_kind` field in `MinHashSignatureEntry` lets v2 introduce SimHash, weighted MinHash, etc., under one segment type.
