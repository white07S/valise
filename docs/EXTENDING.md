# Extending Valise: new scoring algorithms and codec families

This guide covers the two supported extension points of the reference
implementation:

1. **Lexical (text) scoring algorithms** — BM25-style scorers over the
   canonical persisted statistics.
2. **Vector codec families** — quantization codecs persisted alongside
   the vectors they encode.

Both are designed so that the persisted format does **not** change for
the common case. Read the invariants section first; it is the contract
every extension must keep.

---

## 1. Adding a lexical scoring algorithm

Today's scorers (BM25, TF-IDF/count cosine + approx variants, Dice,
overlap, containment) are free functions in `src/retrieval/` selected by
enum dispatch. Exact Jaccard also lives there but is reachable only
through an engine-level retrieval profile — it is deliberately not in the
application `Search` builder; use Dice there instead. Adding, say, LM-Dirichlet or
a DFR variant:

### Files to touch

| Step | File | What |
|---|---|---|
| 1 | `src/retrieval/<algo>.rs` | The scorer (see skeleton below). |
| 2 | `src/retrieval.rs` | `pub mod <algo>;` |
| 3 | `src/file/query_types.rs` | New `QueryAlgorithm` variant; dispatch arm in `query_text` (`src/file/text_ops.rs`); channel classification arm in `text_algorithm_channel` (or hybrid queries will error). |
| 4 | `src/db/query.rs` | New `TextScorer` variant + its `to_algorithm()` arm — this is what applications see. |
| 5 | tests | Unit tests in the scorer module (build state via `build_flush_output`), an integration case in `tests/db_search.rs`. |

**Prefer profile-free variants** (`QueryAlgorithm::Bm25 { k1, b }`-style,
parameters inline). Persisting a new `RetrievalProfileParams` variant in
the catalog is a bincode enum addition → a format-minor event → a
`FORMAT_MINOR` bump and `MIGRATION.md` entry. Only do that when the
parameters must be durable file defaults.

### The canonical statistics contract

Every scorer receives `(&TextSpaceState, &[FrameStub], query_tokens,
top_k, channel_k)` and may rely on exactly these persisted /
open-time-rebuilt statistics (`src/file/text_indexing.rs`):

| Statistic | Accessor |
|---|---|
| token bytes → `TermId` | `state.lookup_term(&[u8])` |
| document frequency | `state.df_for(term_id)` |
| collection frequency (for LM/DFR smoothing) | `state.cf_for(term_id)` — persisted and rebuilt; currently has no consumer, it is kept for exactly this purpose |
| posting list (frame-id-sorted SoA) | `state.lookup_postings_sparse(term_id)` |
| impact-sorted posting (vote phase) | `retrieval::tfidf::lookup_or_build_impact_posting` |
| per-doc total terms | `state.doc_lengths[frame_id.0]` (dense) |
| per-doc unique terms | `state.doc_unique_terms[frame_id.0]` (dense) |
| corpus aggregates (n_active, total_len → avg_dl) | `retrieval::bm25::active_corpus_stats` (epoch-cached) |
| liveness / tombstones | `retrieval::bm25::build_tombstone_set(frames)` |

**Not available** (persisted format has no slot for them): positions,
doc-side learned term weights. A SPLADE-style scorer can weight the
*query* side only.

### Scorer skeleton

The shared shape every scorer follows (the duplication across bm25.rs /
tfidf.rs is being narrowed — start from the shared pieces, don't copy a
whole file):

```text
1. resolve terms      retrieval::resolve_query_terms(state, tokens)
                      → Vec<(TermId, qf)> first-seen order, dict-only
2. tombstones         build_tombstone_set(frames)  (only if any_tombstoned)
3. corpus stats       active_corpus_stats(state, &tombstoned)
4. per-term metadata  df/cf → idf-like weight per (term, qf)
5. optional budget    tfidf::water_fill_budget (channel_k vote mode)
6. vote loop          state.with_arena(|arena| …)  — accumulate into the
                      generation-stamped QueryArena (zero alloc/memset)
7. exact or rerank    full postings, or binary-search the frame-sorted
                      posting for the channel_k survivors
8. top-k              retrieval::top_k::select_top_k_from_touched
                      (use the `_signed` variant if scores can go
                      negative — e.g. negative IDF under tombstones)
```

### Hot-loop constraints

- **Static dispatch.** Keep the inner contribution branch-minimal and
  inlineable; no `dyn` in the per-posting loop. The SoA posting loop
  shapes are deliberately auto-vectorizable.
- **No per-query allocation** beyond what the arena/caches already
  amortize. Per-doc precomputes (normalizers, L2 norms) belong in an
  `Arc`-cached vector keyed by parameter bit patterns — see
  `DocNormalizerCache` in `text_indexing.rs` and invalidate via
  `invalidate_bm25_cache`.
- **Honor `channel_k`.** The vote→rerank contract (and its over-fetch
  sizing in `db/reader.rs`) assumes a scorer caps posting reads when
  `channel_k` is `Some`.

---

## 2. Adding a vector codec family

Family dispatch is centralized in [`CodecParams`](../src/codec.rs): one
enum variant covers persistence, fingerprinting, and instantiation. The
sketch-scan search path is family-generic through the `VectorCodec`
trait; only a bespoke rerank kernel needs extra wiring.

### Files to touch

| Step | File | What |
|---|---|---|
| 1 | `src/format/catalog.rs` | **Append** a `CodecFamily` variant (bincode positional — never reorder). |
| 2 | `src/format/<family>_params.rs` | Params struct + hand-rolled little-endian wire codec (magic, version, validate; mirror `upq_params.rs`). Register in `src/format.rs`. |
| 3 | `src/codec/<family>.rs` | The codec. Implement `VectorCodec` **fully**: `family`, `dimension`, `base_bytes_per_vector`, `encode`, `decode_lossy`, `sign_sketch`, `asymmetric_distance`, `prepare_query` + `asymmetric_distance_prepared`. Plus inherent `from_params` / `to_params(calibration_id)` and a `fit_from_sample` calibration entry. Register in `src/codec.rs`. |
| 4 | `src/codec.rs` | Add the `CodecParams` variant; the compiler surfaces every match arm (`family`, `wire_version`, `family_discriminant` — append-only!, `encode`, `decode`, `instantiate`). |
| 5 | `src/file/codec_io.rs` | Optional convenience wrappers (`register_codec_<family>_from_sample`). The generic `register_codec(CodecParams)` already handles the crash-safe plumbing. |
| 6 | `src/format.rs` + `MIGRATION.md` | Bump `FORMAT_MINOR` (the header check is exact-match) and document. |
| 7 | tests | `tests/<family>_vector.rs` (round-trip, reopen, crash-safety, search recall) — mirror `tests/upq_vector.rs`. Regenerate nothing in `tests/golden_format_v2.rs`: the golden fixture must keep passing unchanged (it registers no new-family codec). |
| 8 | `src/db/schema.rs` | Store-layer surface: a `Codec::<family>()` constructor (+ parameterized variants) and the matching private `CodecSpec` variant. Constructor-only by design — no public enum variant, so this is non-breaking. |
| 9 | `src/db/space.rs` | A dispatch arm in `register_codec_for` — the **one** place both eager (`Calibrate::now`) and deferred (first-commit flush) calibration lower a `CodecSpec` to an engine registration. |
| 10 | `src/db/schema_doc.rs` | A `CodecDoc` variant (family-tagged JSON: `{"family":"<family>",...}`) + its `from_spec`/`to_spec` arms, so the per-field choice survives reopen via the persisted schema doc. Extend the golden/round-trip tests. |
| 11 | `bindings/valise-py` | The Python dataclass mirror (e.g. `Qam`/`Upq` in `schema.py`) + lowering in the PyO3 store; a row in `docs/PARITY.md`. Defaults live in Rust only — Python passes the values through. |

### How a family surfaces at the Store layer (steps 8–11)

Steps 1–7 make the family work at the engine level (`register_codec` +
search). Steps 8–11 make it *declarable*:

```rust
store.collection("notes", Schema::new()
    .vector("dense", Vector::dim(768).codec(Codec::upq())))?;
```

`Codec` is opaque and constructor-only (`docs/SIMPLE_API_SPEC.md` §2), so a
new family is a new constructor, never a new public variant. The constructor
maps to a private `CodecSpec` variant; `register_codec_for` in
`src/db/space.rs` lowers it via your `fit_from_sample` entry; `CodecDoc` in
`src/db/schema_doc.rs` gives it a family-tagged JSON shape so a deferred
choice survives reopen. Until you add the Python dataclass, the family is
still reachable from Python through the prebuilt escape hatch
(`Codec::from_params` ↔ `{"family":"prebuilt","params_family":...,"params_hex":...}`).

### What you get for free

- **Registration / readback / open**: `register_codec`,
  `codec_params(codec_id)`, and the open-time codec cache are generic
  over `CodecParams`.
- **Sketch candidate generation**: `vector_search.rs` derives the
  per-space dense sign-sketch via `VectorCodec::sign_sketch` for every
  family. Your `sign_sketch` must bit-match the query-side
  `retrieval::sketch::pack_query_sketch` packing (LSB-first u64 words,
  `dim.div_ceil(64)` words, padding bits zero).
- **Compaction**: `db/compact.rs` re-registers any family via
  `codec_params` and preserves its operating point on recalibrate.
- **Brute-force + generic rerank**: the trait `prepare_query` /
  `asymmetric_distance_prepared` path scores any family.

### What still needs bespoke wiring (optional, for speed)

A family-specific rerank kernel (like UPQ's decoded-i8 contiguous dot)
needs:

- an open-time cache: a downcast arm in
  `vector_search.rs::build_vector_base_ptrs` populating a per-space
  structure (mirror `upq_i8_by_space`);
- a search path module mirroring `src/file/upq_search.rs`, dispatched
  from `vector_search` by downcast.

Without it, the family still searches correctly through the generic
trait path — measure before adding a kernel.

### Determinism requirements

- `encode` must be byte-identical across runs **and architectures**
  (golden + cross-arch replay discipline).
- Anything derived from a seed (rotation sign masks, codebooks) must
  use the canonical `SplitMix64` derivations — never persist what can
  be re-derived.

---

## 3. Invariants you must not break

- **`tests/golden_format_v2.rs` is the format contract.** Its BLAKE3
  hash pins every persisted byte of the fixture. If it fails, you
  changed the format — either revert, or you are knowingly doing a
  versioned format change (bump `FORMAT_MINOR`, update the golden hash
  *in the same commit*, and document in `MIGRATION.md`).
- **Bincode enums are positional.** `CodecFamily`, catalog descriptor
  enums, `RetrievalProfileParams`: append variants only; placeholders
  like `_LegacyInt4PcaZero` exist to hold ordinals and must stay.
- **Deterministic-id discriminants are append-only**
  (`CodecParams::family_discriminant`: legacy=0, QAM=1, UPQ=2, …).
- **No sidecar files, no WAL, no in-place mutation.** Everything is
  append-only segments + the TOC footer commit point.
- **Score conventions**: engine `VectorHit.score` is smaller = better;
  `retrieval::Hit` and `db::Hit` are higher = better. Don't mix them.
