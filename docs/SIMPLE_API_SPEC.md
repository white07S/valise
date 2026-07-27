# Simple API v2 — synthesized design spec (implementation contract)

Status: IMPLEMENTED in Rust and mirrored by the Python binding facade. The
local Python parity suite is the packaging/readiness gate; published wheel
status is tracked outside this design contract. Originally an APPROVED design
(3-proposal judge panel, unanimous winner = first-timer lens + grafts from the
evolution and symmetry proposals). This file is the contract for the Rust core,
the Python mirror, docs, and tests. Deviations require updating this file in the
same commit — see §9 for the deviations recorded by the landed implementation.

## 1. Goal

One self-explanatory call declares a collection; reopen needs no re-declaration;
Rust and Python expose the same nouns, defaults, and capabilities; codec choice
is per-field with progressive disclosure; every public surface is extensible
without breaking changes.

```rust
// Rust
use valise::prelude::*;
let store = Store::open("kb.vls")?;                       // open-or-create
store.collection("notes", Schema::new()
    .text("body")                                         // English BM25 (default)
    .vector("embedding", Vector::dim(768)))?;             // cosine, auto-calibrating QAM(5,6)
let mut w = store.writer();
w.put("notes", "doc-1", Record::new()
    .text("body", "...").vector("embedding", &emb))?;
w.commit()?;                                              // durability + first calibration
let hits = store.search("notes", Search::new()
    .text("body", "single file vector store")
    .vector("embedding", &emb)
    .top_k(10))?;                                         // RRF-fused hybrid by default
// second process:
let store = Store::open("kb.vls")?;
store.get("notes", "doc-1")?;                             // just works (schema persisted)
```

```python
# Python — same nouns, same defaults
from valise import Store, Schema, Vector, Record, Search
store = Store.open("kb.vls")
store.collection("notes", Schema().text("body").vector("embedding", Vector(dim=768)))
with store.writer() as w:
    w.put("notes", "doc-1", Record().text("body", "...").vector("embedding", emb))
    w.commit()
hits = store.search("notes", Search().text("body", "...").vector("embedding", emb).top_k(10))
```

## 2. Public types (Rust; Python mirrors 1:1 via dataclass-style value objects)

### Field specs (new module `src/db/schema.rs`; old `Shape` in collection.rs is replaced)

```rust
pub struct Schema { /* fields: Vec<(String, FieldSpec)> — private */ }
impl Schema {
    pub fn new() -> Schema;
    pub fn text(self, name: &str) -> Schema;                  // Text::english()
    pub fn text_with(self, name: &str, spec: Text) -> Schema;
    pub fn vector(self, name: &str, spec: Vector) -> Schema;  // dim is mandatory → spec always explicit
}

#[non_exhaustive] pub struct Text { /* private: Inline{lang} | Shared(Space) */ }
impl Text { pub fn english() -> Text; pub fn raw() -> Text; pub fn space(s: &Space) -> Text; }

#[non_exhaustive] pub struct Vector { /* private: Inline{dim, metric, codec, calibrate} | Shared(Space) */ }
impl Vector {
    pub fn dim(dim: u32) -> Vector;            // cosine, Codec::default(), Calibrate::default()
    pub fn space(s: &Space) -> Vector;         // codec/calibrate REJECTED on shared (configured at define_space)
    pub fn metric(self, m: Metric) -> Vector;
    pub fn codec(self, c: Codec) -> Vector;
    pub fn calibrate(self, c: Calibrate) -> Vector;
}

/// OPAQUE — constructor-only (evolution graft: adding a param to a family is
/// non-breaking; future families are new constructors, not new public variants).
pub struct Codec { /* private enum CodecSpec { Qam{amp_bits,phase_bits}, Upq{cells,design}, Prebuilt(CodecParams) } */ }
impl Codec {
    pub fn qam() -> Codec;                       // (5, 6) — the production default
    pub fn qam_bits(amp_bits: u8, phase_bits: u8) -> Codec;
    pub fn upq() -> Codec;                       // (2048 cells, Empirical)
    pub fn upq_cells(cells: u32) -> Codec;
    pub fn upq_with(cells: u32, design: UpqDesign) -> Codec;
    pub fn from_params(params: CodecParams) -> Codec;   // engine escape hatch (replaces Calibration::Params)
}
impl Default for Codec { /* qam() */ }
#[non_exhaustive] pub enum UpqDesign { Empirical, Rayleigh }

/// OPAQUE constructor-only, same reasoning.
pub struct Calibrate { /* private: Auto{sample} | Now(Vec<Vec<f32>>) */ }
impl Calibrate {
    pub fn auto() -> Calibrate;                  // sample = 50_000, fit at first commit
    pub fn auto_sample(sample: usize) -> Calibrate;
    pub fn now(sample: Vec<Vec<f32>>) -> Calibrate;   // eager fit at declaration
}
impl Default for Calibrate { /* auto() */ }

pub struct Space { /* name + kind; replaces SpaceRef (temporary alias only) */ }
```

Python: `Schema`, `Text(lang=Lang.ENGLISH, space=None)`, `Vector(dim=None, *,
metric=Metric.COSINE, codec=None, calibrate=None, space=None)` (exactly one of
dim/space), `Qam(amp_bits=5, phase_bits=6)`, `Upq(cells=2048,
design=Design.EMPIRICAL)`, `Auto(sample=50_000)`, `Now(ndarray)` in new
`python/valise/schema.py`. `Schema().vector(name, Vector(...))` also accepts
`Schema().vector(name, 768)` int shorthand.

### Store

```rust
impl Store {
    pub fn open(path) -> Result<Store>;          // open-or-create, Durability::Buffered
    pub fn open_with(path, StoreOptions) -> Result<Store>;
    pub fn create(path) -> Result<Store>;        // fail-if-exists
    pub fn create_with(path, StoreOptions) -> Result<Store>;
    pub fn collection(&self, name, Schema) -> Result<Collection>;   // create-or-open + persist schema
    pub fn define_space(&self, name, spec: impl Into<FieldSpec>) -> Result<Space>;  // shared-space tier
    pub fn space(&self, name) -> Option<Space>;
    pub fn spaces(&self) -> Vec<SpaceInfo>;      // lists shared + auto (auto flagged)
    pub fn collections(&self) -> Vec<CollectionInfo>;   // excludes ~valise.schema
    // writer()/writer_owned()/try_writer_owned()/reader() unchanged
    pub fn get(&self, coll, key: impl Into<Key>) -> Result<Option<Stored>>;
    pub fn search(&self, coll, Search) -> Result<SearchResult>;
    pub fn search_view(&self, &View, Search) -> Result<SearchResult>;
    // partitioned()/compact()/stats()/raw() unchanged (partitioned takes Schema)
}
```

`define_vector_space`/`define_text_space`/`VectorSpace`/`TextSpace`/
`Calibration`/`open_collection` are REMOVED. `Dtype` leaves the db surface
(engine is F32-only in v1).

### Search & results

```rust
impl Search {
    pub fn text(self, field, query) -> Search;                 // Bm25 { k1: 1.2, b: 0.75 }
    pub fn text_with(self, field, query, TextScorer) -> Search;
    pub fn vector(self, field, query) -> Search;               // Rerank::Accurate
    pub fn vector_with(self, field, query, Rerank) -> Search;
    // fuse()/recency()/top_k()/now() unchanged; defaults Rrf{k:60}, k=10
}

/// Evolution + symmetry graft: result-level metadata becomes additive.
pub struct SearchResult { pub hits: Vec<Hit> /* non_exhaustive-ish via private ext slot */ }
impl Deref<Target = [Hit]> for SearchResult; impl IntoIterator for SearchResult / &SearchResult;
```

`#[non_exhaustive]` sweep: `TextScorer`, `Fusion`, `Recency`, `Rerank`,
`UpqDesign`, plus constructor fns so struct-variant literals stop being the
construction idiom (existing variant fields may stay readable; construction via
fns: `TextScorer::bm25()`, `Fusion::rrf(60)`, etc.).

### Errors

```rust
Error::NotCalibrated { space: String },   // vector search before first vector commit;
    // message: "vector space '{space}' is not calibrated yet — commit a batch containing vectors first"
Error::SchemaMismatch { collection: String, detail: String },  // divergent redeclare, field-level diff
```
Python: `NotCalibratedError`, `SchemaMismatchError` in errors.py.

## 3. Spaces story (three tiers)

1. **Hidden/auto (default)**: inline field specs auto-define one private space
   per field named `~auto/{collection}/{field}` (real persisted engine spaces;
   provider stays "valise-db", model carries the name). No silent dedupe across
   collections. Users never see "space" unless they list `spaces()` or read a
   NotCalibrated error (which names the auto space).
2. **Explicit shared**: `store.define_space("emb", Vector::dim(768).codec(Codec::upq()))`
   → bind with `Vector::space(&emb)` / `Text::space(&en)` in any schema.
   Codec/calibrate on a shared-space binding is a declaration-time error.
3. **Engine**: `store.raw()` → `ValiseFile::register_codec(CodecParams)` etc., unchanged.

User collection/space names starting with `~` are rejected by validation.

## 4. Schema persistence (zero format change)

- Reserved engine collection `~valise.schema` holds ONE ordinary Payload frame per
  user collection (constants + codec in new `src/db/schema_doc.rs`).
- Frame uri = `[0xF5] ++ collection_name` — 0xF5 is not a Key tag (1/2/3), so
  `Key::decode` → None and `IdentityIndex::rebuild` already skips it unchanged.
- Payload = versioned JSON: `{"valise_schema":1,"fields":[{"name","kind","space","spec"}...]}`
  with codec/calibrate as tagged objects; shared-space fields carry `"spec":null`.
  Decoder MUST ignore unknown keys (forward-compat; pinned by test).
- created_at = watermark = `max(0, max created_at over active frames)` — the
  time-index encoder requires created_at non-decreasing in frame-id order
  globally; the watermark provably adds no new ingest constraint. The 0 floor
  forbids pre-1970 backfill on a fresh file ONLY for frames put after a schema
  doc — document this; engine already errors at commit with its standard message.
- Durability: **stage-only, one deterministic rule** — the schema frame is a
  pending engine mutation that persists at the NEXT commit (exactly like
  `create_collection` itself today). Documented + pinned by test.
- Redeclare semantics: identical (canonical, ordered field comparison) → no-op;
  additive → tombstone old doc frame + put new; divergent → `SchemaMismatch`.
- Reopen: `SchemaRegistry::rebuild` gains a pass scanning `~valise.schema` active
  frames, restoring every collection's Schema INCLUDING the codec+calibrate spec
  of not-yet-calibrated vector fields (reconstructing `VecState::Deferred`).
- `compact.rs`: schema frames stream through verbatim (all active frames are
  carried). Switch the `coll_text` map derivation from in-process shapes to the
  persisted docs — this FIXES the latent bug where compact-after-reopen silently
  drops text indexing. Delete the in-process shape carry-forward hack ONLY after
  the compact round-trip test proves the rebuild path.
- Rejected alternative (document in schema_doc.rs so it isn't re-litigated):
  `CollectionDesc.metadata_ref` exists in the catalog but has no engine write
  path, and compaction's `create_collection_at` would drop it.
- A db-layer golden test pins the serialized schema-doc JSON bytes.
- Old files (no docs): one-time `collection(name, schema)` re-declare persists
  the doc; afterwards reopen just works. New files remain readable by old code.

## 5. Codec lowering

`SchemaRegistry`'s `VectorSpaceEntry` carries the chosen `Codec` + `Calibrate`.
Eager (`Calibrate::now`) and prebuilt paths dispatch at definition:
`register_codec_qam_from_sample_with_bits` / `register_codec_upq_from_sample_with_options`
/ `register_codec(params)`. `WriterCore::flush_deferred`'s `Plan::Calib` gains
the same family dispatch (today QAM-only). Auto-space and shared-space entries
both record the spec so deferred choices survive reopen via the schema doc.

## 6. Python parity additions (bindings/valise-py)

- New: `schema.py` (Schema/Text/Vector/Qam/Upq/Auto/Now/Design), `partition.py`
  (Partition/Window/Partitioned/View + `forget_before`), `search_view`,
  `NotCalibratedError`/`SchemaMismatchError`, codec+calibrate lowering in
  `src/store.rs`, `Space` typed handles, `collections() -> list[CollectionInfo]`,
  `spaces()`.
- Renames/fixes from the symmetry audit: `Durability.FSYNC` → `FULL_SYNC`
  (current name lies — it maps to FullSync); Python default rerank changes
  FAST → ACCURATE (aligns with Rust); fix the `Range` recency docstring
  (inclusive, not half-open); `SearchResult` already exists — keep, Rust adopts
  the same shape.
- `put_many` stays Python-side sugar (docs note: ≈ Rust `Writer::bulk()`).
- Defaults live in RUST only; Python passes None/sentinels and the native layer
  applies defaults (single source of truth).

## 7. Landing plan (keep diffs reviewable)

1. Rust mechanical rename/reshape: Schema/Text/Vector/Codec/Calibrate/Space,
   Store::open/open_with, Search sugar, SearchResult, error variants; migrate
   tests + quickstart + prelude. Tests green.
2. Rust behavior: codec lowering + flush_deferred dispatch; schema persistence
   (schema_doc.rs, rebuild pass, compact switch); new tests
   (tests/db_schema_persist.rs, codec-choice tests). Tests green.
3. Python mirror + parity (separate commit(s)); pytest green via uv venv + maturin.
4. Docs: README, quickstart(s), EXTENDING.md cross-refs, mkdocs pages, parity
   table (one concept, two tabs); CONTRIBUTING.md module map.
5. Adversarial parity review before final commit.

## 8. Invariants (unchanged from EXTENDING.md)

Golden format test must stay green untouched (it drives the engine only). No
sidecars, no WAL, no format change. Writes transactional through the single
Writer. Engine `VectorHit` smaller-is-better vs db `Hit` higher-is-better.

## 9. Implementation deviations (recorded; the implementation is the truth)

The landed Rust implementation (`src/db/schema.rs`, `src/db/schema_doc.rs`)
deviates from the letter of §2/§4 in these documented ways:

a. **Prebuilt codec doc shape.** `Codec::from_params` persists as
   `{"family":"prebuilt","params_family":"qam"|"upq","params_hex":...}` —
   `params_family` rides alongside `params_hex` so the decoder can route the
   hex bytes to the right wire codec without sniffing magic numbers.
b. **Spec-object key order.** Inline `spec` objects serialize through
   `serde_json::Value`, so their keys are **alphabetical** (e.g.
   `"calibrate"` before `"codec"` before `"dim"`), not declaration order.
   The full doc encoding is byte-deterministic and pinned by the golden test
   `schema_doc.rs::golden_two_field_doc_bytes`.
c. **Shared DEFERRED vector spaces are not doc-reconstructable.** A shared
   binding persists `"spec":null` (per §4), so a shared **vector** space that
   was never calibrated before the last commit cannot be rebuilt at open; its
   collections degrade to shape-less until `define_space` + one re-declare.
   Calibrated shared spaces and all text spaces restore fine; inline/auto
   fields (the default path) always restore, including deferred codec choice.
d. **`Calibrate::now` serializes without its sample.** The sample is
   transient — consumed at declaration — so the doc carries `{"mode":"now"}`
   only.
e. **Python uses default arguments instead of `text_with`/`vector_with`.**
   `Search().text(field, query, scorer=Bm25())` and
   `.vector(field, query, rerank=Rerank.ACCURATE)` — idiomatic kwargs, same
   concepts and same defaults as the Rust `_with` pairs (`Schema().text` /
   `.vector` likewise take optional spec arguments).
