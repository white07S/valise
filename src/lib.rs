// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! **Retrieval in one file.**
//!
//! Your RAG prototype works. Now ship it — and the corpus stops being a
//! thing you have and becomes a thing you *operate*: a vector-database
//! container, an index directory that must travel with the documents and
//! stay consistent with them, a rebuild step in CI. The most valuable
//! artifact you built is the one piece you cannot hand to anyone.
//!
//! SQLite solved this for relational data. Valise does it for retrieval —
//! documents, a BM25 index, compressed vectors, and the schema describing
//! them, in one `.vls` file you copy, version, and query in place.
//!
//! ```
//! use valise::prelude::*;
//! # fn main() -> Result<()> {
//! # let dir = std::env::temp_dir().join("valise_doc_intro");
//! # std::fs::create_dir_all(&dir)?;
//! # let path = dir.join("kb.vls");
//! # let _ = std::fs::remove_file(&path);
//! let store = Store::open(&path)?;              // open-or-create
//! store.collection("notes", Schema::new().text("body"))?;
//!
//! let mut w = store.writer();
//! w.put("notes", "doc-1", Record::new().text("body", "portable retrieval capsules"))?;
//! w.commit()?;                                  // the durability point
//!
//! let hits = store.search("notes", Search::new().text("body", "retrieval").top_k(10))?;
//! assert_eq!(hits.len(), 1);
//! # Ok(())
//! # }
//! # main().unwrap();
//! ```
//!
//! The schema is stored **in the file**. A later run, another process, or
//! another machine calls [`db::Store::open`] and searches immediately —
//! nothing to re-declare and nothing to migrate.
//!
//! # There is no index to build
//!
//! Every ANN library builds a graph before it answers anything, and pays
//! again whenever the corpus changes. Valise builds **nothing**. Vectors
//! are stored only as compact quantized codes, and the candidate-search
//! structure — a one-bit-per-dimension sign sketch — is derived from those
//! same codes when the file is opened. It lives in memory; nothing beyond
//! the codes is ever written.
//!
//! Measured at 100k vectors × 768d against recall-matched baselines:
//! **0.50 s** to ingest and commit, versus 50 s for FAISS HNSW, 91 s for
//! USearch, and 139 s for hnswlib — **90–310× faster to build**, in
//! **5–6× fewer bytes**, with no graph on disk to rebuild, invalidate, or
//! corrupt independently of the data.
//!
//! The bill is that the candidate scan is linear in corpus size, so a
//! mature graph answers individual queries faster. Prefer Valise when the
//! corpus fits your latency budget through a scan: at d=768, 100k vectors
//! take ~1 ms on one modern aarch64 core (p50 1,061 µs, Apple M4 Max) and
//! ~6.7 ms on an older AVX2 x86 laptop even across all its threads (p50
//! 6,722 µs, Intel i7-8550U) — see `bench/REPRODUCE.md` §6. Scale linearly,
//! and size against the slower number unless you know the hardware. Or
//! reach for it at any scale when the corpus is copied, shipped, or rebuilt
//! more often than about once per ten thousand queries. Below ~256
//! dimensions a tuned index wins outright.
//!
//! It is **not** a hosted vector database or a distributed search cluster,
//! and **it does not generate embeddings** — bring your own model.
//!
//! # Two API levels
//!
//! - **[`prelude`] / [`db`] — the application layer. Start here.**
//!   [`db::Store`] gives schema-by-name, keyed records, text / vector /
//!   hybrid search, partitions, and compaction. It is the surface the
//!   Python bindings consume.
//! - **[`ValiseFile`] / [`Database`] — the engine.** Explicit catalog
//!   registration (codecs, embedding spaces, analyzers, text spaces),
//!   frame and vector primitives, raw introspection. Reach for it when
//!   extending the format or embedding Valise inside another storage
//!   system, not when writing applications.
//!
//! # Hybrid search
//!
//! Add a vector field and query both channels; with two or more channels
//! they are fused with reciprocal-rank fusion by default.
//!
//! ```no_run
//! use valise::prelude::*;
//! # fn main() -> Result<()> {
//! # let embedding: Vec<f32> = vec![0.0; 768];
//! # let query: Vec<f32> = vec![0.0; 768];
//! let store = Store::open("kb.vls")?;
//! store.collection(
//!     "notes",
//!     Schema::new()
//!         .text("body")                            // English BM25 by default
//!         .vector("dense", Vector::dim(768)),      // cosine + auto-calibrated QAM
//! )?;
//!
//! let mut w = store.writer();
//! w.put(
//!     "notes",
//!     "doc-1",
//!     Record::new()
//!         .text("body", "portable retrieval capsules")
//!         .vector("dense", &embedding),
//! )?;
//! w.commit()?;
//!
//! let hits = store.search(
//!     "notes",
//!     Search::new()
//!         .text("body", "retrieval capsule")
//!         .vector("dense", &query)
//!         .fuse(Fusion::rrf(60))
//!         .top_k(10),
//! )?;
//! # Ok(())
//! # }
//! ```
//!
//! Valise owns its retrieval primitives rather than embedding an opaque
//! search-engine blob: BM25, TF-IDF cosine, count cosine, approximate
//! cosine variants, and Dice / overlap / containment set scorers over
//! canonical term dictionaries and postings; QAM Lloyd-Max and UPQ polar
//! vector codecs with NEON and AVX2 kernels. There is no persisted HNSW
//! graph or IVF index — so nothing to rebuild, and nothing on disk but the
//! capsule.
//!
//! # Durability
//!
//! [`db::Writer::commit`] is the only durability point. Commits are
//! footer-rooted and atomic: after any crash a reader sees either the
//! previous committed state or the new one, never a mixture. Segment
//! payloads carry BLAKE3 checksums and the table of contents is
//! self-checksummed; if the active footer is unusable, recovery scans the
//! file and reinstates the newest footer that fully validates.
//!
//! Concurrency is N readers and one writer, coordinated across processes
//! through a region inside the file header. [`Database::writer`] blocks
//! for the single writer slot; [`Database::try_writer`] returns `None`
//! rather than waiting.
//!
//! # Platform support
//!
//! Linux and macOS, on x86-64 and aarch64. **Windows is not supported
//! yet** — the commit protocol relies on positional file IO and fcntl OFD
//! advisory locks, and the Windows equivalents are not implemented.
//! Building there fails with an explicit message.
//!
//! # Command line
//!
//! `cargo install valise` also gives you a CLI for inspecting a capsule
//! without writing any code:
//!
//! ```text
//! valise info kb.vls                        # size, collections, record counts
//! valise search kb.vls notes "query" --top-k 5   # text search
//! valise get kb.vls notes doc-1             # one record
//! valise export kb.vls > kb.jsonl           # every record, as JSON Lines
//! ```
//!
//! # Further reading
//!
//! The on-disk format, the commit and recovery protocol, the vector-search
//! design, and the benchmark methodology are documented in the
//! [repository](https://github.com/white07S/valise).

// Fail with something a human can act on rather than a dozen "cannot find
// `unix` in `os`" errors from deep inside the IO layer.
#[cfg(not(unix))]
compile_error!(
    "valise supports Unix platforms only (Linux, macOS, BSD) as of 0.1.\n\
     The commit protocol relies on positional file IO (`FileExt::write_at`) \
     and fcntl OFD advisory locks. The Windows equivalents (`seek_write`, \
     `LockFileEx`) are not implemented yet — see the platform-support note \
     in the README."
);

/// Opt-in profiling output for the `VALISE_*_PROFILE` env vars.
///
/// Writes straight to stderr on purpose: these are developer diagnostics
/// enabled by an env var, and routing them through `tracing` would make
/// them invisible unless the caller had installed a subscriber. The
/// crate otherwise denies `print_stderr`, so this macro is the single
/// sanctioned exception.
macro_rules! prof_eprintln {
    ($($arg:tt)*) => {{
        #[allow(clippy::print_stderr)]
        {
            eprintln!($($arg)*);
        }
    }};
}
pub(crate) use prof_eprintln;

pub mod codec;
pub mod concurrency;
pub mod db;
pub mod error;
pub mod file;
pub mod format;
pub mod io;
pub mod retrieval;
pub mod text;

/// Application-facing API in one import: `use valise::prelude::*;`.
///
/// Re-exports the [`db`] Store layer plus the shared error types —
/// exactly the surface an application (or the Python bindings) needs to
/// open a store, declare collections, ingest records, and search.
pub mod prelude {
    pub use crate::db::{
        Bulk, Calibrate, Codec, Collection, CollectionInfo, CompactOptions, CompactReport,
        Durability, FieldSpec, Fusion, Hit, Key, Lang, Metric, OwnedWriter, Partition, Partitioned,
        Reader, Recency, Record, Rerank, Schema, Search, SearchResult, Space, SpaceInfo, SpaceKind,
        Store, StoreOptions, StoreStats, Stored, Text, TextScorer, TfMode, UpqDesign, Value,
        Vector, View, Window, Writer,
    };
    pub use crate::error::{Error, Result};
    pub use crate::{CodecParams, VERSION};
}

// ---- shared error / version ----
pub use error::{Error, Result};
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

// ---- codec parameters (power-user surface; see docs/EXTENDING.md) ----
pub use codec::CodecParams;
pub use codec::QamLloydMaxParams;
pub use codec::{UpqDesignSource, UpqParams};

// ---- engine: process-shared handle + snapshot reads ----
pub use concurrency::connection::{ReadConnection, WriteConnection};
pub use concurrency::database::Database;
pub use concurrency::snapshot::Snapshot;

// ---- engine: single-writer file API ----
pub use file::{
    AutoPromote, CommitOutcome, CommitProfile, CreateOptions, EmbeddingSpaceSpec, HybridHit,
    HybridQuery, HybridTextChannel, HybridVectorChannel, IngestProfile, OpenMode, PutFrame,
    PutVector, QueryAlgorithm, QueryProfile, ReadVectorResult, Reconstruct, TextIndexBuildProfile,
    TextMode, TextQuery, TimeQuery, ValiseFile, VectorContract, VectorFidelity, VectorHit,
    VectorOpenProfile, VectorSearchQuery, VoteSearchTrace, last_query_profile,
    last_vector_open_profile,
};
pub use format::create_contract::CreateContractV1;
pub use format::dtype::{Dtype, DtypeSet};

// ---- engine: catalog descriptors + raw ids (schema registration) ----
pub use format::catalog::{
    AnalyzerDesc, CodecDesc, CodecFamily, EmbeddingSpaceDesc, FieldDesc, FieldSchemaDesc,
    FieldSource, FusionNormalization, FusionProfileDesc, IdfMode, IdfVariant, NormMode,
    PunctuationPolicy, RetrievalProfileDesc, RetrievalProfileParams, RetrievalProfileType,
    Stemming, StopwordsPolicy, TextSpaceDesc, TfMode, TokenSource, Tokenizer, UnicodeNormalization,
    VectorDesc, VectorMetric, VectorStatus, WeightSource,
};
pub use format::{
    AnalyzerId, CodecId, CollectionId, EmbeddingSpaceId, FieldSchemaId, FrameId, FusionProfileId,
    RetrievalProfileId, SegmentId, TextSpaceId, VectorId,
};

// ---- bench-only adapters (cargo feature `bench`; not public API) ----
#[cfg(any(test, feature = "bench"))]
pub use codec::QamLloydMaxBench;
#[cfg(any(test, feature = "bench"))]
pub use codec::simd_bench;
#[cfg(any(test, feature = "bench"))]
pub use codec::upq::{
    UpqCodec, UpqDesign, dot_i8, extract11, pack11, packed11_len, packed11_stride,
};
#[cfg(any(test, feature = "bench"))]
pub use codec::{SlidingBench, SlidingPrep};
#[cfg(any(test, feature = "bench"))]
pub use retrieval::sketch_bench;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_is_set() {
        assert!(!VERSION.is_empty());
    }
}
