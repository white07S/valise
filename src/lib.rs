// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! valise — the Valise reference implementation: a
//! single-file, append-only, crash-safe, multi-collection archive for
//! AI/retrieval workloads (documents + chunks, canonical lexical
//! statistics, quantized vectors, hybrid search).
//!
//! # Two API levels
//!
//! - **[`prelude`] / [`db`] — the application layer (start here).**
//!   [`db::Store`] gives schema-by-name, keyed records, text / vector /
//!   hybrid search, partitions, and compaction. It is the surface the
//!   Python bindings consume and the recommended way to build on Valise.
//!   See `examples/quickstart.rs`.
//! - **[`ValiseFile`] / [`Database`] — the engine.** Explicit catalog
//!   registration (codecs, embedding spaces, analyzers, text spaces),
//!   frame/vector primitives, and raw introspection. Reach for it when
//!   extending the format or embedding Valise in another storage system,
//!   not when writing applications.

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
