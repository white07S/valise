//! File-global spaces (schema / capability) and the layer's schema registry.
//!
//! Spaces are defined rarely and reused forever (DB-layer plan §2): a text
//! space is an analyzer + field-schema + default BM25 profile; a vector space
//! is a calibrated codec + embedding space. Most applications never define a
//! space explicitly — inline field specs in a [`Schema`] auto-define one
//! private space per field, named `~auto/{collection}/{field}`. The shared
//! tier is [`crate::db::Store::define_space`] + [`Text::space`] /
//! [`Vector::space`] bindings.
//!
//! The `SchemaRegistry` makes definitions idempotent-by-name and lowers
//! each one onto the engine's `register_*` primitives. Vector-space names are
//! persisted in `EmbeddingSpaceDesc.model` (the engine has no name field);
//! text-space names use `TextSpaceDesc.name`. Both are recovered at open by
//! [`SchemaRegistry::rebuild`].

use std::collections::HashMap;

use crate::db::collection::{Field, Shape};
use crate::db::schema::{
    CalibrateMode, CodecSpec, FieldSpec, FieldSpecKind, Schema, TextBinding, VectorBinding,
    VectorInline,
};
use crate::db::schema_doc;
use crate::error::{Error, Result};
use crate::file::{EmbeddingSpaceSpec, ValiseFile};
use crate::format::catalog::{
    AnalyzerDesc, FieldDesc, FieldSchemaDesc, FieldSource, IdfVariant, PunctuationPolicy,
    RetrievalProfileDesc, RetrievalProfileParams, RetrievalProfileType, Stemming, StopwordsPolicy,
    TextSpaceDesc, Tokenizer, UnicodeNormalization, VectorMetric,
};
use crate::format::dtype::Dtype;
use crate::format::{
    AnalyzerId, CodecId, CollectionId, EmbeddingSpaceId, FieldSchemaId, RetrievalProfileId,
    TextSpaceId,
};

/// Marks every embedding space this layer creates, so [`SchemaRegistry::rebuild`]
/// can tell layer-owned spaces apart from raw-engine ones at open.
pub(crate) const PROVIDER: &str = "valise-db";

/// Prefix reserved for layer-internal names; user collection/space names
/// starting with it are rejected.
pub(crate) const RESERVED_PREFIX: char = '~';

/// Prefix of the private spaces auto-defined for inline field specs.
const AUTO_SPACE_PREFIX: &str = "~auto/";

/// The private space name for an inline field spec of `collection`/`field`.
pub(crate) fn auto_space_name(collection: &str, field: &str) -> String {
    format!("{AUTO_SPACE_PREFIX}{collection}/{field}")
}

/// Reject user-facing names that collide with the reserved internal prefix.
pub(crate) fn reject_reserved_name(what: &str, name: &str) -> Result<()> {
    if name.starts_with(RESERVED_PREFIX) {
        return Err(Error::Format(format!(
            "{what} name '{name}' is reserved: names starting with '~' are internal"
        )));
    }
    Ok(())
}

/// Distance metric for a vector space.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Metric {
    Cosine,
    Dot,
    L2,
}

impl Metric {
    pub(crate) fn to_engine(self) -> VectorMetric {
        match self {
            Metric::Cosine => VectorMetric::Cosine,
            Metric::Dot => VectorMetric::InnerProduct,
            Metric::L2 => VectorMetric::L2,
        }
    }
    pub(crate) fn from_engine(m: VectorMetric) -> Self {
        match m {
            VectorMetric::Cosine => Metric::Cosine,
            VectorMetric::InnerProduct => Metric::Dot,
            VectorMetric::L2 => Metric::L2,
        }
    }
}

/// Analyzer family for a text space. Honest about what the engine supports:
/// only English stemming/stopwords and a raw whitespace path exist.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Lang {
    English,
    Raw,
}

/// What a [`Space`] holds: a text index or vectors of a fixed dimension.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum SpaceKind {
    Vector { dim: u32 },
    Text,
}

/// An opaque, cheap-to-clone reference to a defined space. Resolution to a
/// live engine id goes through the `SchemaRegistry` by name, so a deferred
/// vector space's eventual id is always current.
#[derive(Clone, Debug)]
pub struct Space {
    name: String,
    kind: SpaceKind,
}

/// Transitional alias for the pre-redesign name; prefer [`Space`].
pub type SpaceRef = Space;

impl Space {
    pub(crate) fn vector(name: &str, dim: u32) -> Self {
        Self {
            name: name.to_owned(),
            kind: SpaceKind::Vector { dim },
        }
    }
    pub(crate) fn text(name: &str) -> Self {
        Self {
            name: name.to_owned(),
            kind: SpaceKind::Text,
        }
    }
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }
    #[must_use]
    pub fn is_text(&self) -> bool {
        matches!(self.kind, SpaceKind::Text)
    }
    #[must_use]
    pub fn is_vector(&self) -> bool {
        matches!(self.kind, SpaceKind::Vector { .. })
    }
    pub(crate) fn dim(&self) -> Option<u32> {
        match self.kind {
            SpaceKind::Vector { dim } => Some(dim),
            SpaceKind::Text => None,
        }
    }
}

/// One row of [`crate::db::Store::spaces`]: a defined space's name and kind,
/// flagged when it was auto-defined for an inline field spec.
#[derive(Clone, Debug)]
pub struct SpaceInfo {
    pub name: String,
    pub kind: SpaceKind,
    /// `true` for private `~auto/{collection}/{field}` spaces.
    pub auto: bool,
}

/// Calibration state of a registered vector space.
pub(crate) enum VecState {
    /// Codec + embedding space registered in the engine.
    Ready { space_id: EmbeddingSpaceId },
    /// Not yet registered; calibrate at commit from the first `sample_target`
    /// staged vectors, fitting the declared `codec` family/operating point.
    Deferred {
        sample_target: usize,
        codec: CodecSpec,
    },
}

pub(crate) struct VectorSpaceEntry {
    pub dim: u32,
    pub metric: Metric,
    pub state: VecState,
}

pub(crate) struct TextSpaceEntry {
    pub text_space_id: TextSpaceId,
}

pub(crate) struct CollectionRecord {
    pub collection_id: CollectionId,
    /// `None` when only recovered by name at open from a pre-doc file; set
    /// once the caller declares the collection with its schema (or the doc
    /// pass of `rebuild` restores it).
    pub shape: Option<Shape>,
    /// The canonical schema doc backing `shape`. Compared on re-declare so a
    /// divergent declaration errors even before the doc frame has committed
    /// (staged frame payloads are not readable until commit, so `plan_sync`
    /// alone cannot see a same-process pre-commit declare).
    pub doc: Option<schema_doc::SchemaDoc>,
}

/// In-memory schema state: spaces and collections keyed by name. Lives behind
/// the `Store`'s mutex; mutated only by `define_space` / `collection` (rare)
/// and the writer's deferred-calibration flush.
#[derive(Default)]
pub(crate) struct SchemaRegistry {
    pub vector_spaces: HashMap<String, VectorSpaceEntry>,
    pub text_spaces: HashMap<String, TextSpaceEntry>,
    pub collections: HashMap<String, CollectionRecord>,
}

impl SchemaRegistry {
    /// Recover layer-owned spaces and collections from a freshly opened
    /// file, then restore every collection's shape from its persisted
    /// schema doc (`~valise.schema`, see `schema_doc.rs`) — so reopen needs
    /// no re-declaration. Auto vector spaces that were still deferred
    /// (never calibrated) are reconstructed as [`VecState::Deferred`] from
    /// the doc's inline spec, exactly as a fresh declare would leave them.
    ///
    /// Degraded cases stay shape-less (one re-declare fixes them): old
    /// files without docs, docs of an unknown version, and fields bound to
    /// a shared vector space that was never calibrated (shared bindings
    /// persist `"spec":null`, so the space cannot be reconstructed).
    pub fn rebuild(&mut self, valise: &ValiseFile) -> Result<()> {
        for es in valise.embedding_spaces() {
            if es.provider == PROVIDER {
                self.vector_spaces.insert(
                    es.model.clone(),
                    VectorSpaceEntry {
                        dim: es.dimension,
                        metric: Metric::from_engine(es.metric),
                        state: VecState::Ready {
                            space_id: es.embedding_space_id,
                        },
                    },
                );
            }
        }
        for ts in valise.text_spaces() {
            self.text_spaces.insert(
                ts.name.clone(),
                TextSpaceEntry {
                    text_space_id: ts.text_space_id,
                },
            );
        }
        for c in valise.collections() {
            // `~`-prefixed collections are layer-internal (the schema-doc
            // collection); they must never surface through the registry.
            if c.name.starts_with(RESERVED_PREFIX) {
                continue;
            }
            self.collections.insert(
                c.name.clone(),
                CollectionRecord {
                    collection_id: c.collection_id,
                    shape: None,
                    doc: None,
                },
            );
        }
        for (coll_name, doc) in schema_doc::load_all(valise)? {
            if !self.collections.contains_key(&coll_name) {
                continue;
            }
            if let Some(shape) = self.shape_from_doc(&doc)
                && let Some(rec) = self.collections.get_mut(&coll_name)
            {
                rec.shape = Some(shape);
                rec.doc = Some(doc);
            }
        }
        Ok(())
    }

    /// Rebind a persisted schema doc to a [`Shape`], reconstructing any
    /// deferred (not-in-catalog) vector space from the doc's inline spec.
    /// `None` when a field cannot be restored — the collection then stays
    /// shape-less rather than coming back with a partial schema.
    fn shape_from_doc(&mut self, doc: &schema_doc::SchemaDoc) -> Option<Shape> {
        let mut fields = Vec::with_capacity(doc.fields.len());
        for f in &doc.fields {
            let field = match f.kind.as_str() {
                // Text spaces (auto and shared) are always registered in
                // the engine catalog at declare, so they recover in the
                // catalog pass above; a missing one means the doc refers
                // to state this file does not hold.
                schema_doc::KIND_TEXT => {
                    if !self.text_spaces.contains_key(&f.space) {
                        return None;
                    }
                    Field::Text(Space::text(&f.space))
                }
                schema_doc::KIND_VECTOR => {
                    if let Some(entry) = self.vector_spaces.get(&f.space) {
                        Field::Vector(Space::vector(&f.space, entry.dim))
                    } else {
                        // Never-calibrated auto space: reconstruct the
                        // deferred entry from the persisted spec. Prebuilt
                        // codecs are registered eagerly at declare and so
                        // are always in the catalog; a parse failure here
                        // (e.g. a future codec family) degrades.
                        //
                        // A missing spec ("spec":null) is a shared binding
                        // to a space absent from the catalog — not
                        // reconstructable, see the schema_doc module doc.
                        let spec = f.vector_spec()?;
                        let metric = schema_doc::metric_from_str(&spec.metric).ok()?;
                        let codec = spec.codec.to_codec_spec().ok()?;
                        self.vector_spaces.insert(
                            f.space.clone(),
                            VectorSpaceEntry {
                                dim: spec.dim,
                                metric,
                                state: VecState::Deferred {
                                    sample_target: spec.calibrate.sample_target(),
                                    codec,
                                },
                            },
                        );
                        Field::Vector(Space::vector(&f.space, spec.dim))
                    }
                }
                // Future field kinds: degrade rather than guess.
                _ => return None,
            };
            fields.push((f.name.clone(), field));
        }
        Some(Shape { fields })
    }

    /// Lower a [`FieldSpec`] onto a named space definition — the shared-tier
    /// entry point behind [`crate::db::Store::define_space`]. A spec that is
    /// itself a shared binding cannot define a new space.
    pub fn define_space(
        &mut self,
        valise: &mut ValiseFile,
        name: &str,
        spec: &FieldSpec,
    ) -> Result<Space> {
        match &spec.kind {
            FieldSpecKind::Text(t) => match &t.binding {
                TextBinding::Inline { lang } => self.define_text_space(valise, name, *lang),
                TextBinding::Shared(s) => Err(Error::Format(format!(
                    "define_space: '{name}' cannot be defined from a binding to space '{}'",
                    s.name()
                ))),
            },
            FieldSpecKind::Vector(v) => {
                if let Some(method) = v.shared_misuse {
                    return Err(shared_misuse_error("define_space", name, method));
                }
                match &v.binding {
                    VectorBinding::Inline(inline) => self.define_vector_space(valise, name, inline),
                    VectorBinding::Shared(s) => Err(Error::Format(format!(
                        "define_space: '{name}' cannot be defined from a binding to space '{}'",
                        s.name()
                    ))),
                }
            }
        }
    }

    /// Idempotent-by-name vector-space definition. Eager codec registration
    /// for `Calibrate::now` and prebuilt params; `Calibrate::auto` defers
    /// registration to the first commit with vectors.
    pub fn define_vector_space(
        &mut self,
        valise: &mut ValiseFile,
        name: &str,
        spec: &VectorInline,
    ) -> Result<Space> {
        if let Some(existing) = self.vector_spaces.get(name) {
            // Idempotent-by-name only for an identical spec; a divergent
            // redefine is a caller bug, not a silent no-op.
            if existing.dim != spec.dim {
                return Err(Error::Format(format!(
                    "define_space: '{name}' already defined with dim {} (got {})",
                    existing.dim, spec.dim
                )));
            }
            if existing.metric != spec.metric {
                return Err(Error::Format(format!(
                    "define_space: '{name}' already defined with metric {:?} (got {:?})",
                    existing.metric, spec.metric
                )));
            }
            return Ok(Space::vector(name, existing.dim));
        }
        ensure_sketchable_dim(spec.dim)?;

        let state = if let CodecSpec::Prebuilt(params) = &spec.codec.spec {
            // Prebuilt params need no calibration sample: always eager.
            if params.dimension() != spec.dim {
                return Err(Error::Format(format!(
                    "define_space: '{name}' dim {} disagrees with prebuilt params dimension {}",
                    spec.dim,
                    params.dimension()
                )));
            }
            let codec = valise.register_codec(params.clone())?;
            let space_id = register_embedding_space(valise, name, spec.dim, spec.metric, codec)?;
            VecState::Ready { space_id }
        } else {
            match &spec.calibrate.mode {
                CalibrateMode::Auto { sample } => VecState::Deferred {
                    sample_target: (*sample).max(1),
                    codec: spec.codec.spec.clone(),
                },
                CalibrateMode::Now(sample) => {
                    let codec = register_codec_for(valise, spec.dim, &spec.codec.spec, sample)?;
                    let space_id =
                        register_embedding_space(valise, name, spec.dim, spec.metric, codec)?;
                    VecState::Ready { space_id }
                }
            }
        };

        self.vector_spaces.insert(
            name.to_owned(),
            VectorSpaceEntry {
                dim: spec.dim,
                metric: spec.metric,
                state,
            },
        );
        Ok(Space::vector(name, spec.dim))
    }

    /// Idempotent-by-name text-space definition: registers analyzer +
    /// field-schema + default BM25 profile + text space.
    pub fn define_text_space(
        &mut self,
        valise: &mut ValiseFile,
        name: &str,
        lang: Lang,
    ) -> Result<Space> {
        if self.text_spaces.contains_key(name) {
            return Ok(Space::text(name));
        }
        let analyzer_id = valise.register_analyzer(analyzer_desc(lang))?;
        let field_schema_id = valise.register_field_schema(field_schema_desc())?;
        let profile_id = valise.register_retrieval_profile(bm25_profile_desc())?;
        let text_space_id = valise.register_text_space(TextSpaceDesc {
            text_space_id: TextSpaceId(0),
            name: name.to_owned(),
            analyzer_id,
            field_schema_id,
            default_profile_id: profile_id,
            flags: 0,
            enabled_retrievers: 0,
        })?;
        self.text_spaces
            .insert(name.to_owned(), TextSpaceEntry { text_space_id });
        Ok(Space::text(name))
    }

    /// Resolve every field of `schema` for `coll` — auto-defining the private
    /// space of each inline spec, validating shared bindings — and return the
    /// bound [`Shape`]. The schema is structurally validated *before* any
    /// space is defined, so an invalid declaration has no side effects.
    pub fn bind_schema(
        &mut self,
        valise: &mut ValiseFile,
        coll: &str,
        schema: &Schema,
    ) -> Result<Shape> {
        validate_schema(coll, schema)?;
        let mut fields = Vec::with_capacity(schema.fields.len());
        for (fname, spec) in &schema.fields {
            let field = match &spec.kind {
                FieldSpecKind::Text(t) => match &t.binding {
                    TextBinding::Inline { lang } => {
                        let space =
                            self.define_text_space(valise, &auto_space_name(coll, fname), *lang)?;
                        Field::Text(space)
                    }
                    TextBinding::Shared(s) => Field::Text(s.clone()),
                },
                FieldSpecKind::Vector(v) => match &v.binding {
                    VectorBinding::Inline(inline) => {
                        let space = self.define_vector_space(
                            valise,
                            &auto_space_name(coll, fname),
                            inline,
                        )?;
                        Field::Vector(space)
                    }
                    VectorBinding::Shared(s) => Field::Vector(s.clone()),
                },
            };
            fields.push((fname.clone(), field));
        }
        let shape = Shape { fields };
        shape.validate()?;
        Ok(shape)
    }

    /// Resolve a vector space's live engine id, if it is `Ready`.
    pub fn vector_space_id(&self, name: &str) -> Option<EmbeddingSpaceId> {
        match self.vector_spaces.get(name)?.state {
            VecState::Ready { space_id } => Some(space_id),
            VecState::Deferred { .. } => None,
        }
    }

    /// Resolve a text space's live engine id.
    pub fn text_space_id(&self, name: &str) -> Option<TextSpaceId> {
        self.text_spaces.get(name).map(|t| t.text_space_id)
    }

    /// Every defined space, auto-flagged, sorted by name (deterministic).
    pub fn space_infos(&self) -> Vec<SpaceInfo> {
        let mut out: Vec<SpaceInfo> = self
            .vector_spaces
            .iter()
            .map(|(name, e)| SpaceInfo {
                name: name.clone(),
                kind: SpaceKind::Vector { dim: e.dim },
                auto: name.starts_with(AUTO_SPACE_PREFIX),
            })
            .chain(self.text_spaces.keys().map(|name| SpaceInfo {
                name: name.clone(),
                kind: SpaceKind::Text,
                auto: name.starts_with(AUTO_SPACE_PREFIX),
            }))
            .collect();
        out.sort_by(|a, b| a.name.cmp(&b.name));
        out
    }
}

/// Structural validation of a declared schema, run before any space is
/// defined: unique field names, at most one text field, distinct vector
/// spaces (auto spaces are distinct by construction), shared bindings of the
/// right kind, and no configuration on shared bindings.
fn validate_schema(coll: &str, schema: &Schema) -> Result<()> {
    let mut text_fields = 0usize;
    let mut seen_names: Vec<&str> = Vec::with_capacity(schema.fields.len());
    let mut seen_vec_spaces: Vec<String> = Vec::new();
    for (name, spec) in &schema.fields {
        if seen_names.contains(&name.as_str()) {
            return Err(Error::Format(format!(
                "schema: duplicate field name '{name}'"
            )));
        }
        seen_names.push(name);
        match &spec.kind {
            FieldSpecKind::Text(t) => {
                if let TextBinding::Shared(s) = &t.binding
                    && !s.is_text()
                {
                    return Err(Error::Format(format!(
                        "schema: field '{name}' is declared text but bound to vector space '{}'",
                        s.name()
                    )));
                }
                text_fields += 1;
            }
            FieldSpecKind::Vector(v) => {
                if let Some(method) = v.shared_misuse {
                    return Err(shared_misuse_error("schema", name, method));
                }
                let space_name = match &v.binding {
                    VectorBinding::Inline(_) => auto_space_name(coll, name),
                    VectorBinding::Shared(s) => {
                        if !s.is_vector() {
                            return Err(Error::Format(format!(
                                "schema: field '{name}' is declared vector but bound to text space '{}'",
                                s.name()
                            )));
                        }
                        s.name().to_owned()
                    }
                };
                if seen_vec_spaces.contains(&space_name) {
                    return Err(Error::Format(format!(
                        "schema: vector fields must use distinct spaces; '{space_name}' is reused"
                    )));
                }
                seen_vec_spaces.push(space_name);
            }
        }
    }
    if text_fields > 1 {
        return Err(Error::Unsupported(
            "schema: at most one text field per collection in v1 (use the doc→chunk pattern for multiple text fields)".into(),
        ));
    }
    Ok(())
}

fn shared_misuse_error(context: &str, name: &str, method: &'static str) -> Error {
    Error::Format(format!(
        "{context}: field '{name}' binds a shared space — .{method}() is configured on the \
         space at define_space, not on the binding"
    ))
}

/// The canonical per-field signature used for re-declare comparison
/// (spec §4): `(name, kind, space)`, in declared order.
#[derive(Debug, PartialEq, Eq)]
struct FieldSig {
    name: String,
    kind: &'static str,
    space: String,
}

impl std::fmt::Display for FieldSig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} '{}' (space '{}')", self.kind, self.name, self.space)
    }
}

fn schema_signature(coll: &str, schema: &Schema) -> Vec<FieldSig> {
    schema
        .fields
        .iter()
        .map(|(name, spec)| match &spec.kind {
            FieldSpecKind::Text(t) => FieldSig {
                name: name.clone(),
                kind: "text",
                space: match &t.binding {
                    TextBinding::Inline { .. } => auto_space_name(coll, name),
                    TextBinding::Shared(s) => s.name().to_owned(),
                },
            },
            FieldSpecKind::Vector(v) => FieldSig {
                name: name.clone(),
                kind: "vector",
                space: match &v.binding {
                    VectorBinding::Inline(_) => auto_space_name(coll, name),
                    VectorBinding::Shared(s) => s.name().to_owned(),
                },
            },
        })
        .collect()
}

fn shape_signature(shape: &Shape) -> Vec<FieldSig> {
    shape
        .fields
        .iter()
        .map(|(name, field)| match field {
            Field::Text(s) => FieldSig {
                name: name.clone(),
                kind: "text",
                space: s.name().to_owned(),
            },
            Field::Vector(s) => FieldSig {
                name: name.clone(),
                kind: "vector",
                space: s.name().to_owned(),
            },
        })
        .collect()
}

/// Canonical ordered comparison of a re-declared schema against the shape
/// already bound to `coll` (spec §4): identical → ok (no-op), additive (the
/// existing fields are a prefix of the new declaration) → ok, anything else
/// → [`Error::SchemaMismatch`] naming the first divergent field.
pub(crate) fn check_redeclare(coll: &str, existing: &Shape, schema: &Schema) -> Result<()> {
    let old = shape_signature(existing);
    let new = schema_signature(coll, schema);
    if new.len() < old.len() {
        return Err(Error::SchemaMismatch {
            collection: coll.to_owned(),
            detail: format!(
                "re-declared with {} field(s), but {} were previously declared",
                new.len(),
                old.len()
            ),
        });
    }
    for (i, (o, n)) in old.iter().zip(new.iter()).enumerate() {
        if o != n {
            return Err(Error::SchemaMismatch {
                collection: coll.to_owned(),
                detail: format!("field {i} was declared as {o}, re-declared as {n}"),
            });
        }
    }
    Ok(())
}

/// The engine derives an in-memory sign-sketch (1 bit/dim) from the stored
/// phase codes, packing it into 64-bit words. That path requires the
/// dimension to be a positive multiple of 64; reject anything else here
/// rather than let the engine panic on a length mismatch deep in the
/// snapshot-publish path.
fn ensure_sketchable_dim(dim: u32) -> Result<()> {
    if dim == 0 || !dim.is_multiple_of(64) {
        return Err(Error::Unsupported(format!(
            "vector space dimension must be a positive multiple of 64 (got {dim})"
        )));
    }
    Ok(())
}

/// Register the codec a vector space declared, fitting it on `sample`.
/// One family-dispatch point shared by the eager (`Calibrate::now`) path and
/// the writer's deferred flush. `Prebuilt` is handled by the caller (it
/// needs no sample).
pub(crate) fn register_codec_for(
    valise: &mut ValiseFile,
    dim: u32,
    spec: &CodecSpec,
    sample: &[Vec<f32>],
) -> Result<CodecId> {
    match spec {
        CodecSpec::Qam {
            amp_bits,
            phase_bits,
        } => valise.register_codec_qam_from_sample_with_bits(
            dim as usize,
            *amp_bits,
            *phase_bits,
            sample,
        ),
        CodecSpec::Upq { cells, design } => valise.register_codec_upq_from_sample_with_options(
            dim as usize,
            *cells,
            design.to_engine(),
            sample,
        ),
        CodecSpec::Prebuilt(params) => valise.register_codec(params.clone()),
    }
}

pub(crate) fn register_embedding_space(
    valise: &mut ValiseFile,
    name: &str,
    dim: u32,
    metric: Metric,
    codec: CodecId,
) -> Result<EmbeddingSpaceId> {
    valise.register_embedding_space(EmbeddingSpaceSpec {
        provider: PROVIDER.to_owned(),
        model: name.to_owned(),
        dimension: dim,
        metric: metric.to_engine(),
        normalized: matches!(metric, Metric::Cosine),
        // The engine stores f32 in v1; dtype intentionally left off the db
        // surface.
        dtype: Dtype::F32,
        primary_codec_id: Some(codec),
        secondary_codec_id: None,
    })
}

/// The analyzer for a [`Lang`]. English mirrors the BEIR-comparable Lucene
/// pipeline; Raw is a pass-through whitespace tokenizer.
pub(crate) fn analyzer_desc(lang: Lang) -> AnalyzerDesc {
    match lang {
        Lang::English => AnalyzerDesc {
            analyzer_id: AnalyzerId(0),
            unicode_normalization: UnicodeNormalization::Nfkc,
            case_fold: true,
            accent_fold: false,
            tokenizer: Tokenizer::UnicodeWords,
            stemming: Stemming::Porter2English,
            stopword_set_ref: None,
            stopword_query_only: false,
            stopwords: StopwordsPolicy::EnglishLucene,
            english_possessive_strip: true,
            min_token_len: 1,
            max_token_len: 64,
            shingle_size: 0,
            ngram_min: None,
            ngram_max: None,
            punctuation_policy: PunctuationPolicy::Drop,
        },
        Lang::Raw => AnalyzerDesc {
            analyzer_id: AnalyzerId(0),
            unicode_normalization: UnicodeNormalization::None,
            case_fold: false,
            accent_fold: false,
            tokenizer: Tokenizer::Whitespace,
            stemming: Stemming::None,
            stopword_set_ref: None,
            stopword_query_only: true,
            stopwords: StopwordsPolicy::None,
            english_possessive_strip: false,
            min_token_len: 0,
            max_token_len: 0,
            shingle_size: 0,
            ngram_min: None,
            ngram_max: None,
            punctuation_policy: PunctuationPolicy::Keep,
        },
    }
}

/// Single-field `SearchText` schema — the v1 one-indexed-field-per-frame model.
pub(crate) fn field_schema_desc() -> FieldSchemaDesc {
    FieldSchemaDesc {
        field_schema_id: FieldSchemaId(0),
        fields: vec![FieldDesc {
            field_id: 1,
            name: "body".to_owned(),
            source: FieldSource::SearchText,
            store_positions: true,
            store_term_freq: true,
            store_set_membership: true,
            weight: 1.0,
        }],
    }
}

/// Default BM25 profile (k1=1.2, b=0.75). The profile-free
/// `QueryAlgorithm::Bm25 { k1, b }` read path (P2) overrides these per query;
/// this exists so `register_text_space` has a `default_profile_id` to bind.
pub(crate) fn bm25_profile_desc() -> RetrievalProfileDesc {
    RetrievalProfileDesc {
        profile_id: RetrievalProfileId(0),
        profile_type: RetrievalProfileType::Bm25,
        params: RetrievalProfileParams::Bm25 {
            k1: 1.2,
            b: 0.75,
            idf_variant: IdfVariant::RobertsonSparckJones,
        },
    }
}
