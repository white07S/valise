//! Persisted collection schemas — the hidden `~valise.schema` collection
//! (`docs/SIMPLE_API_SPEC.md` §4, zero format change).
//!
//! Every user collection gets ONE ordinary `Payload` frame in the reserved
//! engine collection [`SCHEMA_COLLECTION`]. The frame's uri is
//! `[0xF5] ++ collection_name` — `0xF5` is not a [`Key`](crate::db::Key)
//! tag (those are 1/2/3), so `Key::decode` returns `None` and the identity
//! index, search-hit resolution, and every other key-driven surface skip
//! schema docs without special-casing. The payload is versioned JSON:
//!
//! ```json
//! {"valise_schema":1,"fields":[{"name":..,"kind":"text"|"vector",
//!   "space":..,"spec":{..}|null}, ..]}
//! ```
//!
//! `space` is the bound space name (the private `~auto/{owner}/{field}` for
//! inline specs, or the shared name); `spec` is the inline configuration
//! (text: `{"lang"}`; vector: `{"dim","metric","codec":{tagged by family},
//! "calibrate":{tagged by mode}}`) and `null` for shared-space bindings.
//! Spec objects serialize through `serde_json::Value`, so their keys are
//! alphabetical; the full doc encoding is byte-deterministic and pinned by
//! the golden test below. The decoder MUST ignore unknown JSON keys, and a
//! doc whose `valise_schema` version is unknown (or whose spec fails to parse)
//! degrades to a shape-less collection instead of failing the open — new
//! files stay readable by old code.
//!
//! ## created_at watermark
//!
//! The doc frame's `created_at` is the watermark
//! `max(0, max created_at over ACTIVE frames)`. The engine's time index
//! requires `created_at` to be non-decreasing in frame-id order globally
//! (over active frames), so the doc — which always takes the newest frame
//! id — must carry a timestamp at least as large as every active frame's.
//! The watermark is exactly that minimum, so it provably adds no new ingest
//! constraint: any later frame already had to be `≥ max` to commit. The one
//! edge is the `0` floor: on a fresh (or all-pre-1970) file the doc lands at
//! `created_at = 0`, so **pre-1970 backfill after a schema declaration is
//! rejected** at commit with the engine's standard time-index ordering
//! error. Declare-then-backfill workloads must use timestamps `≥ 0`.
//!
//! ## Durability
//!
//! Stage-only, one deterministic rule: the schema frame is a pending engine
//! mutation that persists at the NEXT commit — exactly like
//! `create_collection` itself. A declare with no following commit leaves
//! nothing in the file (and nothing partially declared: the collection and
//! its doc ride the same commit).
//!
//! ## Rejected alternative (do not re-litigate)
//!
//! `CollectionDesc.metadata_ref` exists in the catalog and looks purpose-
//! built for this, but (a) it has **no engine write path** — nothing fills
//! a collection's `metadata_ref`, and (b) compaction's
//! `create_collection_at` reconstructs collections from `(name, created_at)`
//! only, silently dropping it. Do not migrate the schema doc onto
//! `metadata_ref` without solving both.
//!
//! ## Known caveat: shared spaces are not (re)constructable from docs
//!
//! A shared-binding field carries `"spec":null`, so a shared **vector**
//! space that was still deferred (never calibrated) at the last commit
//! cannot be reconstructed at open; its collections come back shape-less
//! until `define_space` + one re-declare. Calibrated shared spaces (and all
//! text spaces) live in the engine catalog and restore fine. Inline/auto
//! fields — the default path — always restore, including their deferred
//! codec + calibration choice.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::codec::CodecParams;
use crate::db::schema::{
    CalibrateMode, CodecSpec, DEFAULT_CALIBRATE_SAMPLE, FieldSpecKind, Schema, TextBinding,
    UpqDesign, VectorBinding,
};
use crate::db::space::{Lang, Metric, auto_space_name};
use crate::error::{Error, Result};
use crate::file::{PutFrame, ValiseFile};
use crate::format::catalog::CodecFamily;
use crate::format::catalog::FrameRole;
use crate::format::{CollectionId, FrameId};

/// Reserved engine collection holding one schema doc frame per user
/// collection. The `~` prefix is rejected for user names, so it can never
/// collide.
pub(crate) const SCHEMA_COLLECTION: &str = "~valise.schema";

/// Leading uri byte of a schema doc frame. Deliberately NOT a `Key` tag
/// (1/2/3): `Key::decode` returns `None`, so key-driven surfaces skip docs.
pub(crate) const SCHEMA_URI_TAG: u8 = 0xF5;

/// Version of the JSON doc layout. Docs with a different version are
/// ignored (shape-less degrade), never an open error.
pub(crate) const SCHEMA_DOC_VERSION: u32 = 1;

pub(crate) const KIND_TEXT: &str = "text";
pub(crate) const KIND_VECTOR: &str = "vector";

/// The persisted schema document for one collection.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct SchemaDoc {
    pub valise_schema: u32,
    pub fields: Vec<FieldDoc>,
}

/// One declared field. `spec` stays a raw [`Value`] so unknown future spec
/// shapes (new codec families, new kinds) never fail the decode; the typed
/// views below parse it on demand.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct FieldDoc {
    pub name: String,
    pub kind: String,
    pub space: String,
    pub spec: Option<Value>,
}

/// Typed view of a text field's inline spec.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct TextSpecDoc {
    pub lang: String,
}

/// Typed view of a vector field's inline spec.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct VectorSpecDoc {
    pub dim: u32,
    pub metric: String,
    pub codec: CodecDoc,
    pub calibrate: CalibrateDoc,
}

/// Family-tagged codec choice, mirroring [`CodecSpec`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "family", rename_all = "lowercase")]
pub(crate) enum CodecDoc {
    Qam {
        amp_bits: u8,
        phase_bits: u8,
    },
    Upq {
        cells: u32,
        design: String,
    },
    Prebuilt {
        params_family: String,
        params_hex: String,
    },
}

/// Mode-tagged calibration choice, mirroring [`CalibrateMode`]. `Now`
/// carries no sample (it is transient and already consumed at declaration).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "lowercase")]
pub(crate) enum CalibrateDoc {
    Auto { sample: u64 },
    Now,
}

impl SchemaDoc {
    /// Lower a declared [`Schema`] to its canonical persisted doc.
    /// `space_owner` is the name auto spaces are derived from — the
    /// collection itself, or the partitioned *base* for partition
    /// collections (which share the base's `~auto/{base}/{field}` spaces).
    pub(crate) fn from_schema(space_owner: &str, schema: &Schema) -> Result<SchemaDoc> {
        let mut fields = Vec::with_capacity(schema.fields.len());
        for (name, spec) in &schema.fields {
            let field = match &spec.kind {
                FieldSpecKind::Text(t) => match &t.binding {
                    TextBinding::Inline { lang } => FieldDoc {
                        name: name.clone(),
                        kind: KIND_TEXT.to_owned(),
                        space: auto_space_name(space_owner, name),
                        spec: Some(to_value(&TextSpecDoc {
                            lang: lang_str(*lang).to_owned(),
                        })?),
                    },
                    TextBinding::Shared(s) => FieldDoc {
                        name: name.clone(),
                        kind: KIND_TEXT.to_owned(),
                        space: s.name().to_owned(),
                        spec: None,
                    },
                },
                FieldSpecKind::Vector(v) => match &v.binding {
                    VectorBinding::Inline(i) => FieldDoc {
                        name: name.clone(),
                        kind: KIND_VECTOR.to_owned(),
                        space: auto_space_name(space_owner, name),
                        spec: Some(to_value(&VectorSpecDoc {
                            dim: i.dim,
                            metric: metric_str(i.metric).to_owned(),
                            codec: CodecDoc::from_spec(&i.codec.spec)?,
                            calibrate: CalibrateDoc::from_mode(&i.calibrate.mode),
                        })?),
                    },
                    VectorBinding::Shared(s) => FieldDoc {
                        name: name.clone(),
                        kind: KIND_VECTOR.to_owned(),
                        space: s.name().to_owned(),
                        spec: None,
                    },
                },
            };
            fields.push(field);
        }
        Ok(SchemaDoc {
            valise_schema: SCHEMA_DOC_VERSION,
            fields,
        })
    }

    /// Serialize to the canonical (byte-deterministic) JSON payload.
    pub(crate) fn encode(&self) -> Result<Vec<u8>> {
        serde_json::to_vec(self).map_err(|e| Error::Format(format!("schema doc encode: {e}")))
    }

    /// Decode a payload. `Ok(None)` for a doc of an unknown version
    /// (forward compat: degrade, don't fail the open); `Err` only for
    /// bytes that are not a JSON object with our envelope at all.
    pub(crate) fn decode(bytes: &[u8]) -> Result<Option<SchemaDoc>> {
        /// Envelope probe: read the version while ignoring everything else.
        #[derive(Deserialize)]
        struct Probe {
            #[serde(default)]
            valise_schema: u32,
        }
        let probe: Probe = serde_json::from_slice(bytes)
            .map_err(|e| Error::Format(format!("schema doc: invalid JSON: {e}")))?;
        if probe.valise_schema != SCHEMA_DOC_VERSION {
            return Ok(None);
        }
        let doc: SchemaDoc = serde_json::from_slice(bytes)
            .map_err(|e| Error::Format(format!("schema doc: malformed v1 doc: {e}")))?;
        Ok(Some(doc))
    }
}

impl FieldDoc {
    /// Parse the typed text spec, `None` for shared (`null`) or
    /// unparseable (future-shaped) specs.
    pub(crate) fn text_spec(&self) -> Option<TextSpecDoc> {
        let v = self.spec.as_ref()?;
        serde_json::from_value(v.clone()).ok()
    }

    /// Parse the typed vector spec, `None` for shared (`null`) or
    /// unparseable (future-shaped) specs.
    pub(crate) fn vector_spec(&self) -> Option<VectorSpecDoc> {
        let v = self.spec.as_ref()?;
        serde_json::from_value(v.clone()).ok()
    }
}

impl CodecDoc {
    pub(crate) fn from_spec(spec: &CodecSpec) -> Result<CodecDoc> {
        Ok(match spec {
            CodecSpec::Qam {
                amp_bits,
                phase_bits,
            } => CodecDoc::Qam {
                amp_bits: *amp_bits,
                phase_bits: *phase_bits,
            },
            CodecSpec::Upq { cells, design } => CodecDoc::Upq {
                cells: *cells,
                design: design_str(*design).to_owned(),
            },
            CodecSpec::Prebuilt(params) => CodecDoc::Prebuilt {
                params_family: family_str(params.family())?.to_owned(),
                params_hex: hex::encode(params.encode()?),
            },
        })
    }

    pub(crate) fn to_codec_spec(&self) -> Result<CodecSpec> {
        match self {
            CodecDoc::Qam {
                amp_bits,
                phase_bits,
            } => Ok(CodecSpec::Qam {
                amp_bits: *amp_bits,
                phase_bits: *phase_bits,
            }),
            CodecDoc::Upq { cells, design } => Ok(CodecSpec::Upq {
                cells: *cells,
                design: design_from_str(design)?,
            }),
            CodecDoc::Prebuilt {
                params_family,
                params_hex,
            } => {
                let family = family_from_str(params_family)?;
                let bytes = hex::decode(params_hex)
                    .map_err(|e| Error::Format(format!("schema doc: bad params_hex: {e}")))?;
                Ok(CodecSpec::Prebuilt(CodecParams::decode(family, &bytes)?))
            }
        }
    }
}

impl CalibrateDoc {
    pub(crate) fn from_mode(mode: &CalibrateMode) -> CalibrateDoc {
        match mode {
            CalibrateMode::Auto { sample } => CalibrateDoc::Auto {
                sample: *sample as u64,
            },
            CalibrateMode::Now(_) => CalibrateDoc::Now,
        }
    }

    /// The deferred sample target this calibration choice resolves to,
    /// exactly as a fresh declare would record it. `Now` is normally eager
    /// (its space lives in the engine catalog), so the `Now` arm only fires
    /// in degenerate states; it falls back to the auto default.
    pub(crate) fn sample_target(&self) -> usize {
        match self {
            CalibrateDoc::Auto { sample } => (*sample as usize).max(1),
            CalibrateDoc::Now => DEFAULT_CALIBRATE_SAMPLE,
        }
    }
}

// ---- string lowerings (part of the persisted doc format — append-only) ----

pub(crate) fn lang_str(lang: Lang) -> &'static str {
    match lang {
        Lang::English => "english",
        Lang::Raw => "raw",
    }
}

pub(crate) fn metric_str(m: Metric) -> &'static str {
    match m {
        Metric::Cosine => "cosine",
        Metric::Dot => "dot",
        Metric::L2 => "l2",
    }
}

pub(crate) fn metric_from_str(s: &str) -> Result<Metric> {
    match s {
        "cosine" => Ok(Metric::Cosine),
        "dot" => Ok(Metric::Dot),
        "l2" => Ok(Metric::L2),
        other => Err(Error::Format(format!(
            "schema doc: unknown metric '{other}'"
        ))),
    }
}

fn design_str(d: UpqDesign) -> &'static str {
    match d {
        UpqDesign::Empirical => "empirical",
        UpqDesign::Rayleigh => "rayleigh",
    }
}

fn design_from_str(s: &str) -> Result<UpqDesign> {
    match s {
        "empirical" => Ok(UpqDesign::Empirical),
        "rayleigh" => Ok(UpqDesign::Rayleigh),
        other => Err(Error::Format(format!(
            "schema doc: unknown UPQ design '{other}'"
        ))),
    }
}

fn family_str(f: CodecFamily) -> Result<&'static str> {
    match f {
        CodecFamily::QamLloydMax => Ok("qam_lloyd_max"),
        CodecFamily::Upq => Ok("upq"),
        CodecFamily::_LegacyInt4PcaZero => Err(Error::Unsupported(
            "schema doc: legacy codec family cannot be persisted".into(),
        )),
    }
}

fn family_from_str(s: &str) -> Result<CodecFamily> {
    match s {
        "qam_lloyd_max" => Ok(CodecFamily::QamLloydMax),
        "upq" => Ok(CodecFamily::Upq),
        other => Err(Error::Format(format!(
            "schema doc: unknown codec family '{other}'"
        ))),
    }
}

fn to_value<T: Serialize>(t: &T) -> Result<Value> {
    serde_json::to_value(t).map_err(|e| Error::Format(format!("schema doc encode: {e}")))
}

// ---- canonical comparison (redeclare semantics, spec §4) ----

/// Outcome of comparing a declaration against the persisted doc.
#[derive(Debug)]
pub(crate) enum DocSync {
    /// Persisted doc already matches — nothing to write.
    UpToDate,
    /// Write the new doc, tombstoning the prior doc frame when present
    /// (additive redeclare, missing doc, or first declare).
    Write { replace: Option<FrameId> },
}

/// Canonical ordered field comparison. Equal docs compare via the *parsed*
/// typed specs when both sides parse (so unknown JSON keys written by a
/// newer version are ignored, per the forward-compat rule), falling back to
/// raw `Value` equality otherwise.
fn field_same(a: &FieldDoc, b: &FieldDoc) -> bool {
    if a.name != b.name || a.kind != b.kind || a.space != b.space {
        return false;
    }
    match (&a.spec, &b.spec) {
        (None, None) => true,
        (Some(_), Some(_)) => match a.kind.as_str() {
            KIND_TEXT => match (a.text_spec(), b.text_spec()) {
                (Some(x), Some(y)) => x == y,
                _ => a.spec == b.spec,
            },
            KIND_VECTOR => match (a.vector_spec(), b.vector_spec()) {
                (Some(x), Some(y)) => x == y,
                _ => a.spec == b.spec,
            },
            _ => a.spec == b.spec,
        },
        _ => false,
    }
}

fn spec_json(f: &FieldDoc) -> String {
    f.spec
        .as_ref()
        .map(Value::to_string)
        .unwrap_or_else(|| "null".to_owned())
}

fn field_diff_detail(i: usize, old: &FieldDoc, new: &FieldDoc) -> String {
    if old.name != new.name {
        format!(
            "field {i} was declared as '{}', re-declared as '{}'",
            old.name, new.name
        )
    } else if old.kind != new.kind {
        format!(
            "field {i} ('{}') was declared {}, re-declared as {}",
            old.name, old.kind, new.kind
        )
    } else if old.space != new.space {
        format!(
            "field {i} ('{}') was bound to space '{}', re-declared on '{}'",
            old.name, old.space, new.space
        )
    } else {
        format!(
            "field {i} ('{}') changed its {} spec: {} → {}",
            old.name,
            old.kind,
            spec_json(old),
            spec_json(new)
        )
    }
}

/// Identical → `Ok(true)`; purely additive (persisted fields are an
/// entry-wise prefix of the declaration) → `Ok(false)`; anything else →
/// [`Error::SchemaMismatch`] naming the first divergent field.
pub(crate) fn check_compatible(coll: &str, old: &SchemaDoc, new: &SchemaDoc) -> Result<bool> {
    if new.fields.len() < old.fields.len() {
        return Err(Error::SchemaMismatch {
            collection: coll.to_owned(),
            detail: format!(
                "re-declared with {} field(s), but {} are persisted",
                new.fields.len(),
                old.fields.len()
            ),
        });
    }
    for (i, (o, n)) in old.fields.iter().zip(new.fields.iter()).enumerate() {
        if !field_same(o, n) {
            return Err(Error::SchemaMismatch {
                collection: coll.to_owned(),
                detail: field_diff_detail(i, o, n),
            });
        }
    }
    Ok(new.fields.len() == old.fields.len())
}

// ---- engine IO ----

/// The uri bytes of `coll`'s schema doc frame.
pub(crate) fn doc_uri(coll: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(1 + coll.len());
    out.push(SCHEMA_URI_TAG);
    out.extend_from_slice(coll.as_bytes());
    out
}

/// Inverse of [`doc_uri`]; `None` for uris this module didn't write.
pub(crate) fn decode_doc_uri(uri: &[u8]) -> Option<String> {
    let (&tag, rest) = uri.split_first()?;
    if tag != SCHEMA_URI_TAG {
        return None;
    }
    String::from_utf8(rest.to_vec()).ok()
}

/// The reserved collection's id, if it exists (live, not soft-deleted).
pub(crate) fn schema_collection_id(valise: &ValiseFile) -> Option<CollectionId> {
    valise
        .collections()
        .iter()
        .find(|c| c.name == SCHEMA_COLLECTION && c.flags & 1 == 0)
        .map(|c| c.collection_id)
}

/// Create-or-get the reserved collection. Goes straight to the engine —
/// the `~` validation applies to USER names only, and the reserved
/// collection is deliberately kept out of the layer's registry.
fn ensure_schema_collection(valise: &mut ValiseFile) -> Result<CollectionId> {
    if let Some(cid) = schema_collection_id(valise) {
        return Ok(cid);
    }
    valise.create_collection(SCHEMA_COLLECTION)
}

/// `max(0, max created_at over active frames)` — see the module doc.
pub(crate) fn watermark(valise: &ValiseFile) -> i64 {
    valise
        .active_frames()
        .map(|f| f.created_at)
        .max()
        .unwrap_or(0)
        .max(0)
}

/// Find `coll`'s active schema doc frame and decode it. `Ok(None)` when
/// there is no doc (or it is of an unknown version — forward compat).
pub(crate) fn find_doc(valise: &ValiseFile, coll: &str) -> Result<Option<(FrameId, SchemaDoc)>> {
    let Some(cid) = schema_collection_id(valise) else {
        return Ok(None);
    };
    let want = doc_uri(coll);
    let frame_ids: Vec<FrameId> = valise
        .active_frames()
        .filter(|f| f.collection_id == cid)
        .map(|f| f.frame_id)
        .collect();
    for fid in frame_ids {
        if valise.read_frame_uri(fid)?.as_deref() == Some(want.as_slice()) {
            let payload = valise.read_payload(fid)?;
            return Ok(SchemaDoc::decode(&payload)?.map(|doc| (fid, doc)));
        }
    }
    Ok(None)
}

/// Decode every active schema doc as `(collection_name, doc)`. Docs of an
/// unknown version are skipped (their collections degrade to shape-less).
pub(crate) fn load_all(valise: &ValiseFile) -> Result<Vec<(String, SchemaDoc)>> {
    let Some(cid) = schema_collection_id(valise) else {
        return Ok(Vec::new());
    };
    let frame_ids: Vec<FrameId> = valise
        .active_frames()
        .filter(|f| f.collection_id == cid)
        .map(|f| f.frame_id)
        .collect();
    let mut out = Vec::with_capacity(frame_ids.len());
    for fid in frame_ids {
        let Some(uri) = valise.read_frame_uri(fid)? else {
            continue;
        };
        let Some(name) = decode_doc_uri(&uri) else {
            continue;
        };
        let payload = valise.read_payload(fid)?;
        if let Some(doc) = SchemaDoc::decode(&payload)? {
            out.push((name, doc));
        }
    }
    Ok(out)
}

/// Compare `new_doc` against the persisted doc and plan the write.
/// Read-only — safe to run before any binding side effects, so a divergent
/// declaration ([`Error::SchemaMismatch`]) leaves the store untouched.
pub(crate) fn plan_sync(valise: &ValiseFile, coll: &str, new_doc: &SchemaDoc) -> Result<DocSync> {
    match find_doc(valise, coll)? {
        None => Ok(DocSync::Write { replace: None }),
        Some((fid, old)) => {
            if check_compatible(coll, &old, new_doc)? {
                Ok(DocSync::UpToDate)
            } else {
                Ok(DocSync::Write { replace: Some(fid) })
            }
        }
    }
}

/// Apply a [`plan_sync`] decision: tombstone the replaced doc frame (if
/// any) and stage the new one at the watermark timestamp. Stage-only — the
/// frame persists at the next commit (see the module doc).
pub(crate) fn apply_sync(
    valise: &mut ValiseFile,
    coll: &str,
    new_doc: &SchemaDoc,
    plan: &DocSync,
) -> Result<()> {
    let DocSync::Write { replace } = plan else {
        return Ok(());
    };
    let cid = ensure_schema_collection(valise)?;
    if let Some(old_fid) = replace {
        valise.delete_frame(*old_fid)?;
    }
    let payload = new_doc.encode()?;
    let uri = doc_uri(coll);
    // Computed after the delete: if the replaced doc held the unique max it
    // no longer pins the watermark, and the new doc still sorts after every
    // remaining active frame.
    let created_at = watermark(valise);
    valise.put_frame(PutFrame {
        collection_id: cid,
        role: FrameRole::Document,
        payload: &payload,
        created_at: Some(created_at),
        updated_at: Some(created_at),
        parent_frame_id: None,
        chunk_index: None,
        chunk_count: None,
        uri: Some(&uri),
    })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::schema::{Calibrate, Codec, Vector};
    use crate::format::qam_lloyd_max_params::QamLloydMaxParams;

    fn two_field_schema() -> Schema {
        Schema::new()
            .text("body")
            .vector("embedding", Vector::dim(768))
    }

    /// DB-layer golden: the exact serialized bytes of a representative
    /// two-field schema doc. A diff here is a persisted-format change —
    /// bump [`SCHEMA_DOC_VERSION`] or keep the encoding stable.
    #[test]
    fn golden_two_field_doc_bytes() {
        let doc = SchemaDoc::from_schema("notes", &two_field_schema()).unwrap();
        let bytes = doc.encode().unwrap();
        let expected = concat!(
            r#"{"valise_schema":1,"fields":["#,
            r#"{"name":"body","kind":"text","space":"~auto/notes/body","spec":{"lang":"english"}},"#,
            r#"{"name":"embedding","kind":"vector","space":"~auto/notes/embedding","spec":{"#,
            r#""calibrate":{"mode":"auto","sample":50000},"#,
            r#""codec":{"amp_bits":5,"family":"qam","phase_bits":6},"#,
            r#""dim":768,"metric":"cosine"}}"#,
            r#"]}"#,
        );
        assert_eq!(
            std::str::from_utf8(&bytes).unwrap(),
            expected,
            "schema doc encoding drifted"
        );
    }

    #[test]
    fn doc_roundtrip_all_spec_kinds() {
        let prebuilt = CodecParams::QamLloydMax(QamLloydMaxParams {
            dimension: 64,
            num_pairs: 32,
            block_size: 64,
            rotation_seed: 0x0123_4567_89AB_CDEF,
            amp_bits: 5,
            phase_bits: 6,
            renormalize_at_decode: true,
            sigma_per_pair: vec![0.025_f32; 32],
            calibration_id: [0xAB; 32],
        });
        let schema = Schema::new()
            .text("body")
            .vector(
                "q",
                Vector::dim(128)
                    .metric(Metric::Dot)
                    .codec(Codec::qam_bits(8, 8))
                    .calibrate(Calibrate::auto_sample(9)),
            )
            .vector(
                "u",
                Vector::dim(64)
                    .metric(Metric::L2)
                    .codec(Codec::upq_with(512, UpqDesign::Rayleigh))
                    .calibrate(Calibrate::now(vec![vec![0.0; 64]])),
            )
            .vector(
                "p",
                Vector::dim(64).codec(Codec::from_params(prebuilt.clone())),
            );
        let doc = SchemaDoc::from_schema("kb", &schema).unwrap();
        let bytes = doc.encode().unwrap();
        let back = SchemaDoc::decode(&bytes).unwrap().expect("v1 doc");
        assert_eq!(back.fields.len(), 4);
        for (a, b) in doc.fields.iter().zip(back.fields.iter()) {
            assert!(field_same(a, b), "field {} drifted", a.name);
        }
        // Typed views parse back to the declared choices.
        let q = back.fields[1].vector_spec().unwrap();
        assert_eq!(q.dim, 128);
        assert_eq!(metric_from_str(&q.metric).unwrap(), Metric::Dot);
        assert_eq!(
            q.codec.to_codec_spec().unwrap(),
            CodecSpec::Qam {
                amp_bits: 8,
                phase_bits: 8
            }
        );
        assert_eq!(q.calibrate.sample_target(), 9);
        let u = back.fields[2].vector_spec().unwrap();
        assert_eq!(
            u.codec.to_codec_spec().unwrap(),
            CodecSpec::Upq {
                cells: 512,
                design: UpqDesign::Rayleigh
            }
        );
        assert_eq!(u.calibrate, CalibrateDoc::Now);
        let p = back.fields[3].vector_spec().unwrap();
        assert_eq!(
            p.codec.to_codec_spec().unwrap(),
            CodecSpec::Prebuilt(prebuilt)
        );
    }

    #[test]
    fn decoder_ignores_unknown_keys() {
        let json = concat!(
            r#"{"valise_schema":1,"from_the_future":[1,2,3],"fields":["#,
            r#"{"name":"body","kind":"text","space":"~auto/n/body","mystery":true,"#,
            r#""spec":{"lang":"english","novel_knob":7}},"#,
            r#"{"name":"v","kind":"vector","space":"~auto/n/v","spec":{"#,
            r#""calibrate":{"mode":"auto","sample":10,"extra":1},"#,
            r#""codec":{"amp_bits":5,"family":"qam","phase_bits":6,"new_param":0.5},"#,
            r#""dim":64,"metric":"cosine","weighting":"none"}}]}"#,
        );
        let doc = SchemaDoc::decode(json.as_bytes()).unwrap().expect("v1 doc");
        assert_eq!(doc.fields[0].text_spec().unwrap().lang, "english");
        let v = doc.fields[1].vector_spec().unwrap();
        assert_eq!(v.dim, 64);
        assert_eq!(v.calibrate.sample_target(), 10);
        // Unknown keys are also invisible to the canonical comparison.
        let clean = SchemaDoc::from_schema(
            "n",
            &Schema::new()
                .text("body")
                .vector("v", Vector::dim(64).calibrate(Calibrate::auto_sample(10))),
        )
        .unwrap();
        assert!(check_compatible("n", &doc, &clean).unwrap());
    }

    #[test]
    fn unknown_version_degrades_to_none() {
        let json = r#"{"valise_schema":2,"fields":"something else entirely"}"#;
        assert!(SchemaDoc::decode(json.as_bytes()).unwrap().is_none());
        assert!(SchemaDoc::decode(b"not json").is_err());
    }

    #[test]
    fn compatibility_identical_additive_divergent() {
        let base = SchemaDoc::from_schema(
            "kb",
            &Schema::new().text("body").vector("v", Vector::dim(64)),
        )
        .unwrap();
        // Identical.
        assert!(check_compatible("kb", &base, &base.clone()).unwrap());
        // Additive: old is a prefix.
        let grown = SchemaDoc::from_schema(
            "kb",
            &Schema::new()
                .text("body")
                .vector("v", Vector::dim(64))
                .vector("w", Vector::dim(128)),
        )
        .unwrap();
        assert!(!check_compatible("kb", &base, &grown).unwrap());
        // Dropping a field.
        let dropped = SchemaDoc::from_schema("kb", &Schema::new().text("body")).unwrap();
        let e = check_compatible("kb", &base, &dropped).unwrap_err();
        assert!(matches!(e, Error::SchemaMismatch { .. }), "{e}");
        // Divergent spec on an existing field (codec change) names the field.
        let recodec = SchemaDoc::from_schema(
            "kb",
            &Schema::new()
                .text("body")
                .vector("v", Vector::dim(64).codec(Codec::upq())),
        )
        .unwrap();
        let e = check_compatible("kb", &base, &recodec).unwrap_err();
        let Error::SchemaMismatch { detail, .. } = &e else {
            panic!("expected SchemaMismatch, got {e}");
        };
        assert!(detail.contains("'v'"), "{detail}");
        assert!(detail.contains("spec"), "{detail}");
    }

    #[test]
    fn doc_uri_roundtrip_and_key_disjointness() {
        let uri = doc_uri("notes");
        assert_eq!(uri[0], SCHEMA_URI_TAG);
        assert_eq!(decode_doc_uri(&uri).as_deref(), Some("notes"));
        // 0xF5 must never be a Key tag, or identity rebuild would ingest
        // schema docs as records.
        assert!(crate::db::record::Key::decode(&uri).is_none());
        assert!(decode_doc_uri(b"\x01notes").is_none());
    }
}
