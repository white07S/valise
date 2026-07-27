// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Declarative collection schemas: [`Schema`] and its field specs
//! ([`Text`] / [`Vector`]), plus the per-field codec ([`Codec`]) and
//! calibration ([`Calibrate`]) choices.
//!
//! This is the surface described in `docs/SIMPLE_API_SPEC.md` §2. A
//! `Schema` is purely declarative — it is resolved against the store's
//! [`SchemaRegistry`](crate::db::space::SchemaRegistry) when bound via
//! [`crate::db::Store::collection`], where inline field specs auto-define
//! private `~auto/{collection}/{field}` spaces and shared bindings
//! ([`Text::space`] / [`Vector::space`]) reuse a space from
//! [`crate::db::Store::define_space`].
//!
//! [`Codec`] and [`Calibrate`] are deliberately **opaque, constructor-only**
//! types: adding a parameter to a codec family, or a new family, is then a
//! non-breaking change (new constructors instead of new public enum
//! variants).

use crate::codec::CodecParams;
use crate::db::space::{Lang, Metric, Space};

/// Default deferred-calibration sample size (vectors), per spec §2.
pub(crate) const DEFAULT_CALIBRATE_SAMPLE: usize = 50_000;

/// The set of named fields a collection's records may fill. Built fluently
/// and bound (with validation) by [`crate::db::Store::collection`].
///
/// ```no_run
/// use valise::prelude::*;
/// # fn demo(store: &Store) -> Result<()> {
/// store.collection("notes", Schema::new()
///     .text("body")
///     .vector("embedding", Vector::dim(768)))?;
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, Default)]
pub struct Schema {
    pub(crate) fields: Vec<(String, FieldSpec)>,
}

impl Schema {
    #[must_use]
    pub fn new() -> Schema {
        Schema { fields: Vec::new() }
    }

    /// Add a BM25-indexed text field with the default analyzer
    /// ([`Text::english`]). At most one text field per collection in v1.
    #[must_use]
    pub fn text(self, name: &str) -> Schema {
        self.text_with(name, Text::english())
    }

    /// Add a text field with an explicit [`Text`] spec.
    #[must_use]
    pub fn text_with(mut self, name: &str, spec: Text) -> Schema {
        self.fields.push((name.to_owned(), FieldSpec::from(spec)));
        self
    }

    /// Add a vector field. The dimension is mandatory, so the spec is always
    /// explicit: `Vector::dim(768)` or `Vector::space(&shared)`.
    #[must_use]
    pub fn vector(mut self, name: &str, spec: Vector) -> Schema {
        self.fields.push((name.to_owned(), FieldSpec::from(spec)));
        self
    }
}

/// A field specification — either text or vector. Obtained via
/// `impl Into<FieldSpec>` from [`Text`] or [`Vector`]; also what
/// [`crate::db::Store::define_space`] accepts.
#[derive(Clone, Debug)]
pub struct FieldSpec {
    pub(crate) kind: FieldSpecKind,
}

#[derive(Clone, Debug)]
pub(crate) enum FieldSpecKind {
    Text(Text),
    Vector(Vector),
}

impl From<Text> for FieldSpec {
    fn from(spec: Text) -> Self {
        FieldSpec {
            kind: FieldSpecKind::Text(spec),
        }
    }
}

impl From<Vector> for FieldSpec {
    fn from(spec: Vector) -> Self {
        FieldSpec {
            kind: FieldSpecKind::Vector(spec),
        }
    }
}

/// Text field spec: an inline analyzer choice (auto-defines a private text
/// space) or a binding to a shared space from
/// [`crate::db::Store::define_space`].
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct Text {
    pub(crate) binding: TextBinding,
}

#[derive(Clone, Debug)]
pub(crate) enum TextBinding {
    Inline { lang: Lang },
    Shared(Space),
}

impl Text {
    /// The English analyzer pipeline (NFKC, case-fold, Porter2 stemming,
    /// Lucene stopwords) — the default for [`Schema::text`].
    #[must_use]
    pub fn english() -> Text {
        Text {
            binding: TextBinding::Inline {
                lang: Lang::English,
            },
        }
    }

    /// A pass-through whitespace tokenizer (no normalization/stemming).
    #[must_use]
    pub fn raw() -> Text {
        Text {
            binding: TextBinding::Inline { lang: Lang::Raw },
        }
    }

    /// Bind to a shared text space previously created with
    /// [`crate::db::Store::define_space`].
    #[must_use]
    pub fn space(s: &Space) -> Text {
        Text {
            binding: TextBinding::Shared(s.clone()),
        }
    }
}

/// Vector field spec: an inline `(dim, metric, codec, calibrate)` choice
/// (auto-defines a private vector space) or a binding to a shared space.
///
/// On a shared binding ([`Vector::space`]) the configuration methods are
/// rejected at declaration time — metric/codec/calibration belong to the
/// space and are configured at [`crate::db::Store::define_space`].
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct Vector {
    pub(crate) binding: VectorBinding,
    /// Set when a configuration method was called on a shared-space binding;
    /// surfaced as an error when the schema is declared.
    pub(crate) shared_misuse: Option<&'static str>,
}

#[derive(Clone, Debug)]
pub(crate) enum VectorBinding {
    Inline(VectorInline),
    Shared(Space),
}

/// The resolved inline vector-field configuration.
#[derive(Clone, Debug)]
pub(crate) struct VectorInline {
    pub dim: u32,
    pub metric: Metric,
    pub codec: Codec,
    pub calibrate: Calibrate,
}

impl Vector {
    /// A `dim`-dimensional cosine field with the default codec
    /// ([`Codec::qam`]) and auto-calibration ([`Calibrate::auto`]). `dim`
    /// must be a positive multiple of 64 (checked at declaration).
    #[must_use]
    pub fn dim(dim: u32) -> Vector {
        Vector {
            binding: VectorBinding::Inline(VectorInline {
                dim,
                metric: Metric::Cosine,
                codec: Codec::default(),
                calibrate: Calibrate::default(),
            }),
            shared_misuse: None,
        }
    }

    /// Bind to a shared vector space previously created with
    /// [`crate::db::Store::define_space`]. Configuration methods are not
    /// allowed on a shared binding.
    #[must_use]
    pub fn space(s: &Space) -> Vector {
        Vector {
            binding: VectorBinding::Shared(s.clone()),
            shared_misuse: None,
        }
    }

    /// Set the distance metric (default [`Metric::Cosine`]).
    #[must_use]
    pub fn metric(mut self, m: Metric) -> Vector {
        match &mut self.binding {
            VectorBinding::Inline(i) => i.metric = m,
            VectorBinding::Shared(_) => {
                self.shared_misuse.get_or_insert("metric");
            }
        }
        self
    }

    /// Set the quantization codec (default [`Codec::qam`]).
    #[must_use]
    pub fn codec(mut self, c: Codec) -> Vector {
        match &mut self.binding {
            VectorBinding::Inline(i) => i.codec = c,
            VectorBinding::Shared(_) => {
                self.shared_misuse.get_or_insert("codec");
            }
        }
        self
    }

    /// Set the calibration mode (default [`Calibrate::auto`]).
    #[must_use]
    pub fn calibrate(mut self, c: Calibrate) -> Vector {
        match &mut self.binding {
            VectorBinding::Inline(i) => i.calibrate = c,
            VectorBinding::Shared(_) => {
                self.shared_misuse.get_or_insert("calibrate");
            }
        }
        self
    }
}

/// Per-field vector codec choice. Opaque and constructor-only: families
/// gain parameters (and new families appear) as new constructors, never as
/// public enum variants — so codec evolution is non-breaking.
#[derive(Clone, Debug, PartialEq)]
pub struct Codec {
    pub(crate) spec: CodecSpec,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum CodecSpec {
    Qam { amp_bits: u8, phase_bits: u8 },
    Upq { cells: u32, design: UpqDesign },
    Prebuilt(CodecParams),
}

impl Codec {
    /// QAM Lloyd-Max at the production default `(5, 6)` bits — the
    /// [`Default`] codec.
    #[must_use]
    pub fn qam() -> Codec {
        Codec {
            spec: CodecSpec::Qam {
                amp_bits: crate::codec::qam_lloyd_max::DEFAULT_AMP_BITS,
                phase_bits: crate::codec::qam_lloyd_max::DEFAULT_PHASE_BITS,
            },
        }
    }

    /// QAM Lloyd-Max at an explicit `(amp_bits, phase_bits)` budget. Higher
    /// bits trade storage for recall.
    #[must_use]
    pub fn qam_bits(amp_bits: u8, phase_bits: u8) -> Codec {
        Codec {
            spec: CodecSpec::Qam {
                amp_bits,
                phase_bits,
            },
        }
    }

    /// UPQ (unrestricted polar quantization) at the default operating point
    /// (2048 cells, empirical ring design).
    #[must_use]
    pub fn upq() -> Codec {
        Codec::upq_with(crate::codec::upq::DEFAULT_UPQ_CELLS, UpqDesign::Empirical)
    }

    /// UPQ at an explicit cell budget (empirical ring design).
    #[must_use]
    pub fn upq_cells(cells: u32) -> Codec {
        Codec::upq_with(cells, UpqDesign::Empirical)
    }

    /// UPQ at an explicit cell budget and ring design.
    #[must_use]
    pub fn upq_with(cells: u32, design: UpqDesign) -> Codec {
        Codec {
            spec: CodecSpec::Upq { cells, design },
        }
    }

    /// Engine escape hatch: a prebuilt, family-tagged parameter set (e.g.
    /// from [`crate::ValiseFile::codec_params`]). Registered eagerly at
    /// declaration — no calibration sample is needed or used.
    #[must_use]
    pub fn from_params(params: CodecParams) -> Codec {
        Codec {
            spec: CodecSpec::Prebuilt(params),
        }
    }
}

impl Default for Codec {
    fn default() -> Self {
        Codec::qam()
    }
}

/// UPQ ring-design source. See `docs/FORMAT.md` §14 / `upq_params.rs`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum UpqDesign {
    /// Rings fitted on the calibration sample's empirical distribution
    /// (data-adaptive; the default).
    Empirical,
    /// Rings designed on the analytic unit-Rayleigh density.
    Rayleigh,
}

impl UpqDesign {
    pub(crate) fn to_engine(&self) -> crate::format::upq_params::UpqDesignSource {
        match self {
            UpqDesign::Empirical => crate::format::upq_params::UpqDesignSource::Empirical,
            UpqDesign::Rayleigh => crate::format::upq_params::UpqDesignSource::Rayleigh,
        }
    }
}

/// When and from what data a vector field's codec is calibrated. Opaque and
/// constructor-only for the same evolution reasons as [`Codec`].
#[derive(Clone, Debug)]
pub struct Calibrate {
    pub(crate) mode: CalibrateMode,
}

#[derive(Clone, Debug)]
pub(crate) enum CalibrateMode {
    Auto { sample: usize },
    Now(Vec<Vec<f32>>),
}

impl Calibrate {
    /// Fit the codec at the first commit that contains vectors for the
    /// field, from up to 50 000 staged vectors (the [`Default`]).
    #[must_use]
    pub fn auto() -> Calibrate {
        Calibrate::auto_sample(DEFAULT_CALIBRATE_SAMPLE)
    }

    /// [`Calibrate::auto`] with an explicit sample-size cap.
    #[must_use]
    pub fn auto_sample(sample: usize) -> Calibrate {
        Calibrate {
            mode: CalibrateMode::Auto { sample },
        }
    }

    /// Calibrate eagerly at declaration from a caller-supplied
    /// representative sample.
    #[must_use]
    pub fn now(sample: Vec<Vec<f32>>) -> Calibrate {
        Calibrate {
            mode: CalibrateMode::Now(sample),
        }
    }
}

impl Default for Calibrate {
    fn default() -> Self {
        Calibrate::auto()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codec_constructors_and_default() {
        // The default is the production QAM (5, 6).
        assert_eq!(Codec::default(), Codec::qam());
        assert_eq!(
            Codec::qam().spec,
            CodecSpec::Qam {
                amp_bits: 5,
                phase_bits: 6
            }
        );
        assert_eq!(
            Codec::qam_bits(8, 8).spec,
            CodecSpec::Qam {
                amp_bits: 8,
                phase_bits: 8
            }
        );
        // The UPQ default is 2048 cells, empirical design.
        assert_eq!(Codec::upq(), Codec::upq_with(2048, UpqDesign::Empirical));
        assert_eq!(
            Codec::upq_cells(512).spec,
            CodecSpec::Upq {
                cells: 512,
                design: UpqDesign::Empirical
            }
        );
        assert_eq!(
            Codec::upq_with(1024, UpqDesign::Rayleigh).spec,
            CodecSpec::Upq {
                cells: 1024,
                design: UpqDesign::Rayleigh
            }
        );
    }

    #[test]
    fn calibrate_default_is_auto_50k() {
        let c = Calibrate::default();
        assert!(matches!(c.mode, CalibrateMode::Auto { sample: 50_000 }));
        let c = Calibrate::auto_sample(8);
        assert!(matches!(c.mode, CalibrateMode::Auto { sample: 8 }));
        let c = Calibrate::now(vec![vec![0.0; 4]]);
        assert!(matches!(c.mode, CalibrateMode::Now(ref s) if s.len() == 1));
    }

    #[test]
    fn vector_records_shared_binding_misuse() {
        let s = Space::vector("emb", 64);
        // First misuse wins (named in the eventual declaration error).
        let v = Vector::space(&s)
            .codec(Codec::upq())
            .calibrate(Calibrate::auto());
        assert_eq!(v.shared_misuse, Some("codec"));
        let v = Vector::space(&s).metric(Metric::Dot);
        assert_eq!(v.shared_misuse, Some("metric"));
        // Inline bindings accept configuration.
        let v = Vector::dim(64).metric(Metric::Dot).codec(Codec::upq());
        assert!(v.shared_misuse.is_none());
    }

    #[test]
    fn schema_builder_collects_fields_in_order() {
        let s = Space::vector("emb", 64);
        let schema = Schema::new()
            .text("body")
            .vector("dense", Vector::dim(64))
            .vector("shared", Vector::space(&s));
        let names: Vec<&str> = schema.fields.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, ["body", "dense", "shared"]);
        assert!(matches!(
            schema.fields[0].1.kind,
            FieldSpecKind::Text(Text {
                binding: TextBinding::Inline {
                    lang: Lang::English
                },
                ..
            })
        ));
    }
}
