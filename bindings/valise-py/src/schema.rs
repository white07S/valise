//! Schema lowering: the native [`PySchema`] builder and [`PySpace`] handle,
//! plus the shared helpers that turn the facade's flat keyword arguments
//! into core `Text` / `Vector` / `Codec` / `Calibrate` values.
//!
//! Defaults live in Rust: every parameter the Python layer leaves as `None`
//! routes to the corresponding core constructor/default (`Text::english()`,
//! `Codec::qam()`, `Calibrate::auto()`, the implicit `Metric::Cosine` of
//! `Vector::dim`), so the facade never duplicates a default value.

use numpy::PyReadonlyArray2;
use pyo3::prelude::*;

use valise::db::{Calibrate, Codec, Lang, Schema, Text, Vector};

use crate::convert::{parse_design, parse_lang, parse_metric};
use crate::errors::{PyResultX, invalid};

/// Opaque handle to a defined space (returned by `Store::define_space` /
/// `Store::space`, bound via `Text(space=...)` / `Vector(space=...)`).
#[pyclass(name = "Space", skip_from_py_object)]
#[derive(Clone)]
pub struct PySpace {
    pub(crate) inner: valise::db::Space,
}

#[pymethods]
impl PySpace {
    #[getter]
    fn name(&self) -> &str {
        self.inner.name()
    }

    #[getter]
    fn is_text(&self) -> bool {
        self.inner.is_text()
    }

    #[getter]
    fn is_vector(&self) -> bool {
        self.inner.is_vector()
    }

    fn __repr__(&self) -> String {
        let kind = if self.inner.is_text() {
            "text"
        } else {
            "vector"
        };
        format!("Space(name='{}', kind='{kind}')", self.inner.name())
    }
}

/// Build a core [`Text`] spec from flat facade args.
pub(crate) fn build_text(lang: Option<&str>, space: Option<&PySpace>) -> PyResultX<Text> {
    match (lang, space) {
        (Some(_), Some(_)) => Err(invalid("Text: lang and space are mutually exclusive")),
        (None, Some(s)) => Ok(Text::space(&s.inner)),
        (None, None) => Ok(Text::english()),
        (Some(l), None) => Ok(match parse_lang(l)? {
            Lang::English => Text::english(),
            Lang::Raw => Text::raw(),
        }),
    }
}

/// Build a core [`Codec`] from flat facade args. `(None, None)` per family
/// resolves to the core default constructor.
pub(crate) fn build_codec(
    family: &str,
    qam_amp_bits: Option<u8>,
    qam_phase_bits: Option<u8>,
    upq_cells: Option<u32>,
    upq_design: Option<&str>,
) -> PyResultX<Codec> {
    match family.to_ascii_lowercase().as_str() {
        "qam" => match (qam_amp_bits, qam_phase_bits) {
            (None, None) => Ok(Codec::qam()),
            (Some(a), Some(p)) => Ok(Codec::qam_bits(a, p)),
            _ => Err(invalid(
                "Qam: specify both amp_bits and phase_bits, or neither",
            )),
        },
        "upq" => match (upq_cells, upq_design) {
            (None, None) => Ok(Codec::upq()),
            (Some(c), None) => Ok(Codec::upq_cells(c)),
            (Some(c), Some(d)) => Ok(Codec::upq_with(c, parse_design(d)?)),
            (None, Some(_)) => Err(invalid(
                "Upq: a design requires an explicit cells budget (mirror of Codec::upq_with)",
            )),
        },
        other => Err(invalid(format!(
            "unknown codec family '{other}' (expected qam or upq)"
        ))),
    }
}

/// Build a core [`Calibrate`] from flat facade args.
pub(crate) fn build_calibrate(
    mode: &str,
    sample: Option<u64>,
    vectors: Option<PyReadonlyArray2<'_, f32>>,
) -> PyResultX<Calibrate> {
    match mode.to_ascii_lowercase().as_str() {
        "auto" => Ok(match sample {
            None => Calibrate::auto(),
            Some(n) => Calibrate::auto_sample(n as usize),
        }),
        "now" => {
            let arr =
                vectors.ok_or_else(|| invalid("Now: a 2-D float32 sample array is required"))?;
            arr.as_slice()
                .map_err(|_| invalid("Now: sample must be a C-contiguous float32 2-D array"))?;
            let rows: Vec<Vec<f32>> = arr
                .as_array()
                .outer_iter()
                .map(|row| row.to_vec())
                .collect();
            Ok(Calibrate::now(rows))
        }
        other => Err(invalid(format!(
            "unknown calibrate mode '{other}' (expected auto or now)"
        ))),
    }
}

/// Build a core [`Vector`] spec from flat facade args. Exactly one of
/// `dim`/`space`; configuration on a shared binding is passed through so the
/// core raises its declaration-time error.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_vector(
    dim: Option<u32>,
    space: Option<&PySpace>,
    metric: Option<&str>,
    codec_family: Option<&str>,
    qam_amp_bits: Option<u8>,
    qam_phase_bits: Option<u8>,
    upq_cells: Option<u32>,
    upq_design: Option<&str>,
    cal_mode: Option<&str>,
    cal_sample: Option<u64>,
    cal_vectors: Option<PyReadonlyArray2<'_, f32>>,
) -> PyResultX<Vector> {
    let mut v = match (dim, space) {
        (Some(d), None) => Vector::dim(d),
        (None, Some(s)) => Vector::space(&s.inner),
        (Some(_), Some(_)) => return Err(invalid("Vector: dim and space are mutually exclusive")),
        (None, None) => return Err(invalid("Vector: one of dim or space is required")),
    };
    if let Some(m) = metric {
        v = v.metric(parse_metric(m)?);
    }
    if let Some(f) = codec_family {
        v = v.codec(build_codec(
            f,
            qam_amp_bits,
            qam_phase_bits,
            upq_cells,
            upq_design,
        )?);
    }
    if let Some(m) = cal_mode {
        v = v.calibrate(build_calibrate(m, cal_sample, cal_vectors)?);
    }
    Ok(v)
}

/// Native schema builder. The facade lowers its declarative field list into
/// one `text`/`vector` call per field, then hands the whole object to
/// `Store::collection` / `Store::partitioned`.
#[pyclass(name = "Schema")]
pub struct PySchema {
    pub(crate) inner: Schema,
}

#[pymethods]
impl PySchema {
    #[new]
    fn new() -> Self {
        PySchema {
            inner: Schema::new(),
        }
    }

    /// Add a text field. `lang=None, space=None` is the core default
    /// (`Text::english()`).
    #[pyo3(signature = (name, lang=None, space=None))]
    fn text(
        &mut self,
        name: &str,
        lang: Option<&str>,
        space: Option<PyRef<'_, PySpace>>,
    ) -> PyResultX<()> {
        let spec = build_text(lang, space.as_deref())?;
        self.inner = std::mem::take(&mut self.inner).text_with(name, spec);
        Ok(())
    }

    /// Add a vector field from flat spec args (see [`build_vector`]).
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (name, dim=None, space=None, metric=None, codec_family=None,
        qam_amp_bits=None, qam_phase_bits=None, upq_cells=None, upq_design=None,
        cal_mode=None, cal_sample=None, cal_vectors=None))]
    fn vector(
        &mut self,
        name: &str,
        dim: Option<u32>,
        space: Option<PyRef<'_, PySpace>>,
        metric: Option<&str>,
        codec_family: Option<&str>,
        qam_amp_bits: Option<u8>,
        qam_phase_bits: Option<u8>,
        upq_cells: Option<u32>,
        upq_design: Option<&str>,
        cal_mode: Option<&str>,
        cal_sample: Option<u64>,
        cal_vectors: Option<PyReadonlyArray2<'_, f32>>,
    ) -> PyResultX<()> {
        let spec = build_vector(
            dim,
            space.as_deref(),
            metric,
            codec_family,
            qam_amp_bits,
            qam_phase_bits,
            upq_cells,
            upq_design,
            cal_mode,
            cal_sample,
            cal_vectors,
        )?;
        self.inner = std::mem::take(&mut self.inner).vector(name, spec);
        Ok(())
    }
}
