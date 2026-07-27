// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! `Key` <-> Python value conversion and string -> enum parsers.
//!
//! These conversions are kept at the native edge so the facade can pass enum
//! `.value` strings straight through.

use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes};

use valise::db::{Durability, Key, Lang, Metric, Rerank, TextScorer, TfMode, UpqDesign};

use crate::errors::{PyResultX, invalid};

/// `Py<PyAny>` — an owned, GIL-independent reference to an arbitrary Python
/// object. (`pyo3::PyObject` is not re-exported uniformly across 0.28 builds, so
/// we spell the alias locally and share it crate-wide.)
pub(crate) type PyObject = Py<PyAny>;

/// Convert a Python `str | int | bytes` into a [`Key`].
///
/// Extraction order matters: `bool` is an `int` subclass and would otherwise
/// become `Key::U64(0/1)`, so it is rejected explicitly. A negative int fails
/// `u64` extraction and falls through to a clean `ValidationError` (it is not a
/// `bytes`).
pub(crate) fn key_from_py(obj: &Bound<'_, PyAny>) -> PyResultX<Key> {
    if let Ok(s) = obj.extract::<String>() {
        return Ok(Key::Str(s));
    }
    if obj.is_instance_of::<PyBool>() {
        return Err(invalid("key must be str, int, or bytes (got bool)"));
    }
    if let Ok(n) = obj.extract::<u64>() {
        return Ok(Key::U64(n));
    }
    if let Ok(b) = obj.extract::<Vec<u8>>() {
        return Ok(Key::Bytes(b));
    }
    Err(invalid(
        "key must be str, int (non-negative, fits u64), or bytes",
    ))
}

/// Convert a [`Key`] back into the exact Python variant the user supplied.
pub(crate) fn key_to_py(py: Python<'_>, key: &Key) -> PyObject {
    match key {
        Key::Str(s) => s.into_pyobject(py).map(Bound::into_any),
        Key::U64(n) => n.into_pyobject(py).map(Bound::into_any),
        Key::Bytes(b) => PyBytes::new(py, b).into_pyobject(py).map(Bound::into_any),
    }
    .expect("BUG: key_to_py conversion is infallible")
    .unbind()
}

pub(crate) fn parse_metric(s: &str) -> PyResultX<Metric> {
    match s.to_ascii_lowercase().as_str() {
        "cosine" => Ok(Metric::Cosine),
        "dot" | "inner" | "ip" => Ok(Metric::Dot),
        "l2" | "euclidean" => Ok(Metric::L2),
        other => Err(invalid(format!(
            "unknown metric '{other}' (expected cosine, dot, or l2)"
        ))),
    }
}

pub(crate) fn parse_lang(s: &str) -> PyResultX<Lang> {
    match s.to_ascii_lowercase().as_str() {
        "english" | "en" => Ok(Lang::English),
        "raw" => Ok(Lang::Raw),
        other => Err(invalid(format!(
            "unknown lang '{other}' (expected english or raw)"
        ))),
    }
}

pub(crate) fn parse_durability(s: &str) -> PyResultX<Durability> {
    match s.to_ascii_lowercase().as_str() {
        "buffered" => Ok(Durability::Buffered),
        "fsync" | "fullsync" | "full_sync" => Ok(Durability::FullSync),
        "syncall" | "sync_all" | "sync" => Ok(Durability::SyncAll),
        other => Err(invalid(format!(
            "unknown durability '{other}' (expected buffered, full_sync, or sync_all)"
        ))),
    }
}

pub(crate) fn parse_design(s: &str) -> PyResultX<UpqDesign> {
    match s.to_ascii_lowercase().as_str() {
        "empirical" => Ok(UpqDesign::Empirical),
        "rayleigh" => Ok(UpqDesign::Rayleigh),
        other => Err(invalid(format!(
            "unknown UPQ design '{other}' (expected empirical or rayleigh)"
        ))),
    }
}

pub(crate) fn parse_rerank(s: &str) -> PyResultX<Rerank> {
    match s.to_ascii_lowercase().as_str() {
        "fast" => Ok(Rerank::Fast),
        "accurate" => Ok(Rerank::Accurate),
        other => Err(invalid(format!(
            "unknown rerank '{other}' (expected fast or accurate)"
        ))),
    }
}

pub(crate) fn parse_tf_mode(s: &str) -> PyResultX<TfMode> {
    match s.to_ascii_lowercase().as_str() {
        "raw" => Ok(TfMode::Raw),
        "log" => Ok(TfMode::Log),
        other => Err(invalid(format!(
            "unknown tf_mode '{other}' (expected raw or log)"
        ))),
    }
}

pub(crate) fn parse_text_scorer(
    s: &str,
    k1: Option<f32>,
    b: Option<f32>,
    tf_mode: Option<&str>,
) -> PyResultX<TextScorer> {
    match s.to_ascii_lowercase().as_str() {
        "bm25" => {
            // Defaults live in Rust: pull the canonical (k1, b) from the core
            // constructor so partial overrides never duplicate the numbers.
            let TextScorer::Bm25 { k1: dk1, b: db } = TextScorer::bm25() else {
                return Err(invalid("BUG: TextScorer::bm25() is not Bm25"));
            };
            Ok(TextScorer::bm25_with(k1.unwrap_or(dk1), b.unwrap_or(db)))
        }
        "tfidf_cosine" | "tfidfcosine" => Ok(TextScorer::tfidf_cosine(parse_tf_mode(
            tf_mode.unwrap_or("log"),
        )?)),
        "tfidf_cosine_approx" => Ok(TextScorer::tfidf_cosine_approx(parse_tf_mode(
            tf_mode.unwrap_or("log"),
        )?)),
        "count_cosine" | "countcosine" => Ok(TextScorer::count_cosine()),
        "count_cosine_approx" => Ok(TextScorer::count_cosine_approx()),
        "dice" => Ok(TextScorer::dice()),
        "overlap" => Ok(TextScorer::overlap()),
        "containment" => Ok(TextScorer::containment()),
        other => Err(invalid(format!(
            "unknown text scorer '{other}' (expected bm25, tfidf_cosine, tfidf_cosine_approx, count_cosine, count_cosine_approx, dice, overlap, or containment)"
        ))),
    }
}
