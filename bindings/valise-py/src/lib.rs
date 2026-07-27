//! Python bindings for the `valise` application (`db`) layer.
//!
//! Thin `#[pyclass]` wrappers over `Store` / `Reader` / `OwnedWriter`, the
//! `Schema` / `Record` / `Search` builders, and the partition handles
//! (`Partitioned` / `View`). The ergonomic surface (typed value objects,
//! kwargs defaults, context managers, docstrings) lives in
//! `python/valise/`; these classes are deliberately mechanical.
//!
//! Design notes:
//! - No `#[pyclass]` carries a Rust lifetime. `PyWriter` holds an
//!   [`OwnedWriter`](valise::db::OwnedWriter) (whose single-writer lock guard
//!   is `'static`) inside an `Option` so `close()` can drop it deterministically
//!   and release the lock.
//! - The orphan rule forbids `impl From<valise::Error> for PyErr` (both
//!   foreign), so we go through the `PyValiseError` newtype; every `#[pymethods]`
//!   returns `PyResultX<T>`.
//! - Defaults live in Rust: facade-level `None` sentinels route to the core
//!   constructors/defaults in `schema.rs` lowering, never to duplicated
//!   Python-side numbers.
//! - The zero-copy ingest path is `PyWriter::put_many`, which borrows a
//!   C-contiguous `f32` 2-D NumPy buffer and slices each row into a `Record`
//!   with no per-row vector copy.

mod builders;
mod convert;
mod errors;
mod partition;
mod reader;
mod results;
mod schema;
mod store;
mod writer;

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use builders::{PyRecord, PySearch};
use errors::{
    BusyError, CorruptionError, NotCalibratedError, SchemaMismatchError, UnsupportedError,
    ValidationError, ValiseError,
};
use partition::{PyPartitioned, PyView};
use reader::PyReader;
use results::PyStored;
use schema::{PySchema, PySpace};
use store::PyStore;
use writer::PyWriter;

/// Return the core `valise` crate version string.
#[pyfunction]
fn format_version() -> &'static str {
    valise::VERSION
}

#[pymodule]
fn _native(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(format_version, m)?)?;

    m.add_class::<PyStore>()?;
    m.add_class::<PyWriter>()?;
    m.add_class::<PyReader>()?;
    m.add_class::<PySchema>()?;
    m.add_class::<PySpace>()?;
    m.add_class::<PyRecord>()?;
    m.add_class::<PySearch>()?;
    m.add_class::<PyStored>()?;
    m.add_class::<PyPartitioned>()?;
    m.add_class::<PyView>()?;

    m.add("ValiseError", py.get_type::<ValiseError>())?;
    m.add("CorruptionError", py.get_type::<CorruptionError>())?;
    m.add("BusyError", py.get_type::<BusyError>())?;
    m.add("UnsupportedError", py.get_type::<UnsupportedError>())?;
    m.add("ValidationError", py.get_type::<ValidationError>())?;
    m.add("NotCalibratedError", py.get_type::<NotCalibratedError>())?;
    m.add("SchemaMismatchError", py.get_type::<SchemaMismatchError>())?;

    Ok(())
}
