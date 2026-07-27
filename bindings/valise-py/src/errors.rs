// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Exception hierarchy + the `valise::Error` → `PyErr` classification newtype.
//!
//! The orphan rule forbids `impl From<valise::Error> for PyErr` (both
//! foreign), so we go through the [`PyValiseError`] newtype; every `#[pymethods]`
//! returns [`PyResultX`] so `?` flows cleanly.

use pyo3::create_exception;
use pyo3::exceptions::PyIOError;
use pyo3::prelude::*;

create_exception!(_native, ValiseError, pyo3::exceptions::PyException);
create_exception!(_native, CorruptionError, ValiseError);
create_exception!(_native, BusyError, ValiseError);
create_exception!(_native, UnsupportedError, ValiseError);
create_exception!(_native, ValidationError, ValiseError);
create_exception!(_native, NotCalibratedError, ValiseError);
create_exception!(_native, SchemaMismatchError, ValiseError);

/// Newtype around [`valise::Error`] so we can satisfy the orphan rule and
/// convert into a classified [`PyErr`]. All `#[pymethods]` return
/// `Result<T, PyValiseError>` (aliased [`PyResultX`]) so `?` flows cleanly.
pub(crate) struct PyValiseError(pub(crate) valise::Error);

impl From<valise::Error> for PyValiseError {
    fn from(e: valise::Error) -> Self {
        PyValiseError(e)
    }
}

impl From<PyValiseError> for PyErr {
    fn from(e: PyValiseError) -> PyErr {
        use valise::Error as E;
        match e.0 {
            E::Io(io) => PyIOError::new_err(io.to_string()),
            E::Integrity(s) => CorruptionError::new_err(s),
            E::Signature => CorruptionError::new_err("signature verification failed"),
            E::Busy(s) => BusyError::new_err(s),
            E::Unsupported(s) => UnsupportedError::new_err(s),
            E::Format(s) | E::Serde(s) => ValidationError::new_err(s),
            E::Compression(s) => ValiseError::new_err(format!("compression: {s}")),
            // The Display impls carry the structured fields (space / collection
            // + detail) in the message, which is what Python surfaces.
            e @ E::NotCalibrated { .. } => NotCalibratedError::new_err(e.to_string()),
            e @ E::SchemaMismatch { .. } => SchemaMismatchError::new_err(e.to_string()),
            E::Other(s) => ValiseError::new_err(s),
        }
    }
}

pub(crate) type PyResultX<T> = Result<T, PyValiseError>;

/// Build a `ValidationError` `PyValiseError` from a message (for binding-local
/// validation, not engine errors).
pub(crate) fn invalid(msg: impl Into<String>) -> PyValiseError {
    PyValiseError(valise::Error::Format(msg.into()))
}
