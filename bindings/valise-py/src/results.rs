//! `Stored` (read-back record).

use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;

use crate::convert::{PyObject, key_to_py};

#[pyclass(name = "Stored")]
pub struct PyStored {
    #[pyo3(get)]
    key: PyObject,
    #[pyo3(get)]
    collection: String,
    #[pyo3(get)]
    created_at: i64,
    #[pyo3(get)]
    text: Option<String>,
    #[pyo3(get)]
    vectors: Vec<(String, Py<PyArray1<f32>>)>,
}

impl PyStored {
    // TODO(perf, BINDINGS_PLAN §11): the `get` return path overshoots the plan's
    // <1% overhead target (measured ~1.6-3.2% at dim=768, ~1.5 µs/call) because
    // it eagerly builds a Python result object per call — decoding the text
    // payload to `str`, moving each reconstructed vector into a NumPy array, and
    // converting the key. The plan wants `get` to be a lean transfer/move. Fix
    // later by materializing lazily (e.g. expose `text`/`vectors` as cached
    // properties that reconstruct on first access) or returning columnar buffers
    // so a key-only `get` pays nothing for the vector/text it never touches.
    pub(crate) fn from_stored(py: Python<'_>, s: valise::db::Stored) -> PyStored {
        let vectors = s
            .vectors
            .into_iter()
            .map(|(name, v)| (name, v.into_pyarray(py).unbind()))
            .collect();
        PyStored {
            key: key_to_py(py, &s.key),
            collection: s.collection,
            created_at: s.created_at,
            text: s.text,
            vectors,
        }
    }
}
