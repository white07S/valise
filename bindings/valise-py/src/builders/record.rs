//! Record builder (owns its data so `Record<'_>` can borrow it for one put).

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

use valise::db::{Key, Record};

use crate::convert::key_from_py;
use crate::errors::{PyResultX, invalid};

#[pyclass(name = "Record")]
pub struct PyRecord {
    texts: Vec<(String, String)>,
    vectors: Vec<(String, Vec<f32>)>,
    created_at: Option<i64>,
    parent: Option<Key>,
}

impl PyRecord {
    /// Borrow this record's owned buffers as a core [`Record`] for one `put`.
    pub(crate) fn to_record(&self) -> Record<'_> {
        let mut r = Record::new();
        for (f, v) in &self.texts {
            r = r.text(f, v);
        }
        for (f, v) in &self.vectors {
            r = r.vector(f, v);
        }
        if let Some(ts) = self.created_at {
            r = r.at(ts);
        }
        if let Some(p) = &self.parent {
            r = r.child_of(p.clone());
        }
        r
    }
}

#[pymethods]
impl PyRecord {
    #[new]
    fn new() -> Self {
        PyRecord {
            texts: Vec::new(),
            vectors: Vec::new(),
            created_at: None,
            parent: None,
        }
    }

    fn text(mut slf: PyRefMut<'_, Self>, field: String, value: String) -> PyRefMut<'_, Self> {
        slf.texts.push((field, value));
        slf
    }

    fn vector<'py>(
        mut slf: PyRefMut<'py, Self>,
        field: String,
        values: PyReadonlyArray1<'py, f32>,
    ) -> PyResultX<PyRefMut<'py, Self>> {
        let v = values
            .as_slice()
            .map_err(|_| invalid("vector must be a C-contiguous float32 array"))?
            .to_vec();
        slf.vectors.push((field, v));
        Ok(slf)
    }

    fn at(mut slf: PyRefMut<'_, Self>, unix_secs: i64) -> PyRefMut<'_, Self> {
        slf.created_at = Some(unix_secs);
        slf
    }

    fn child_of<'py>(
        mut slf: PyRefMut<'py, Self>,
        parent: &Bound<'py, PyAny>,
    ) -> PyResultX<PyRefMut<'py, Self>> {
        slf.parent = Some(key_from_py(parent)?);
        Ok(slf)
    }
}
