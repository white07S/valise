// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Reader handle (cheap, cloneable read access).

use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;
use pyo3::types::PyList;

use crate::builders::PySearch;
use crate::convert::{PyObject, key_from_py, key_to_py};
use crate::errors::PyResultX;
use crate::partition::PyView;
use crate::results::PyStored;

#[pyclass(name = "Reader")]
pub struct PyReader {
    pub(crate) inner: valise::db::Reader,
}

impl PyReader {
    /// Crate-visible entry to the search path so `PyStore::search` can forward
    /// to it across module boundaries (the `#[pymethods]` `search` below is
    /// private to PyO3's generated impl).
    pub(crate) fn search_impl(
        &self,
        py: Python<'_>,
        coll: String,
        search: &PySearch,
    ) -> PyResultX<(Vec<PyObject>, Py<PyArray1<f32>>)> {
        let q = search.build();
        let reader = self.inner.clone();
        let hits = py.detach(move || reader.search(&coll, q))?;
        let keys = hits.iter().map(|h| key_to_py(py, &h.key)).collect();
        let scores: Vec<f32> = hits.iter().map(|h| h.score).collect();
        Ok((keys, scores.into_pyarray(py).unbind()))
    }

    /// Crate-visible view-search path (`PyStore::search_view` forwards here).
    /// Hits span partitions, so the per-hit collection name is returned as a
    /// third column.
    pub(crate) fn search_view_impl(
        &self,
        py: Python<'_>,
        view: &PyView,
        search: &PySearch,
    ) -> PyResultX<(Vec<PyObject>, Py<PyArray1<f32>>, Vec<String>)> {
        let q = search.build();
        let reader = self.inner.clone();
        let v = view.inner.clone();
        let hits = py.detach(move || reader.search_view(&v, q))?;
        let keys = hits.iter().map(|h| key_to_py(py, &h.key)).collect();
        let scores: Vec<f32> = hits.iter().map(|h| h.score).collect();
        let colls: Vec<String> = hits.iter().map(|h| h.collection.clone()).collect();
        Ok((keys, scores.into_pyarray(py).unbind(), colls))
    }
}

#[pymethods]
impl PyReader {
    fn get(
        &self,
        py: Python<'_>,
        coll: String,
        key: &Bound<'_, PyAny>,
    ) -> PyResultX<Option<PyStored>> {
        let k = key_from_py(key)?;
        Ok(self
            .inner
            .get(&coll, &k)?
            .map(|s| PyStored::from_stored(py, s)))
    }

    fn get_many(
        &self,
        py: Python<'_>,
        coll: String,
        keys: &Bound<'_, PyList>,
    ) -> PyResultX<Vec<Option<PyStored>>> {
        let mut ks = Vec::with_capacity(keys.len());
        for item in keys.iter() {
            ks.push(key_from_py(&item)?);
        }
        let stored = self.inner.get_many(&coll, &ks)?;
        Ok(stored
            .into_iter()
            .map(|opt| opt.map(|s| PyStored::from_stored(py, s)))
            .collect())
    }

    /// Run a search. Returns columnar `(keys, scores)` — `keys` is a list of the
    /// exact key variants the user wrote; `scores` is a float32 NumPy array. The
    /// engine work runs with the GIL released.
    fn search(
        &self,
        py: Python<'_>,
        coll: String,
        search: &PySearch,
    ) -> PyResultX<(Vec<PyObject>, Py<PyArray1<f32>>)> {
        self.search_impl(py, coll, search)
    }

    /// Search across a partitioned view as one globally-fused query. Returns
    /// columnar `(keys, scores, collections)`.
    fn search_view(
        &self,
        py: Python<'_>,
        view: &PyView,
        search: &PySearch,
    ) -> PyResultX<(Vec<PyObject>, Py<PyArray1<f32>>, Vec<String>)> {
        self.search_view_impl(py, view, search)
    }
}
