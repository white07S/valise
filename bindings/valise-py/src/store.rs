//! Store handle — the entry point: open/create, schema declaration, shared
//! spaces, partitioning, reader/writer acquisition, and convenience
//! get/search/search_view/compact/stats.

use std::path::PathBuf;

use numpy::PyReadonlyArray2;
use pyo3::prelude::*;

use valise::db::{Partition, Store, StoreOptions};

use crate::builders::PySearch;
use crate::convert::{PyObject, key_from_py, parse_durability};
use crate::errors::{PyResultX, invalid};
use crate::partition::{PyPartitioned, PyView};
use crate::reader::PyReader;
use crate::results::PyStored;
use crate::schema::{PySchema, PySpace, build_text, build_vector};
use crate::writer::PyWriter;

#[pyclass(name = "Store")]
pub struct PyStore {
    inner: Store,
}

#[pymethods]
impl PyStore {
    /// Open an existing store or create one if the path does not exist.
    #[staticmethod]
    #[pyo3(signature = (path, durability="buffered"))]
    fn open(path: PathBuf, durability: &str) -> PyResultX<PyStore> {
        let opts = StoreOptions {
            durability: parse_durability(durability)?,
        };
        Ok(PyStore {
            inner: Store::open_with(path, opts)?,
        })
    }

    /// Create a new store, failing if the path already exists.
    #[staticmethod]
    #[pyo3(signature = (path, durability="buffered"))]
    fn create(path: PathBuf, durability: &str) -> PyResultX<PyStore> {
        let opts = StoreOptions {
            durability: parse_durability(durability)?,
        };
        Ok(PyStore {
            inner: Store::create_with(path, opts)?,
        })
    }

    /// Create-or-open a collection, bind the schema, and persist it in-file
    /// (reopen needs no re-declaration). Divergent re-declares raise
    /// `SchemaMismatchError`.
    fn collection(&self, name: String, schema: &PySchema) -> PyResultX<()> {
        self.inner.collection(&name, schema.inner.clone())?;
        Ok(())
    }

    /// Define (or return the existing) shared **text** space.
    #[pyo3(signature = (name, lang=None))]
    fn define_space_text(&self, name: String, lang: Option<&str>) -> PyResultX<PySpace> {
        let spec = build_text(lang, None)?;
        Ok(PySpace {
            inner: self.inner.define_space(&name, spec)?,
        })
    }

    /// Define (or return the existing) shared **vector** space from flat
    /// spec args (same lowering as `Schema.vector`).
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (name, dim, metric=None, codec_family=None,
        qam_amp_bits=None, qam_phase_bits=None, upq_cells=None, upq_design=None,
        cal_mode=None, cal_sample=None, cal_vectors=None))]
    fn define_space_vector(
        &self,
        name: String,
        dim: u32,
        metric: Option<&str>,
        codec_family: Option<&str>,
        qam_amp_bits: Option<u8>,
        qam_phase_bits: Option<u8>,
        upq_cells: Option<u32>,
        upq_design: Option<&str>,
        cal_mode: Option<&str>,
        cal_sample: Option<u64>,
        cal_vectors: Option<PyReadonlyArray2<'_, f32>>,
    ) -> PyResultX<PySpace> {
        let spec = build_vector(
            Some(dim),
            None,
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
        Ok(PySpace {
            inner: self.inner.define_space(&name, spec)?,
        })
    }

    /// Look up a previously defined space by name.
    fn space(&self, name: String) -> Option<PySpace> {
        self.inner.space(&name).map(|s| PySpace { inner: s })
    }

    /// All defined spaces as `(name, dim_or_None, auto)` rows — `dim=None`
    /// means a text space. The facade wraps these into `SpaceInfo`.
    fn spaces(&self) -> Vec<(String, Option<u32>, bool)> {
        self.inner
            .spaces()
            .into_iter()
            .map(|info| {
                let dim = match info.kind {
                    valise::db::SpaceKind::Vector { dim } => Some(dim),
                    valise::db::SpaceKind::Text => None,
                    _ => None,
                };
                (info.name, dim, info.auto)
            })
            .collect()
    }

    /// All user collections as `(name, id)` rows (the hidden `~valise.schema`
    /// store is excluded by the core).
    fn collections(&self) -> Vec<(String, u32)> {
        self.inner
            .collections()
            .into_iter()
            .map(|c| (c.name, c.id.0))
            .collect()
    }

    /// Create-or-open a logical partitioned collection routed by `by`
    /// (`"day"` or `"month"`).
    fn partitioned(&self, base: String, schema: &PySchema, by: &str) -> PyResultX<PyPartitioned> {
        let by = match by.to_ascii_lowercase().as_str() {
            "day" => Partition::ByDay,
            "month" => Partition::ByMonth,
            other => {
                return Err(invalid(format!(
                    "unknown partition period '{other}' (expected day or month)"
                )));
            }
        };
        Ok(PyPartitioned {
            inner: self.inner.partitioned(&base, schema.inner.clone(), by)?,
        })
    }

    /// A cheap, cloneable read handle.
    fn reader(&self) -> PyReader {
        PyReader {
            inner: self.inner.reader(),
        }
    }

    /// Acquire the single serialized writer (owned; blocks until any prior
    /// writer drops). Always `close()` it (or use it as a context manager).
    ///
    /// The single-writer lock is acquired non-blockingly in a poll loop that
    /// releases the GIL (`py.detach`) while it sleeps. A plain blocking acquire
    /// would park this thread *with the GIL held*, so a writer already held on
    /// another Python thread could never reacquire the GIL to run its
    /// `close()`/Drop and release the lock — an interpreter-wide deadlock.
    /// `OwnedWriter` is `!Send`, so it cannot be returned out of a `py.detach`
    /// closure; we keep the acquire on this thread and only sleep detached.
    fn writer(&self, py: Python<'_>) -> PyWriter {
        loop {
            if let Some(w) = self.inner.try_writer_owned() {
                return PyWriter { inner: Some(w) };
            }
            py.detach(|| std::thread::sleep(std::time::Duration::from_micros(50)));
        }
    }

    fn get(
        &self,
        py: Python<'_>,
        coll: String,
        key: &Bound<'_, PyAny>,
    ) -> PyResultX<Option<PyStored>> {
        let k = key_from_py(key)?;
        Ok(self
            .inner
            .get(&coll, k)?
            .map(|s| PyStored::from_stored(py, s)))
    }

    fn search(
        &self,
        py: Python<'_>,
        coll: String,
        search: &PySearch,
    ) -> PyResultX<(Vec<PyObject>, Py<numpy::PyArray1<f32>>)> {
        self.reader().search_impl(py, coll, search)
    }

    /// Search across a partitioned view (one global fusion). Returns
    /// `(keys, scores, collections)` with a per-hit collection name.
    fn search_view(
        &self,
        py: Python<'_>,
        view: &PyView,
        search: &PySearch,
    ) -> PyResultX<(Vec<PyObject>, Py<numpy::PyArray1<f32>>, Vec<String>)> {
        self.reader().search_view_impl(py, view, search)
    }

    /// Reclaim tombstoned bytes. Returns a dict of before/after counts.
    #[pyo3(signature = (recalibrate=false))]
    fn compact(&self, py: Python<'_>, recalibrate: bool) -> PyResultX<PyObject> {
        let report = self
            .inner
            .compact(valise::db::CompactOptions { recalibrate })?;
        let d = pyo3::types::PyDict::new(py);
        let set = |k: &str, v: u64| {
            d.set_item(k, v).expect("BUG: dict set_item failed");
        };
        d.set_item("compacted", report.compacted)
            .expect("BUG: dict set_item failed");
        set("frames_before", report.frames_before as u64);
        set("frames_after", report.frames_after as u64);
        set("vectors_before", report.vectors_before as u64);
        set("vectors_after", report.vectors_after as u64);
        set("bytes_before", report.bytes_before);
        set("bytes_after", report.bytes_after);
        Ok(d.into())
    }

    /// Live/total occupancy counts as a dict.
    fn stats(&self, py: Python<'_>) -> PyResultX<PyObject> {
        let s = self.inner.stats();
        let d = pyo3::types::PyDict::new(py);
        let set = |k: &str, v: u64| {
            d.set_item(k, v).expect("BUG: dict set_item failed");
        };
        set("frames_total", s.frames_total as u64);
        set("frames_active", s.frames_active as u64);
        set("vectors_total", s.vectors_total as u64);
        set("vectors_active", s.vectors_active as u64);
        set("file_bytes", s.file_bytes);
        d.set_item("tombstone_ratio", s.tombstone_ratio())
            .expect("BUG: dict set_item failed");
        Ok(d.into())
    }
}
