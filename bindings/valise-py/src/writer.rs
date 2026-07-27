//! Writer handle (owns `OwnedWriter`; `close()` drops it to release the
//! single-writer lock).

use numpy::PyReadonlyArray2;
use pyo3::prelude::*;
use pyo3::types::PyList;

use valise::db::Record;

use crate::builders::PyRecord;
use crate::convert::{PyObject, key_from_py, key_to_py};
use crate::errors::{PyResultX, PyValiseError, invalid};

/// `unsendable`: `OwnedWriter` holds a `parking_lot::ArcMutexGuard` whose guard
/// is `!Send` (it must be released on the acquiring thread). `unsendable` makes
/// PyO3 enforce same-thread access at runtime, which is exactly the contract a
/// lock guard needs.
#[pyclass(name = "Writer", unsendable)]
pub struct PyWriter {
    pub(crate) inner: Option<valise::db::OwnedWriter>,
}

impl PyWriter {
    fn writer(&mut self) -> PyResultX<&mut valise::db::OwnedWriter> {
        self.inner
            .as_mut()
            .ok_or_else(|| PyValiseError(valise::Error::Unsupported("writer is closed".into())))
    }
}

#[pymethods]
impl PyWriter {
    fn put(&mut self, coll: String, key: &Bound<'_, PyAny>, record: &PyRecord) -> PyResultX<()> {
        let k = key_from_py(key)?;
        let rec = record.to_record();
        self.writer()?.put(&coll, k, rec)?;
        Ok(())
    }

    fn put_auto(&mut self, py: Python<'_>, coll: String, record: &PyRecord) -> PyResultX<PyObject> {
        let rec = record.to_record();
        let key = self.writer()?.put_auto(&coll, rec)?;
        Ok(key_to_py(py, &key))
    }

    /// Route a record into a partitioned logical collection (the partition is
    /// derived from the record's `created_at`, defaulting to now).
    fn put_into(
        &mut self,
        partitioned: PyRef<'_, crate::partition::PyPartitioned>,
        key: &Bound<'_, PyAny>,
        record: &PyRecord,
    ) -> PyResultX<()> {
        let k = key_from_py(key)?;
        let rec = record.to_record();
        self.writer()?.put_into(&partitioned.inner, k, rec)?;
        Ok(())
    }

    /// Zero-copy bulk ingest. `vectors` is an `[N, dim]` C-contiguous float32
    /// array borrowed without per-row copy; `keys` (and optional `texts`) are
    /// length-`N` Python lists. `field` is the vector field; `text_field` is the
    /// text field the optional `texts` land in (defaults to `"body"`).
    #[pyo3(signature = (coll, keys, vectors, field="dense", texts=None, text_field="body"))]
    fn put_many(
        &mut self,
        coll: String,
        keys: &Bound<'_, PyList>,
        vectors: PyReadonlyArray2<'_, f32>,
        field: &str,
        texts: Option<&Bound<'_, PyList>>,
        text_field: &str,
    ) -> PyResultX<()> {
        // Validate C-contiguity of the whole 2-D buffer up front so this is an
        // all-or-nothing boundary check: a non-contiguous array fails here,
        // before any row is staged on the writer (avoids leaving a partial batch
        // staged that a later commit would silently persist).
        vectors
            .as_slice()
            .map_err(|_| invalid("put_many: vectors must be a C-contiguous float32 2-D array"))?;
        let arr = vectors.as_array();
        let n = arr.nrows();
        if keys.len() != n {
            return Err(invalid(format!(
                "put_many: keys has {} items but vectors has {n} rows",
                keys.len()
            )));
        }
        if let Some(t) = texts {
            if t.len() != n {
                return Err(invalid(format!(
                    "put_many: texts has {} items but vectors has {n} rows",
                    t.len()
                )));
            }
        }

        // Pre-extract keys/texts to owned Rust values so the per-row loop does
        // not re-enter Python after the row borrow.
        let mut row_keys = Vec::with_capacity(n);
        for item in keys.iter() {
            row_keys.push(key_from_py(&item)?);
        }
        let row_texts: Option<Vec<String>> = match texts {
            Some(t) => {
                let mut v = Vec::with_capacity(n);
                for item in t.iter() {
                    v.push(
                        item.extract::<String>()
                            .map_err(|_| invalid("put_many: texts must be a list of str"))?,
                    );
                }
                Some(v)
            }
            None => None,
        };

        let w = self.writer()?;
        for (i, key) in row_keys.into_iter().enumerate() {
            let row = arr
                .row(i)
                .to_slice()
                .ok_or_else(|| invalid("put_many: vectors must be C-contiguous"))?;
            let mut rec = Record::new().vector(field, row);
            if let Some(tx) = row_texts.as_ref() {
                rec = rec.text(text_field, &tx[i]);
            }
            w.put(&coll, key, rec)?;
        }
        Ok(())
    }

    fn delete(&mut self, coll: String, key: &Bound<'_, PyAny>) -> PyResultX<bool> {
        let k = key_from_py(key)?;
        Ok(self.writer()?.delete(&coll, &k)?)
    }

    /// Commit all staged mutations. Returns the new snapshot generation.
    ///
    /// NB: the GIL is held across the commit. `OwnedWriter` carries the
    /// single-writer lock guard (`parking_lot::ArcMutexGuard`), which is `!Send`,
    /// so the commit closure cannot cross a `py.detach` boundary. The durable
    /// barrier still runs under the writer lock, just GIL-attached.
    fn commit(&mut self) -> PyResultX<u64> {
        let outcome = self.writer()?.commit()?;
        Ok(outcome.snapshot_generation)
    }

    /// Drop the underlying writer, releasing the single-writer lock. Idempotent.
    fn close(&mut self) -> PyResultX<()> {
        self.inner = None;
        Ok(())
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    #[pyo3(signature = (*_args))]
    fn __exit__(&mut self, _args: &Bound<'_, PyAny>) -> PyResultX<bool> {
        self.close()?;
        Ok(false)
    }
}
