//! Partition natives: [`PyPartitioned`] (time-routed logical collection) and
//! [`PyView`] (a resolved window of physical partitions for `search_view`).
//!
//! The facade's `Window` value object dispatches to one of the `view_*`
//! methods; `Partition.BY_DAY` / `BY_MONTH` arrive as strings. The core's
//! `Partition::Custom` (an arbitrary Rust closure) is intentionally not
//! mirrored — see the parity notes in `python/valise/partition.py`.

use pyo3::prelude::*;

use valise::db::{Partitioned, View, Window};

use crate::errors::PyResultX;

#[pyclass(name = "Partitioned")]
pub struct PyPartitioned {
    pub(crate) inner: Partitioned,
}

#[pymethods]
impl PyPartitioned {
    /// Partitions intersecting `[now - days, now]`. `now=None` uses the
    /// system clock (mirror of `Partitioned::view` vs `view_as_of`).
    #[pyo3(signature = (days, now=None))]
    fn view_last_days(&self, days: u32, now: Option<i64>) -> PyView {
        let w = Window::LastDays(days);
        PyView {
            inner: match now {
                Some(n) => self.inner.view_as_of(w, n),
                None => self.inner.view(w),
            },
        }
    }

    /// Partitions intersecting `[from, to]` (inclusive unix seconds).
    fn view_range(&self, from: i64, to: i64) -> PyView {
        PyView {
            inner: self.inner.view(Window::Range { from, to }),
        }
    }

    /// Explicit period suffixes (e.g. `"2026-05-25"`), resolved to existing
    /// partitions.
    fn view_partitions(&self, suffixes: Vec<String>) -> PyView {
        PyView {
            inner: self.inner.view(Window::Partitions(suffixes)),
        }
    }

    /// A view over every existing partition.
    fn all(&self) -> PyView {
        PyView {
            inner: self.inner.all(),
        }
    }

    /// Soft-forget: tombstone every record in partitions whose entire period
    /// precedes `cutoff`; returns the number of frames tombstoned.
    ///
    /// Acquires the single writer internally, so it runs detached (GIL
    /// released) — a blocking GIL-held acquire could deadlock against a
    /// writer held on another Python thread (same hazard as `Store.writer`).
    fn forget_before(&self, py: Python<'_>, cutoff: i64) -> PyResultX<usize> {
        let p = self.inner.clone();
        Ok(py.detach(move || p.forget_before(cutoff))?)
    }
}

/// A resolved set of physical partition collections (input to
/// `Store.search_view` / `Reader.search_view`).
#[pyclass(name = "View", skip_from_py_object)]
#[derive(Clone)]
pub struct PyView {
    pub(crate) inner: View,
}

#[pymethods]
impl PyView {
    /// The physical collection names in this view.
    fn collections(&self) -> Vec<String> {
        self.inner.collections().to_vec()
    }

    fn __repr__(&self) -> String {
        format!("View(collections={:?})", self.inner.collections())
    }
}
