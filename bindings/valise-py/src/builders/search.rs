// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Search builder (stores owned components; assembles a fresh `Search` per call
//! because core `Search` is not `Clone`).

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

use valise::db::{Fusion, Recency, Rerank, Search, TextScorer};

use crate::convert::{parse_rerank, parse_text_scorer};
use crate::errors::{PyResultX, invalid};

struct TextSpec {
    field: String,
    query: String,
    scorer: TextScorer,
}

struct VecSpec {
    field: String,
    query: Vec<f32>,
    rerank: Rerank,
}

#[pyclass(name = "Search")]
pub struct PySearch {
    text: Option<TextSpec>,
    vectors: Vec<VecSpec>,
    fusion: Fusion,
    recency: Option<Recency>,
    top_k: usize,
    now: Option<i64>,
}

impl PySearch {
    /// Assemble a fresh core [`Search`]. Called right before a (detached)
    /// `reader.search`, which takes `Search` by value.
    pub(crate) fn build(&self) -> Search {
        let mut s = Search::new().top_k(self.top_k).fuse(self.fusion);
        if let Some(t) = &self.text {
            s = s.text_with(&t.field, &t.query, t.scorer);
        }
        for v in &self.vectors {
            s = s.vector_with(&v.field, &v.query, v.rerank);
        }
        if let Some(r) = self.recency {
            s = s.recency(r);
        }
        if let Some(n) = self.now {
            s = s.now(n);
        }
        s
    }
}

#[pymethods]
impl PySearch {
    #[new]
    fn new() -> Self {
        PySearch {
            text: None,
            vectors: Vec::new(),
            fusion: Fusion::default(),
            recency: None,
            top_k: 10,
            now: None,
        }
    }

    #[pyo3(signature = (field, query, scorer="bm25", k1=None, b=None, tf_mode=None))]
    fn text<'py>(
        mut slf: PyRefMut<'py, Self>,
        field: String,
        query: String,
        scorer: &str,
        k1: Option<f32>,
        b: Option<f32>,
        tf_mode: Option<&str>,
    ) -> PyResultX<PyRefMut<'py, Self>> {
        let scorer = parse_text_scorer(scorer, k1, b, tf_mode)?;
        slf.text = Some(TextSpec {
            field,
            query,
            scorer,
        });
        Ok(slf)
    }

    /// Defaults live in Rust: `rerank="accurate"` is the core default of
    /// `Search::vector` (`Rerank::Accurate`).
    #[pyo3(signature = (field, query, rerank="accurate"))]
    fn vector<'py>(
        mut slf: PyRefMut<'py, Self>,
        field: String,
        query: PyReadonlyArray1<'py, f32>,
        rerank: &str,
    ) -> PyResultX<PyRefMut<'py, Self>> {
        let rerank = parse_rerank(rerank)?;
        let q = query
            .as_slice()
            .map_err(|_| invalid("query vector must be a C-contiguous float32 array"))?
            .to_vec();
        slf.vectors.push(VecSpec {
            field,
            query: q,
            rerank,
        });
        Ok(slf)
    }

    /// `k=None` resolves to the core default (`Fusion::default()` = RRF 60).
    #[pyo3(signature = (k=None))]
    fn fuse_rrf(mut slf: PyRefMut<'_, Self>, k: Option<u32>) -> PyRefMut<'_, Self> {
        slf.fusion = match k {
            Some(k) => Fusion::rrf(k),
            None => Fusion::default(),
        };
        slf
    }

    fn fuse_weighted(mut slf: PyRefMut<'_, Self>, text: f32, vector: f32) -> PyRefMut<'_, Self> {
        slf.fusion = Fusion::Weighted { text, vector };
        slf
    }

    fn recency_range(mut slf: PyRefMut<'_, Self>, from: i64, to: i64) -> PyRefMut<'_, Self> {
        slf.recency = Some(Recency::Range { from, to });
        slf
    }

    fn recency_half_life(mut slf: PyRefMut<'_, Self>, days: f64) -> PyRefMut<'_, Self> {
        slf.recency = Some(Recency::HalfLife { days });
        slf
    }

    fn recency_rrf_channel(
        mut slf: PyRefMut<'_, Self>,
        half_life_days: f64,
        weight: f32,
    ) -> PyRefMut<'_, Self> {
        slf.recency = Some(Recency::rrf_channel(half_life_days, weight));
        slf
    }

    fn top_k(mut slf: PyRefMut<'_, Self>, k: usize) -> PyRefMut<'_, Self> {
        slf.top_k = k;
        slf
    }

    fn now(mut slf: PyRefMut<'_, Self>, unix_secs: i64) -> PyRefMut<'_, Self> {
        slf.now = Some(unix_secs);
        slf
    }
}
