//! PyO3 bindings, built by maturin into the `ann_vector_search_rs` extension module.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::{Hnsw as Core, Metric};

/// HNSW index. Levels are supplied by the caller so the Python wrapper can draw them
/// from the same seeded generator as the pure-Python index.
#[pyclass(name = "HNSW")]
struct PyHnsw {
    inner: Core,
}

#[pymethods]
impl PyHnsw {
    #[new]
    #[pyo3(signature = (dim, m, ef_construction, metric))]
    fn new(dim: usize, m: usize, ef_construction: usize, metric: &str) -> PyResult<Self> {
        let metric = Metric::parse(metric).map_err(PyValueError::new_err)?;
        Ok(Self {
            inner: Core::new(dim, m, ef_construction, metric),
        })
    }

    fn add(&mut self, node_id: i64, vector: Vec<f64>, level: usize) -> PyResult<()> {
        self.inner
            .add(node_id, vector, level)
            .map_err(PyValueError::new_err)
    }

    #[pyo3(signature = (query, k, ef=None))]
    fn search(
        &self,
        py: Python<'_>,
        query: Vec<f64>,
        k: usize,
        ef: Option<usize>,
    ) -> PyResult<Vec<(i64, f64)>> {
        py.detach(|| self.inner.search(&query, k, ef))
            .map_err(PyValueError::new_err)
    }

    /// Search many queries in parallel with the GIL released.
    #[pyo3(signature = (queries, k, ef=None))]
    fn search_batch(
        &self,
        py: Python<'_>,
        queries: Vec<Vec<f64>>,
        k: usize,
        ef: Option<usize>,
    ) -> PyResult<Vec<Vec<(i64, f64)>>> {
        py.detach(|| self.inner.search_batch(&queries, k, ef))
            .map_err(PyValueError::new_err)
    }

    #[getter]
    fn distance_count(&self) -> u64 {
        self.inner.distance_count()
    }

    fn reset_count(&self) {
        self.inner.reset_count();
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }
}

#[pymodule]
fn ann_vector_search_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyHnsw>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
