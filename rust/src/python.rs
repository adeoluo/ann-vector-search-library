//! PyO3 bindings, built by maturin into the `ann_vector_search_rs` extension module.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::{Hnsw as Core, Metric};

/// HNSW index. Levels are supplied by the caller so the Python wrapper can draw them
/// from the same seeded generator as the Python index. `precision` is `"f64"` (exact
/// reproduction of the Python index) or `"f32"` (lane-parallel, approximate sums).
enum Inner {
    F64(Core<f64>),
    F32(Core<f32>),
}

macro_rules! with_inner {
    ($self:expr, $core:ident => $body:expr) => {
        match &$self.inner {
            Inner::F64($core) => $body,
            Inner::F32($core) => $body,
        }
    };
}

#[pyclass(name = "HNSW")]
struct PyHnsw {
    inner: Inner,
}

#[pymethods]
impl PyHnsw {
    #[new]
    #[pyo3(signature = (dim, m, ef_construction, metric, precision="f64"))]
    fn new(
        dim: usize,
        m: usize,
        ef_construction: usize,
        metric: &str,
        precision: &str,
    ) -> PyResult<Self> {
        let metric = Metric::parse(metric).map_err(PyValueError::new_err)?;
        let inner = match precision {
            "f64" => Inner::F64(Core::new(dim, m, ef_construction, metric)),
            "f32" => Inner::F32(Core::new(dim, m, ef_construction, metric)),
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown precision: {other:?} (expected 'f64' or 'f32')"
                )))
            }
        };
        Ok(Self { inner })
    }

    #[getter]
    fn precision(&self) -> &'static str {
        match self.inner {
            Inner::F64(_) => "f64",
            Inner::F32(_) => "f32",
        }
    }

    fn add(&mut self, node_id: i64, vector: Vec<f64>, level: usize) -> PyResult<()> {
        match &mut self.inner {
            Inner::F64(core) => core.add(node_id, &vector, level),
            Inner::F32(core) => core.add(node_id, &vector, level),
        }
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
        py.detach(|| with_inner!(self, core => core.search(&query, k, ef)))
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
        py.detach(|| with_inner!(self, core => core.search_batch(&queries, k, ef)))
            .map_err(PyValueError::new_err)
    }

    #[getter]
    fn distance_count(&self) -> u64 {
        with_inner!(self, core => core.distance_count())
    }

    fn reset_count(&self) {
        with_inner!(self, core => core.reset_count())
    }

    fn __len__(&self) -> usize {
        with_inner!(self, core => core.len())
    }
}

#[pymodule]
fn ann_vector_search_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyHnsw>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
