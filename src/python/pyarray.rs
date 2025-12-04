use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use crate::array::{DType, RumpyArray};

/// Python-visible ndarray class.
#[pyclass(name = "ndarray", module = "rumpy")]
pub struct PyRumpyArray {
    pub(crate) inner: RumpyArray,
}

impl PyRumpyArray {
    pub fn new(inner: RumpyArray) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyRumpyArray {
    /// Array shape as tuple.
    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.inner.shape().to_vec()
    }

    /// Array strides in bytes.
    #[getter]
    fn strides(&self) -> Vec<isize> {
        self.inner.strides().to_vec()
    }

    /// Number of dimensions.
    #[getter]
    fn ndim(&self) -> usize {
        self.inner.ndim()
    }

    /// Data type string.
    #[getter]
    fn dtype(&self) -> &'static str {
        self.inner.dtype().typestr()
    }

    /// Total number of elements.
    #[getter]
    fn size(&self) -> usize {
        self.inner.size()
    }

    /// Size of one element in bytes.
    #[getter]
    fn itemsize(&self) -> usize {
        self.inner.itemsize()
    }

    /// Total size in bytes.
    #[getter]
    fn nbytes(&self) -> usize {
        self.inner.nbytes()
    }

    /// NumPy array interface for zero-copy interop.
    #[getter]
    fn __array_interface__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        // Shape and strides must be tuples for numpy
        let shape_tuple = PyTuple::new(py, self.inner.shape())?;
        let strides_tuple = PyTuple::new(py, self.inner.strides())?;
        dict.set_item("shape", shape_tuple)?;
        dict.set_item("typestr", self.inner.dtype().typestr())?;
        let data_ptr = self.inner.data_ptr() as usize;
        dict.set_item("data", (data_ptr, false))?; // (ptr, readonly=false)
        dict.set_item("strides", strides_tuple)?;
        dict.set_item("version", 3)?;
        Ok(dict)
    }

    fn __repr__(&self) -> String {
        format!(
            "rumpy.ndarray(shape={:?}, dtype='{}')",
            self.inner.shape(),
            self.inner.dtype().typestr()
        )
    }
}

/// Parse dtype string to DType enum.
pub fn parse_dtype(s: &str) -> PyResult<DType> {
    DType::from_str(s).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!("Unknown dtype: {}", s))
    })
}
