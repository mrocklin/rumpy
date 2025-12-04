use pyo3::prelude::*;
use pyo3::types::{PyDict, PySlice, PyTuple};

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

    /// Indexing and slicing.
    fn __getitem__<'py>(&self, py: Python<'py>, key: &Bound<'py, PyAny>) -> PyResult<PyObject> {
        // Handle tuple for multi-dimensional indexing
        if let Ok(tuple) = key.downcast::<PyTuple>() {
            // Check if all indices are integers (scalar access)
            let all_ints = tuple.iter().all(|item| item.extract::<isize>().is_ok());

            if all_ints && tuple.len() == self.inner.ndim() {
                let indices: Vec<usize> = tuple
                    .iter()
                    .enumerate()
                    .map(|(axis, item)| {
                        let idx: isize = item.extract().unwrap();
                        normalize_index(idx, self.inner.shape()[axis])
                    })
                    .collect();
                let val = self.inner.get_element(&indices);
                return Ok(val.into_pyobject(py)?.into_any().unbind());
            }

            // Otherwise, handle slices
            let mut result = self.inner.clone();
            for (axis, item) in tuple.iter().enumerate() {
                if axis >= result.ndim() {
                    return Err(pyo3::exceptions::PyIndexError::new_err(
                        "too many indices for array",
                    ));
                }
                if let Ok(idx) = item.extract::<isize>() {
                    // Integer index: slice to single element, then squeeze later
                    let idx = normalize_index(idx, result.shape()[axis]);
                    result = result.slice_axis(axis, idx as isize, idx as isize + 1, 1);
                } else if let Ok(slice) = item.downcast::<PySlice>() {
                    let (start, stop, step) = extract_slice_indices(&slice, result.shape()[axis])?;
                    result = result.slice_axis(axis, start, stop, step);
                } else {
                    return Err(pyo3::exceptions::PyTypeError::new_err(
                        "indices must be integers or slices",
                    ));
                }
            }
            return Ok(Self::new(result).into_pyobject(py)?.into_any().unbind());
        }

        // Handle single integer index
        if let Ok(idx) = key.extract::<isize>() {
            let idx = normalize_index(idx, self.inner.shape()[0]);
            let result = self.inner.slice_axis(0, idx as isize, idx as isize + 1, 1);
            return Ok(Self::new(result).into_pyobject(py)?.into_any().unbind());
        }

        // Handle single slice
        if let Ok(slice) = key.downcast::<PySlice>() {
            let (start, stop, step) = extract_slice_indices(&slice, self.inner.shape()[0])?;
            let result = self.inner.slice_axis(0, start, stop, step);
            return Ok(Self::new(result).into_pyobject(py)?.into_any().unbind());
        }

        Err(pyo3::exceptions::PyTypeError::new_err(
            "indices must be integers or slices",
        ))
    }

    /// Reshape array. Returns a view if possible.
    fn reshape(&self, shape: Vec<usize>) -> PyResult<Self> {
        self.inner.reshape(shape).map(Self::new).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "cannot reshape array (size mismatch or non-contiguous)",
            )
        })
    }

    /// Transpose the array.
    fn transpose(&self) -> Self {
        Self::new(self.inner.transpose())
    }

    /// Transposed view (same as transpose()).
    #[getter]
    #[allow(non_snake_case)]
    fn T(&self) -> Self {
        self.transpose()
    }
}

/// Parse dtype string to DType enum.
pub fn parse_dtype(s: &str) -> PyResult<DType> {
    DType::from_str(s).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!("Unknown dtype: {}", s))
    })
}

/// Extract start, stop, step from a Python slice object.
fn extract_slice_indices(
    slice: &Bound<'_, PySlice>,
    length: usize,
) -> PyResult<(isize, isize, isize)> {
    let indices = slice.indices(length as isize)?;
    Ok((indices.start as isize, indices.stop as isize, indices.step as isize))
}

/// Normalize a negative index to positive.
fn normalize_index(idx: isize, len: usize) -> usize {
    if idx < 0 {
        (len as isize + idx) as usize
    } else {
        idx as usize
    }
}
