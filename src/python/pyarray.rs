use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PySlice, PyTuple};

use crate::array::{DType, RumpyArray};
use crate::ops::{BinaryOp, ComparisonOp};

/// Parse shape from int, tuple, or list.
pub fn parse_shape(obj: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
    if let Ok(n) = obj.extract::<usize>() {
        return Ok(vec![n]);
    }
    if let Ok(tuple) = obj.downcast::<PyTuple>() {
        return tuple.iter().map(|x| x.extract::<usize>()).collect();
    }
    if let Ok(list) = obj.downcast::<PyList>() {
        return list.iter().map(|x| x.extract::<usize>()).collect();
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "shape must be an int, tuple, or list",
    ))
}

/// Parse reshape arguments: handles reshape(3, 4), reshape((3, 4)), reshape([3, 4]).
fn parse_reshape_args(args: &Bound<'_, PyTuple>) -> PyResult<Vec<usize>> {
    if args.len() == 0 {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "reshape requires at least one argument",
        ));
    }
    // Single argument: delegate to parse_shape
    if args.len() == 1 {
        return parse_shape(&args.get_item(0)?);
    }
    // Multiple arguments: treat as shape dimensions
    args.iter().map(|x| x.extract::<usize>()).collect()
}

/// Result type for reductions that can return scalar or array.
pub enum ReductionResult {
    Scalar(f64),
    Array(PyRumpyArray),
}

impl<'py> IntoPyObject<'py> for ReductionResult {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        match self {
            ReductionResult::Scalar(v) => Ok(v.into_pyobject(py)?.into_any()),
            ReductionResult::Array(arr) => Ok(arr.into_pyobject(py)?.into_any()),
        }
    }
}

/// Check axis is valid for array.
fn check_axis(axis: usize, ndim: usize) -> PyResult<()> {
    if axis >= ndim {
        Err(pyo3::exceptions::PyValueError::new_err(format!(
            "axis {} is out of bounds for array of dimension {}",
            axis, ndim
        )))
    } else {
        Ok(())
    }
}

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
    fn shape<'py>(&self, py: Python<'py>) -> Bound<'py, PyTuple> {
        PyTuple::new(py, self.inner.shape()).unwrap()
    }

    /// Array strides in bytes.
    #[getter]
    fn strides<'py>(&self, py: Python<'py>) -> Bound<'py, PyTuple> {
        PyTuple::new(py, self.inner.strides()).unwrap()
    }

    /// Number of dimensions.
    #[getter]
    fn ndim(&self) -> usize {
        self.inner.ndim()
    }

    /// Data type string (human-readable name).
    #[getter]
    fn dtype(&self) -> &'static str {
        self.inner.dtype().ops().name()
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
        // Handle array indexing (boolean or integer)
        if let Ok(idx_arr) = key.extract::<PyRef<'_, PyRumpyArray>>() {
            if idx_arr.inner.dtype().kind() == crate::array::dtype::DTypeKind::Bool {
                // Boolean indexing
                return self.inner.select_by_mask(&idx_arr.inner)
                    .map(|arr| Self::new(arr).into_pyobject(py).unwrap().into_any().unbind())
                    .ok_or_else(|| {
                        pyo3::exceptions::PyIndexError::new_err(
                            "boolean index shape must match array shape"
                        )
                    });
            } else {
                // Fancy indexing (integer array)
                return self.inner.select_by_indices(&idx_arr.inner)
                    .map(|arr| Self::new(arr).into_pyobject(py).unwrap().into_any().unbind())
                    .ok_or_else(|| {
                        pyo3::exceptions::PyIndexError::new_err(
                            "index out of bounds"
                        )
                    });
            }
        }

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
    /// Accepts reshape(3, 4) or reshape((3, 4)) or reshape([3, 4]).
    #[pyo3(signature = (*args))]
    fn reshape(&self, args: &Bound<'_, PyTuple>) -> PyResult<Self> {
        let shape = parse_reshape_args(args)?;
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

    /// Create a contiguous copy of the array.
    fn copy(&self) -> Self {
        Self::new(self.inner.copy())
    }

    /// Remove single-dimensional entries from the shape.
    fn squeeze(&self) -> Self {
        Self::new(self.inner.squeeze())
    }

    /// Convert array to new dtype.
    fn astype(&self, dtype: &str) -> PyResult<Self> {
        let dt = parse_dtype(dtype)?;
        Ok(Self::new(self.inner.astype(dt)))
    }

    // Binary operations (array op array)

    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        binary_op_dispatch(&self.inner, other, BinaryOp::Add)
    }

    fn __radd__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        binary_op_dispatch(&self.inner, other, BinaryOp::Add)
    }

    fn __sub__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        binary_op_dispatch(&self.inner, other, BinaryOp::Sub)
    }

    fn __rsub__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        rbinary_op_dispatch(&self.inner, other, BinaryOp::Sub)
    }

    fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        binary_op_dispatch(&self.inner, other, BinaryOp::Mul)
    }

    fn __rmul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        binary_op_dispatch(&self.inner, other, BinaryOp::Mul)
    }

    fn __truediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        binary_op_dispatch(&self.inner, other, BinaryOp::Div)
    }

    fn __rtruediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        rbinary_op_dispatch(&self.inner, other, BinaryOp::Div)
    }

    // Unary operations

    fn __neg__(&self) -> Self {
        Self::new(self.inner.neg())
    }

    fn __abs__(&self) -> Self {
        Self::new(self.inner.abs())
    }

    // Comparison operations

    fn __gt__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        comparison_op_dispatch(&self.inner, other, ComparisonOp::Gt)
    }

    fn __lt__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        comparison_op_dispatch(&self.inner, other, ComparisonOp::Lt)
    }

    fn __ge__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        comparison_op_dispatch(&self.inner, other, ComparisonOp::Ge)
    }

    fn __le__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        comparison_op_dispatch(&self.inner, other, ComparisonOp::Le)
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        comparison_op_dispatch(&self.inner, other, ComparisonOp::Eq)
    }

    fn __ne__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        comparison_op_dispatch(&self.inner, other, ComparisonOp::Ne)
    }

    // Reductions

    #[pyo3(signature = (axis=None))]
    fn sum(&self, axis: Option<usize>) -> PyResult<ReductionResult> {
        match axis {
            None => Ok(ReductionResult::Scalar(self.inner.sum())),
            Some(ax) => {
                check_axis(ax, self.inner.ndim())?;
                Ok(ReductionResult::Array(Self::new(self.inner.sum_axis(ax))))
            }
        }
    }

    #[pyo3(signature = (axis=None))]
    fn prod(&self, axis: Option<usize>) -> PyResult<ReductionResult> {
        match axis {
            None => Ok(ReductionResult::Scalar(self.inner.prod())),
            Some(ax) => {
                check_axis(ax, self.inner.ndim())?;
                Ok(ReductionResult::Array(Self::new(self.inner.prod_axis(ax))))
            }
        }
    }

    #[pyo3(signature = (axis=None))]
    fn max(&self, axis: Option<usize>) -> PyResult<ReductionResult> {
        match axis {
            None => Ok(ReductionResult::Scalar(self.inner.max())),
            Some(ax) => {
                check_axis(ax, self.inner.ndim())?;
                Ok(ReductionResult::Array(Self::new(self.inner.max_axis(ax))))
            }
        }
    }

    #[pyo3(signature = (axis=None))]
    fn min(&self, axis: Option<usize>) -> PyResult<ReductionResult> {
        match axis {
            None => Ok(ReductionResult::Scalar(self.inner.min())),
            Some(ax) => {
                check_axis(ax, self.inner.ndim())?;
                Ok(ReductionResult::Array(Self::new(self.inner.min_axis(ax))))
            }
        }
    }

    #[pyo3(signature = (axis=None))]
    fn mean(&self, axis: Option<usize>) -> PyResult<ReductionResult> {
        match axis {
            None => Ok(ReductionResult::Scalar(self.inner.mean())),
            Some(ax) => {
                check_axis(ax, self.inner.ndim())?;
                Ok(ReductionResult::Array(Self::new(self.inner.mean_axis(ax))))
            }
        }
    }

    #[pyo3(signature = (axis=None))]
    fn var(&self, axis: Option<usize>) -> PyResult<ReductionResult> {
        match axis {
            None => Ok(ReductionResult::Scalar(self.inner.var())),
            Some(ax) => {
                check_axis(ax, self.inner.ndim())?;
                Ok(ReductionResult::Array(Self::new(self.inner.var_axis(ax))))
            }
        }
    }

    #[pyo3(signature = (axis=None))]
    fn std(&self, axis: Option<usize>) -> PyResult<ReductionResult> {
        match axis {
            None => Ok(ReductionResult::Scalar(self.inner.std())),
            Some(ax) => {
                check_axis(ax, self.inner.ndim())?;
                Ok(ReductionResult::Array(Self::new(self.inner.std_axis(ax))))
            }
        }
    }

    fn argmax(&self) -> usize {
        self.inner.argmax()
    }

    fn argmin(&self) -> usize {
        self.inner.argmin()
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

/// Dispatch binary operation: array op (array or scalar).
fn binary_op_dispatch(
    arr: &RumpyArray,
    other: &Bound<'_, PyAny>,
    op: BinaryOp,
) -> PyResult<PyRumpyArray> {
    // Try array first
    if let Ok(other_arr) = other.extract::<PyRef<'_, PyRumpyArray>>() {
        arr.binary_op(&other_arr.inner, op)
            .map(PyRumpyArray::new)
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("operands have incompatible shapes")
            })
    } else if let Ok(scalar) = other.extract::<f64>() {
        Ok(PyRumpyArray::new(arr.scalar_op(scalar, op)))
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "operand must be ndarray or number",
        ))
    }
}

/// Dispatch reverse binary operation: scalar op array.
fn rbinary_op_dispatch(
    arr: &RumpyArray,
    other: &Bound<'_, PyAny>,
    op: BinaryOp,
) -> PyResult<PyRumpyArray> {
    if let Ok(scalar) = other.extract::<f64>() {
        Ok(PyRumpyArray::new(arr.rscalar_op(scalar, op)))
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "operand must be a number",
        ))
    }
}

/// Dispatch comparison operation: array cmp (array or scalar).
fn comparison_op_dispatch(
    arr: &RumpyArray,
    other: &Bound<'_, PyAny>,
    op: ComparisonOp,
) -> PyResult<PyRumpyArray> {
    if let Ok(other_arr) = other.extract::<PyRef<'_, PyRumpyArray>>() {
        arr.compare(&other_arr.inner, op)
            .map(PyRumpyArray::new)
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("operands have incompatible shapes")
            })
    } else if let Ok(scalar) = other.extract::<f64>() {
        Ok(PyRumpyArray::new(arr.compare_scalar(scalar, op)))
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "operand must be ndarray or number",
        ))
    }
}
