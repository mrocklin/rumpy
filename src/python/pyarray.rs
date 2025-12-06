use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PySlice, PyTuple};

use crate::array::{DType, RumpyArray};
use crate::ops::matmul::matmul;
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
/// Allows -1 for one dimension to be inferred.
fn parse_reshape_args_isize(args: &Bound<'_, PyTuple>) -> PyResult<Vec<isize>> {
    if args.len() == 0 {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "reshape requires at least one argument",
        ));
    }
    // Single argument: could be int, tuple, or list
    if args.len() == 1 {
        let obj = args.get_item(0)?;
        if let Ok(n) = obj.extract::<isize>() {
            return Ok(vec![n]);
        }
        if let Ok(tuple) = obj.downcast::<PyTuple>() {
            return tuple.iter().map(|x| x.extract::<isize>()).collect();
        }
        if let Ok(list) = obj.downcast::<PyList>() {
            return list.iter().map(|x| x.extract::<isize>()).collect();
        }
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "shape must be an int, tuple, or list",
        ));
    }
    // Multiple arguments: treat as shape dimensions
    args.iter().map(|x| x.extract::<isize>()).collect()
}

/// Resolve -1 in shape given total size.
fn resolve_reshape_shape(shape: Vec<isize>, size: usize) -> PyResult<Vec<usize>> {
    let mut neg_idx: Option<usize> = None;
    let mut known_product: usize = 1;

    for (i, &dim) in shape.iter().enumerate() {
        if dim < -1 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "negative dimensions not allowed (except -1)",
            ));
        } else if dim == -1 {
            if neg_idx.is_some() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "can only specify one unknown dimension (-1)",
                ));
            }
            neg_idx = Some(i);
        } else {
            known_product *= dim as usize;
        }
    }

    let mut result: Vec<usize> = shape.iter().map(|&d| d as usize).collect();

    if let Some(idx) = neg_idx {
        if known_product == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "cannot reshape with zero-sized known dimensions and -1",
            ));
        }
        if size % known_product != 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "cannot reshape array: size not divisible",
            ));
        }
        result[idx] = size / known_product;
    }

    Ok(result)
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

/// Apply keepdims to a reduction result.
fn apply_keepdims(arr: RumpyArray, axis: usize) -> RumpyArray {
    // Insert a dimension of size 1 at the reduced axis position
    arr.expand_dims(axis).unwrap_or(arr)
}

/// Helper for scalar reductions with keepdims support.
fn scalar_reduction_with_keepdims(
    arr: &RumpyArray,
    value: f64,
    keepdims: bool,
    dtype: DType,
) -> ReductionResult {
    if keepdims {
        let ones_shape = vec![1; arr.ndim()];
        let mut result = RumpyArray::zeros(ones_shape, dtype.clone());
        let buffer = result.buffer_mut();
        let result_buffer = std::sync::Arc::get_mut(buffer).expect("buffer must be unique");
        let ptr = result_buffer.as_mut_ptr();
        unsafe { dtype.ops().write_f64(ptr, 0, value); }
        ReductionResult::Array(PyRumpyArray::new(result))
    } else {
        ReductionResult::Scalar(value)
    }
}

/// Helper for axis reductions with keepdims support.
fn axis_reduction_with_keepdims(
    result: RumpyArray,
    axis: usize,
    keepdims: bool,
) -> ReductionResult {
    if keepdims {
        ReductionResult::Array(PyRumpyArray::new(apply_keepdims(result, axis)))
    } else {
        ReductionResult::Array(PyRumpyArray::new(result))
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
        self.inner.format_repr()
    }

    fn __str__(&self) -> String {
        self.inner.format_str()
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
            // Check for None (newaxis) in tuple - if present, we can't do scalar access
            let has_none = tuple.iter().any(|item| item.is_none());

            // Check if all indices are integers (scalar access) - only if no None
            let all_ints = !has_none && tuple.iter().all(|item| item.extract::<isize>().is_ok());

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

            // Otherwise, handle slices and None (newaxis)
            let mut result = self.inner.clone();
            let mut axis = 0usize;  // Track current axis in result array

            for item in tuple.iter() {
                // Handle None (newaxis) - insert new axis
                if item.is_none() {
                    result = result.expand_dims(axis).ok_or_else(|| {
                        pyo3::exceptions::PyIndexError::new_err("cannot expand dims")
                    })?;
                    axis += 1;  // Move past the newly inserted axis
                    continue;
                }

                if axis >= result.ndim() {
                    return Err(pyo3::exceptions::PyIndexError::new_err(
                        "too many indices for array",
                    ));
                }
                if let Ok(idx) = item.extract::<isize>() {
                    // Integer index: slice to single element, then squeeze later
                    let idx = normalize_index(idx, result.shape()[axis]);
                    result = result.slice_axis(axis, idx as isize, idx as isize + 1, 1);
                    axis += 1;
                } else if let Ok(slice) = item.downcast::<PySlice>() {
                    let (start, stop, step) = extract_slice_indices(&slice, result.shape()[axis])?;
                    result = result.slice_axis(axis, start, stop, step);
                    axis += 1;
                } else {
                    return Err(pyo3::exceptions::PyTypeError::new_err(
                        "indices must be integers, slices, or None",
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
    /// Use -1 for one dimension to be automatically inferred.
    #[pyo3(signature = (*args))]
    fn reshape(&self, args: &Bound<'_, PyTuple>) -> PyResult<Self> {
        let shape_isize = parse_reshape_args_isize(args)?;
        let shape = resolve_reshape_shape(shape_isize, self.inner.size())?;
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

    /// Real part of the array.
    #[getter]
    fn real(&self) -> Self {
        Self::new(self.inner.real())
    }

    /// Imaginary part of the array.
    #[getter]
    fn imag(&self) -> Self {
        Self::new(self.inner.imag())
    }

    /// Complex conjugate of the array.
    fn conj(&self) -> Self {
        Self::new(self.inner.conj())
    }

    /// Extract diagonal from 2D array.
    fn diagonal(&self) -> Self {
        Self::new(self.inner.diagonal())
    }

    /// Sum of diagonal elements.
    fn trace(&self) -> f64 {
        self.inner.trace()
    }

    /// Swap two axes.
    fn swapaxes(&self, axis1: usize, axis2: usize) -> Self {
        Self::new(self.inner.swapaxes(axis1, axis2))
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

    fn __pow__(&self, other: &Bound<'_, PyAny>, _modulo: &Bound<'_, PyAny>) -> PyResult<Self> {
        binary_op_dispatch(&self.inner, other, BinaryOp::Pow)
    }

    fn __rpow__(&self, other: &Bound<'_, PyAny>, _modulo: &Bound<'_, PyAny>) -> PyResult<Self> {
        rbinary_op_dispatch(&self.inner, other, BinaryOp::Pow)
    }

    fn __mod__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        binary_op_dispatch(&self.inner, other, BinaryOp::Mod)
    }

    fn __rmod__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        rbinary_op_dispatch(&self.inner, other, BinaryOp::Mod)
    }

    fn __floordiv__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        binary_op_dispatch(&self.inner, other, BinaryOp::FloorDiv)
    }

    fn __rfloordiv__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        rbinary_op_dispatch(&self.inner, other, BinaryOp::FloorDiv)
    }

    fn __matmul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(other_arr) = other.extract::<PyRef<'_, PyRumpyArray>>() {
            matmul(&self.inner, &other_arr.inner)
                .map(Self::new)
                .ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err("matmul: incompatible shapes")
                })
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "matmul operand must be ndarray",
            ))
        }
    }

    fn __rmatmul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(other_arr) = other.extract::<PyRef<'_, PyRumpyArray>>() {
            matmul(&other_arr.inner, &self.inner)
                .map(Self::new)
                .ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err("matmul: incompatible shapes")
                })
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "matmul operand must be ndarray",
            ))
        }
    }

    // Unary operations

    fn __neg__(&self) -> Self {
        Self::new(self.inner.neg().expect("neg always succeeds"))
    }

    fn __abs__(&self) -> Self {
        Self::new(self.inner.abs().expect("abs always succeeds"))
    }

    fn __float__(&self) -> PyResult<f64> {
        if self.inner.size() != 1 {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "only size-1 arrays can be converted to Python scalars",
            ));
        }
        Ok(self.inner.get_element(&vec![0; self.inner.ndim()]))
    }

    fn __int__(&self) -> PyResult<i64> {
        if self.inner.size() != 1 {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "only size-1 arrays can be converted to Python scalars",
            ));
        }
        Ok(self.inner.get_element(&vec![0; self.inner.ndim()]) as i64)
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

    #[pyo3(signature = (axis=None, keepdims=false))]
    fn sum(&self, axis: Option<usize>, keepdims: bool) -> PyResult<ReductionResult> {
        match axis {
            None => Ok(scalar_reduction_with_keepdims(&self.inner, self.inner.sum(), keepdims, self.inner.dtype())),
            Some(ax) => {
                check_axis(ax, self.inner.ndim())?;
                Ok(axis_reduction_with_keepdims(self.inner.sum_axis(ax), ax, keepdims))
            }
        }
    }

    #[pyo3(signature = (axis=None, keepdims=false))]
    fn prod(&self, axis: Option<usize>, keepdims: bool) -> PyResult<ReductionResult> {
        match axis {
            None => Ok(scalar_reduction_with_keepdims(&self.inner, self.inner.prod(), keepdims, self.inner.dtype())),
            Some(ax) => {
                check_axis(ax, self.inner.ndim())?;
                Ok(axis_reduction_with_keepdims(self.inner.prod_axis(ax), ax, keepdims))
            }
        }
    }

    #[pyo3(signature = (axis=None, keepdims=false))]
    fn max(&self, axis: Option<usize>, keepdims: bool) -> PyResult<ReductionResult> {
        match axis {
            None => Ok(scalar_reduction_with_keepdims(&self.inner, self.inner.max(), keepdims, self.inner.dtype())),
            Some(ax) => {
                check_axis(ax, self.inner.ndim())?;
                Ok(axis_reduction_with_keepdims(self.inner.max_axis(ax), ax, keepdims))
            }
        }
    }

    #[pyo3(signature = (axis=None, keepdims=false))]
    fn min(&self, axis: Option<usize>, keepdims: bool) -> PyResult<ReductionResult> {
        match axis {
            None => Ok(scalar_reduction_with_keepdims(&self.inner, self.inner.min(), keepdims, self.inner.dtype())),
            Some(ax) => {
                check_axis(ax, self.inner.ndim())?;
                Ok(axis_reduction_with_keepdims(self.inner.min_axis(ax), ax, keepdims))
            }
        }
    }

    #[pyo3(signature = (axis=None, keepdims=false))]
    fn mean(&self, axis: Option<usize>, keepdims: bool) -> PyResult<ReductionResult> {
        match axis {
            None => Ok(scalar_reduction_with_keepdims(&self.inner, self.inner.mean(), keepdims, self.inner.dtype())),
            Some(ax) => {
                check_axis(ax, self.inner.ndim())?;
                Ok(axis_reduction_with_keepdims(self.inner.mean_axis(ax), ax, keepdims))
            }
        }
    }

    #[pyo3(signature = (axis=None, keepdims=false))]
    fn var(&self, axis: Option<usize>, keepdims: bool) -> PyResult<ReductionResult> {
        match axis {
            None => Ok(scalar_reduction_with_keepdims(&self.inner, self.inner.var(), keepdims, self.inner.dtype())),
            Some(ax) => {
                check_axis(ax, self.inner.ndim())?;
                Ok(axis_reduction_with_keepdims(self.inner.var_axis(ax), ax, keepdims))
            }
        }
    }

    #[pyo3(signature = (axis=None, keepdims=false))]
    fn std(&self, axis: Option<usize>, keepdims: bool) -> PyResult<ReductionResult> {
        match axis {
            None => Ok(scalar_reduction_with_keepdims(&self.inner, self.inner.std(), keepdims, self.inner.dtype())),
            Some(ax) => {
                check_axis(ax, self.inner.ndim())?;
                Ok(axis_reduction_with_keepdims(self.inner.std_axis(ax), ax, keepdims))
            }
        }
    }

    #[pyo3(signature = (axis=None))]
    fn argmax(&self, axis: Option<usize>) -> PyResult<ReductionResult> {
        match axis {
            None => Ok(ReductionResult::Scalar(self.inner.argmax() as f64)),
            Some(ax) => {
                check_axis(ax, self.inner.ndim())?;
                Ok(ReductionResult::Array(Self::new(self.inner.argmax_axis(ax))))
            }
        }
    }

    #[pyo3(signature = (axis=None))]
    fn argmin(&self, axis: Option<usize>) -> PyResult<ReductionResult> {
        match axis {
            None => Ok(ReductionResult::Scalar(self.inner.argmin() as f64)),
            Some(ax) => {
                check_axis(ax, self.inner.ndim())?;
                Ok(ReductionResult::Array(Self::new(self.inner.argmin_axis(ax))))
            }
        }
    }

    /// Dot product with numpy semantics.
    fn dot(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(other_arr) = other.extract::<PyRef<'_, PyRumpyArray>>() {
            crate::ops::dot::dot(&self.inner, &other_arr.inner)
                .map(Self::new)
                .ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err("dot: incompatible shapes")
                })
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "dot operand must be ndarray",
            ))
        }
    }

    /// Return a flattened copy of the array.
    fn flatten(&self) -> Self {
        let size = self.inner.size();
        Self::new(self.inner.copy().reshape(vec![size]).unwrap())
    }

    /// Return a flattened array (view if possible, copy otherwise).
    fn ravel(&self) -> Self {
        let size = self.inner.size();
        // Try to return a view, fall back to copy
        if let Some(view) = self.inner.reshape(vec![size]) {
            Self::new(view)
        } else {
            self.flatten()
        }
    }

    /// Return the array as a (possibly nested) Python list.
    fn tolist<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        self.inner.to_pylist(py)
    }

    /// Extract a scalar from a size-1 array.
    fn item(&self) -> PyResult<f64> {
        if self.inner.size() != 1 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "can only convert an array of size 1 to a Python scalar",
            ));
        }
        Ok(self.inner.get_element(&vec![0; self.inner.ndim()]))
    }

    /// Test if all elements evaluate to True.
    #[pyo3(signature = (axis=None, keepdims=false))]
    fn all(&self, axis: Option<usize>, keepdims: bool) -> PyResult<ReductionResult> {
        let val = if self.inner.all() { 1.0 } else { 0.0 };
        match axis {
            None => Ok(scalar_reduction_with_keepdims(&self.inner, val, keepdims, DType::bool())),
            Some(ax) => {
                check_axis(ax, self.inner.ndim())?;
                Ok(axis_reduction_with_keepdims(self.inner.all_axis(ax), ax, keepdims))
            }
        }
    }

    /// Test if any element evaluates to True.
    #[pyo3(signature = (axis=None, keepdims=false))]
    fn any(&self, axis: Option<usize>, keepdims: bool) -> PyResult<ReductionResult> {
        let val = if self.inner.any() { 1.0 } else { 0.0 };
        match axis {
            None => Ok(scalar_reduction_with_keepdims(&self.inner, val, keepdims, DType::bool())),
            Some(ax) => {
                check_axis(ax, self.inner.ndim())?;
                Ok(axis_reduction_with_keepdims(self.inner.any_axis(ax), ax, keepdims))
            }
        }
    }

    /// Clip values to a range.
    #[pyo3(signature = (a_min=None, a_max=None))]
    fn clip(&self, a_min: Option<f64>, a_max: Option<f64>) -> Self {
        Self::new(self.inner.clip(a_min, a_max))
    }

    /// Round to the given number of decimals.
    #[pyo3(signature = (decimals=0))]
    fn round(&self, decimals: i32) -> PyResult<Self> {
        Ok(Self::new(self.inner.round(decimals)))
    }

    /// Cumulative sum along axis (or flattened if axis is None).
    #[pyo3(signature = (axis=None))]
    fn cumsum(&self, axis: Option<usize>) -> PyResult<Self> {
        if let Some(ax) = axis {
            if ax >= self.inner.ndim() {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "axis {} is out of bounds for array of dimension {}",
                    ax, self.inner.ndim()
                )));
            }
        }
        Ok(Self::new(self.inner.cumsum(axis)))
    }

    /// Cumulative product along axis (or flattened if axis is None).
    #[pyo3(signature = (axis=None))]
    fn cumprod(&self, axis: Option<usize>) -> PyResult<Self> {
        if let Some(ax) = axis {
            if ax >= self.inner.ndim() {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "axis {} is out of bounds for array of dimension {}",
                    ax, self.inner.ndim()
                )));
            }
        }
        Ok(Self::new(self.inner.cumprod(axis)))
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
        use crate::ops::BinaryOpError;
        arr.binary_op(&other_arr.inner, op)
            .map(PyRumpyArray::new)
            .map_err(|e| match e {
                BinaryOpError::ShapeMismatch => {
                    pyo3::exceptions::PyValueError::new_err("operands have incompatible shapes")
                }
                BinaryOpError::UnsupportedDtype => {
                    pyo3::exceptions::PyTypeError::new_err("operation not supported for these dtypes")
                }
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
