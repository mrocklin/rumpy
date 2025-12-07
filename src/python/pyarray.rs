use pyo3::prelude::*;
use pyo3::types::{PyDict, PyEllipsis, PyList, PySlice, PyString, PyTuple};

use crate::array::{promote_dtype, DType, RumpyArray};
use crate::ops::matmul::matmul;
use crate::ops::{map_binary_op_inplace, BinaryOp, ComparisonOp};

/// Minimum array size (in elements) for temporary elision optimization.
/// Arrays smaller than this won't have their buffers reused in-place.
/// 32768 f64 elements = 256KB, matching NumPy's threshold.
const ELISION_SIZE_THRESHOLD: usize = 32768;

/// Get Python reference count for an object via raw pointer.
/// Used to detect ephemeral intermediates that can be modified in-place.
#[inline]
fn py_refcount_ptr(ptr: *mut pyo3::ffi::PyObject) -> isize {
    unsafe { pyo3::ffi::Py_REFCNT(ptr) }
}

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
        if !size.is_multiple_of(known_product) {
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

    /// NumPy ufunc protocol - tells numpy to defer operations to us.
    /// This makes `np.float64(2.0) * rp.array([1,2,3])` return a rumpy array.
    #[pyo3(signature = (ufunc, method, *inputs, **kwargs))]
    fn __array_ufunc__<'py>(
        &self,
        py: Python<'py>,
        ufunc: &Bound<'py, PyAny>,
        method: &Bound<'py, PyString>,
        inputs: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<PyObject> {
        // Only handle __call__ method
        let method_str: String = method.extract()?;
        if method_str != "__call__" {
            return Ok(py.NotImplemented());
        }

        // Check for 'out' kwarg - we don't support it yet
        if let Some(kw) = kwargs {
            if kw.contains("out")? {
                return Ok(py.NotImplemented());
            }
        }

        // Get ufunc name and look up corresponding function in rumpy module
        let ufunc_name: String = ufunc.getattr("__name__")?.extract()?;

        // Map numpy names to rumpy names where they differ
        let rumpy_name = match ufunc_name.as_str() {
            "true_divide" => "divide",
            "absolute" => "abs",
            "negative" => "negative",
            name => name,
        };

        let rumpy_module = py.import("rumpy")?;
        let our_func = match rumpy_module.getattr(rumpy_name) {
            Ok(f) => f,
            Err(_) => return Ok(py.NotImplemented()),
        };

        // Call our function with the inputs
        match kwargs {
            Some(kw) => our_func.call(inputs, Some(kw)),
            None => our_func.call1(inputs),
        }.map(|r| r.unbind())
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

        // Handle Python list indexing (convert to array for fancy indexing)
        if let Ok(list) = key.downcast::<PyList>() {
            let indices: Vec<i64> = list.iter()
                .map(|x| x.extract::<i64>())
                .collect::<PyResult<Vec<_>>>()?;
            let idx_arr = RumpyArray::from_slice_i64(&indices);
            return self.inner.select_by_indices(&idx_arr)
                .map(|arr| Self::new(arr).into_pyobject(py).unwrap().into_any().unbind())
                .ok_or_else(|| {
                    pyo3::exceptions::PyIndexError::new_err("index out of bounds")
                });
        }

        // Handle single None (newaxis)
        if key.is_none() {
            let result = self.inner.expand_dims(0).ok_or_else(|| {
                pyo3::exceptions::PyIndexError::new_err("cannot expand dims")
            })?;
            return Ok(Self::new(result).into_pyobject(py)?.into_any().unbind());
        }

        // Handle single ellipsis
        if key.downcast::<PyEllipsis>().is_ok() {
            return Ok(Self::new(self.inner.clone()).into_pyobject(py)?.into_any().unbind());
        }

        // Handle tuple for multi-dimensional indexing
        if let Ok(tuple) = key.downcast::<PyTuple>() {
            // Pre-process: expand ellipsis and parse all items
            let items = expand_ellipsis(&tuple, self.inner.ndim())?;

            // Fast path: all integers and matches ndim -> scalar access
            if items.len() == self.inner.ndim() &&
               items.iter().all(|i| matches!(i, IndexItem::Int(_))) {
                let indices: Vec<usize> = items.iter()
                    .enumerate()
                    .map(|(axis, item)| {
                        if let IndexItem::Int(idx) = item {
                            normalize_index(*idx, self.inner.shape()[axis])
                        } else {
                            unreachable!()
                        }
                    })
                    .collect();
                let val = self.inner.get_element(&indices);
                return Ok(val.into_pyobject(py)?.into_any().unbind());
            }

            // Process indices, tracking which axes had integer indices (to squeeze)
            let mut result = self.inner.clone();
            let mut axis = 0usize;
            let mut squeeze_axes: Vec<usize> = Vec::new();

            for item in items.iter() {
                match item {
                    IndexItem::NewAxis => {
                        result = result.expand_dims(axis).ok_or_else(|| {
                            pyo3::exceptions::PyIndexError::new_err("cannot expand dims")
                        })?;
                        axis += 1;
                    }
                    IndexItem::Int(idx) => {
                        if axis >= result.ndim() {
                            return Err(pyo3::exceptions::PyIndexError::new_err(
                                "too many indices for array",
                            ));
                        }
                        let idx = normalize_index(*idx, result.shape()[axis]);
                        result = result.slice_axis(axis, idx as isize, idx as isize + 1, 1);
                        squeeze_axes.push(axis);
                        axis += 1;
                    }
                    IndexItem::Slice(start, stop, step) => {
                        if axis >= result.ndim() {
                            return Err(pyo3::exceptions::PyIndexError::new_err(
                                "too many indices for array",
                            ));
                        }
                        let len = result.shape()[axis] as isize;
                        let (norm_start, norm_stop) = normalize_slice(*start, *stop, *step, len);
                        result = result.slice_axis(axis, norm_start, norm_stop, *step);
                        axis += 1;
                    }
                }
            }

            // Squeeze out axes that had integer indices (in reverse order to preserve indices)
            for &ax in squeeze_axes.iter().rev() {
                result = result.squeeze_axis(ax);
            }

            return Ok(Self::new(result).into_pyobject(py)?.into_any().unbind());
        }

        // Handle single integer index
        if let Ok(idx) = key.extract::<isize>() {
            let idx = normalize_index(idx, self.inner.shape()[0]);
            let result = self.inner.slice_axis(0, idx as isize, idx as isize + 1, 1);
            let result = result.squeeze_axis(0);
            return Ok(Self::new(result).into_pyobject(py)?.into_any().unbind());
        }

        // Handle single slice
        if let Ok(slice) = key.downcast::<PySlice>() {
            let (start, stop, step) = extract_slice_indices(slice, self.inner.shape()[0])?;
            let result = self.inner.slice_axis(0, start, stop, step);
            return Ok(Self::new(result).into_pyobject(py)?.into_any().unbind());
        }

        Err(pyo3::exceptions::PyTypeError::new_err(
            "indices must be integers, slices, or None",
        ))
    }

    /// Item assignment (slice or element).
    fn __setitem__(&mut self, key: &Bound<'_, PyAny>, value: &Bound<'_, PyAny>) -> PyResult<()> {
        // Extract value as scalar or array
        let value_scalar = value.extract::<f64>().ok();
        let value_array = value.extract::<PyRef<'_, PyRumpyArray>>().ok();

        // Handle tuple for multi-dimensional indexing
        if let Ok(tuple) = key.downcast::<PyTuple>() {
            // Check if all indices are integers (single element assignment)
            let all_ints = tuple.iter().all(|item| item.extract::<isize>().is_ok());

            if all_ints && tuple.len() == self.inner.ndim() {
                // Single element assignment
                let indices: Vec<usize> = tuple
                    .iter()
                    .enumerate()
                    .map(|(axis, item)| {
                        let idx: isize = item.extract().unwrap();
                        normalize_index(idx, self.inner.shape()[axis])
                    })
                    .collect();

                if let Some(scalar) = value_scalar {
                    self.inner.set_element(&indices, scalar);
                    return Ok(());
                } else {
                    return Err(pyo3::exceptions::PyTypeError::new_err(
                        "can only assign scalar to single element",
                    ));
                }
            }

            // Slice assignment - get view and fill
            let mut view = self.inner.clone();
            let mut axis = 0usize;

            for item in tuple.iter() {
                if axis >= view.ndim() {
                    return Err(pyo3::exceptions::PyIndexError::new_err(
                        "too many indices for array",
                    ));
                }
                if let Ok(idx) = item.extract::<isize>() {
                    let idx = normalize_index(idx, view.shape()[axis]);
                    view = view.slice_axis(axis, idx as isize, idx as isize + 1, 1);
                    axis += 1;
                } else if let Ok(slice) = item.downcast::<PySlice>() {
                    let (start, stop, step) = extract_slice_indices(slice, view.shape()[axis])?;
                    view = view.slice_axis(axis, start, stop, step);
                    axis += 1;
                } else {
                    return Err(pyo3::exceptions::PyTypeError::new_err(
                        "indices must be integers or slices",
                    ));
                }
            }

            return fill_view(&mut view, value_scalar, value_array.as_deref());
        }

        // Handle single integer index (row assignment for 2D+, or single element for 1D)
        if let Ok(idx) = key.extract::<isize>() {
            let idx = normalize_index(idx, self.inner.shape()[0]);

            if self.inner.ndim() == 1 {
                // Single element in 1D array
                if let Some(scalar) = value_scalar {
                    self.inner.set_element(&[idx], scalar);
                    return Ok(());
                } else {
                    return Err(pyo3::exceptions::PyTypeError::new_err(
                        "can only assign scalar to single element",
                    ));
                }
            } else {
                // Row/slice assignment
                let mut view = self.inner.slice_axis(0, idx as isize, idx as isize + 1, 1);
                return fill_view(&mut view, value_scalar, value_array.as_deref());
            }
        }

        // Handle single slice
        if let Ok(slice) = key.downcast::<PySlice>() {
            let (start, stop, step) = extract_slice_indices(slice, self.inner.shape()[0])?;
            let mut view = self.inner.slice_axis(0, start, stop, step);
            return fill_view(&mut view, value_scalar, value_array.as_deref());
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
    // Uses elision-aware dispatch for potential in-place buffer reuse

    fn __add__(slf: PyRef<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        binary_op_dispatch_with_elision(slf.as_ptr(), &slf.inner, other, BinaryOp::Add)
    }

    fn __radd__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        // Reverse ops don't benefit from elision (self is the non-ephemeral one)
        binary_op_dispatch(&self.inner, other, BinaryOp::Add)
    }

    fn __sub__(slf: PyRef<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        binary_op_dispatch_with_elision(slf.as_ptr(), &slf.inner, other, BinaryOp::Sub)
    }

    fn __rsub__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        rbinary_op_dispatch(&self.inner, other, BinaryOp::Sub)
    }

    fn __mul__(slf: PyRef<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        binary_op_dispatch_with_elision(slf.as_ptr(), &slf.inner, other, BinaryOp::Mul)
    }

    fn __rmul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        binary_op_dispatch(&self.inner, other, BinaryOp::Mul)
    }

    fn __truediv__(slf: PyRef<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        binary_op_dispatch_with_elision(slf.as_ptr(), &slf.inner, other, BinaryOp::Div)
    }

    fn __rtruediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        rbinary_op_dispatch(&self.inner, other, BinaryOp::Div)
    }

    fn __pow__(slf: PyRef<'_, Self>, other: &Bound<'_, PyAny>, _modulo: &Bound<'_, PyAny>) -> PyResult<Self> {
        binary_op_dispatch_with_elision(slf.as_ptr(), &slf.inner, other, BinaryOp::Pow)
    }

    fn __rpow__(&self, other: &Bound<'_, PyAny>, _modulo: &Bound<'_, PyAny>) -> PyResult<Self> {
        rbinary_op_dispatch(&self.inner, other, BinaryOp::Pow)
    }

    fn __mod__(slf: PyRef<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        binary_op_dispatch_with_elision(slf.as_ptr(), &slf.inner, other, BinaryOp::Mod)
    }

    fn __rmod__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        rbinary_op_dispatch(&self.inner, other, BinaryOp::Mod)
    }

    fn __floordiv__(slf: PyRef<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        binary_op_dispatch_with_elision(slf.as_ptr(), &slf.inner, other, BinaryOp::FloorDiv)
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

    // Bitwise operations

    fn __and__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        bitwise_binary_op_dispatch(&self.inner, other, crate::ops::bitwise_and)
    }

    fn __rand__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        // AND is commutative, so order doesn't matter
        rbitwise_binary_op_dispatch(&self.inner, other, crate::ops::bitwise_and)
    }

    fn __or__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        bitwise_binary_op_dispatch(&self.inner, other, crate::ops::bitwise_or)
    }

    fn __ror__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        // OR is commutative, so order doesn't matter
        rbitwise_binary_op_dispatch(&self.inner, other, crate::ops::bitwise_or)
    }

    fn __xor__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        bitwise_binary_op_dispatch(&self.inner, other, crate::ops::bitwise_xor)
    }

    fn __rxor__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        // XOR is commutative, so order doesn't matter
        rbitwise_binary_op_dispatch(&self.inner, other, crate::ops::bitwise_xor)
    }

    fn __invert__(&self) -> PyResult<Self> {
        crate::ops::bitwise_not(&self.inner)
            .map(Self::new)
            .ok_or_else(|| pyo3::exceptions::PyTypeError::new_err("bitwise_not not supported for this dtype"))
    }

    fn __lshift__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        bitwise_binary_op_dispatch(&self.inner, other, crate::ops::left_shift)
    }

    fn __rlshift__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        rbitwise_binary_op_dispatch(&self.inner, other, crate::ops::left_shift)
    }

    fn __rshift__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        bitwise_binary_op_dispatch(&self.inner, other, crate::ops::right_shift)
    }

    fn __rrshift__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        rbitwise_binary_op_dispatch(&self.inner, other, crate::ops::right_shift)
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

    /// Central moment of order k.
    #[pyo3(signature = (k, axis=None, keepdims=false))]
    fn moment(&self, k: usize, axis: Option<usize>, keepdims: bool) -> PyResult<ReductionResult> {
        match axis {
            None => Ok(scalar_reduction_with_keepdims(&self.inner, self.inner.moment(k), keepdims, self.inner.dtype())),
            Some(ax) => {
                check_axis(ax, self.inner.ndim())?;
                Ok(axis_reduction_with_keepdims(self.inner.moment_axis(k, ax), ax, keepdims))
            }
        }
    }

    /// Skewness (Fisher's definition).
    #[pyo3(signature = (axis=None, keepdims=false))]
    fn skew(&self, axis: Option<usize>, keepdims: bool) -> PyResult<ReductionResult> {
        match axis {
            None => Ok(scalar_reduction_with_keepdims(&self.inner, self.inner.skew(), keepdims, self.inner.dtype())),
            Some(ax) => {
                check_axis(ax, self.inner.ndim())?;
                Ok(axis_reduction_with_keepdims(self.inner.skew_axis(ax), ax, keepdims))
            }
        }
    }

    /// Kurtosis (Fisher's definition: excess kurtosis, normal = 0).
    #[pyo3(signature = (axis=None, keepdims=false))]
    fn kurtosis(&self, axis: Option<usize>, keepdims: bool) -> PyResult<ReductionResult> {
        match axis {
            None => Ok(scalar_reduction_with_keepdims(&self.inner, self.inner.kurtosis(), keepdims, self.inner.dtype())),
            Some(ax) => {
                check_axis(ax, self.inner.ndim())?;
                Ok(axis_reduction_with_keepdims(self.inner.kurtosis_axis(ax), ax, keepdims))
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
    DType::parse(s).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!("Unknown dtype: {}", s))
    })
}

/// Extract start, stop, step from a Python slice object.
fn extract_slice_indices(
    slice: &Bound<'_, PySlice>,
    length: usize,
) -> PyResult<(isize, isize, isize)> {
    let indices = slice.indices(length as isize)?;
    Ok((indices.start, indices.stop, indices.step))
}

/// Normalize a negative index to positive.
fn normalize_index(idx: isize, len: usize) -> usize {
    if idx < 0 {
        (len as isize + idx) as usize
    } else {
        idx as usize
    }
}

/// Normalize slice start/stop values with proper defaults and clamping.
fn normalize_slice(start: isize, stop: isize, step: isize, len: isize) -> (isize, isize) {
    let (default_start, default_stop) = if step > 0 {
        (0, len)
    } else {
        (len - 1, -len - 1)
    };

    // Apply defaults for sentinel values
    let start = if start == isize::MIN { default_start } else { start };
    let stop = if stop == isize::MAX { default_stop } else { stop };

    // Normalize negative indices
    let start = if start < 0 { (start + len).max(0) } else { start.min(len) };
    let stop = if stop < 0 { (stop + len).max(-1) } else { stop.min(len) };

    (start, stop)
}

/// Represents an indexing item after ellipsis expansion.
#[derive(Clone, Debug)]
enum IndexItem {
    NewAxis,
    Int(isize),
    Slice(isize, isize, isize), // start, stop, step
}

/// Expand ellipsis in a tuple of indices.
/// Returns a Vec of IndexItem representing each axis operation.
fn expand_ellipsis(tuple: &Bound<'_, PyTuple>, ndim: usize) -> PyResult<Vec<IndexItem>> {
    let mut items: Vec<IndexItem> = Vec::new();
    let mut ellipsis_pos: Option<usize> = None;
    let mut explicit_axis_count = 0usize;

    // First pass: count elements and find ellipsis
    for (i, item) in tuple.iter().enumerate() {
        if item.downcast::<PyEllipsis>().is_ok() {
            if ellipsis_pos.is_some() {
                return Err(pyo3::exceptions::PyIndexError::new_err(
                    "an index can only have a single ellipsis",
                ));
            }
            ellipsis_pos = Some(i);
        } else if item.is_none() {
            // newaxis doesn't consume a dimension
        } else {
            explicit_axis_count += 1;
        }
    }

    // Calculate how many axes the ellipsis expands to
    let ellipsis_axes = if ellipsis_pos.is_some() {
        if explicit_axis_count > ndim {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                "too many indices for array",
            ));
        }
        ndim - explicit_axis_count
    } else {
        0
    };

    // Second pass: build items list
    for item in tuple.iter() {
        if item.downcast::<PyEllipsis>().is_ok() {
            // Expand ellipsis to full slices
            for _ in 0..ellipsis_axes {
                items.push(IndexItem::Slice(0, isize::MAX, 1)); // Will be clipped by slice_axis
            }
        } else if item.is_none() {
            items.push(IndexItem::NewAxis);
        } else if let Ok(idx) = item.extract::<isize>() {
            items.push(IndexItem::Int(idx));
        } else if let Ok(slice) = item.downcast::<PySlice>() {
            // We'll extract proper indices later when we know the axis size
            // For now, store special marker values
            let start = slice.getattr("start").ok()
                .and_then(|v| if v.is_none() { None } else { v.extract::<isize>().ok() });
            let stop = slice.getattr("stop").ok()
                .and_then(|v| if v.is_none() { None } else { v.extract::<isize>().ok() });
            let step = slice.getattr("step").ok()
                .and_then(|v| if v.is_none() { None } else { v.extract::<isize>().ok() })
                .unwrap_or(1);

            // Use sentinel values that will be processed during application
            items.push(IndexItem::Slice(
                start.unwrap_or(isize::MIN),
                stop.unwrap_or(isize::MAX),
                step
            ));
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "indices must be integers, slices, ellipsis, or None",
            ));
        }
    }

    Ok(items)
}

/// Fill a view with a scalar or array value.
fn fill_view(
    view: &mut RumpyArray,
    scalar: Option<f64>,
    array: Option<&PyRumpyArray>,
) -> PyResult<()> {
    use crate::array::increment_indices;

    if let Some(val) = scalar {
        // Fill with scalar
        let mut indices = vec![0usize; view.ndim()];
        let shape = view.shape().to_vec();
        for _ in 0..view.size() {
            view.set_element(&indices, val);
            increment_indices(&mut indices, &shape);
        }
        Ok(())
    } else if let Some(src) = array {
        // Copy from array (shapes must match)
        if view.shape() != src.inner.shape() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "could not broadcast input array to shape",
            ));
        }
        let mut indices = vec![0usize; view.ndim()];
        let shape = view.shape().to_vec();
        for _ in 0..view.size() {
            let val = src.inner.get_element(&indices);
            view.set_element(&indices, val);
            increment_indices(&mut indices, &shape);
        }
        Ok(())
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "value must be a number or array",
        ))
    }
}

/// Check if an array qualifies for temporary elision (in-place buffer reuse).
///
/// Returns the array to use as output buffer if elision is possible.
fn try_elision_candidate(
    self_ptr: *mut pyo3::ffi::PyObject,
    arr: &RumpyArray,
    result_shape: &[usize],
    result_dtype: &DType,
) -> Option<RumpyArray> {
    // Check Python refcount - we need exactly 1 (this PyObject)
    // Note: refcount may be higher due to method call overhead, so we check <= 2
    // In practice, `x + 1 + 2` will have refcount 1 for the intermediate
    if py_refcount_ptr(self_ptr) > 2 {
        return None;
    }

    // Check size threshold
    if arr.size() < ELISION_SIZE_THRESHOLD {
        return None;
    }

    // Check buffer compatibility
    if !arr.can_reuse_for_output(result_shape, result_dtype) {
        return None;
    }

    // Clone the RumpyArray - this is cheap if Arc is sole owner
    // (just increments refcount), expensive if shared (materializes buffer)
    Some(arr.clone())
}

/// Dispatch binary operation: array op (array or scalar).
/// Takes raw PyObject pointer for refcount checking to enable temporary elision.
fn binary_op_dispatch_with_elision(
    self_ptr: *mut pyo3::ffi::PyObject,
    arr: &RumpyArray,
    other: &Bound<'_, PyAny>,
    op: BinaryOp,
) -> PyResult<PyRumpyArray> {
    use crate::array::broadcast_shapes;
    use crate::ops::BinaryOpError;

    // Try array first
    if let Ok(other_arr) = other.extract::<PyRef<'_, PyRumpyArray>>() {
        let other_inner = &other_arr.inner;

        // Compute result shape and dtype for elision check
        let result_shape = broadcast_shapes(arr.shape(), other_inner.shape())
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("operands have incompatible shapes"))?;
        let result_dtype = promote_dtype(&arr.dtype(), &other_inner.dtype());

        // Try to reuse self's buffer if it's an ephemeral intermediate
        let out = try_elision_candidate(self_ptr, arr, &result_shape, &result_dtype);

        map_binary_op_inplace(arr, other_inner, op, out)
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
        // For scalar ops, result shape == arr shape, result dtype == arr dtype (usually)
        // TODO: Consider elision for scalar ops too
        Ok(PyRumpyArray::new(arr.scalar_op(scalar, op)))
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "operand must be ndarray or number",
        ))
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

/// Dispatch bitwise binary operation: array op (array or scalar).
fn bitwise_binary_op_dispatch<F>(
    arr: &RumpyArray,
    other: &Bound<'_, PyAny>,
    op: F,
) -> PyResult<PyRumpyArray>
where
    F: Fn(&RumpyArray, &RumpyArray) -> Option<RumpyArray>,
{
    if let Ok(other_arr) = other.extract::<PyRef<'_, PyRumpyArray>>() {
        op(arr, &other_arr.inner)
            .map(PyRumpyArray::new)
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("operands have incompatible shapes")
            })
    } else if let Ok(scalar) = other.extract::<i64>() {
        let scalar_arr = RumpyArray::full(vec![1], scalar as f64, DType::int64());
        op(arr, &scalar_arr)
            .map(PyRumpyArray::new)
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("operands have incompatible shapes")
            })
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "operand must be ndarray or integer",
        ))
    }
}

/// Dispatch reverse bitwise binary operation: scalar op array.
fn rbitwise_binary_op_dispatch<F>(
    arr: &RumpyArray,
    other: &Bound<'_, PyAny>,
    op: F,
) -> PyResult<PyRumpyArray>
where
    F: Fn(&RumpyArray, &RumpyArray) -> Option<RumpyArray>,
{
    if let Ok(scalar) = other.extract::<i64>() {
        let scalar_arr = RumpyArray::full(vec![1], scalar as f64, DType::int64());
        op(&scalar_arr, arr)
            .map(PyRumpyArray::new)
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("operands have incompatible shapes")
            })
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "operand must be an integer",
        ))
    }
}
