//! PyRumpyArray: Python-visible ndarray class.
//!
//! Split into submodules:
//! - `dunder_ops`: arithmetic, comparison, bitwise operators
//! - `dunder_item`: __getitem__, __setitem__
//! - `methods_reductions`: sum, mean, var, std, etc.

mod dunder_item;
mod dunder_ops;
mod methods_reductions;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyEllipsis, PyList, PySlice, PyString, PyTuple};

use crate::array::{promote_dtype, DType, RumpyArray};
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
pub fn resolve_reshape_shape(shape: Vec<isize>, size: usize) -> PyResult<Vec<usize>> {
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

    /// Return indices of non-zero elements as tuple of arrays.
    fn nonzero(&self) -> Vec<Self> {
        self.inner.nonzero().into_iter().map(Self::new).collect()
    }

    /// Return indices that would sort the array.
    #[pyo3(signature = (axis=None))]
    fn argsort(&self, axis: Option<isize>) -> Self {
        let ax = normalize_axis_option(axis, self.inner.ndim(), false);
        Self::new(self.inner.argsort(ax))
    }

    /// Sort array in-place.
    #[pyo3(signature = (axis=None))]
    fn sort(&mut self, axis: Option<isize>) {
        let ax = normalize_axis_option(axis, self.inner.ndim(), true);
        self.inner = self.inner.sort(ax);
    }

    /// Find indices where elements should be inserted to maintain order.
    #[pyo3(signature = (v, side="left"))]
    fn searchsorted<'py>(&self, py: Python<'py>, v: &Bound<'py, PyAny>, side: &str) -> PyResult<PyObject> {
        // Handle scalar input - always use float64 for the search value to avoid truncation
        if let Ok(scalar) = v.extract::<f64>() {
            let v_arr = RumpyArray::from_vec(vec![scalar], DType::float64());
            let result = crate::ops::indexing::searchsorted(&self.inner, &v_arr, side)
                .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("searchsorted failed"))?;
            // Return scalar for scalar input
            let idx = result.get_element(&[0]) as i64;
            return Ok(idx.into_pyobject(py)?.into_any().unbind());
        }
        // Handle array input
        if let Ok(arr) = v.extract::<PyRef<'_, PyRumpyArray>>() {
            let result = crate::ops::indexing::searchsorted(&self.inner, &arr.inner, side)
                .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("searchsorted failed"))?;
            return Ok(Self::new(result).into_pyobject(py)?.into_any().unbind());
        }
        Err(pyo3::exceptions::PyTypeError::new_err("v must be a scalar or array"))
    }

    /// Repeat elements of an array.
    #[pyo3(signature = (repeats, axis=None))]
    fn repeat(&self, repeats: usize, axis: Option<usize>) -> PyResult<Self> {
        let arr = &self.inner;
        let dtype = arr.dtype();

        // For axis=None, flatten first; for axis=Some, validate bounds
        let (src, ax) = match axis {
            None => (arr.copy().reshape(vec![arr.size()]).unwrap(), 0usize),
            Some(ax) => {
                if ax >= arr.ndim() {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "axis {} is out of bounds for array of dimension {}",
                        ax, arr.ndim()
                    )));
                }
                (arr.clone(), ax)
            }
        };

        let shape = src.shape();
        let mut new_shape = shape.to_vec();
        new_shape[ax] *= repeats;

        let mut result = RumpyArray::zeros(new_shape.clone(), dtype.clone());
        let result_buffer = std::sync::Arc::get_mut(result.buffer_mut()).expect("unique");
        let result_ptr = result_buffer.as_mut_ptr();

        // Fast path for 1D contiguous
        if src.ndim() == 1 && src.is_c_contiguous() {
            let src_ptr = src.data_ptr();
            let itemsize = dtype.itemsize();
            for i in 0..shape[0] {
                let val = unsafe { dtype.ops().read_f64(src_ptr, (i * itemsize) as isize) }.unwrap_or(0.0);
                let base = i * repeats;
                for j in 0..repeats {
                    unsafe { dtype.ops().write_f64(result_ptr, base + j, val); }
                }
            }
        } else {
            // General case
            let mut out_indices = vec![0usize; src.ndim()];
            for i in 0..result.size() {
                let mut src_indices = out_indices.clone();
                src_indices[ax] /= repeats;
                let val = src.get_element(&src_indices);
                unsafe { dtype.ops().write_f64(result_ptr, i, val); }
                crate::array::increment_indices(&mut out_indices, &new_shape);
            }
        }

        Ok(Self::new(result))
    }

    /// Take elements from an array along an axis.
    /// If axis is None, the array is flattened before taking.
    #[pyo3(signature = (indices, axis=None))]
    fn take(&self, indices: &PyRumpyArray, axis: Option<usize>) -> PyResult<Self> {
        let (arr, ax) = match axis {
            None => (self.inner.copy().reshape(vec![self.inner.size()]).unwrap(), 0),
            Some(ax) => (self.inner.clone(), ax),
        };
        crate::ops::indexing::take(&arr, &indices.inner, Some(ax))
            .map(Self::new)
            .ok_or_else(|| pyo3::exceptions::PyIndexError::new_err("index out of bounds"))
    }

    /// Set values at indices (in-place).
    fn put(&mut self, indices: &Bound<'_, PyAny>, values: &Bound<'_, PyAny>) -> PyResult<()> {
        let idx_vec = extract_i64_vec(indices, "indices")?;
        let val_vec = extract_f64_vec(values, "values")?;
        crate::ops::indexing::put(&mut self.inner, &idx_vec, &val_vec)
            .ok_or_else(|| pyo3::exceptions::PyIndexError::new_err("index out of bounds"))
    }

    /// Fill the array with a scalar value (in-place).
    fn fill(&mut self, value: f64) {
        let size = self.inner.size();
        if size == 0 {
            return;
        }

        // Fast path: contiguous float64 - direct slice fill
        if self.inner.is_c_contiguous() && self.inner.dtype().ops().name() == "float64" {
            let ptr = self.inner.data_ptr_mut() as *mut f64;
            let slice = unsafe { std::slice::from_raw_parts_mut(ptr, size) };
            slice.fill(value);
            return;
        }

        // Fast path: other contiguous types
        if self.inner.is_c_contiguous() {
            let dtype = self.inner.dtype();
            let ptr = self.inner.data_ptr_mut();
            let ops = dtype.ops();
            for i in 0..size {
                unsafe { ops.write_f64(ptr, i, value); }
            }
        } else {
            // Slow path: strided iteration
            let ndim = self.inner.ndim();
            let shape = self.inner.shape().to_vec();
            let mut indices = vec![0usize; ndim];
            for _ in 0..size {
                self.inner.set_element(&indices, value);
                crate::array::increment_indices(&mut indices, &shape);
            }
        }
    }

    /// Return the array data as a bytes object.
    fn tobytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        // Need contiguous array
        let arr = if self.inner.is_c_contiguous() {
            self.inner.clone()
        } else {
            self.inner.copy()
        };

        let nbytes = arr.nbytes();
        let ptr = arr.data_ptr();
        let bytes = unsafe { std::slice::from_raw_parts(ptr, nbytes) };
        Ok(pyo3::types::PyBytes::new(py, bytes))
    }

    /// View array with a different dtype.
    fn view(&self, dtype: &str) -> PyResult<Self> {
        let new_dtype = parse_dtype(dtype)?;
        let old_itemsize = self.inner.dtype().itemsize();
        let new_itemsize = new_dtype.itemsize();

        // Check that array is contiguous
        if !self.inner.is_c_contiguous() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "view requires contiguous array"
            ));
        }

        let shape = self.inner.shape();
        let nbytes = self.inner.nbytes();

        // Calculate new shape - last dimension changes
        if shape.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "cannot view 0-d array with different dtype"
            ));
        }

        let last_dim_bytes = shape[shape.len() - 1] * old_itemsize;
        if !last_dim_bytes.is_multiple_of(new_itemsize) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "view cannot reshape array"
            ));
        }

        let mut new_shape = shape.to_vec();
        new_shape[shape.len() - 1] = last_dim_bytes / new_itemsize;

        // Create new strides
        let mut new_strides = Vec::with_capacity(new_shape.len());
        let mut stride = new_itemsize as isize;
        for &dim in new_shape.iter().rev() {
            new_strides.push(stride);
            stride *= dim as isize;
        }
        new_strides.reverse();

        // Verify total bytes match
        let new_size: usize = new_shape.iter().product();
        if new_size * new_itemsize != nbytes {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "view cannot reshape array"
            ));
        }

        // Create view with shared buffer
        Ok(Self::new(self.inner.view_with_dtype(new_shape, new_strides, new_dtype)))
    }

    /// Partially sort array in-place so element at kth position is in sorted position.
    fn partition(&mut self, kth: usize) -> PyResult<()> {
        let size = self.inner.size();
        if kth >= size {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "kth(={}) out of bounds ({})", kth, size
            )));
        }

        // Fast path: contiguous float64 array - work directly on buffer
        if self.inner.is_c_contiguous() && self.inner.dtype().ops().name() == "float64" {
            let ptr = self.inner.data_ptr_mut() as *mut f64;
            let slice = unsafe { std::slice::from_raw_parts_mut(ptr, size) };
            partition_select(slice, kth);
            return Ok(());
        }

        // General path: copy, partition, write back
        let mut values: Vec<f64> = Vec::with_capacity(size);
        for offset in self.inner.iter_offsets() {
            let val = unsafe { self.inner.dtype().ops().read_f64(self.inner.data_ptr(), offset) }.unwrap_or(0.0);
            values.push(val);
        }
        partition_select(&mut values, kth);

        let shape = self.inner.shape().to_vec();
        let mut indices = vec![0usize; self.inner.ndim()];
        for (i, &val) in values.iter().enumerate() {
            self.inner.set_element(&indices, val);
            if i + 1 < size {
                crate::array::increment_indices(&mut indices, &shape);
            }
        }

        Ok(())
    }

    /// Return indices that would partition the array.
    fn argpartition(&self, kth: usize) -> PyResult<Self> {
        let size = self.inner.size();
        if kth >= size {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "kth(={}) out of bounds ({})", kth, size
            )));
        }

        // Collect values with original indices
        let mut indexed: Vec<(f64, usize)> = Vec::with_capacity(size);
        for (i, offset) in self.inner.iter_offsets().enumerate() {
            let val = unsafe { self.inner.dtype().ops().read_f64(self.inner.data_ptr(), offset) }.unwrap_or(0.0);
            indexed.push((val, i));
        }

        // Partition by value
        partition_select_indexed(&mut indexed, kth);

        // Extract indices
        let indices: Vec<f64> = indexed.iter().map(|(_, idx)| *idx as f64).collect();
        Ok(Self::new(RumpyArray::from_vec(indices, DType::int64())))
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

/// Normalize axis for sort/argsort operations.
/// - If axis is None and default_last is false, returns None (flatten behavior)
/// - If axis is None and default_last is true, returns last axis
/// - Handles negative axis values
fn normalize_axis_option(axis: Option<isize>, ndim: usize, default_last: bool) -> Option<usize> {
    match axis {
        None if default_last => Some(ndim - 1),
        None => None,
        Some(a) if a >= 0 => Some(a as usize),
        Some(a) => Some((ndim as isize + a) as usize),
    }
}

/// Extract i64 values from a Python list or array.
fn extract_i64_vec(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<i64>> {
    if let Ok(list) = obj.downcast::<PyList>() {
        list.iter().map(|x| x.extract::<i64>()).collect()
    } else if let Ok(arr) = obj.extract::<PyRef<'_, PyRumpyArray>>() {
        Ok(arr.inner.iter_offsets()
            .map(|offset| unsafe { arr.inner.dtype().ops().read_f64(arr.inner.data_ptr(), offset) }.unwrap_or(0.0) as i64)
            .collect())
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err(format!("{} must be list or array", name)))
    }
}

/// Extract f64 values from a Python list or array.
fn extract_f64_vec(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<f64>> {
    if let Ok(list) = obj.downcast::<PyList>() {
        list.iter().map(|x| x.extract::<f64>()).collect()
    } else if let Ok(arr) = obj.extract::<PyRef<'_, PyRumpyArray>>() {
        Ok(arr.inner.iter_offsets()
            .map(|offset| unsafe { arr.inner.dtype().ops().read_f64(arr.inner.data_ptr(), offset) }.unwrap_or(0.0))
            .collect())
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err(format!("{} must be list or array", name)))
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

/// Quickselect-style partition: rearrange so that element at kth position is in sorted position.
fn partition_select(arr: &mut [f64], kth: usize) {
    if arr.len() <= 1 {
        return;
    }

    // Simple quickselect
    let mut left = 0;
    let mut right = arr.len() - 1;

    while left < right {
        let pivot_idx = (left + right) / 2;
        let pivot = arr[pivot_idx];

        // Move pivot to end
        arr.swap(pivot_idx, right);

        let mut store = left;
        for i in left..right {
            if arr[i] < pivot {
                arr.swap(i, store);
                store += 1;
            }
        }
        arr.swap(store, right);

        if store == kth {
            return;
        } else if store < kth {
            left = store + 1;
        } else {
            right = store.saturating_sub(1);
        }
    }
}

/// Quickselect for indexed values: partition so kth smallest is at position kth.
fn partition_select_indexed(arr: &mut [(f64, usize)], kth: usize) {
    if arr.len() <= 1 {
        return;
    }

    let mut left = 0;
    let mut right = arr.len() - 1;

    while left < right {
        let pivot_idx = (left + right) / 2;
        let pivot = arr[pivot_idx].0;

        arr.swap(pivot_idx, right);

        let mut store = left;
        for i in left..right {
            if arr[i].0 < pivot {
                arr.swap(i, store);
                store += 1;
            }
        }
        arr.swap(store, right);

        if store == kth {
            return;
        } else if store < kth {
            left = store + 1;
        } else {
            right = store.saturating_sub(1);
        }
    }
}
