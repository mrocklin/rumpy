pub mod fft;
pub mod linalg;
pub mod pyarray;
pub mod random;

use pyo3::prelude::*;
use pyo3::types::PyList;

pub use pyarray::{parse_dtype, parse_shape, PyRumpyArray};

use crate::array::{increment_indices, DType, RumpyArray};

/// Create an array filled with zeros.
#[pyfunction]
#[pyo3(signature = (shape, dtype=None))]
pub fn zeros(shape: &Bound<'_, PyAny>, dtype: Option<&str>) -> PyResult<PyRumpyArray> {
    let shape = parse_shape(shape)?;
    let dtype = parse_dtype(dtype.unwrap_or("float64"))?;
    Ok(PyRumpyArray::new(RumpyArray::zeros(shape, dtype)))
}

/// Create an array filled with ones.
#[pyfunction]
#[pyo3(signature = (shape, dtype=None))]
pub fn ones(shape: &Bound<'_, PyAny>, dtype: Option<&str>) -> PyResult<PyRumpyArray> {
    let shape = parse_shape(shape)?;
    let dtype = parse_dtype(dtype.unwrap_or("float64"))?;
    Ok(PyRumpyArray::new(RumpyArray::ones(shape, dtype)))
}

/// Create array with evenly spaced values.
#[pyfunction]
#[pyo3(signature = (start_or_stop, stop=None, step=None, dtype=None))]
pub fn arange(
    start_or_stop: f64,
    stop: Option<f64>,
    step: Option<f64>,
    dtype: Option<&str>,
) -> PyResult<PyRumpyArray> {
    let (start, stop, step) = match stop {
        Some(stop) => (start_or_stop, stop, step.unwrap_or(1.0)),
        None => (0.0, start_or_stop, step.unwrap_or(1.0)),
    };
    let dtype = parse_dtype(dtype.unwrap_or("int64"))?;
    Ok(PyRumpyArray::new(RumpyArray::arange(start, stop, step, dtype)))
}

/// Create array with evenly spaced values over interval [start, stop].
#[pyfunction]
#[pyo3(signature = (start, stop, num=50, dtype=None))]
pub fn linspace(
    start: f64,
    stop: f64,
    num: usize,
    dtype: Option<&str>,
) -> PyResult<PyRumpyArray> {
    let dtype = parse_dtype(dtype.unwrap_or("float64"))?;
    Ok(PyRumpyArray::new(RumpyArray::linspace(start, stop, num, dtype)))
}

/// Create an identity matrix.
#[pyfunction]
#[pyo3(signature = (n, dtype=None))]
pub fn eye(n: usize, dtype: Option<&str>) -> PyResult<PyRumpyArray> {
    let dtype = parse_dtype(dtype.unwrap_or("float64"))?;
    Ok(PyRumpyArray::new(RumpyArray::eye(n, dtype)))
}

/// Create array filled with given value.
#[pyfunction]
#[pyo3(signature = (shape, fill_value, dtype=None))]
pub fn full(shape: &Bound<'_, PyAny>, fill_value: f64, dtype: Option<&str>) -> PyResult<PyRumpyArray> {
    let shape = parse_shape(shape)?;
    let dtype = parse_dtype(dtype.unwrap_or("float64"))?;
    Ok(PyRumpyArray::new(RumpyArray::full(shape, fill_value, dtype)))
}

/// Create uninitialized array.
#[pyfunction]
#[pyo3(signature = (shape, dtype=None))]
pub fn empty(shape: &Bound<'_, PyAny>, dtype: Option<&str>) -> PyResult<PyRumpyArray> {
    let shape = parse_shape(shape)?;
    let dtype = parse_dtype(dtype.unwrap_or("float64"))?;
    // For simplicity, we just create zeros - truly uninitialized would require unsafe
    Ok(PyRumpyArray::new(RumpyArray::zeros(shape, dtype)))
}

/// Create array of zeros with same shape and dtype as input.
#[pyfunction]
#[pyo3(signature = (a, dtype=None))]
pub fn zeros_like(a: &PyRumpyArray, dtype: Option<&str>) -> PyResult<PyRumpyArray> {
    let shape = a.inner.shape().to_vec();
    let dtype = match dtype {
        Some(dt) => parse_dtype(dt)?,
        None => a.inner.dtype(),
    };
    Ok(PyRumpyArray::new(RumpyArray::zeros(shape, dtype)))
}

/// Create array of ones with same shape and dtype as input.
#[pyfunction]
#[pyo3(signature = (a, dtype=None))]
pub fn ones_like(a: &PyRumpyArray, dtype: Option<&str>) -> PyResult<PyRumpyArray> {
    let shape = a.inner.shape().to_vec();
    let dtype = match dtype {
        Some(dt) => parse_dtype(dt)?,
        None => a.inner.dtype(),
    };
    Ok(PyRumpyArray::new(RumpyArray::ones(shape, dtype)))
}

/// Create uninitialized array with same shape and dtype as input.
#[pyfunction]
#[pyo3(signature = (a, dtype=None))]
pub fn empty_like(a: &PyRumpyArray, dtype: Option<&str>) -> PyResult<PyRumpyArray> {
    let shape = a.inner.shape().to_vec();
    let dtype = match dtype {
        Some(dt) => parse_dtype(dt)?,
        None => a.inner.dtype(),
    };
    // For simplicity, we just create zeros - truly uninitialized would require unsafe
    Ok(PyRumpyArray::new(RumpyArray::zeros(shape, dtype)))
}

/// Return a contiguous copy of the array.
#[pyfunction]
pub fn copy(a: &PyRumpyArray) -> PyRumpyArray {
    PyRumpyArray::new(a.inner.copy())
}

/// Convert input to an array.
/// Supports: PyRumpyArray, objects with __array_interface__, and Python lists.
#[pyfunction]
#[pyo3(signature = (obj, dtype=None))]
pub fn asarray(py: Python<'_>, obj: &Bound<'_, PyAny>, dtype: Option<&str>) -> PyResult<PyRumpyArray> {
    array_impl(py, obj, dtype)
}

/// Create an array (alias for asarray).
#[pyfunction]
#[pyo3(signature = (obj, dtype=None))]
pub fn array(py: Python<'_>, obj: &Bound<'_, PyAny>, dtype: Option<&str>) -> PyResult<PyRumpyArray> {
    array_impl(py, obj, dtype)
}

/// Implementation for array/asarray.
fn array_impl(py: Python<'_>, obj: &Bound<'_, PyAny>, dtype: Option<&str>) -> PyResult<PyRumpyArray> {
    // Already a rumpy array?
    if let Ok(arr) = obj.extract::<PyRef<'_, PyRumpyArray>>() {
        // TODO: handle dtype conversion if requested
        return Ok(PyRumpyArray::new(arr.inner.clone()));
    }

    // Has __array_interface__? (numpy arrays, etc.)
    if let Ok(interface) = obj.getattr("__array_interface__") {
        return from_array_interface(py, &interface, dtype);
    }

    // Fall back to Python list
    if let Ok(list) = obj.downcast::<PyList>() {
        return from_list(list, dtype);
    }

    // Try to iterate as a sequence
    if let Ok(seq) = obj.try_iter() {
        let items: Vec<f64> = seq
            .map(|item| item.and_then(|i| i.extract::<f64>()))
            .collect::<PyResult<Vec<f64>>>()?;
        let dtype = parse_dtype(dtype.unwrap_or("float64"))?;
        return Ok(PyRumpyArray::new(RumpyArray::from_vec(items, dtype)));
    }

    Err(pyo3::exceptions::PyTypeError::new_err(
        "Cannot convert object to array",
    ))
}

/// Create array from __array_interface__ dict.
fn from_array_interface(
    _py: Python<'_>,
    interface: &Bound<'_, PyAny>,
    dtype_override: Option<&str>,
) -> PyResult<PyRumpyArray> {
    use pyo3::types::PyTuple;

    // Extract shape
    let shape_obj = interface.get_item("shape")?;
    let shape: Vec<usize> = if let Ok(tuple) = shape_obj.downcast::<PyTuple>() {
        tuple.iter().map(|x| x.extract::<usize>()).collect::<PyResult<Vec<_>>>()?
    } else {
        return Err(pyo3::exceptions::PyValueError::new_err("Invalid shape"));
    };

    // Extract typestr to determine dtype
    let typestr: String = interface.get_item("typestr")?.extract()?;
    let dtype = if let Some(dt) = dtype_override {
        parse_dtype(dt)?
    } else {
        dtype_from_typestr(&typestr)?
    };

    // Extract data pointer
    let data_obj = interface.get_item("data")?;
    let (data_ptr, _readonly): (usize, bool) = data_obj.extract()?;

    // Extract strides (optional)
    let strides: Option<Vec<isize>> = interface
        .get_item("strides")
        .ok()
        .and_then(|s| {
            if s.is_none() {
                None
            } else if let Ok(tuple) = s.downcast::<PyTuple>() {
                tuple.iter().map(|x| x.extract::<isize>()).collect::<PyResult<Vec<_>>>().ok()
            } else {
                None
            }
        });

    // Calculate size and copy data
    let size: usize = shape.iter().product();
    let itemsize = dtype.itemsize();
    let mut arr = RumpyArray::zeros(shape.clone(), dtype);

    if size == 0 {
        return Ok(PyRumpyArray::new(arr));
    }

    // Copy data respecting strides
    let src_ptr = data_ptr as *const u8;
    let src_strides = strides.unwrap_or_else(|| {
        crate::array::compute_c_strides(&shape, itemsize)
    });

    let dtype = arr.dtype();
    // Use get_element pattern - iterate and copy
    let buffer = arr.buffer_mut();
    let result_buffer = std::sync::Arc::get_mut(buffer).expect("buffer must be unique");
    let dst_ptr = result_buffer.as_mut_ptr();

    let mut indices = vec![0usize; shape.len()];
    for i in 0..size {
        // Calculate source offset
        let src_offset: isize = indices
            .iter()
            .zip(src_strides.iter())
            .map(|(&idx, &stride)| idx as isize * stride)
            .sum();

        // Copy element from source to destination
        unsafe { dtype.ops().copy_element(src_ptr, src_offset, dst_ptr, i); }

        increment_indices(&mut indices, &shape);
    }

    Ok(PyRumpyArray::new(arr))
}

/// Infer dtype from first element of a Python list.
fn infer_dtype_from_list(list: &Bound<'_, PyList>) -> Option<DType> {
    if list.is_empty() {
        return Some(DType::float64());
    }
    let first = list.get_item(0).ok()?;
    // Recurse into nested lists
    if let Ok(sublist) = first.downcast::<PyList>() {
        return infer_dtype_from_list(sublist);
    }
    // Check type of first element - order matters: bool before int!
    if first.is_instance_of::<pyo3::types::PyBool>() {
        return Some(DType::bool());
    }
    if first.is_instance_of::<pyo3::types::PyInt>() {
        return Some(DType::int64());
    }
    if first.is_instance_of::<pyo3::types::PyFloat>() {
        return Some(DType::float64());
    }
    Some(DType::float64())
}

/// Create array from Python list.
fn from_list(list: &Bound<'_, PyList>, dtype: Option<&str>) -> PyResult<PyRumpyArray> {
    let dtype = match dtype {
        Some(dt) => parse_dtype(dt)?,
        None => infer_dtype_from_list(list).unwrap_or(DType::float64()),
    };

    // Check if nested (2D+)
    if list.is_empty() {
        return Ok(PyRumpyArray::new(RumpyArray::zeros(vec![0], dtype)));
    }

    let first = list.get_item(0)?;
    if first.downcast::<PyList>().is_ok() {
        // Nested list - recursively get shape and flatten
        let (shape, data) = flatten_nested_list(list)?;
        return Ok(PyRumpyArray::new(RumpyArray::from_vec_with_shape(data, shape, dtype)));
    }

    // 1D list
    let data: Vec<f64> = list
        .iter()
        .map(|item| item.extract::<f64>())
        .collect::<PyResult<Vec<f64>>>()?;

    Ok(PyRumpyArray::new(RumpyArray::from_vec(data, dtype)))
}

/// Flatten nested Python list, returning shape and data.
fn flatten_nested_list(list: &Bound<'_, PyList>) -> PyResult<(Vec<usize>, Vec<f64>)> {
    let mut shape = vec![list.len()];
    let mut data = Vec::new();

    for item in list.iter() {
        if let Ok(sublist) = item.downcast::<PyList>() {
            let (sub_shape, sub_data) = flatten_nested_list(sublist)?;
            if shape.len() == 1 {
                shape.extend(sub_shape);
            }
            data.extend(sub_data);
        } else {
            data.push(item.extract::<f64>()?);
        }
    }

    Ok((shape, data))
}

/// Parse dtype from numpy typestr.
fn dtype_from_typestr(typestr: &str) -> PyResult<DType> {
    // Handle datetime64: "<M8[ns]", "<M8[us]", etc.
    if typestr.starts_with("<M8[") || typestr.starts_with(">M8[") {
        return match typestr {
            "<M8[ns]" | ">M8[ns]" => Ok(DType::datetime64_ns()),
            "<M8[us]" | ">M8[us]" => Ok(DType::datetime64_us()),
            "<M8[ms]" | ">M8[ms]" => Ok(DType::datetime64_ms()),
            "<M8[s]" | ">M8[s]" => Ok(DType::datetime64_s()),
            _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unsupported datetime64 unit: {}",
                typestr
            ))),
        };
    }

    // Format: "<f8" (little-endian float64), ">i4" (big-endian int32), etc.
    let kind = typestr.chars().nth(1).unwrap_or('f');
    let size: usize = typestr[2..].parse().unwrap_or(8);

    match (kind, size) {
        ('f', 8) => Ok(DType::float64()),
        ('f', 4) => Ok(DType::float32()),
        ('i', 8) => Ok(DType::int64()),
        ('i', 4) => Ok(DType::int32()),
        ('u', 8) => Ok(DType::uint64()),
        ('u', 4) => Ok(DType::uint32()),
        ('u', 1) => Ok(DType::uint8()),
        ('b', 1) => Ok(DType::bool()),
        ('c', 8) => Ok(DType::complex64()),
        ('c', 16) => Ok(DType::complex128()),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unsupported dtype: {}",
            typestr
        ))),
    }
}

// Math ufuncs (module-level functions like np.sqrt, np.exp, etc.)
// These accept either array or scalar, returning the same type.

/// Result type for ufuncs that can return scalar or array.
pub enum UnaryResult {
    Scalar(f64),
    Array(PyRumpyArray),
}

impl<'py> IntoPyObject<'py> for UnaryResult {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        match self {
            UnaryResult::Scalar(v) => Ok(v.into_pyobject(py)?.into_any()),
            UnaryResult::Array(arr) => Ok(arr.into_pyobject(py)?.into_any()),
        }
    }
}

/// Apply a unary ufunc to either scalar or array input.
fn apply_unary<F, G>(x: &Bound<'_, PyAny>, scalar_op: F, array_op: G) -> PyResult<UnaryResult>
where
    F: FnOnce(f64) -> f64,
    G: FnOnce(&RumpyArray) -> Result<RumpyArray, crate::ops::UnaryOpError>,
{
    // Try array first
    if let Ok(arr) = x.extract::<PyRef<'_, PyRumpyArray>>() {
        return array_op(&arr.inner)
            .map(|a| UnaryResult::Array(PyRumpyArray::new(a)))
            .map_err(|_| pyo3::exceptions::PyTypeError::new_err("ufunc not supported for this dtype"));
    }
    // Try scalar
    if let Ok(scalar) = x.extract::<f64>() {
        return Ok(UnaryResult::Scalar(scalar_op(scalar)));
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "input must be ndarray or number",
    ))
}

#[pyfunction]
pub fn sqrt(x: &Bound<'_, PyAny>) -> PyResult<UnaryResult> {
    apply_unary(x, |v| v.sqrt(), |a| a.sqrt())
}

#[pyfunction]
pub fn exp(x: &Bound<'_, PyAny>) -> PyResult<UnaryResult> {
    apply_unary(x, |v| v.exp(), |a| a.exp())
}

#[pyfunction]
pub fn log(x: &Bound<'_, PyAny>) -> PyResult<UnaryResult> {
    apply_unary(x, |v| v.ln(), |a| a.log())
}

#[pyfunction]
pub fn sin(x: &Bound<'_, PyAny>) -> PyResult<UnaryResult> {
    apply_unary(x, |v| v.sin(), |a| a.sin())
}

#[pyfunction]
pub fn cos(x: &Bound<'_, PyAny>) -> PyResult<UnaryResult> {
    apply_unary(x, |v| v.cos(), |a| a.cos())
}

#[pyfunction]
pub fn tan(x: &Bound<'_, PyAny>) -> PyResult<UnaryResult> {
    apply_unary(x, |v| v.tan(), |a| a.tan())
}

#[pyfunction]
pub fn floor(x: &Bound<'_, PyAny>) -> PyResult<UnaryResult> {
    apply_unary(x, |v| v.floor(), |a| a.floor())
}

#[pyfunction]
pub fn ceil(x: &Bound<'_, PyAny>) -> PyResult<UnaryResult> {
    apply_unary(x, |v| v.ceil(), |a| a.ceil())
}

#[pyfunction]
pub fn arcsin(x: &Bound<'_, PyAny>) -> PyResult<UnaryResult> {
    apply_unary(x, |v| v.asin(), |a| a.arcsin())
}

#[pyfunction]
pub fn arccos(x: &Bound<'_, PyAny>) -> PyResult<UnaryResult> {
    apply_unary(x, |v| v.acos(), |a| a.arccos())
}

#[pyfunction]
pub fn arctan(x: &Bound<'_, PyAny>) -> PyResult<UnaryResult> {
    apply_unary(x, |v| v.atan(), |a| a.arctan())
}

#[pyfunction]
pub fn log10(x: &Bound<'_, PyAny>) -> PyResult<UnaryResult> {
    apply_unary(x, |v| v.log10(), |a| a.log10())
}

#[pyfunction]
pub fn log2(x: &Bound<'_, PyAny>) -> PyResult<UnaryResult> {
    apply_unary(x, |v| v.log2(), |a| a.log2())
}

#[pyfunction]
pub fn sinh(x: &Bound<'_, PyAny>) -> PyResult<UnaryResult> {
    apply_unary(x, |v| v.sinh(), |a| a.sinh())
}

#[pyfunction]
pub fn cosh(x: &Bound<'_, PyAny>) -> PyResult<UnaryResult> {
    apply_unary(x, |v| v.cosh(), |a| a.cosh())
}

#[pyfunction]
pub fn tanh(x: &Bound<'_, PyAny>) -> PyResult<UnaryResult> {
    apply_unary(x, |v| v.tanh(), |a| a.tanh())
}

#[pyfunction]
pub fn sign(x: &Bound<'_, PyAny>) -> PyResult<UnaryResult> {
    apply_unary(x, |v| if v > 0.0 { 1.0 } else if v < 0.0 { -1.0 } else { 0.0 }, |a| a.sign())
}

#[pyfunction]
pub fn isnan(x: &Bound<'_, PyAny>) -> PyResult<UnaryResult> {
    apply_unary(x, |v| if v.is_nan() { 1.0 } else { 0.0 }, |a| a.isnan())
}

#[pyfunction]
pub fn isinf(x: &Bound<'_, PyAny>) -> PyResult<UnaryResult> {
    apply_unary(x, |v| if v.is_infinite() { 1.0 } else { 0.0 }, |a| a.isinf())
}

#[pyfunction]
pub fn isfinite(x: &Bound<'_, PyAny>) -> PyResult<UnaryResult> {
    apply_unary(x, |v| if v.is_finite() { 1.0 } else { 0.0 }, |a| a.isfinite())
}

#[pyfunction]
pub fn abs(x: &Bound<'_, PyAny>) -> PyResult<UnaryResult> {
    apply_unary(x, |v| v.abs(), |a| a.abs())
}

#[pyfunction]
pub fn square(x: &Bound<'_, PyAny>) -> PyResult<UnaryResult> {
    apply_unary(x, |v| v * v, |a| a.square())
}

#[pyfunction]
pub fn positive(x: &Bound<'_, PyAny>) -> PyResult<UnaryResult> {
    apply_unary(x, |v| v, |a| a.positive())
}

#[pyfunction]
pub fn negative(x: &Bound<'_, PyAny>) -> PyResult<UnaryResult> {
    apply_unary(x, |v| -v, |a| a.neg())
}

#[pyfunction]
pub fn reciprocal(x: &Bound<'_, PyAny>) -> PyResult<UnaryResult> {
    apply_unary(x, |v| 1.0 / v, |a| a.reciprocal())
}

#[pyfunction]
pub fn exp2(x: &Bound<'_, PyAny>) -> PyResult<UnaryResult> {
    apply_unary(x, |v| 2.0_f64.powf(v), |a| a.exp2())
}

#[pyfunction]
pub fn expm1(x: &Bound<'_, PyAny>) -> PyResult<UnaryResult> {
    apply_unary(x, |v| v.exp_m1(), |a| a.expm1())
}

#[pyfunction]
pub fn log1p(x: &Bound<'_, PyAny>) -> PyResult<UnaryResult> {
    apply_unary(x, |v| v.ln_1p(), |a| a.log1p())
}

#[pyfunction]
pub fn cbrt(x: &Bound<'_, PyAny>) -> PyResult<UnaryResult> {
    apply_unary(x, |v| v.cbrt(), |a| a.cbrt())
}

#[pyfunction]
pub fn trunc(x: &Bound<'_, PyAny>) -> PyResult<UnaryResult> {
    apply_unary(x, |v| v.trunc(), |a| a.trunc())
}

#[pyfunction]
pub fn rint(x: &Bound<'_, PyAny>) -> PyResult<UnaryResult> {
    apply_unary(x, |v| v.round(), |a| a.rint())
}

#[pyfunction]
pub fn arcsinh(x: &Bound<'_, PyAny>) -> PyResult<UnaryResult> {
    apply_unary(x, |v| v.asinh(), |a| a.arcsinh())
}

#[pyfunction]
pub fn arccosh(x: &Bound<'_, PyAny>) -> PyResult<UnaryResult> {
    apply_unary(x, |v| v.acosh(), |a| a.arccosh())
}

#[pyfunction]
pub fn arctanh(x: &Bound<'_, PyAny>) -> PyResult<UnaryResult> {
    apply_unary(x, |v| v.atanh(), |a| a.arctanh())
}

#[pyfunction]
pub fn signbit(x: &Bound<'_, PyAny>) -> PyResult<UnaryResult> {
    apply_unary(x, |v| if v.is_sign_negative() { 1.0 } else { 0.0 }, |a| a.signbit())
}

#[pyfunction]
#[pyo3(signature = (x, nan=None, posinf=None, neginf=None))]
pub fn nan_to_num(
    x: &PyRumpyArray,
    nan: Option<f64>,
    posinf: Option<f64>,
    neginf: Option<f64>,
) -> PyRumpyArray {
    let nan_val = nan.unwrap_or(0.0);
    PyRumpyArray::new(x.inner.nan_to_num(nan_val, posinf, neginf))
}

// Element-wise binary functions that accept array or scalar

/// Apply a binary ufunc to either array or scalar inputs.
fn apply_binary_ufunc(
    x1: &Bound<'_, PyAny>,
    x2: &Bound<'_, PyAny>,
    op: crate::array::dtype::BinaryOp,
) -> PyResult<PyRumpyArray> {
    use crate::ops::BinaryOpError;

    // Try array-array first
    let arr1 = if let Ok(arr) = x1.extract::<PyRef<'_, PyRumpyArray>>() {
        arr.inner.clone()
    } else if let Ok(scalar) = x1.extract::<f64>() {
        RumpyArray::full(vec![1], scalar, DType::float64())
    } else {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "operand must be ndarray or number",
        ));
    };

    let arr2 = if let Ok(arr) = x2.extract::<PyRef<'_, PyRumpyArray>>() {
        arr.inner.clone()
    } else if let Ok(scalar) = x2.extract::<f64>() {
        RumpyArray::full(vec![1], scalar, DType::float64())
    } else {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "operand must be ndarray or number",
        ));
    };

    arr1.binary_op(&arr2, op)
        .map(PyRumpyArray::new)
        .map_err(|e| match e {
            BinaryOpError::ShapeMismatch => {
                pyo3::exceptions::PyValueError::new_err("operands have incompatible shapes")
            }
            BinaryOpError::UnsupportedDtype => {
                pyo3::exceptions::PyTypeError::new_err("operation not supported for these dtypes")
            }
        })
}

#[pyfunction]
pub fn maximum(x1: &Bound<'_, PyAny>, x2: &Bound<'_, PyAny>) -> PyResult<PyRumpyArray> {
    apply_binary_ufunc(x1, x2, crate::array::dtype::BinaryOp::Maximum)
}

#[pyfunction]
pub fn minimum(x1: &Bound<'_, PyAny>, x2: &Bound<'_, PyAny>) -> PyResult<PyRumpyArray> {
    apply_binary_ufunc(x1, x2, crate::array::dtype::BinaryOp::Minimum)
}

#[pyfunction]
pub fn add(x1: &Bound<'_, PyAny>, x2: &Bound<'_, PyAny>) -> PyResult<PyRumpyArray> {
    apply_binary_ufunc(x1, x2, crate::array::dtype::BinaryOp::Add)
}

#[pyfunction]
pub fn subtract(x1: &Bound<'_, PyAny>, x2: &Bound<'_, PyAny>) -> PyResult<PyRumpyArray> {
    apply_binary_ufunc(x1, x2, crate::array::dtype::BinaryOp::Sub)
}

#[pyfunction]
pub fn multiply(x1: &Bound<'_, PyAny>, x2: &Bound<'_, PyAny>) -> PyResult<PyRumpyArray> {
    apply_binary_ufunc(x1, x2, crate::array::dtype::BinaryOp::Mul)
}

#[pyfunction]
pub fn divide(x1: &Bound<'_, PyAny>, x2: &Bound<'_, PyAny>) -> PyResult<PyRumpyArray> {
    apply_binary_ufunc(x1, x2, crate::array::dtype::BinaryOp::Div)
}

#[pyfunction]
pub fn power(x1: &Bound<'_, PyAny>, x2: &Bound<'_, PyAny>) -> PyResult<PyRumpyArray> {
    apply_binary_ufunc(x1, x2, crate::array::dtype::BinaryOp::Pow)
}

#[pyfunction]
pub fn floor_divide(x1: &Bound<'_, PyAny>, x2: &Bound<'_, PyAny>) -> PyResult<PyRumpyArray> {
    apply_binary_ufunc(x1, x2, crate::array::dtype::BinaryOp::FloorDiv)
}

#[pyfunction]
pub fn remainder(x1: &Bound<'_, PyAny>, x2: &Bound<'_, PyAny>) -> PyResult<PyRumpyArray> {
    apply_binary_ufunc(x1, x2, crate::array::dtype::BinaryOp::Mod)
}

// Stream 2: Binary math operations

#[pyfunction]
pub fn arctan2(y: &Bound<'_, PyAny>, x: &Bound<'_, PyAny>) -> PyResult<PyRumpyArray> {
    apply_binary_ufunc(y, x, crate::array::dtype::BinaryOp::Arctan2)
}

#[pyfunction]
pub fn hypot(x1: &Bound<'_, PyAny>, x2: &Bound<'_, PyAny>) -> PyResult<PyRumpyArray> {
    apply_binary_ufunc(x1, x2, crate::array::dtype::BinaryOp::Hypot)
}

#[pyfunction]
pub fn fmax(x1: &Bound<'_, PyAny>, x2: &Bound<'_, PyAny>) -> PyResult<PyRumpyArray> {
    apply_binary_ufunc(x1, x2, crate::array::dtype::BinaryOp::FMax)
}

#[pyfunction]
pub fn fmin(x1: &Bound<'_, PyAny>, x2: &Bound<'_, PyAny>) -> PyResult<PyRumpyArray> {
    apply_binary_ufunc(x1, x2, crate::array::dtype::BinaryOp::FMin)
}

#[pyfunction]
pub fn copysign(x1: &Bound<'_, PyAny>, x2: &Bound<'_, PyAny>) -> PyResult<PyRumpyArray> {
    apply_binary_ufunc(x1, x2, crate::array::dtype::BinaryOp::Copysign)
}

#[pyfunction]
pub fn logaddexp(x1: &Bound<'_, PyAny>, x2: &Bound<'_, PyAny>) -> PyResult<PyRumpyArray> {
    apply_binary_ufunc(x1, x2, crate::array::dtype::BinaryOp::Logaddexp)
}

#[pyfunction]
pub fn logaddexp2(x1: &Bound<'_, PyAny>, x2: &Bound<'_, PyAny>) -> PyResult<PyRumpyArray> {
    apply_binary_ufunc(x1, x2, crate::array::dtype::BinaryOp::Logaddexp2)
}

#[pyfunction]
pub fn nextafter(x1: &Bound<'_, PyAny>, x2: &Bound<'_, PyAny>) -> PyResult<PyRumpyArray> {
    apply_binary_ufunc(x1, x2, crate::array::dtype::BinaryOp::Nextafter)
}

// deg2rad and rad2deg as module-level functions

#[pyfunction]
pub fn deg2rad(x: &Bound<'_, PyAny>) -> PyResult<UnaryResult> {
    let deg_to_rad = std::f64::consts::PI / 180.0;
    apply_unary(x, |v| v * deg_to_rad, |a| Ok(a.scalar_op(deg_to_rad, crate::array::dtype::BinaryOp::Mul)))
}

#[pyfunction]
pub fn rad2deg(x: &Bound<'_, PyAny>) -> PyResult<UnaryResult> {
    let rad_to_deg = 180.0 / std::f64::consts::PI;
    apply_unary(x, |v| v * rad_to_deg, |a| Ok(a.scalar_op(rad_to_deg, crate::array::dtype::BinaryOp::Mul)))
}

// Complex accessors

#[pyfunction]
pub fn real(x: &PyRumpyArray) -> PyRumpyArray {
    PyRumpyArray::new(x.inner.real())
}

#[pyfunction]
pub fn imag(x: &PyRumpyArray) -> PyRumpyArray {
    PyRumpyArray::new(x.inner.imag())
}

#[pyfunction]
pub fn conj(x: &PyRumpyArray) -> PyRumpyArray {
    PyRumpyArray::new(x.inner.conj())
}

#[pyfunction]
pub fn diagonal(x: &PyRumpyArray) -> PyRumpyArray {
    PyRumpyArray::new(x.inner.diagonal())
}

#[pyfunction]
pub fn count_nonzero(x: &PyRumpyArray) -> usize {
    x.inner.count_nonzero()
}

#[pyfunction]
pub fn swapaxes(x: &PyRumpyArray, axis1: usize, axis2: usize) -> PyRumpyArray {
    PyRumpyArray::new(x.inner.swapaxes(axis1, axis2))
}

#[pyfunction]
#[pyo3(signature = (x, axis=-1))]
pub fn sort(x: &PyRumpyArray, axis: Option<isize>) -> PyRumpyArray {
    let resolved_axis = axis.map(|a| resolve_axis(a, x.inner.ndim()));
    PyRumpyArray::new(x.inner.sort(resolved_axis))
}

#[pyfunction]
#[pyo3(signature = (x, axis=-1))]
pub fn argsort(x: &PyRumpyArray, axis: Option<isize>) -> PyRumpyArray {
    let resolved_axis = axis.map(|a| resolve_axis(a, x.inner.ndim()));
    PyRumpyArray::new(x.inner.argsort(resolved_axis))
}

#[pyfunction]
#[pyo3(signature = (x, n=1, axis=-1))]
pub fn diff(x: &PyRumpyArray, n: usize, axis: isize) -> PyRumpyArray {
    PyRumpyArray::new(x.inner.diff(n, resolve_axis(axis, x.inner.ndim())))
}

// Module-level reduction functions

/// Helper for axis bounds checking.
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

/// Resolve a potentially negative axis to a positive index.
fn resolve_axis(axis: isize, ndim: usize) -> usize {
    if axis < 0 {
        (ndim as isize + axis) as usize
    } else {
        axis as usize
    }
}

/// Test if all elements evaluate to True.
#[pyfunction]
#[pyo3(signature = (x, axis=None))]
pub fn all(x: &PyRumpyArray, axis: Option<usize>) -> PyResult<pyarray::ReductionResult> {
    match axis {
        None => Ok(pyarray::ReductionResult::Scalar(if x.inner.all() { 1.0 } else { 0.0 })),
        Some(ax) => {
            check_axis(ax, x.inner.ndim())?;
            Ok(pyarray::ReductionResult::Array(PyRumpyArray::new(x.inner.all_axis(ax))))
        }
    }
}

/// Test if any element evaluates to True.
#[pyfunction]
#[pyo3(signature = (x, axis=None))]
pub fn any(x: &PyRumpyArray, axis: Option<usize>) -> PyResult<pyarray::ReductionResult> {
    match axis {
        None => Ok(pyarray::ReductionResult::Scalar(if x.inner.any() { 1.0 } else { 0.0 })),
        Some(ax) => {
            check_axis(ax, x.inner.ndim())?;
            Ok(pyarray::ReductionResult::Array(PyRumpyArray::new(x.inner.any_axis(ax))))
        }
    }
}

/// Clip values to a range.
#[pyfunction]
#[pyo3(signature = (x, a_min=None, a_max=None))]
pub fn clip(x: &PyRumpyArray, a_min: Option<f64>, a_max: Option<f64>) -> PyRumpyArray {
    PyRumpyArray::new(x.inner.clip(a_min, a_max))
}

/// Sum of array elements.
#[pyfunction]
#[pyo3(signature = (x, axis=None))]
pub fn sum(x: &PyRumpyArray, axis: Option<usize>) -> PyResult<pyarray::ReductionResult> {
    match axis {
        None => Ok(pyarray::ReductionResult::Scalar(x.inner.sum())),
        Some(ax) => {
            check_axis(ax, x.inner.ndim())?;
            Ok(pyarray::ReductionResult::Array(PyRumpyArray::new(x.inner.sum_axis(ax))))
        }
    }
}

/// Product of array elements.
#[pyfunction]
#[pyo3(signature = (x, axis=None))]
pub fn prod(x: &PyRumpyArray, axis: Option<usize>) -> PyResult<pyarray::ReductionResult> {
    match axis {
        None => Ok(pyarray::ReductionResult::Scalar(x.inner.prod())),
        Some(ax) => {
            check_axis(ax, x.inner.ndim())?;
            Ok(pyarray::ReductionResult::Array(PyRumpyArray::new(x.inner.prod_axis(ax))))
        }
    }
}

/// Mean of array elements.
#[pyfunction]
#[pyo3(signature = (x, axis=None))]
pub fn mean(x: &PyRumpyArray, axis: Option<usize>) -> PyResult<pyarray::ReductionResult> {
    match axis {
        None => Ok(pyarray::ReductionResult::Scalar(x.inner.mean())),
        Some(ax) => {
            check_axis(ax, x.inner.ndim())?;
            Ok(pyarray::ReductionResult::Array(PyRumpyArray::new(x.inner.mean_axis(ax))))
        }
    }
}

/// Variance of array elements.
#[pyfunction]
#[pyo3(signature = (x, axis=None))]
pub fn var(x: &PyRumpyArray, axis: Option<usize>) -> PyResult<pyarray::ReductionResult> {
    match axis {
        None => Ok(pyarray::ReductionResult::Scalar(x.inner.var())),
        Some(ax) => {
            check_axis(ax, x.inner.ndim())?;
            Ok(pyarray::ReductionResult::Array(PyRumpyArray::new(x.inner.var_axis(ax))))
        }
    }
}

/// Standard deviation of array elements.
#[pyfunction]
#[pyo3(name = "std", signature = (x, axis=None))]
pub fn std_fn(x: &PyRumpyArray, axis: Option<usize>) -> PyResult<pyarray::ReductionResult> {
    match axis {
        None => Ok(pyarray::ReductionResult::Scalar(x.inner.std())),
        Some(ax) => {
            check_axis(ax, x.inner.ndim())?;
            Ok(pyarray::ReductionResult::Array(PyRumpyArray::new(x.inner.std_axis(ax))))
        }
    }
}

/// Maximum of array elements.
#[pyfunction]
#[pyo3(signature = (x, axis=None))]
pub fn max(x: &PyRumpyArray, axis: Option<usize>) -> PyResult<pyarray::ReductionResult> {
    match axis {
        None => Ok(pyarray::ReductionResult::Scalar(x.inner.max())),
        Some(ax) => {
            check_axis(ax, x.inner.ndim())?;
            Ok(pyarray::ReductionResult::Array(PyRumpyArray::new(x.inner.max_axis(ax))))
        }
    }
}

/// Minimum of array elements.
#[pyfunction]
#[pyo3(signature = (x, axis=None))]
pub fn min(x: &PyRumpyArray, axis: Option<usize>) -> PyResult<pyarray::ReductionResult> {
    match axis {
        None => Ok(pyarray::ReductionResult::Scalar(x.inner.min())),
        Some(ax) => {
            check_axis(ax, x.inner.ndim())?;
            Ok(pyarray::ReductionResult::Array(PyRumpyArray::new(x.inner.min_axis(ax))))
        }
    }
}

/// Index of maximum element.
#[pyfunction]
#[pyo3(signature = (x, axis=None))]
pub fn argmax(x: &PyRumpyArray, axis: Option<usize>) -> PyResult<pyarray::ReductionResult> {
    match axis {
        None => Ok(pyarray::ReductionResult::Scalar(x.inner.argmax() as f64)),
        Some(ax) => {
            check_axis(ax, x.inner.ndim())?;
            Ok(pyarray::ReductionResult::Array(PyRumpyArray::new(x.inner.argmax_axis(ax))))
        }
    }
}

/// Index of minimum element.
#[pyfunction]
#[pyo3(signature = (x, axis=None))]
pub fn argmin(x: &PyRumpyArray, axis: Option<usize>) -> PyResult<pyarray::ReductionResult> {
    match axis {
        None => Ok(pyarray::ReductionResult::Scalar(x.inner.argmin() as f64)),
        Some(ax) => {
            check_axis(ax, x.inner.ndim())?;
            Ok(pyarray::ReductionResult::Array(PyRumpyArray::new(x.inner.argmin_axis(ax))))
        }
    }
}

// ============================================================================
// NaN-aware reduction functions
// ============================================================================

/// Sum ignoring NaN values.
#[pyfunction]
#[pyo3(signature = (x, axis=None))]
pub fn nansum(x: &PyRumpyArray, axis: Option<usize>) -> PyResult<pyarray::ReductionResult> {
    match axis {
        None => Ok(pyarray::ReductionResult::Scalar(x.inner.nansum())),
        Some(ax) => {
            check_axis(ax, x.inner.ndim())?;
            Ok(pyarray::ReductionResult::Array(PyRumpyArray::new(x.inner.nansum_axis(ax))))
        }
    }
}

/// Product ignoring NaN values.
#[pyfunction]
#[pyo3(signature = (x, axis=None))]
pub fn nanprod(x: &PyRumpyArray, axis: Option<usize>) -> PyResult<pyarray::ReductionResult> {
    match axis {
        None => Ok(pyarray::ReductionResult::Scalar(x.inner.nanprod())),
        Some(ax) => {
            check_axis(ax, x.inner.ndim())?;
            Ok(pyarray::ReductionResult::Array(PyRumpyArray::new(x.inner.nanprod_axis(ax))))
        }
    }
}

/// Mean ignoring NaN values.
#[pyfunction]
#[pyo3(signature = (x, axis=None))]
pub fn nanmean(x: &PyRumpyArray, axis: Option<usize>) -> PyResult<pyarray::ReductionResult> {
    match axis {
        None => Ok(pyarray::ReductionResult::Scalar(x.inner.nanmean())),
        Some(ax) => {
            check_axis(ax, x.inner.ndim())?;
            Ok(pyarray::ReductionResult::Array(PyRumpyArray::new(x.inner.nanmean_axis(ax))))
        }
    }
}

/// Variance ignoring NaN values.
#[pyfunction]
#[pyo3(signature = (x, axis=None))]
pub fn nanvar(x: &PyRumpyArray, axis: Option<usize>) -> PyResult<pyarray::ReductionResult> {
    match axis {
        None => Ok(pyarray::ReductionResult::Scalar(x.inner.nanvar())),
        Some(ax) => {
            check_axis(ax, x.inner.ndim())?;
            Ok(pyarray::ReductionResult::Array(PyRumpyArray::new(x.inner.nanvar_axis(ax))))
        }
    }
}

/// Standard deviation ignoring NaN values.
#[pyfunction]
#[pyo3(signature = (x, axis=None))]
pub fn nanstd(x: &PyRumpyArray, axis: Option<usize>) -> PyResult<pyarray::ReductionResult> {
    match axis {
        None => Ok(pyarray::ReductionResult::Scalar(x.inner.nanstd())),
        Some(ax) => {
            check_axis(ax, x.inner.ndim())?;
            Ok(pyarray::ReductionResult::Array(PyRumpyArray::new(x.inner.nanstd_axis(ax))))
        }
    }
}

/// Minimum ignoring NaN values.
#[pyfunction]
#[pyo3(signature = (x, axis=None))]
pub fn nanmin(x: &PyRumpyArray, axis: Option<usize>) -> PyResult<pyarray::ReductionResult> {
    match axis {
        None => Ok(pyarray::ReductionResult::Scalar(x.inner.nanmin())),
        Some(ax) => {
            check_axis(ax, x.inner.ndim())?;
            Ok(pyarray::ReductionResult::Array(PyRumpyArray::new(x.inner.nanmin_axis(ax))))
        }
    }
}

/// Maximum ignoring NaN values.
#[pyfunction]
#[pyo3(signature = (x, axis=None))]
pub fn nanmax(x: &PyRumpyArray, axis: Option<usize>) -> PyResult<pyarray::ReductionResult> {
    match axis {
        None => Ok(pyarray::ReductionResult::Scalar(x.inner.nanmax())),
        Some(ax) => {
            check_axis(ax, x.inner.ndim())?;
            Ok(pyarray::ReductionResult::Array(PyRumpyArray::new(x.inner.nanmax_axis(ax))))
        }
    }
}

/// Index of minimum ignoring NaN values.
#[pyfunction]
#[pyo3(signature = (x, axis=None))]
pub fn nanargmin(x: &PyRumpyArray, axis: Option<usize>) -> PyResult<pyarray::ReductionResult> {
    match axis {
        None => {
            match x.inner.nanargmin() {
                Some(idx) => Ok(pyarray::ReductionResult::Scalar(idx as f64)),
                None => Err(pyo3::exceptions::PyValueError::new_err(
                    "All-NaN slice encountered"
                )),
            }
        }
        Some(ax) => {
            check_axis(ax, x.inner.ndim())?;
            Ok(pyarray::ReductionResult::Array(PyRumpyArray::new(x.inner.nanargmin_axis(ax))))
        }
    }
}

/// Index of maximum ignoring NaN values.
#[pyfunction]
#[pyo3(signature = (x, axis=None))]
pub fn nanargmax(x: &PyRumpyArray, axis: Option<usize>) -> PyResult<pyarray::ReductionResult> {
    match axis {
        None => {
            match x.inner.nanargmax() {
                Some(idx) => Ok(pyarray::ReductionResult::Scalar(idx as f64)),
                None => Err(pyo3::exceptions::PyValueError::new_err(
                    "All-NaN slice encountered"
                )),
            }
        }
        Some(ax) => {
            check_axis(ax, x.inner.ndim())?;
            Ok(pyarray::ReductionResult::Array(PyRumpyArray::new(x.inner.nanargmax_axis(ax))))
        }
    }
}

/// Round to the given number of decimals.
#[pyfunction]
#[pyo3(signature = (x, decimals=0))]
pub fn round(x: &PyRumpyArray, decimals: i32) -> PyRumpyArray {
    PyRumpyArray::new(x.inner.round(decimals))
}

/// Cumulative sum along axis (or flattened if axis is None).
#[pyfunction]
#[pyo3(signature = (x, axis=None))]
pub fn cumsum(x: &PyRumpyArray, axis: Option<usize>) -> PyResult<PyRumpyArray> {
    if let Some(ax) = axis {
        if ax >= x.inner.ndim() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "axis {} is out of bounds for array of dimension {}",
                ax, x.inner.ndim()
            )));
        }
    }
    Ok(PyRumpyArray::new(x.inner.cumsum(axis)))
}

/// Cumulative product along axis (or flattened if axis is None).
#[pyfunction]
#[pyo3(signature = (x, axis=None))]
pub fn cumprod(x: &PyRumpyArray, axis: Option<usize>) -> PyResult<PyRumpyArray> {
    if let Some(ax) = axis {
        if ax >= x.inner.ndim() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "axis {} is out of bounds for array of dimension {}",
                ax, x.inner.ndim()
            )));
        }
    }
    Ok(PyRumpyArray::new(x.inner.cumprod(axis)))
}

/// Convert a Python object to RumpyArray (for concatenate, etc.)
fn to_rumpy_array(obj: &Bound<'_, pyo3::PyAny>) -> PyResult<RumpyArray> {
    // Try PyRumpyArray first
    if let Ok(arr) = obj.extract::<pyo3::PyRef<'_, PyRumpyArray>>() {
        return Ok(arr.inner.clone());
    }

    // Try Python list
    if let Ok(list) = obj.downcast::<PyList>() {
        let arr = from_list(list, None)?;
        return Ok(arr.inner);
    }

    // Try scalar
    if let Ok(val) = obj.extract::<f64>() {
        return Ok(RumpyArray::from_vec(vec![val], DType::float64()));
    }

    Err(pyo3::exceptions::PyTypeError::new_err(
        "cannot convert to array"
    ))
}

/// Concatenate arrays along an axis.
#[pyfunction]
#[pyo3(signature = (arrays, axis=0))]
pub fn concatenate(arrays: &Bound<'_, PyList>, axis: usize) -> PyResult<PyRumpyArray> {
    let mut inner_arrays: Vec<RumpyArray> = Vec::with_capacity(arrays.len());
    for item in arrays.iter() {
        inner_arrays.push(to_rumpy_array(&item)?);
    }

    crate::array::concatenate(&inner_arrays, axis)
        .map(PyRumpyArray::new)
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "arrays must have same shape except in concatenation axis"
            )
        })
}

/// Stack arrays along a new axis.
#[pyfunction]
#[pyo3(signature = (arrays, axis=0))]
pub fn stack(arrays: &Bound<'_, PyList>, axis: usize) -> PyResult<PyRumpyArray> {
    let mut inner_arrays: Vec<RumpyArray> = Vec::with_capacity(arrays.len());
    for item in arrays.iter() {
        inner_arrays.push(to_rumpy_array(&item)?);
    }
    crate::array::stack(&inner_arrays, axis)
        .map(PyRumpyArray::new)
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("arrays must have same shape for stack")
        })
}

/// Stack arrays vertically (row-wise).
#[pyfunction]
pub fn vstack(arrays: &Bound<'_, PyList>) -> PyResult<PyRumpyArray> {
    if arrays.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err("need at least one array"));
    }

    // For 1D arrays, reshape to (1, N) first
    let mut inner_arrays: Vec<RumpyArray> = Vec::with_capacity(arrays.len());
    for item in arrays.iter() {
        let arr = to_rumpy_array(&item)?;
        if arr.ndim() == 1 {
            inner_arrays.push(arr.reshape(vec![1, arr.size()]).unwrap_or_else(|| arr.clone()));
        } else {
            inner_arrays.push(arr);
        }
    }

    crate::array::concatenate(&inner_arrays, 0)
        .map(PyRumpyArray::new)
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("arrays must have same shape for vstack")
        })
}

/// Stack arrays horizontally (column-wise).
#[pyfunction]
pub fn hstack(arrays: &Bound<'_, PyList>) -> PyResult<PyRumpyArray> {
    if arrays.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err("need at least one array"));
    }

    let mut inner_arrays: Vec<RumpyArray> = Vec::with_capacity(arrays.len());
    for item in arrays.iter() {
        inner_arrays.push(to_rumpy_array(&item)?);
    }

    let first = &inner_arrays[0];
    let axis = if first.ndim() == 1 { 0 } else { 1 };

    crate::array::concatenate(&inner_arrays, axis)
        .map(PyRumpyArray::new)
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("arrays must have same shape for hstack")
        })
}

/// Split array into equal parts.
#[pyfunction]
#[pyo3(signature = (arr, num_sections, axis=0))]
pub fn split(arr: &PyRumpyArray, num_sections: usize, axis: usize) -> PyResult<Vec<PyRumpyArray>> {
    crate::array::split(&arr.inner, num_sections, axis)
        .map(|sections| sections.into_iter().map(PyRumpyArray::new).collect())
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "array split does not result in an equal division"
            )
        })
}

/// Split array into sections, allowing unequal sizes.
#[pyfunction]
#[pyo3(signature = (arr, num_sections, axis=0))]
pub fn array_split(arr: &PyRumpyArray, num_sections: usize, axis: usize) -> PyResult<Vec<PyRumpyArray>> {
    crate::array::array_split(&arr.inner, num_sections, axis)
        .map(|sections| sections.into_iter().map(PyRumpyArray::new).collect())
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("invalid split parameters")
        })
}

/// Expand array dimensions at specified axis.
#[pyfunction]
pub fn expand_dims(arr: &PyRumpyArray, axis: isize) -> PyResult<PyRumpyArray> {
    // Handle negative axis
    let ndim = arr.inner.ndim();
    let axis = if axis < 0 {
        (ndim as isize + axis + 1) as usize
    } else {
        axis as usize
    };

    arr.inner.expand_dims(axis)
        .map(PyRumpyArray::new)
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "axis {} is out of bounds for array of dimension {}",
                axis, ndim
            ))
        })
}

/// Remove single-dimensional entries from array shape.
#[pyfunction]
pub fn squeeze(arr: &PyRumpyArray) -> PyRumpyArray {
    PyRumpyArray::new(arr.inner.squeeze())
}

/// Reverse the order of elements along given axis.
#[pyfunction]
#[pyo3(signature = (arr, axis=None))]
pub fn flip(arr: &PyRumpyArray, axis: Option<isize>) -> PyResult<PyRumpyArray> {
    match axis {
        None => {
            // Flip all axes
            let mut result = arr.inner.clone();
            for ax in 0..result.ndim() {
                result = result.flip(ax).unwrap();
            }
            Ok(PyRumpyArray::new(result))
        }
        Some(ax) => {
            let ndim = arr.inner.ndim();
            let axis = if ax < 0 {
                (ndim as isize + ax) as usize
            } else {
                ax as usize
            };
            arr.inner.flip(axis)
                .map(PyRumpyArray::new)
                .ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "axis {} is out of bounds for array of dimension {}",
                        axis, ndim
                    ))
                })
        }
    }
}

/// Flip array vertically (axis=0).
#[pyfunction]
pub fn flipud(arr: &PyRumpyArray) -> PyResult<PyRumpyArray> {
    if arr.inner.ndim() < 1 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "flipud requires array with at least 1 dimension"
        ));
    }
    Ok(PyRumpyArray::new(arr.inner.flip(0).unwrap()))
}

/// Flip array horizontally (axis=1).
#[pyfunction]
pub fn fliplr(arr: &PyRumpyArray) -> PyResult<PyRumpyArray> {
    if arr.inner.ndim() < 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "fliplr requires array with at least 2 dimensions"
        ));
    }
    Ok(PyRumpyArray::new(arr.inner.flip(1).unwrap()))
}

// sort and argsort are defined earlier with axis parameter

/// Return unique sorted values.
#[pyfunction]
pub fn unique(arr: &PyRumpyArray) -> PyRumpyArray {
    PyRumpyArray::new(arr.inner.unique())
}

/// Return indices of non-zero elements.
#[pyfunction]
pub fn nonzero(arr: &PyRumpyArray) -> Vec<PyRumpyArray> {
    arr.inner.nonzero().into_iter().map(PyRumpyArray::new).collect()
}

/// Count occurrences of each value in an array of non-negative integers.
#[pyfunction]
#[pyo3(signature = (x, minlength=0))]
pub fn bincount(x: &PyRumpyArray, minlength: usize) -> PyResult<PyRumpyArray> {
    crate::array::bincount(&x.inner, minlength)
        .map(PyRumpyArray::new)
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("bincount requires 1D array of non-negative integers")
        })
}

/// Compute the q-th percentile of the data along the specified axis.
#[pyfunction]
#[pyo3(signature = (a, q, axis=None))]
pub fn percentile(a: &PyRumpyArray, q: &Bound<'_, PyAny>, axis: Option<usize>) -> PyResult<PyRumpyArray> {
    // Parse q - can be scalar or array-like
    let q_values: Vec<f64> = if let Ok(val) = q.extract::<f64>() {
        vec![val]
    } else if let Ok(list) = q.extract::<Vec<f64>>() {
        list
    } else {
        return Err(pyo3::exceptions::PyTypeError::new_err("q must be a number or list of numbers"));
    };

    // Validate q values are in [0, 100]
    for &qv in &q_values {
        if !(0.0..=100.0).contains(&qv) {
            return Err(pyo3::exceptions::PyValueError::new_err("percentiles must be in range [0, 100]"));
        }
    }

    crate::array::percentile(&a.inner, &q_values, axis)
        .map(PyRumpyArray::new)
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("percentile computation failed")
        })
}

/// Compute the q-th quantile of the data (q in [0, 1]).
#[pyfunction]
#[pyo3(signature = (a, q, axis=None))]
pub fn quantile(a: &PyRumpyArray, q: &Bound<'_, PyAny>, axis: Option<usize>) -> PyResult<PyRumpyArray> {
    // Parse q - can be scalar or array-like
    let q_values: Vec<f64> = if let Ok(val) = q.extract::<f64>() {
        vec![val]
    } else if let Ok(list) = q.extract::<Vec<f64>>() {
        list
    } else {
        return Err(pyo3::exceptions::PyTypeError::new_err("q must be a number or list of numbers"));
    };

    // Validate q values are in [0, 1]
    for &qv in &q_values {
        if !(0.0..=1.0).contains(&qv) {
            return Err(pyo3::exceptions::PyValueError::new_err("quantiles must be in range [0, 1]"));
        }
    }

    // Convert to percentiles
    let pct_values: Vec<f64> = q_values.iter().map(|&q| q * 100.0).collect();

    crate::array::percentile(&a.inner, &pct_values, axis)
        .map(PyRumpyArray::new)
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("quantile computation failed")
        })
}

/// 1D discrete convolution.
#[pyfunction]
#[pyo3(signature = (a, v, mode="full"))]
pub fn convolve(a: &PyRumpyArray, v: &PyRumpyArray, mode: &str) -> PyResult<PyRumpyArray> {
    crate::array::convolve(&a.inner, &v.inner, mode)
        .map(PyRumpyArray::new)
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("convolve requires 1D arrays and valid mode")
        })
}

/// Matrix multiplication.
#[pyfunction]
pub fn matmul(a: &PyRumpyArray, b: &PyRumpyArray) -> PyResult<PyRumpyArray> {
    crate::ops::matmul::matmul(&a.inner, &b.inner)
        .map(PyRumpyArray::new)
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("matmul: incompatible shapes")
        })
}

/// Dot product with numpy semantics.
#[pyfunction]
pub fn dot(a: &PyRumpyArray, b: &PyRumpyArray) -> PyResult<PyRumpyArray> {
    crate::ops::dot::dot(&a.inner, &b.inner)
        .map(PyRumpyArray::new)
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("dot: incompatible shapes")
        })
}

/// Inner product of two arrays.
#[pyfunction]
pub fn inner(a: &PyRumpyArray, b: &PyRumpyArray) -> PyResult<PyRumpyArray> {
    crate::ops::inner::inner(&a.inner, &b.inner)
        .map(PyRumpyArray::new)
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("inner: incompatible shapes")
        })
}

/// Outer product of two arrays.
#[pyfunction]
pub fn outer(a: &PyRumpyArray, b: &PyRumpyArray) -> PyResult<PyRumpyArray> {
    crate::ops::outer::outer(&a.inner, &b.inner)
        .map(PyRumpyArray::new)
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("outer: incompatible shapes")
        })
}

/// Solve linear system Ax = b.
#[pyfunction]
pub fn solve(a: &PyRumpyArray, b: &PyRumpyArray) -> PyResult<PyRumpyArray> {
    crate::ops::solve::solve(&a.inner, &b.inner)
        .map(PyRumpyArray::new)
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("solve: invalid dimensions or singular matrix")
        })
}

/// Compute trace of a matrix (sum of diagonal elements).
#[pyfunction]
pub fn trace(a: &PyRumpyArray) -> PyResult<f64> {
    crate::ops::linalg::trace(&a.inner)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("trace requires 2D array"))
}

/// Compute determinant of a square matrix.
#[pyfunction]
pub fn det(a: &PyRumpyArray) -> PyResult<f64> {
    crate::ops::linalg::det(&a.inner)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("det requires square 2D array"))
}

/// Compute matrix/vector norm.
#[pyfunction]
#[pyo3(signature = (a, ord=None))]
pub fn norm(a: &PyRumpyArray, ord: Option<&str>) -> PyResult<f64> {
    crate::ops::linalg::norm(&a.inner, ord)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("unsupported norm type"))
}

/// QR decomposition: A = QR.
#[pyfunction]
pub fn qr(a: &PyRumpyArray) -> PyResult<(PyRumpyArray, PyRumpyArray)> {
    crate::ops::linalg::qr(&a.inner)
        .map(|(q, r)| (PyRumpyArray::new(q), PyRumpyArray::new(r)))
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("qr requires 2D array"))
}

/// SVD decomposition: A = U @ diag(S) @ Vt.
#[pyfunction]
pub fn svd(a: &PyRumpyArray) -> PyResult<(PyRumpyArray, PyRumpyArray, PyRumpyArray)> {
    crate::ops::linalg::svd(&a.inner)
        .map(|(u, s, vt)| (PyRumpyArray::new(u), PyRumpyArray::new(s), PyRumpyArray::new(vt)))
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("svd requires 2D array"))
}

/// Matrix inverse.
#[pyfunction]
pub fn inv(a: &PyRumpyArray) -> PyResult<PyRumpyArray> {
    crate::ops::linalg::inv(&a.inner)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("inv requires square 2D array"))
}

/// Eigendecomposition of symmetric matrix.
#[pyfunction]
pub fn eigh(a: &PyRumpyArray) -> PyResult<(PyRumpyArray, PyRumpyArray)> {
    crate::ops::linalg::eigh(&a.inner)
        .map(|(w, v)| (PyRumpyArray::new(w), PyRumpyArray::new(v)))
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("eigh requires square 2D array"))
}

/// Extract diagonal or construct diagonal matrix.
#[pyfunction]
pub fn diag(a: &PyRumpyArray) -> PyResult<PyRumpyArray> {
    crate::ops::linalg::diag(&a.inner)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("diag requires 1D or 2D array"))
}

/// Conditional selection: where(condition, x, y).
/// Returns elements from x where condition is true, else from y.
#[pyfunction]
#[pyo3(name = "where")]
pub fn where_fn(
    condition: &PyRumpyArray,
    x: &Bound<'_, pyo3::PyAny>,
    y: &Bound<'_, pyo3::PyAny>,
) -> PyResult<PyRumpyArray> {
    // Extract x - could be array or scalar
    let x_arr = if let Ok(arr) = x.extract::<pyo3::PyRef<'_, PyRumpyArray>>() {
        arr.inner.clone()
    } else if let Ok(scalar) = x.extract::<f64>() {
        RumpyArray::full(vec![1], scalar, DType::float64())
    } else {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "x must be ndarray or number",
        ));
    };

    // Extract y - could be array or scalar
    let y_arr = if let Ok(arr) = y.extract::<pyo3::PyRef<'_, PyRumpyArray>>() {
        arr.inner.clone()
    } else if let Ok(scalar) = y.extract::<f64>() {
        RumpyArray::full(vec![1], scalar, DType::float64())
    } else {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "y must be ndarray or number",
        ));
    };

    crate::ops::where_select(&condition.inner, &x_arr, &y_arr)
        .map(PyRumpyArray::new)
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("operands could not be broadcast together")
        })
}

// Logical operations

/// Element-wise logical AND.
/// Treats any non-zero value as true, returns bool array.
#[pyfunction]
pub fn logical_and(x1: &PyRumpyArray, x2: &PyRumpyArray) -> PyResult<PyRumpyArray> {
    crate::ops::logical_and(&x1.inner, &x2.inner)
        .map(PyRumpyArray::new)
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("operands could not be broadcast together")
        })
}

/// Element-wise logical OR.
/// Treats any non-zero value as true, returns bool array.
#[pyfunction]
pub fn logical_or(x1: &PyRumpyArray, x2: &PyRumpyArray) -> PyResult<PyRumpyArray> {
    crate::ops::logical_or(&x1.inner, &x2.inner)
        .map(PyRumpyArray::new)
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("operands could not be broadcast together")
        })
}

/// Element-wise logical XOR.
/// Treats any non-zero value as true, returns bool array.
#[pyfunction]
pub fn logical_xor(x1: &PyRumpyArray, x2: &PyRumpyArray) -> PyResult<PyRumpyArray> {
    crate::ops::logical_xor(&x1.inner, &x2.inner)
        .map(PyRumpyArray::new)
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("operands could not be broadcast together")
        })
}

/// Element-wise logical NOT.
/// Treats any non-zero value as true, returns bool array.
#[pyfunction]
pub fn logical_not(x: &PyRumpyArray) -> PyRumpyArray {
    PyRumpyArray::new(crate::ops::logical_not(&x.inner))
}

// Comparison operations

/// Element-wise equality test.
#[pyfunction]
pub fn equal(x1: &Bound<'_, PyAny>, x2: &Bound<'_, PyAny>) -> PyResult<PyRumpyArray> {
    let a = to_rumpy_array(x1)?;
    let b = to_rumpy_array(x2)?;
    crate::ops::equal(&a, &b)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("operands could not be broadcast together"))
}

/// Element-wise not-equal test.
#[pyfunction]
pub fn not_equal(x1: &Bound<'_, PyAny>, x2: &Bound<'_, PyAny>) -> PyResult<PyRumpyArray> {
    let a = to_rumpy_array(x1)?;
    let b = to_rumpy_array(x2)?;
    crate::ops::not_equal(&a, &b)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("operands could not be broadcast together"))
}

/// Element-wise less-than test.
#[pyfunction]
pub fn less(x1: &Bound<'_, PyAny>, x2: &Bound<'_, PyAny>) -> PyResult<PyRumpyArray> {
    let a = to_rumpy_array(x1)?;
    let b = to_rumpy_array(x2)?;
    crate::ops::less(&a, &b)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("operands could not be broadcast together"))
}

/// Element-wise less-than-or-equal test.
#[pyfunction]
pub fn less_equal(x1: &Bound<'_, PyAny>, x2: &Bound<'_, PyAny>) -> PyResult<PyRumpyArray> {
    let a = to_rumpy_array(x1)?;
    let b = to_rumpy_array(x2)?;
    crate::ops::less_equal(&a, &b)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("operands could not be broadcast together"))
}

/// Element-wise greater-than test.
#[pyfunction]
pub fn greater(x1: &Bound<'_, PyAny>, x2: &Bound<'_, PyAny>) -> PyResult<PyRumpyArray> {
    let a = to_rumpy_array(x1)?;
    let b = to_rumpy_array(x2)?;
    crate::ops::greater(&a, &b)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("operands could not be broadcast together"))
}

/// Element-wise greater-than-or-equal test.
#[pyfunction]
pub fn greater_equal(x1: &Bound<'_, PyAny>, x2: &Bound<'_, PyAny>) -> PyResult<PyRumpyArray> {
    let a = to_rumpy_array(x1)?;
    let b = to_rumpy_array(x2)?;
    crate::ops::greater_equal(&a, &b)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("operands could not be broadcast together"))
}

/// Element-wise approximate equality test: |a - b| <= atol + rtol * |b|.
#[pyfunction]
#[pyo3(signature = (a, b, rtol=1e-5, atol=1e-8, equal_nan=false))]
pub fn isclose(
    a: &Bound<'_, PyAny>,
    b: &Bound<'_, PyAny>,
    rtol: f64,
    atol: f64,
    equal_nan: bool,
) -> PyResult<PyRumpyArray> {
    let arr_a = to_rumpy_array(a)?;
    let arr_b = to_rumpy_array(b)?;
    crate::ops::isclose(&arr_a, &arr_b, rtol, atol, equal_nan)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("operands could not be broadcast together"))
}

/// Test if all elements are approximately equal.
#[pyfunction]
#[pyo3(signature = (a, b, rtol=1e-5, atol=1e-8, equal_nan=false))]
pub fn allclose(
    a: &Bound<'_, PyAny>,
    b: &Bound<'_, PyAny>,
    rtol: f64,
    atol: f64,
    equal_nan: bool,
) -> PyResult<bool> {
    let arr_a = to_rumpy_array(a)?;
    let arr_b = to_rumpy_array(b)?;
    crate::ops::allclose(&arr_a, &arr_b, rtol, atol, equal_nan)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("operands could not be broadcast together"))
}

/// Test if two arrays have the same shape and elements.
#[pyfunction]
pub fn array_equal(x1: &PyRumpyArray, x2: &PyRumpyArray) -> bool {
    crate::ops::array_equal(&x1.inner, &x2.inner)
}

// Bitwise operations

/// Element-wise bitwise AND.
#[pyfunction]
pub fn bitwise_and(x1: &PyRumpyArray, x2: &PyRumpyArray) -> PyResult<PyRumpyArray> {
    crate::ops::bitwise_and(&x1.inner, &x2.inner)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("operands could not be broadcast together"))
}

/// Element-wise bitwise OR.
#[pyfunction]
pub fn bitwise_or(x1: &PyRumpyArray, x2: &PyRumpyArray) -> PyResult<PyRumpyArray> {
    crate::ops::bitwise_or(&x1.inner, &x2.inner)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("operands could not be broadcast together"))
}

/// Element-wise bitwise XOR.
#[pyfunction]
pub fn bitwise_xor(x1: &PyRumpyArray, x2: &PyRumpyArray) -> PyResult<PyRumpyArray> {
    crate::ops::bitwise_xor(&x1.inner, &x2.inner)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("operands could not be broadcast together"))
}

/// Element-wise bitwise NOT (invert).
#[pyfunction]
pub fn bitwise_not(x: &PyRumpyArray) -> PyResult<PyRumpyArray> {
    crate::ops::bitwise_not(&x.inner)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyTypeError::new_err("bitwise_not not supported for this dtype"))
}

/// Alias for bitwise_not.
#[pyfunction]
pub fn invert(x: &PyRumpyArray) -> PyResult<PyRumpyArray> {
    crate::ops::bitwise_not(&x.inner)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyTypeError::new_err("bitwise_not not supported for this dtype"))
}

/// Element-wise left shift.
#[pyfunction]
pub fn left_shift(x1: &PyRumpyArray, x2: &PyRumpyArray) -> PyResult<PyRumpyArray> {
    crate::ops::left_shift(&x1.inner, &x2.inner)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("operands could not be broadcast together"))
}

/// Element-wise right shift.
#[pyfunction]
pub fn right_shift(x1: &PyRumpyArray, x2: &PyRumpyArray) -> PyResult<PyRumpyArray> {
    crate::ops::right_shift(&x1.inner, &x2.inner)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("operands could not be broadcast together"))
}

/// Register Python module contents.
pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRumpyArray>()?;
    // Register submodules
    random::register_submodule(m)?;
    fft::register_submodule(m)?;
    linalg::register_submodule(m)?;
    // Constructors
    m.add_function(wrap_pyfunction!(zeros, m)?)?;
    m.add_function(wrap_pyfunction!(ones, m)?)?;
    m.add_function(wrap_pyfunction!(arange, m)?)?;
    m.add_function(wrap_pyfunction!(linspace, m)?)?;
    m.add_function(wrap_pyfunction!(eye, m)?)?;
    m.add_function(wrap_pyfunction!(full, m)?)?;
    m.add_function(wrap_pyfunction!(empty, m)?)?;
    m.add_function(wrap_pyfunction!(zeros_like, m)?)?;
    m.add_function(wrap_pyfunction!(ones_like, m)?)?;
    m.add_function(wrap_pyfunction!(empty_like, m)?)?;
    m.add_function(wrap_pyfunction!(copy, m)?)?;
    m.add_function(wrap_pyfunction!(asarray, m)?)?;
    m.add_function(wrap_pyfunction!(array, m)?)?;
    // Reductions
    m.add_function(wrap_pyfunction!(sum, m)?)?;
    m.add_function(wrap_pyfunction!(prod, m)?)?;
    m.add_function(wrap_pyfunction!(mean, m)?)?;
    m.add_function(wrap_pyfunction!(var, m)?)?;
    m.add_function(wrap_pyfunction!(std_fn, m)?)?;
    m.add_function(wrap_pyfunction!(max, m)?)?;
    m.add_function(wrap_pyfunction!(min, m)?)?;
    m.add_function(wrap_pyfunction!(argmax, m)?)?;
    m.add_function(wrap_pyfunction!(argmin, m)?)?;
    // NaN-aware reductions
    m.add_function(wrap_pyfunction!(nansum, m)?)?;
    m.add_function(wrap_pyfunction!(nanprod, m)?)?;
    m.add_function(wrap_pyfunction!(nanmean, m)?)?;
    m.add_function(wrap_pyfunction!(nanvar, m)?)?;
    m.add_function(wrap_pyfunction!(nanstd, m)?)?;
    m.add_function(wrap_pyfunction!(nanmin, m)?)?;
    m.add_function(wrap_pyfunction!(nanmax, m)?)?;
    m.add_function(wrap_pyfunction!(nanargmin, m)?)?;
    m.add_function(wrap_pyfunction!(nanargmax, m)?)?;
    // Math ufuncs
    m.add_function(wrap_pyfunction!(sqrt, m)?)?;
    m.add_function(wrap_pyfunction!(exp, m)?)?;
    m.add_function(wrap_pyfunction!(log, m)?)?;
    m.add_function(wrap_pyfunction!(sin, m)?)?;
    m.add_function(wrap_pyfunction!(cos, m)?)?;
    m.add_function(wrap_pyfunction!(tan, m)?)?;
    m.add_function(wrap_pyfunction!(floor, m)?)?;
    m.add_function(wrap_pyfunction!(ceil, m)?)?;
    m.add_function(wrap_pyfunction!(arcsin, m)?)?;
    m.add_function(wrap_pyfunction!(arccos, m)?)?;
    m.add_function(wrap_pyfunction!(arctan, m)?)?;
    m.add_function(wrap_pyfunction!(log10, m)?)?;
    m.add_function(wrap_pyfunction!(log2, m)?)?;
    m.add_function(wrap_pyfunction!(sinh, m)?)?;
    m.add_function(wrap_pyfunction!(cosh, m)?)?;
    m.add_function(wrap_pyfunction!(tanh, m)?)?;
    m.add_function(wrap_pyfunction!(sign, m)?)?;
    m.add_function(wrap_pyfunction!(isnan, m)?)?;
    m.add_function(wrap_pyfunction!(isinf, m)?)?;
    m.add_function(wrap_pyfunction!(isfinite, m)?)?;
    m.add_function(wrap_pyfunction!(abs, m)?)?;

    m.add_function(wrap_pyfunction!(square, m)?)?;
    m.add_function(wrap_pyfunction!(positive, m)?)?;
    m.add_function(wrap_pyfunction!(negative, m)?)?;
    m.add_function(wrap_pyfunction!(reciprocal, m)?)?;
    m.add_function(wrap_pyfunction!(exp2, m)?)?;
    m.add_function(wrap_pyfunction!(expm1, m)?)?;
    m.add_function(wrap_pyfunction!(log1p, m)?)?;
    m.add_function(wrap_pyfunction!(cbrt, m)?)?;
    m.add_function(wrap_pyfunction!(trunc, m)?)?;
    m.add_function(wrap_pyfunction!(rint, m)?)?;
    m.add_function(wrap_pyfunction!(arcsinh, m)?)?;
    m.add_function(wrap_pyfunction!(arccosh, m)?)?;
    m.add_function(wrap_pyfunction!(arctanh, m)?)?;
    m.add_function(wrap_pyfunction!(signbit, m)?)?;
    m.add_function(wrap_pyfunction!(nan_to_num, m)?)?;
    m.add_function(wrap_pyfunction!(maximum, m)?)?;
    m.add_function(wrap_pyfunction!(minimum, m)?)?;
    m.add_function(wrap_pyfunction!(add, m)?)?;
    m.add_function(wrap_pyfunction!(subtract, m)?)?;
    m.add_function(wrap_pyfunction!(multiply, m)?)?;
    m.add_function(wrap_pyfunction!(divide, m)?)?;
    m.add_function(wrap_pyfunction!(power, m)?)?;
    m.add_function(wrap_pyfunction!(floor_divide, m)?)?;
    m.add_function(wrap_pyfunction!(remainder, m)?)?;
    // Stream 2: Binary math operations
    m.add_function(wrap_pyfunction!(arctan2, m)?)?;
    m.add_function(wrap_pyfunction!(hypot, m)?)?;
    m.add_function(wrap_pyfunction!(fmax, m)?)?;
    m.add_function(wrap_pyfunction!(fmin, m)?)?;
    m.add_function(wrap_pyfunction!(copysign, m)?)?;
    m.add_function(wrap_pyfunction!(logaddexp, m)?)?;
    m.add_function(wrap_pyfunction!(logaddexp2, m)?)?;
    m.add_function(wrap_pyfunction!(nextafter, m)?)?;
    m.add_function(wrap_pyfunction!(deg2rad, m)?)?;
    m.add_function(wrap_pyfunction!(rad2deg, m)?)?;
    m.add_function(wrap_pyfunction!(real, m)?)?;
    m.add_function(wrap_pyfunction!(imag, m)?)?;
    m.add_function(wrap_pyfunction!(conj, m)?)?;
    m.add_function(wrap_pyfunction!(diagonal, m)?)?;
    m.add_function(wrap_pyfunction!(count_nonzero, m)?)?;
    m.add_function(wrap_pyfunction!(swapaxes, m)?)?;
    m.add_function(wrap_pyfunction!(sort, m)?)?;
    m.add_function(wrap_pyfunction!(argsort, m)?)?;
    m.add_function(wrap_pyfunction!(diff, m)?)?;
    m.add_function(wrap_pyfunction!(all, m)?)?;
    m.add_function(wrap_pyfunction!(any, m)?)?;
    m.add_function(wrap_pyfunction!(clip, m)?)?;
    m.add_function(wrap_pyfunction!(round, m)?)?;
    m.add_function(wrap_pyfunction!(cumsum, m)?)?;
    m.add_function(wrap_pyfunction!(cumprod, m)?)?;
    // Conditional
    m.add_function(wrap_pyfunction!(where_fn, m)?)?;
    // Logical operations
    m.add_function(wrap_pyfunction!(logical_and, m)?)?;
    m.add_function(wrap_pyfunction!(logical_or, m)?)?;
    m.add_function(wrap_pyfunction!(logical_xor, m)?)?;
    m.add_function(wrap_pyfunction!(logical_not, m)?)?;
    // Comparison operations
    m.add_function(wrap_pyfunction!(equal, m)?)?;
    m.add_function(wrap_pyfunction!(not_equal, m)?)?;
    m.add_function(wrap_pyfunction!(less, m)?)?;
    m.add_function(wrap_pyfunction!(less_equal, m)?)?;
    m.add_function(wrap_pyfunction!(greater, m)?)?;
    m.add_function(wrap_pyfunction!(greater_equal, m)?)?;
    m.add_function(wrap_pyfunction!(isclose, m)?)?;
    m.add_function(wrap_pyfunction!(allclose, m)?)?;
    m.add_function(wrap_pyfunction!(array_equal, m)?)?;
    // Bitwise operations
    m.add_function(wrap_pyfunction!(bitwise_and, m)?)?;
    m.add_function(wrap_pyfunction!(bitwise_or, m)?)?;
    m.add_function(wrap_pyfunction!(bitwise_xor, m)?)?;
    m.add_function(wrap_pyfunction!(bitwise_not, m)?)?;
    m.add_function(wrap_pyfunction!(invert, m)?)?;
    m.add_function(wrap_pyfunction!(left_shift, m)?)?;
    m.add_function(wrap_pyfunction!(right_shift, m)?)?;
    // Shape manipulation
    m.add_function(wrap_pyfunction!(expand_dims, m)?)?;
    m.add_function(wrap_pyfunction!(squeeze, m)?)?;
    m.add_function(wrap_pyfunction!(flip, m)?)?;
    m.add_function(wrap_pyfunction!(flipud, m)?)?;
    m.add_function(wrap_pyfunction!(fliplr, m)?)?;
    // Concatenation
    m.add_function(wrap_pyfunction!(concatenate, m)?)?;
    m.add_function(wrap_pyfunction!(stack, m)?)?;
    m.add_function(wrap_pyfunction!(vstack, m)?)?;
    m.add_function(wrap_pyfunction!(hstack, m)?)?;
    // Splitting
    m.add_function(wrap_pyfunction!(split, m)?)?;
    m.add_function(wrap_pyfunction!(array_split, m)?)?;
    // Sorting - sort and argsort are registered earlier with axis parameter
    m.add_function(wrap_pyfunction!(unique, m)?)?;
    m.add_function(wrap_pyfunction!(nonzero, m)?)?;
    // Counting and statistics
    m.add_function(wrap_pyfunction!(bincount, m)?)?;
    m.add_function(wrap_pyfunction!(percentile, m)?)?;
    m.add_function(wrap_pyfunction!(quantile, m)?)?;
    // Signal processing
    m.add_function(wrap_pyfunction!(convolve, m)?)?;
    // Linear algebra
    m.add_function(wrap_pyfunction!(matmul, m)?)?;
    m.add_function(wrap_pyfunction!(dot, m)?)?;
    m.add_function(wrap_pyfunction!(inner, m)?)?;
    m.add_function(wrap_pyfunction!(outer, m)?)?;
    m.add_function(wrap_pyfunction!(solve, m)?)?;
    m.add_function(wrap_pyfunction!(trace, m)?)?;
    m.add_function(wrap_pyfunction!(det, m)?)?;
    m.add_function(wrap_pyfunction!(norm, m)?)?;
    m.add_function(wrap_pyfunction!(qr, m)?)?;
    m.add_function(wrap_pyfunction!(svd, m)?)?;
    m.add_function(wrap_pyfunction!(inv, m)?)?;
    m.add_function(wrap_pyfunction!(eigh, m)?)?;
    m.add_function(wrap_pyfunction!(diag, m)?)?;
    // Dtype constants (as strings, compatible with our dtype= parameters)
    m.add("float32", "float32")?;
    m.add("float64", "float64")?;
    m.add("int16", "int16")?;
    m.add("int32", "int32")?;
    m.add("int64", "int64")?;
    m.add("uint8", "uint8")?;
    m.add("uint16", "uint16")?;
    m.add("uint32", "uint32")?;
    m.add("uint64", "uint64")?;
    m.add("bool_", "bool")?;  // bool_ to avoid Python keyword conflict
    m.add("complex64", "complex64")?;
    m.add("complex128", "complex128")?;
    // newaxis is None in numpy (used for broadcasting)
    m.add("newaxis", m.py().None())?;
    Ok(())
}
