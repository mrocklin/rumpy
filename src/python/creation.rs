// Python bindings for array creation functions.

use pyo3::prelude::*;
use pyo3::types::PyList;

use crate::array::{increment_indices, DType, RumpyArray};
use super::{parse_dtype, parse_shape, PyRumpyArray};

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

/// Create array filled with given value, with same shape and dtype as input.
#[pyfunction]
#[pyo3(signature = (a, fill_value, dtype=None))]
pub fn full_like(a: &PyRumpyArray, fill_value: f64, dtype: Option<&str>) -> PyResult<PyRumpyArray> {
    let shape = a.inner.shape().to_vec();
    let dtype = match dtype {
        Some(dt) => parse_dtype(dt)?,
        None => a.inner.dtype(),
    };
    Ok(PyRumpyArray::new(RumpyArray::full(shape, fill_value, dtype)))
}

/// Create an n x n identity matrix.
#[pyfunction]
#[pyo3(signature = (n, dtype=None))]
pub fn identity(n: usize, dtype: Option<&str>) -> PyResult<PyRumpyArray> {
    let dtype = parse_dtype(dtype.unwrap_or("float64"))?;
    Ok(PyRumpyArray::new(RumpyArray::eye(n, dtype)))
}

/// Create logarithmically spaced array.
#[pyfunction]
#[pyo3(signature = (start, stop, num=50, base=10.0, dtype=None))]
pub fn logspace(
    start: f64,
    stop: f64,
    num: usize,
    base: f64,
    dtype: Option<&str>,
) -> PyResult<PyRumpyArray> {
    let dtype = parse_dtype(dtype.unwrap_or("float64"))?;
    Ok(PyRumpyArray::new(crate::array::logspace(start, stop, num, base, dtype)))
}

/// Create geometrically spaced array.
#[pyfunction]
#[pyo3(signature = (start, stop, num=50, dtype=None))]
pub fn geomspace(
    start: f64,
    stop: f64,
    num: usize,
    dtype: Option<&str>,
) -> PyResult<PyRumpyArray> {
    let dtype = parse_dtype(dtype.unwrap_or("float64"))?;
    crate::array::geomspace(start, stop, num, dtype)
        .map(PyRumpyArray::new)
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("geomspace requires start and stop to have same sign")
        })
}

/// Create a triangular matrix of ones.
#[pyfunction]
#[pyo3(signature = (n, m=None, k=0, dtype=None))]
pub fn tri(n: usize, m: Option<usize>, k: isize, dtype: Option<&str>) -> PyResult<PyRumpyArray> {
    let m = m.unwrap_or(n);
    let dtype = parse_dtype(dtype.unwrap_or("float64"))?;
    Ok(PyRumpyArray::new(crate::array::tri(n, m, k, dtype)))
}

/// Return lower triangle of an array.
#[pyfunction]
#[pyo3(signature = (m, k=0))]
pub fn tril(m: &PyRumpyArray, k: isize) -> PyResult<PyRumpyArray> {
    crate::array::tril(&m.inner, k)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("tril requires 2D array"))
}

/// Return upper triangle of an array.
#[pyfunction]
#[pyo3(signature = (m, k=0))]
pub fn triu(m: &PyRumpyArray, k: isize) -> PyResult<PyRumpyArray> {
    crate::array::triu(&m.inner, k)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("triu requires 2D array"))
}

/// Create a 2D array with flattened input as diagonal.
#[pyfunction]
#[pyo3(signature = (v, k=0))]
pub fn diagflat(v: &PyRumpyArray, k: isize) -> PyRumpyArray {
    PyRumpyArray::new(crate::array::diagflat(&v.inner, k))
}

/// Return coordinate matrices from coordinate vectors.
#[pyfunction]
#[pyo3(signature = (*xi, indexing="xy"))]
pub fn meshgrid(xi: &Bound<'_, pyo3::types::PyTuple>, indexing: &str) -> PyResult<Vec<PyRumpyArray>> {
    let mut arrays: Vec<RumpyArray> = Vec::with_capacity(xi.len());
    for item in xi.iter() {
        let arr = if let Ok(arr) = item.extract::<PyRef<'_, PyRumpyArray>>() {
            arr.inner.clone()
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err("meshgrid requires array arguments"));
        };
        arrays.push(arr);
    }

    crate::array::meshgrid(&arrays, indexing)
        .map(|arrs| arrs.into_iter().map(PyRumpyArray::new).collect())
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("meshgrid failed"))
}

/// Return an array representing indices of a grid.
#[pyfunction]
#[pyo3(signature = (dimensions, dtype=None))]
pub fn indices(dimensions: Vec<usize>, dtype: Option<&str>) -> PyResult<PyRumpyArray> {
    let dtype = parse_dtype(dtype.unwrap_or("int64"))?;
    Ok(PyRumpyArray::new(crate::array::indices(&dimensions, dtype)))
}

/// Construct an array by executing a function over each coordinate.
#[pyfunction]
#[pyo3(signature = (shape, function, dtype=None))]
pub fn fromfunction(
    py: Python<'_>,
    shape: Vec<usize>,
    function: &Bound<'_, pyo3::PyAny>,
    dtype: Option<&str>,
) -> PyResult<PyRumpyArray> {
    let dtype = parse_dtype(dtype.unwrap_or("float64"))?;
    let ndim = shape.len();
    let size: usize = shape.iter().product();

    if size == 0 {
        let result = RumpyArray::zeros(shape.clone(), dtype);
        return Ok(PyRumpyArray::new(result));
    }

    // Create index arrays for each dimension
    let index_arrays: Vec<PyRumpyArray> = (0..ndim)
        .map(|dim| {
            let mut arr = RumpyArray::zeros(shape.clone(), DType::float64());
            let buffer = std::sync::Arc::get_mut(arr.buffer_mut()).expect("unique");
            let ptr = buffer.as_mut_ptr() as *mut f64;

            let mut indices = vec![0usize; ndim];
            for i in 0..size {
                unsafe { *ptr.add(i) = indices[dim] as f64; }
                increment_indices(&mut indices, &shape);
            }
            PyRumpyArray::new(arr)
        })
        .collect();

    // Convert to Python tuple of arrays
    let args = pyo3::types::PyTuple::new(py, index_arrays)?;

    // Call the function
    let result_obj = function.call1(args)?;

    // Extract result
    if let Ok(arr) = result_obj.extract::<PyRef<'_, PyRumpyArray>>() {
        return Ok(PyRumpyArray::new(arr.inner.astype(dtype)));
    }

    Err(pyo3::exceptions::PyTypeError::new_err(
        "fromfunction callback must return an array",
    ))
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
pub fn array_impl(py: Python<'_>, obj: &Bound<'_, PyAny>, dtype: Option<&str>) -> PyResult<PyRumpyArray> {
    // Already a rumpy array?
    if let Ok(arr) = obj.extract::<PyRef<'_, PyRumpyArray>>() {
        // Convert dtype if requested
        if let Some(dtype_str) = dtype {
            let target_dtype = parse_dtype(dtype_str)?;
            if arr.inner.dtype() != target_dtype {
                return Ok(PyRumpyArray::new(arr.inner.astype(target_dtype)));
            }
        }
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
pub fn from_list(list: &Bound<'_, PyList>, dtype: Option<&str>) -> PyResult<PyRumpyArray> {
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
        // Try integer path for integer dtypes, fall back to f64
        if dtype.is_integer() {
            if let Ok((shape, data)) = flatten_nested_list_i64(list) {
                return Ok(PyRumpyArray::new(RumpyArray::from_vec_i64_with_shape(data, shape, dtype)));
            }
        }
        let (shape, data) = flatten_nested_list(list)?;
        return Ok(PyRumpyArray::new(RumpyArray::from_vec_with_shape(data, shape, dtype)));
    }

    // 1D list - try integer extraction for integer dtypes, fall back to f64
    if dtype.is_integer() {
        // Try extracting as i64 first (preserves large int precision)
        let data_i64: Result<Vec<i64>, _> = list
            .iter()
            .map(|item| item.extract::<i64>())
            .collect();
        if let Ok(data) = data_i64 {
            return Ok(PyRumpyArray::new(RumpyArray::from_vec_i64(data, dtype)));
        }
        // Fall back to f64 (handles float inputs like [5.0, 2.0, ...])
    }
    let data: Vec<f64> = list
        .iter()
        .map(|item| item.extract::<f64>())
        .collect::<PyResult<Vec<f64>>>()?;
    Ok(PyRumpyArray::new(RumpyArray::from_vec(data, dtype)))
}

/// Flatten nested Python list, returning shape and data as f64.
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

/// Flatten nested Python list, returning shape and data as i64.
fn flatten_nested_list_i64(list: &Bound<'_, PyList>) -> PyResult<(Vec<usize>, Vec<i64>)> {
    let mut shape = vec![list.len()];
    let mut data = Vec::new();

    for item in list.iter() {
        if let Ok(sublist) = item.downcast::<PyList>() {
            let (sub_shape, sub_data) = flatten_nested_list_i64(sublist)?;
            if shape.len() == 1 {
                shape.extend(sub_shape);
            }
            data.extend(sub_data);
        } else {
            data.push(item.extract::<i64>()?);
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
        ('f', 2) => Ok(DType::float16()),
        ('i', 8) => Ok(DType::int64()),
        ('i', 4) => Ok(DType::int32()),
        ('i', 2) => Ok(DType::int16()),
        ('i', 1) => Ok(DType::int8()),
        ('u', 8) => Ok(DType::uint64()),
        ('u', 4) => Ok(DType::uint32()),
        ('u', 2) => Ok(DType::uint16()),
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

// ============================================================================
// Window functions
// ============================================================================

/// Return the Bartlett window.
///
/// The Bartlett window is a triangular window with endpoints at zero.
#[pyfunction]
pub fn bartlett(m: i64) -> PyRumpyArray {
    if m < 1 {
        return PyRumpyArray::new(RumpyArray::zeros(vec![0], DType::float64()));
    }
    if m == 1 {
        return PyRumpyArray::new(RumpyArray::ones(vec![1], DType::float64()));
    }

    let m_f = m as f64;
    let size = m as usize;
    let mut data = Vec::with_capacity(size);

    // n = arange(1-M, M, 2), then:
    // where(n <= 0, 1 + n/(M-1), 1 - n/(M-1))
    let denom = m_f - 1.0;
    for i in 0..size {
        let n = (1 - m + 2 * i as i64) as f64;
        let val = if n <= 0.0 {
            1.0 + n / denom
        } else {
            1.0 - n / denom
        };
        data.push(val);
    }

    PyRumpyArray::new(RumpyArray::from_vec(data, DType::float64()))
}

/// Return the Blackman window.
///
/// The Blackman window is a taper formed by using the first three terms
/// of a summation of cosines, designed to have close to minimal leakage.
#[pyfunction]
pub fn blackman(m: i64) -> PyRumpyArray {
    if m < 1 {
        return PyRumpyArray::new(RumpyArray::zeros(vec![0], DType::float64()));
    }
    if m == 1 {
        return PyRumpyArray::new(RumpyArray::ones(vec![1], DType::float64()));
    }

    let m_f = m as f64;
    let size = m as usize;
    let mut data = Vec::with_capacity(size);

    // n = arange(1-M, M, 2)
    // 0.42 + 0.5*cos(pi*n/(M-1)) + 0.08*cos(2*pi*n/(M-1))
    let denom = m_f - 1.0;
    for i in 0..size {
        let n = (1 - m + 2 * i as i64) as f64;
        let val = 0.42
            + 0.5 * (std::f64::consts::PI * n / denom).cos()
            + 0.08 * (2.0 * std::f64::consts::PI * n / denom).cos();
        data.push(val);
    }

    PyRumpyArray::new(RumpyArray::from_vec(data, DType::float64()))
}

/// Return the Hamming window.
///
/// The Hamming window is a taper formed by using a raised cosine
/// with non-zero endpoints.
#[pyfunction]
pub fn hamming(m: i64) -> PyRumpyArray {
    if m < 1 {
        return PyRumpyArray::new(RumpyArray::zeros(vec![0], DType::float64()));
    }
    if m == 1 {
        return PyRumpyArray::new(RumpyArray::ones(vec![1], DType::float64()));
    }

    let m_f = m as f64;
    let size = m as usize;
    let mut data = Vec::with_capacity(size);

    // n = arange(1-M, M, 2)
    // 0.54 + 0.46*cos(pi*n/(M-1))
    let denom = m_f - 1.0;
    for i in 0..size {
        let n = (1 - m + 2 * i as i64) as f64;
        let val = 0.54 + 0.46 * (std::f64::consts::PI * n / denom).cos();
        data.push(val);
    }

    PyRumpyArray::new(RumpyArray::from_vec(data, DType::float64()))
}

/// Return the Hann window (also known as Hanning).
///
/// The Hann window is a taper formed by using a raised cosine
/// with zero endpoints.
#[pyfunction]
pub fn hanning(m: i64) -> PyRumpyArray {
    if m < 1 {
        return PyRumpyArray::new(RumpyArray::zeros(vec![0], DType::float64()));
    }
    if m == 1 {
        return PyRumpyArray::new(RumpyArray::ones(vec![1], DType::float64()));
    }

    let m_f = m as f64;
    let size = m as usize;
    let mut data = Vec::with_capacity(size);

    // n = arange(1-M, M, 2)
    // 0.5 + 0.5*cos(pi*n/(M-1))
    let denom = m_f - 1.0;
    for i in 0..size {
        let n = (1 - m + 2 * i as i64) as f64;
        let val = 0.5 + 0.5 * (std::f64::consts::PI * n / denom).cos();
        data.push(val);
    }

    PyRumpyArray::new(RumpyArray::from_vec(data, DType::float64()))
}

/// Return the Kaiser window.
///
/// The Kaiser window is a taper formed by using a Bessel function.
/// The beta parameter controls the trade-off between main-lobe width
/// and side-lobe level.
#[pyfunction]
pub fn kaiser(m: i64, beta: f64) -> PyRumpyArray {
    if m < 1 {
        return PyRumpyArray::new(RumpyArray::zeros(vec![0], DType::float64()));
    }
    if m == 1 {
        return PyRumpyArray::new(RumpyArray::ones(vec![1], DType::float64()));
    }

    let m_f = m as f64;
    let size = m as usize;
    let mut data = Vec::with_capacity(size);

    // n = arange(0, M), alpha = (M-1)/2
    // i0(beta * sqrt(1 - ((n-alpha)/alpha)^2)) / i0(beta)
    let alpha = (m_f - 1.0) / 2.0;
    let i0_beta = bessel_i0(beta);

    for i in 0..size {
        let n = i as f64;
        let ratio = (n - alpha) / alpha;
        let arg = beta * (1.0 - ratio * ratio).sqrt();
        let val = bessel_i0(arg) / i0_beta;
        data.push(val);
    }

    PyRumpyArray::new(RumpyArray::from_vec(data, DType::float64()))
}

/// Chebyshev polynomial approximation for modified Bessel function I0.
fn bessel_i0(x: f64) -> f64 {
    let ax = x.abs();
    if ax < 3.75 {
        let y = (x / 3.75).powi(2);
        1.0 + y * (3.5156229 + y * (3.0899424 + y * (1.2067492
            + y * (0.2659732 + y * (0.0360768 + y * 0.0045813)))))
    } else {
        let y = 3.75 / ax;
        (ax.exp() / ax.sqrt())
            * (0.39894228 + y * (0.01328592 + y * (0.00225319
                + y * (-0.00157565 + y * (0.00916281 + y * (-0.02057706
                    + y * (0.02635537 + y * (-0.01647633 + y * 0.00392377))))))))
    }
}
