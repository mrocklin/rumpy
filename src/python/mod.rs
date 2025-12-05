pub mod pyarray;

use pyo3::prelude::*;
use pyo3::types::PyList;

pub use pyarray::{parse_dtype, parse_shape, PyRumpyArray};

use crate::array::{increment_indices, read_element, write_element, DType, RumpyArray};

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

/// Convert input to an array.
/// Supports: PyRumpyArray, objects with __array_interface__, and Python lists.
#[pyfunction]
#[pyo3(signature = (obj, dtype=None))]
pub fn asarray(py: Python<'_>, obj: &Bound<'_, PyAny>, dtype: Option<&str>) -> PyResult<PyRumpyArray> {
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

        // Read from source, write to destination
        let val = unsafe { read_element(src_ptr, src_offset, &dtype) };
        unsafe { write_element(dst_ptr, i, val, &dtype); }

        increment_indices(&mut indices, &shape);
    }

    Ok(PyRumpyArray::new(arr))
}

/// Create array from Python list.
fn from_list(list: &Bound<'_, PyList>, dtype: Option<&str>) -> PyResult<PyRumpyArray> {
    let dtype = parse_dtype(dtype.unwrap_or("float64"))?;

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
            let (sub_shape, sub_data) = flatten_nested_list(&sublist)?;
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
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unsupported dtype: {}",
            typestr
        ))),
    }
}

// Math ufuncs (module-level functions like np.sqrt, np.exp, etc.)

#[pyfunction]
pub fn sqrt(x: &PyRumpyArray) -> PyRumpyArray {
    PyRumpyArray::new(x.inner.sqrt())
}

#[pyfunction]
pub fn exp(x: &PyRumpyArray) -> PyRumpyArray {
    PyRumpyArray::new(x.inner.exp())
}

#[pyfunction]
pub fn log(x: &PyRumpyArray) -> PyRumpyArray {
    PyRumpyArray::new(x.inner.log())
}

#[pyfunction]
pub fn sin(x: &PyRumpyArray) -> PyRumpyArray {
    PyRumpyArray::new(x.inner.sin())
}

#[pyfunction]
pub fn cos(x: &PyRumpyArray) -> PyRumpyArray {
    PyRumpyArray::new(x.inner.cos())
}

#[pyfunction]
pub fn tan(x: &PyRumpyArray) -> PyRumpyArray {
    PyRumpyArray::new(x.inner.tan())
}

/// Concatenate arrays along an axis.
#[pyfunction]
#[pyo3(signature = (arrays, axis=0))]
pub fn concatenate(arrays: Vec<PyRef<'_, PyRumpyArray>>, axis: usize) -> PyResult<PyRumpyArray> {
    let inner_arrays: Vec<RumpyArray> = arrays.iter().map(|a| a.inner.clone()).collect();
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
pub fn stack(arrays: Vec<PyRef<'_, PyRumpyArray>>, axis: usize) -> PyResult<PyRumpyArray> {
    let inner_arrays: Vec<RumpyArray> = arrays.iter().map(|a| a.inner.clone()).collect();
    crate::array::stack(&inner_arrays, axis)
        .map(PyRumpyArray::new)
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("arrays must have same shape for stack")
        })
}

/// Stack arrays vertically (row-wise).
#[pyfunction]
pub fn vstack(arrays: Vec<PyRef<'_, PyRumpyArray>>) -> PyResult<PyRumpyArray> {
    if arrays.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err("need at least one array"));
    }

    // For 1D arrays, reshape to (1, N) first
    let inner_arrays: Vec<RumpyArray> = arrays.iter().map(|a| {
        if a.inner.ndim() == 1 {
            a.inner.reshape(vec![1, a.inner.size()]).unwrap_or_else(|| a.inner.clone())
        } else {
            a.inner.clone()
        }
    }).collect();

    crate::array::concatenate(&inner_arrays, 0)
        .map(PyRumpyArray::new)
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("arrays must have same shape for vstack")
        })
}

/// Stack arrays horizontally (column-wise).
#[pyfunction]
pub fn hstack(arrays: Vec<PyRef<'_, PyRumpyArray>>) -> PyResult<PyRumpyArray> {
    if arrays.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err("need at least one array"));
    }

    let first = &arrays[0].inner;
    let axis = if first.ndim() == 1 { 0 } else { 1 };

    let inner_arrays: Vec<RumpyArray> = arrays.iter().map(|a| a.inner.clone()).collect();
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

/// Sort array (flattened).
#[pyfunction]
pub fn sort(arr: &PyRumpyArray) -> PyRumpyArray {
    PyRumpyArray::new(arr.inner.sort())
}

/// Return indices that would sort the array (flattened).
#[pyfunction]
pub fn argsort(arr: &PyRumpyArray) -> PyRumpyArray {
    PyRumpyArray::new(arr.inner.argsort())
}

/// Return unique sorted values.
#[pyfunction]
pub fn unique(arr: &PyRumpyArray) -> PyRumpyArray {
    PyRumpyArray::new(arr.inner.unique())
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

/// Register Python module contents.
pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRumpyArray>()?;
    // Constructors
    m.add_function(wrap_pyfunction!(zeros, m)?)?;
    m.add_function(wrap_pyfunction!(ones, m)?)?;
    m.add_function(wrap_pyfunction!(arange, m)?)?;
    m.add_function(wrap_pyfunction!(linspace, m)?)?;
    m.add_function(wrap_pyfunction!(eye, m)?)?;
    m.add_function(wrap_pyfunction!(full, m)?)?;
    m.add_function(wrap_pyfunction!(asarray, m)?)?;
    // Math ufuncs
    m.add_function(wrap_pyfunction!(sqrt, m)?)?;
    m.add_function(wrap_pyfunction!(exp, m)?)?;
    m.add_function(wrap_pyfunction!(log, m)?)?;
    m.add_function(wrap_pyfunction!(sin, m)?)?;
    m.add_function(wrap_pyfunction!(cos, m)?)?;
    m.add_function(wrap_pyfunction!(tan, m)?)?;
    // Conditional
    m.add_function(wrap_pyfunction!(where_fn, m)?)?;
    // Shape manipulation
    m.add_function(wrap_pyfunction!(expand_dims, m)?)?;
    m.add_function(wrap_pyfunction!(squeeze, m)?)?;
    // Concatenation
    m.add_function(wrap_pyfunction!(concatenate, m)?)?;
    m.add_function(wrap_pyfunction!(stack, m)?)?;
    m.add_function(wrap_pyfunction!(vstack, m)?)?;
    m.add_function(wrap_pyfunction!(hstack, m)?)?;
    // Splitting
    m.add_function(wrap_pyfunction!(split, m)?)?;
    m.add_function(wrap_pyfunction!(array_split, m)?)?;
    // Sorting
    m.add_function(wrap_pyfunction!(sort, m)?)?;
    m.add_function(wrap_pyfunction!(argsort, m)?)?;
    m.add_function(wrap_pyfunction!(unique, m)?)?;
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
    Ok(())
}
