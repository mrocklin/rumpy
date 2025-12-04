pub mod pyarray;

use pyo3::prelude::*;
use pyo3::types::PyList;

pub use pyarray::{parse_dtype, PyRumpyArray};

use crate::array::{increment_indices, read_element, write_element, DType, RumpyArray};

/// Create an array filled with zeros.
#[pyfunction]
#[pyo3(signature = (shape, dtype=None))]
pub fn zeros(shape: Vec<usize>, dtype: Option<&str>) -> PyResult<PyRumpyArray> {
    let dtype = parse_dtype(dtype.unwrap_or("float64"))?;
    Ok(PyRumpyArray::new(RumpyArray::zeros(shape, dtype)))
}

/// Create an array filled with ones.
#[pyfunction]
#[pyo3(signature = (shape, dtype=None))]
pub fn ones(shape: Vec<usize>, dtype: Option<&str>) -> PyResult<PyRumpyArray> {
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
    let dtype = parse_dtype(dtype.unwrap_or("float64"))?;
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
pub fn full(shape: Vec<usize>, fill_value: f64, dtype: Option<&str>) -> PyResult<PyRumpyArray> {
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
    let mut arr = RumpyArray::zeros(shape.clone(), dtype);

    if size == 0 {
        return Ok(PyRumpyArray::new(arr));
    }

    // Copy data respecting strides
    let src_ptr = data_ptr as *const u8;
    let src_strides = strides.unwrap_or_else(|| {
        crate::array::compute_c_strides(&shape, dtype.itemsize())
    });

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
        let val = unsafe { read_element(src_ptr, src_offset, dtype) };
        unsafe { write_element(dst_ptr, i, val, dtype); }

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
    // Format: "<f8" (little-endian float64), ">i4" (big-endian int32), etc.
    let kind = typestr.chars().nth(1).unwrap_or('f');
    let size: usize = typestr[2..].parse().unwrap_or(8);

    match (kind, size) {
        ('f', 8) => Ok(DType::Float64),
        ('f', 4) => Ok(DType::Float32),
        ('i', 8) => Ok(DType::Int64),
        ('i', 4) => Ok(DType::Int32),
        ('b', 1) | ('u', 1) => Ok(DType::Bool),
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
    Ok(())
}
