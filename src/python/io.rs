//! Python bindings for I/O operations.

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use std::path::PathBuf;

use super::{parse_dtype, PyRumpyArray};
use crate::ops::io as io_ops;

/// Load data from a text file.
#[pyfunction]
#[pyo3(signature = (fname, dtype=None, comments=None, delimiter=None, skiprows=None, usecols=None, max_rows=None))]
pub fn loadtxt(
    fname: &str,
    dtype: Option<&str>,
    comments: Option<&str>,
    delimiter: Option<&str>,
    skiprows: Option<usize>,
    usecols: Option<Vec<usize>>,
    max_rows: Option<usize>,
) -> PyResult<PyRumpyArray> {
    let path = PathBuf::from(fname);
    let dtype = parse_dtype(dtype.unwrap_or("float64"))?;
    let comments = comments.unwrap_or("#");
    let skiprows = skiprows.unwrap_or(0);

    let usecols_ref = usecols.as_deref();

    let arr = io_ops::loadtxt(&path, dtype, comments, delimiter, skiprows, usecols_ref, max_rows)
        .map_err(pyo3::exceptions::PyIOError::new_err)?;

    Ok(PyRumpyArray::new(arr))
}

/// Save array to a text file.
#[pyfunction]
#[pyo3(signature = (fname, x, fmt=None, delimiter=None, newline=None, header=None, footer=None, comments=None))]
pub fn savetxt(
    fname: &str,
    x: &PyRumpyArray,
    fmt: Option<&str>,
    delimiter: Option<&str>,
    newline: Option<&str>,
    header: Option<&str>,
    footer: Option<&str>,
    comments: Option<&str>,
) -> PyResult<()> {
    let path = PathBuf::from(fname);
    let fmt = fmt.unwrap_or("%.18e");
    let delimiter = delimiter.unwrap_or(" ");
    let newline = newline.unwrap_or("\n");
    let header = header.unwrap_or("");
    let footer = footer.unwrap_or("");
    let comments = comments.unwrap_or("# ");

    io_ops::savetxt(&path, &x.inner, delimiter, newline, header, footer, fmt, comments)
        .map_err(pyo3::exceptions::PyIOError::new_err)?;

    Ok(())
}

/// Load data from a text file with missing value handling.
#[pyfunction]
#[pyo3(signature = (fname, dtype=None, comments=None, delimiter=None, skip_header=None, skip_footer=None, usecols=None, missing_values=None, filling_values=None, max_rows=None))]
pub fn genfromtxt(
    fname: &str,
    dtype: Option<&str>,
    comments: Option<&str>,
    delimiter: Option<&str>,
    skip_header: Option<usize>,
    skip_footer: Option<usize>,
    usecols: Option<Vec<usize>>,
    missing_values: Option<&str>,
    filling_values: Option<f64>,
    max_rows: Option<usize>,
) -> PyResult<PyRumpyArray> {
    let path = PathBuf::from(fname);
    let dtype = parse_dtype(dtype.unwrap_or("float64"))?;
    let comments = comments.unwrap_or("#");
    let skip_header = skip_header.unwrap_or(0);
    let skip_footer = skip_footer.unwrap_or(0);
    let missing_values = missing_values.unwrap_or("");
    let filling_values = filling_values.unwrap_or(f64::NAN);

    let usecols_ref = usecols.as_deref();

    let arr = io_ops::genfromtxt(
        &path,
        dtype,
        comments,
        delimiter,
        skip_header,
        skip_footer,
        usecols_ref,
        missing_values,
        filling_values,
        max_rows,
    )
    .map_err(pyo3::exceptions::PyIOError::new_err)?;

    Ok(PyRumpyArray::new(arr))
}

/// Save a single array to a .npy file.
#[pyfunction]
#[pyo3(signature = (file, arr, allow_pickle=None))]
pub fn save(file: &str, arr: &PyRumpyArray, allow_pickle: Option<bool>) -> PyResult<()> {
    let _ = allow_pickle; // Ignored - we don't pickle
    let path = PathBuf::from(file);

    io_ops::save_npy(&path, &arr.inner)
        .map_err(pyo3::exceptions::PyIOError::new_err)?;

    Ok(())
}

/// Load array from a .npy or .npz file.
#[pyfunction]
#[pyo3(signature = (file, mmap_mode=None, allow_pickle=None))]
pub fn load<'py>(
    py: Python<'py>,
    file: &str,
    mmap_mode: Option<&str>,
    allow_pickle: Option<bool>,
) -> PyResult<PyObject> {
    let _ = mmap_mode; // Ignored - we don't support mmap
    let _ = allow_pickle; // Ignored - we don't pickle
    let path = PathBuf::from(file);

    if file.ends_with(".npz") {
        // Load as dict-like object
        let arrays = io_ops::load_npz(&path)
            .map_err(pyo3::exceptions::PyIOError::new_err)?;

        let dict = PyDict::new(py);
        for (name, arr) in arrays {
            let py_arr = Py::new(py, PyRumpyArray::new(arr))?;
            dict.set_item(name, py_arr)?;
        }

        Ok(dict.into())
    } else {
        // Load as single array
        let arr = io_ops::load_npy(&path)
            .map_err(pyo3::exceptions::PyIOError::new_err)?;

        let py_arr = Py::new(py, PyRumpyArray::new(arr))?;
        Ok(py_arr.into_any())
    }
}

/// Extract arrays from args/kwargs for savez functions.
fn extract_arrays_for_savez(
    args: &Bound<'_, pyo3::types::PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Vec<(String, crate::array::RumpyArray)>> {
    let mut arrays = Vec::new();

    // Process positional args (named arr_0, arr_1, etc.)
    for (i, arg) in args.iter().enumerate() {
        let arr: PyRef<'_, PyRumpyArray> = arg.extract()?;
        arrays.push((format!("arr_{}", i), arr.inner.clone()));
    }

    // Process keyword args
    if let Some(kw) = kwargs {
        for (key, value) in kw.iter() {
            let name: String = key.extract()?;
            let arr: PyRef<'_, PyRumpyArray> = value.extract()?;
            arrays.push((name, arr.inner.clone()));
        }
    }

    Ok(arrays)
}

/// Save multiple arrays to a .npz file (uncompressed).
#[pyfunction]
#[pyo3(signature = (file, *args, **kwargs))]
pub fn savez(
    file: &str,
    args: &Bound<'_, pyo3::types::PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<()> {
    let path = PathBuf::from(file);
    let arrays = extract_arrays_for_savez(args, kwargs)?;
    let refs: Vec<_> = arrays.iter().map(|(n, a)| (n.as_str(), a)).collect();

    io_ops::savez(&path, &refs)
        .map_err(pyo3::exceptions::PyIOError::new_err)
}

/// Save multiple arrays to a compressed .npz file.
#[pyfunction]
#[pyo3(signature = (file, *args, **kwargs))]
pub fn savez_compressed(
    file: &str,
    args: &Bound<'_, pyo3::types::PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<()> {
    let path = PathBuf::from(file);
    let arrays = extract_arrays_for_savez(args, kwargs)?;
    let refs: Vec<_> = arrays.iter().map(|(n, a)| (n.as_str(), a)).collect();

    io_ops::savez_compressed(&path, &refs)
        .map_err(pyo3::exceptions::PyIOError::new_err)
}

/// Create array from bytes buffer.
#[pyfunction]
#[pyo3(signature = (buffer, dtype=None, count=None, offset=None))]
pub fn frombuffer(
    buffer: &Bound<'_, PyBytes>,
    dtype: Option<&str>,
    count: Option<isize>,
    offset: Option<usize>,
) -> PyResult<PyRumpyArray> {
    let dtype = parse_dtype(dtype.unwrap_or("float64"))?;
    let count = count.unwrap_or(-1);
    let offset = offset.unwrap_or(0);

    let data = buffer.as_bytes();

    let arr = io_ops::frombuffer(data, dtype, count, offset)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;

    Ok(PyRumpyArray::new(arr))
}

/// Read array from a binary or text file.
#[pyfunction]
#[pyo3(signature = (file, dtype=None, count=None, sep=None, offset=None))]
pub fn fromfile(
    file: &str,
    dtype: Option<&str>,
    count: Option<isize>,
    sep: Option<&str>,
    offset: Option<usize>,
) -> PyResult<PyRumpyArray> {
    let path = PathBuf::from(file);
    let dtype = parse_dtype(dtype.unwrap_or("float64"))?;
    let count = count.unwrap_or(-1);
    let sep = sep.unwrap_or("");
    let offset = offset.unwrap_or(0);

    let arr = io_ops::fromfile(&path, dtype, count, offset, sep)
        .map_err(pyo3::exceptions::PyIOError::new_err)?;

    Ok(PyRumpyArray::new(arr))
}
