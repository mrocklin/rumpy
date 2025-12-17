//! Python bindings for numpy.char module (vectorized string operations).

use pyo3::prelude::*;

use crate::ops::char as char_ops;
use super::pyarray::PyRumpyArray;

/// Concatenate two string arrays element-wise.
#[pyfunction]
#[pyo3(name = "add")]
pub fn char_add(x1: &PyRumpyArray, x2: &PyRumpyArray) -> PyResult<PyRumpyArray> {
    char_ops::add(&x1.inner, &x2.inner)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyTypeError::new_err("add requires string arrays"))
}

/// Repeat each string n times.
#[pyfunction]
#[pyo3(name = "multiply")]
pub fn char_multiply(a: &PyRumpyArray, i: usize) -> PyResult<PyRumpyArray> {
    char_ops::multiply(&a.inner, i)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyTypeError::new_err("multiply requires a string array"))
}

/// Return element-wise string_upper(a).
#[pyfunction]
pub fn upper(a: &PyRumpyArray) -> PyResult<PyRumpyArray> {
    char_ops::upper(&a.inner)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyTypeError::new_err("upper requires a string array"))
}

/// Return element-wise string_lower(a).
#[pyfunction]
pub fn lower(a: &PyRumpyArray) -> PyResult<PyRumpyArray> {
    char_ops::lower(&a.inner)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyTypeError::new_err("lower requires a string array"))
}

/// Strip leading and trailing whitespace.
#[pyfunction]
#[pyo3(signature = (a, chars=None))]
pub fn strip(a: &PyRumpyArray, chars: Option<&str>) -> PyResult<PyRumpyArray> {
    let _ = chars; // TODO: Support custom chars
    char_ops::strip(&a.inner)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyTypeError::new_err("strip requires a string array"))
}

/// Strip leading whitespace.
#[pyfunction]
#[pyo3(signature = (a, chars=None))]
pub fn lstrip(a: &PyRumpyArray, chars: Option<&str>) -> PyResult<PyRumpyArray> {
    let _ = chars;
    char_ops::lstrip(&a.inner)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyTypeError::new_err("lstrip requires a string array"))
}

/// Strip trailing whitespace.
#[pyfunction]
#[pyo3(signature = (a, chars=None))]
pub fn rstrip(a: &PyRumpyArray, chars: Option<&str>) -> PyResult<PyRumpyArray> {
    let _ = chars;
    char_ops::rstrip(&a.inner)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyTypeError::new_err("rstrip requires a string array"))
}

/// Find the lowest index of substring in each string.
#[pyfunction]
#[pyo3(signature = (a, sub, start=None, end=None))]
pub fn find(a: &PyRumpyArray, sub: &str, start: Option<usize>, end: Option<usize>) -> PyResult<PyRumpyArray> {
    let _ = (start, end); // TODO: Support start/end
    char_ops::find(&a.inner, sub)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyTypeError::new_err("find requires a string array"))
}

/// Find the highest index of substring in each string.
#[pyfunction]
#[pyo3(signature = (a, sub, start=None, end=None))]
pub fn rfind(a: &PyRumpyArray, sub: &str, start: Option<usize>, end: Option<usize>) -> PyResult<PyRumpyArray> {
    let _ = (start, end);
    char_ops::rfind(&a.inner, sub)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyTypeError::new_err("rfind requires a string array"))
}

/// Like find, but raises ValueError if substring is not found.
#[pyfunction]
#[pyo3(signature = (a, sub, start=None, end=None))]
pub fn index(a: &PyRumpyArray, sub: &str, start: Option<usize>, end: Option<usize>) -> PyResult<PyRumpyArray> {
    let _ = (start, end);
    let result = char_ops::index(&a.inner, sub)
        .ok_or_else(|| pyo3::exceptions::PyTypeError::new_err("index requires a string array"))?;

    // Check for -1 values directly from buffer (fast path)
    let size = result.size();
    if size > 0 {
        let ptr = result.data_ptr() as *const i64;
        for i in 0..size {
            if unsafe { *ptr.add(i) } == -1 {
                return Err(pyo3::exceptions::PyValueError::new_err("substring not found"));
            }
        }
    }
    Ok(PyRumpyArray::new(result))
}

/// Like rfind, but raises ValueError if substring is not found.
#[pyfunction]
#[pyo3(signature = (a, sub, start=None, end=None))]
pub fn rindex(a: &PyRumpyArray, sub: &str, start: Option<usize>, end: Option<usize>) -> PyResult<PyRumpyArray> {
    let _ = (start, end);
    let result = char_ops::rindex(&a.inner, sub)
        .ok_or_else(|| pyo3::exceptions::PyTypeError::new_err("rindex requires a string array"))?;

    let size = result.size();
    if size > 0 {
        let ptr = result.data_ptr() as *const i64;
        for i in 0..size {
            if unsafe { *ptr.add(i) } == -1 {
                return Err(pyo3::exceptions::PyValueError::new_err("substring not found"));
            }
        }
    }
    Ok(PyRumpyArray::new(result))
}

/// Partition each string at first occurrence of separator.
#[pyfunction]
pub fn partition(a: &PyRumpyArray, sep: &str) -> PyResult<PyRumpyArray> {
    char_ops::partition(&a.inner, sep)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("empty separator"))
}

/// Partition each string at last occurrence of separator.
#[pyfunction]
pub fn rpartition(a: &PyRumpyArray, sep: &str) -> PyResult<PyRumpyArray> {
    char_ops::rpartition(&a.inner, sep)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("empty separator"))
}

/// Join array of strings with separator.
#[pyfunction]
pub fn join(sep: &str, a: &PyRumpyArray) -> PyResult<PyRumpyArray> {
    char_ops::join(sep, &a.inner)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyTypeError::new_err("join requires a string array"))
}

/// Replace occurrences of old with new.
#[pyfunction]
#[pyo3(signature = (a, old, new, count=None))]
pub fn replace(a: &PyRumpyArray, old: &str, new: &str, count: Option<usize>) -> PyResult<PyRumpyArray> {
    char_ops::replace(&a.inner, old, new, count)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyTypeError::new_err("replace requires a string array"))
}

/// Count occurrences of substring in each string.
#[pyfunction]
#[pyo3(name = "count", signature = (a, sub, start=None, end=None))]
pub fn char_count(a: &PyRumpyArray, sub: &str, start: Option<usize>, end: Option<usize>) -> PyResult<PyRumpyArray> {
    let _ = (start, end);
    char_ops::count(&a.inner, sub)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyTypeError::new_err("count requires a string array"))
}

/// Get the length of each string.
#[pyfunction]
pub fn str_len(a: &PyRumpyArray) -> PyResult<PyRumpyArray> {
    char_ops::str_len(&a.inner)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyTypeError::new_err("str_len requires a string array"))
}

/// Check if each string contains only alphabetic characters.
#[pyfunction]
pub fn isalpha(a: &PyRumpyArray) -> PyResult<PyRumpyArray> {
    char_ops::isalpha(&a.inner)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyTypeError::new_err("isalpha requires a string array"))
}

/// Check if each string contains only digits.
#[pyfunction]
pub fn isdigit(a: &PyRumpyArray) -> PyResult<PyRumpyArray> {
    char_ops::isdigit(&a.inner)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyTypeError::new_err("isdigit requires a string array"))
}

/// Check if each string contains only alphanumeric characters.
#[pyfunction]
pub fn isalnum(a: &PyRumpyArray) -> PyResult<PyRumpyArray> {
    char_ops::isalnum(&a.inner)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyTypeError::new_err("isalnum requires a string array"))
}

/// Check if each string is uppercase.
#[pyfunction]
pub fn isupper(a: &PyRumpyArray) -> PyResult<PyRumpyArray> {
    char_ops::isupper(&a.inner)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyTypeError::new_err("isupper requires a string array"))
}

/// Check if each string is lowercase.
#[pyfunction]
pub fn islower(a: &PyRumpyArray) -> PyResult<PyRumpyArray> {
    char_ops::islower(&a.inner)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyTypeError::new_err("islower requires a string array"))
}

/// Check if each string contains only whitespace.
#[pyfunction]
pub fn isspace(a: &PyRumpyArray) -> PyResult<PyRumpyArray> {
    char_ops::isspace(&a.inner)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyTypeError::new_err("isspace requires a string array"))
}

/// Check if each string starts with the given prefix.
#[pyfunction]
#[pyo3(signature = (a, prefix, start=None, end=None))]
pub fn startswith(a: &PyRumpyArray, prefix: &str, start: Option<usize>, end: Option<usize>) -> PyResult<PyRumpyArray> {
    let _ = (start, end);
    char_ops::startswith(&a.inner, prefix)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyTypeError::new_err("startswith requires a string array"))
}

/// Check if each string ends with the given suffix.
#[pyfunction]
#[pyo3(signature = (a, suffix, start=None, end=None))]
pub fn endswith(a: &PyRumpyArray, suffix: &str, start: Option<usize>, end: Option<usize>) -> PyResult<PyRumpyArray> {
    let _ = (start, end);
    char_ops::endswith(&a.inner, suffix)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyTypeError::new_err("endswith requires a string array"))
}

/// Check if each string is decimal.
#[pyfunction]
pub fn isdecimal(a: &PyRumpyArray) -> PyResult<PyRumpyArray> {
    char_ops::isdecimal(&a.inner)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyTypeError::new_err("isdecimal requires a string array"))
}

/// Check if each string is numeric.
#[pyfunction]
pub fn isnumeric(a: &PyRumpyArray) -> PyResult<PyRumpyArray> {
    char_ops::isnumeric(&a.inner)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyTypeError::new_err("isnumeric requires a string array"))
}

/// Check if each string is titlecase.
#[pyfunction]
pub fn istitle(a: &PyRumpyArray) -> PyResult<PyRumpyArray> {
    char_ops::istitle(&a.inner)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyTypeError::new_err("istitle requires a string array"))
}

/// Center each string in a field of given width.
#[pyfunction]
#[pyo3(signature = (a, width, fillchar=None))]
pub fn center(a: &PyRumpyArray, width: usize, fillchar: Option<&str>) -> PyResult<PyRumpyArray> {
    let fillchar = fillchar.and_then(|s| s.chars().next()).unwrap_or(' ');
    char_ops::center(&a.inner, width, fillchar)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyTypeError::new_err("center requires a string array"))
}

/// Left-justify each string in a field of given width.
#[pyfunction]
#[pyo3(signature = (a, width, fillchar=None))]
pub fn ljust(a: &PyRumpyArray, width: usize, fillchar: Option<&str>) -> PyResult<PyRumpyArray> {
    let fillchar = fillchar.and_then(|s| s.chars().next()).unwrap_or(' ');
    char_ops::ljust(&a.inner, width, fillchar)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyTypeError::new_err("ljust requires a string array"))
}

/// Right-justify each string in a field of given width.
#[pyfunction]
#[pyo3(signature = (a, width, fillchar=None))]
pub fn rjust(a: &PyRumpyArray, width: usize, fillchar: Option<&str>) -> PyResult<PyRumpyArray> {
    let fillchar = fillchar.and_then(|s| s.chars().next()).unwrap_or(' ');
    char_ops::rjust(&a.inner, width, fillchar)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyTypeError::new_err("rjust requires a string array"))
}

/// Pad each string with zeros on the left.
#[pyfunction]
pub fn zfill(a: &PyRumpyArray, width: usize) -> PyResult<PyRumpyArray> {
    char_ops::zfill(&a.inner, width)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyTypeError::new_err("zfill requires a string array"))
}

/// Capitalize first character of each string.
#[pyfunction]
pub fn capitalize(a: &PyRumpyArray) -> PyResult<PyRumpyArray> {
    char_ops::capitalize(&a.inner)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyTypeError::new_err("capitalize requires a string array"))
}

/// Titlecase each string.
#[pyfunction]
pub fn title(a: &PyRumpyArray) -> PyResult<PyRumpyArray> {
    char_ops::title(&a.inner)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyTypeError::new_err("title requires a string array"))
}

/// Swap case of each string.
#[pyfunction]
pub fn swapcase(a: &PyRumpyArray) -> PyResult<PyRumpyArray> {
    char_ops::swapcase(&a.inner)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyTypeError::new_err("swapcase requires a string array"))
}

/// Compare two string arrays element-wise for equality.
#[pyfunction]
#[pyo3(name = "equal")]
pub fn char_equal(x1: &PyRumpyArray, x2: &PyRumpyArray) -> PyResult<PyRumpyArray> {
    char_ops::equal(&x1.inner, &x2.inner)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyTypeError::new_err("equal requires string arrays with same shape"))
}

/// Compare two string arrays element-wise for inequality.
#[pyfunction]
#[pyo3(name = "not_equal")]
pub fn char_not_equal(x1: &PyRumpyArray, x2: &PyRumpyArray) -> PyResult<PyRumpyArray> {
    char_ops::not_equal(&x1.inner, &x2.inner)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyTypeError::new_err("not_equal requires string arrays with same shape"))
}

/// Compare two string arrays element-wise (less than).
#[pyfunction]
#[pyo3(name = "less")]
pub fn char_less(x1: &PyRumpyArray, x2: &PyRumpyArray) -> PyResult<PyRumpyArray> {
    char_ops::less(&x1.inner, &x2.inner)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyTypeError::new_err("less requires string arrays with same shape"))
}

/// Compare two string arrays element-wise (less than or equal).
#[pyfunction]
#[pyo3(name = "less_equal")]
pub fn char_less_equal(x1: &PyRumpyArray, x2: &PyRumpyArray) -> PyResult<PyRumpyArray> {
    char_ops::less_equal(&x1.inner, &x2.inner)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyTypeError::new_err("less_equal requires string arrays with same shape"))
}

/// Compare two string arrays element-wise (greater than).
#[pyfunction]
#[pyo3(name = "greater")]
pub fn char_greater(x1: &PyRumpyArray, x2: &PyRumpyArray) -> PyResult<PyRumpyArray> {
    char_ops::greater(&x1.inner, &x2.inner)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyTypeError::new_err("greater requires string arrays with same shape"))
}

/// Compare two string arrays element-wise (greater than or equal).
#[pyfunction]
#[pyo3(name = "greater_equal")]
pub fn char_greater_equal(x1: &PyRumpyArray, x2: &PyRumpyArray) -> PyResult<PyRumpyArray> {
    char_ops::greater_equal(&x1.inner, &x2.inner)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyTypeError::new_err("greater_equal requires string arrays with same shape"))
}

/// Expand tabs to spaces.
#[pyfunction]
#[pyo3(signature = (a, tabsize=None))]
pub fn expandtabs(a: &PyRumpyArray, tabsize: Option<usize>) -> PyResult<PyRumpyArray> {
    let tabsize = tabsize.unwrap_or(8);
    char_ops::expandtabs(&a.inner, tabsize)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyTypeError::new_err("expandtabs requires a string array"))
}

/// Register the char submodule.
pub fn register_char_submodule(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "char")?;

    m.add_function(wrap_pyfunction!(char_add, &m)?)?;
    m.add_function(wrap_pyfunction!(char_multiply, &m)?)?;
    m.add_function(wrap_pyfunction!(upper, &m)?)?;
    m.add_function(wrap_pyfunction!(lower, &m)?)?;
    m.add_function(wrap_pyfunction!(strip, &m)?)?;
    m.add_function(wrap_pyfunction!(lstrip, &m)?)?;
    m.add_function(wrap_pyfunction!(rstrip, &m)?)?;
    m.add_function(wrap_pyfunction!(find, &m)?)?;
    m.add_function(wrap_pyfunction!(rfind, &m)?)?;
    m.add_function(wrap_pyfunction!(index, &m)?)?;
    m.add_function(wrap_pyfunction!(rindex, &m)?)?;
    m.add_function(wrap_pyfunction!(partition, &m)?)?;
    m.add_function(wrap_pyfunction!(rpartition, &m)?)?;
    m.add_function(wrap_pyfunction!(join, &m)?)?;
    m.add_function(wrap_pyfunction!(replace, &m)?)?;
    m.add_function(wrap_pyfunction!(char_count, &m)?)?;
    m.add_function(wrap_pyfunction!(str_len, &m)?)?;
    m.add_function(wrap_pyfunction!(isalpha, &m)?)?;
    m.add_function(wrap_pyfunction!(isdigit, &m)?)?;
    m.add_function(wrap_pyfunction!(isalnum, &m)?)?;
    m.add_function(wrap_pyfunction!(isupper, &m)?)?;
    m.add_function(wrap_pyfunction!(islower, &m)?)?;
    m.add_function(wrap_pyfunction!(isspace, &m)?)?;
    m.add_function(wrap_pyfunction!(startswith, &m)?)?;
    m.add_function(wrap_pyfunction!(endswith, &m)?)?;
    m.add_function(wrap_pyfunction!(isdecimal, &m)?)?;
    m.add_function(wrap_pyfunction!(isnumeric, &m)?)?;
    m.add_function(wrap_pyfunction!(istitle, &m)?)?;
    m.add_function(wrap_pyfunction!(center, &m)?)?;
    m.add_function(wrap_pyfunction!(ljust, &m)?)?;
    m.add_function(wrap_pyfunction!(rjust, &m)?)?;
    m.add_function(wrap_pyfunction!(zfill, &m)?)?;
    m.add_function(wrap_pyfunction!(capitalize, &m)?)?;
    m.add_function(wrap_pyfunction!(title, &m)?)?;
    m.add_function(wrap_pyfunction!(swapcase, &m)?)?;
    m.add_function(wrap_pyfunction!(char_equal, &m)?)?;
    m.add_function(wrap_pyfunction!(char_not_equal, &m)?)?;
    m.add_function(wrap_pyfunction!(char_less, &m)?)?;
    m.add_function(wrap_pyfunction!(char_less_equal, &m)?)?;
    m.add_function(wrap_pyfunction!(char_greater, &m)?)?;
    m.add_function(wrap_pyfunction!(char_greater_equal, &m)?)?;
    m.add_function(wrap_pyfunction!(expandtabs, &m)?)?;

    parent.add_submodule(&m)?;
    Ok(())
}
