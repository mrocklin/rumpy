pub mod pyarray;

use pyo3::prelude::*;

pub use pyarray::{parse_dtype, PyRumpyArray};

use crate::array::RumpyArray;

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

/// Register Python module contents.
pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRumpyArray>()?;
    m.add_function(wrap_pyfunction!(zeros, m)?)?;
    m.add_function(wrap_pyfunction!(ones, m)?)?;
    m.add_function(wrap_pyfunction!(arange, m)?)?;
    Ok(())
}
