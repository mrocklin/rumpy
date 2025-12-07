// Python bindings for numerical operations (gradient, trapezoid, interp, convolve, correlate).

use pyo3::prelude::*;
use pyo3::types::PyList;
use pyo3::IntoPyObject;

use crate::array::RumpyArray;
use super::PyRumpyArray;

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

/// 1D cross-correlation.
#[pyfunction]
#[pyo3(signature = (a, v, mode="full"))]
pub fn correlate(a: &PyRumpyArray, v: &PyRumpyArray, mode: &str) -> PyResult<PyRumpyArray> {
    crate::ops::numerical::correlate(&a.inner, &v.inner, mode)
        .map(PyRumpyArray::new)
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("correlate requires 1D arrays and valid mode")
        })
}

/// Compute numerical gradient.
///
/// For N-D arrays, returns a list of gradients along each axis.
/// If axis is specified, returns a single gradient array.
#[pyfunction]
#[pyo3(signature = (f, *varargs, axis=None))]
pub fn gradient(
    f: &PyRumpyArray,
    varargs: &Bound<'_, pyo3::types::PyTuple>,
    axis: Option<isize>,
) -> PyResult<pyo3::PyObject> {
    let py = varargs.py();

    // Parse varargs for spacing
    let (spacing, coords): (Option<f64>, Option<RumpyArray>) = if varargs.is_empty() {
        (None, None)
    } else if varargs.len() == 1 {
        // Single argument: either scalar spacing or coordinate array
        let arg = varargs.get_item(0)?;
        if let Ok(scalar) = arg.extract::<f64>() {
            (Some(scalar), None)
        } else if let Ok(py_arr) = arg.downcast::<PyRumpyArray>() {
            (None, Some(py_arr.borrow().inner.clone()))
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "gradient spacing must be scalar or array"
            ));
        }
    } else {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "gradient: too many positional arguments"
        ));
    };

    // Normalize axis
    let axis_usize = axis.map(|a| {
        if a < 0 {
            (f.inner.ndim() as isize + a) as usize
        } else {
            a as usize
        }
    });

    // Use coordinate-based gradient if coords provided
    if let Some(ref coord_arr) = coords {
        let ax = axis_usize.unwrap_or(0);
        let result = crate::ops::numerical::gradient_with_coords(&f.inner, coord_arr, ax)
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("gradient: invalid axis or coordinates")
            })?;
        return Ok(PyRumpyArray::new(result).into_pyobject(py)?.into_any().unbind());
    }

    // Use scalar spacing gradient
    let results = crate::ops::numerical::gradient(&f.inner, spacing, axis_usize)
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("gradient: invalid input")
        })?;

    // If single axis or 1D, return single array
    if results.len() == 1 {
        Ok(PyRumpyArray::new(results.into_iter().next().unwrap()).into_pyobject(py)?.into_any().unbind())
    } else {
        // Return tuple of arrays for N-D case
        let py_arrays: Vec<_> = results.into_iter().map(PyRumpyArray::new).collect();
        let list = PyList::new(py, py_arrays)?;
        Ok(list.into_pyobject(py)?.into_any().unbind())
    }
}

/// Trapezoidal integration.
#[pyfunction]
#[pyo3(signature = (y, x=None, dx=1.0, axis=-1))]
pub fn trapezoid(
    y: &PyRumpyArray,
    x: Option<&PyRumpyArray>,
    dx: f64,
    axis: isize,
) -> PyResult<PyRumpyArray> {
    let x_inner = x.map(|a| &a.inner);
    crate::ops::numerical::trapezoid(&y.inner, x_inner, dx, axis)
        .map(PyRumpyArray::new)
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("trapezoid: invalid input")
        })
}

/// 1D linear interpolation.
#[pyfunction]
#[pyo3(signature = (x, xp, fp, left=None, right=None))]
pub fn interp(
    x: &PyRumpyArray,
    xp: &PyRumpyArray,
    fp: &PyRumpyArray,
    left: Option<f64>,
    right: Option<f64>,
) -> PyResult<PyRumpyArray> {
    crate::ops::numerical::interp(&x.inner, &xp.inner, &fp.inner, left, right)
        .map(PyRumpyArray::new)
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("interp: xp and fp must be 1D with same size")
        })
}
