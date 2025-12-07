// Python bindings for polynomial operations (polyval, polyder, polyint, polyfit, roots).

use pyo3::prelude::*;

use super::PyRumpyArray;

/// Evaluate polynomial at given points.
/// Coefficients are in descending order (highest degree first).
#[pyfunction]
pub fn polyval(p: &PyRumpyArray, x: &PyRumpyArray) -> PyRumpyArray {
    PyRumpyArray::new(crate::ops::polyval(&p.inner, &x.inner))
}

/// Compute polynomial derivative.
/// Returns coefficients of d^m/dx^m polynomial.
#[pyfunction]
#[pyo3(signature = (p, m=1))]
pub fn polyder(p: &PyRumpyArray, m: usize) -> PyRumpyArray {
    PyRumpyArray::new(crate::ops::polyder(&p.inner, m))
}

/// Compute polynomial integral.
/// Returns coefficients of m-th order antiderivative with integration constant(s) k.
#[pyfunction]
#[pyo3(signature = (p, m=1, k=None))]
pub fn polyint(p: &PyRumpyArray, m: usize, k: Option<&PyRumpyArray>) -> PyRumpyArray {
    let k_inner = k.map(|arr| &arr.inner);
    PyRumpyArray::new(crate::ops::polyint(&p.inner, m, k_inner))
}

/// Fit polynomial of degree deg to data points (x, y).
/// Returns coefficients in descending order (highest degree first).
#[pyfunction]
#[pyo3(signature = (x, y, deg, w=None))]
pub fn polyfit(x: &PyRumpyArray, y: &PyRumpyArray, deg: usize, w: Option<&PyRumpyArray>) -> PyResult<PyRumpyArray> {
    let w_inner = w.map(|arr| &arr.inner);
    crate::ops::polyfit(&x.inner, &y.inner, deg, w_inner)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("polyfit failed - check inputs"))
}

/// Find roots of polynomial.
/// Coefficients are in descending order (highest degree first).
#[pyfunction]
#[pyo3(name = "roots")]
pub fn roots_fn(p: &PyRumpyArray) -> PyResult<PyRumpyArray> {
    crate::ops::roots(&p.inner)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("roots computation failed"))
}
