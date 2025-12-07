//! Python bindings for linalg submodule (numpy.linalg compatibility).

use pyo3::prelude::*;
use crate::ops::linalg as linalg_ops;
use crate::ops::solve as solve_ops;
use crate::python::PyRumpyArray;

/// Solve linear system Ax = b.
#[pyfunction]
pub fn solve(a: &PyRumpyArray, b: &PyRumpyArray) -> PyResult<PyRumpyArray> {
    solve_ops::solve(&a.inner, &b.inner)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("solve: invalid dimensions or singular matrix"))
}

/// QR decomposition: A = QR.
#[pyfunction]
pub fn qr(a: &PyRumpyArray) -> PyResult<(PyRumpyArray, PyRumpyArray)> {
    linalg_ops::qr(&a.inner)
        .map(|(q, r)| (PyRumpyArray::new(q), PyRumpyArray::new(r)))
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("qr requires 2D array"))
}

/// SVD decomposition: A = U @ diag(S) @ Vt.
#[pyfunction]
#[pyo3(signature = (a, full_matrices=true))]
pub fn svd(a: &PyRumpyArray, full_matrices: bool) -> PyResult<(PyRumpyArray, PyRumpyArray, PyRumpyArray)> {
    // Note: we only have thin SVD, but numpy defaults to full_matrices=True
    // For now, we always return thin SVD regardless of the flag
    let _ = full_matrices; // TODO: implement full matrices support
    linalg_ops::svd(&a.inner)
        .map(|(u, s, vt)| (PyRumpyArray::new(u), PyRumpyArray::new(s), PyRumpyArray::new(vt)))
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("svd requires 2D array"))
}

/// Matrix inverse.
#[pyfunction]
pub fn inv(a: &PyRumpyArray) -> PyResult<PyRumpyArray> {
    linalg_ops::inv(&a.inner)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("inv requires square 2D array"))
}

/// Eigendecomposition of symmetric matrix.
#[pyfunction]
pub fn eigh(a: &PyRumpyArray) -> PyResult<(PyRumpyArray, PyRumpyArray)> {
    linalg_ops::eigh(&a.inner)
        .map(|(w, v)| (PyRumpyArray::new(w), PyRumpyArray::new(v)))
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("eigh requires square 2D array"))
}

/// Compute determinant of a square matrix.
#[pyfunction]
pub fn det(a: &PyRumpyArray) -> PyResult<f64> {
    linalg_ops::det(&a.inner)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("det requires square 2D array"))
}

/// Compute matrix/vector norm.
#[pyfunction]
#[pyo3(signature = (a, ord=None))]
pub fn norm(a: &PyRumpyArray, ord: Option<&str>) -> PyResult<f64> {
    linalg_ops::norm(&a.inner, ord)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("unsupported norm type"))
}

/// Cholesky decomposition: A = L @ L.T for symmetric positive-definite matrix.
#[pyfunction]
pub fn cholesky(a: &PyRumpyArray) -> PyResult<PyRumpyArray> {
    linalg_ops::cholesky(&a.inner)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("cholesky requires SPD matrix"))
}

/// Sign and log of determinant.
#[pyfunction]
pub fn slogdet(a: &PyRumpyArray) -> PyResult<(f64, f64)> {
    linalg_ops::slogdet(&a.inner)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("slogdet requires square 2D array"))
}

/// Condition number.
#[pyfunction]
#[pyo3(signature = (a, p=None))]
pub fn cond(a: &PyRumpyArray, p: Option<&str>) -> PyResult<f64> {
    linalg_ops::cond(&a.inner, p)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("unsupported norm type or non-2D array"))
}

/// Matrix rank.
#[pyfunction]
#[pyo3(signature = (a, tol=None))]
pub fn matrix_rank(a: &PyRumpyArray, tol: Option<f64>) -> PyResult<usize> {
    linalg_ops::matrix_rank(&a.inner, tol)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("matrix_rank requires 2D array"))
}

/// Moore-Penrose pseudo-inverse.
#[pyfunction]
#[pyo3(signature = (a, rcond=None))]
pub fn pinv(a: &PyRumpyArray, rcond: Option<f64>) -> PyResult<PyRumpyArray> {
    linalg_ops::pinv(&a.inner, rcond)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("pinv requires 2D array"))
}

/// Least squares solution.
#[pyfunction]
#[pyo3(signature = (a, b, rcond=None))]
pub fn lstsq(a: &PyRumpyArray, b: &PyRumpyArray, rcond: Option<f64>) -> PyResult<(PyRumpyArray, PyRumpyArray, usize, PyRumpyArray)> {
    linalg_ops::lstsq(&a.inner, &b.inner, rcond)
        .map(|(x, residuals, rank, s)| (PyRumpyArray::new(x), PyRumpyArray::new(residuals), rank, PyRumpyArray::new(s)))
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("lstsq: invalid dimensions"))
}

/// Eigenvalues of a general matrix.
#[pyfunction]
pub fn eigvals(a: &PyRumpyArray) -> PyResult<PyRumpyArray> {
    linalg_ops::eigvals(&a.inner)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("eigvals requires square 2D array"))
}

/// Eigendecomposition of a general matrix.
#[pyfunction]
pub fn eig(a: &PyRumpyArray) -> PyResult<(PyRumpyArray, PyRumpyArray)> {
    linalg_ops::eig(&a.inner)
        .map(|(w, v)| (PyRumpyArray::new(w), PyRumpyArray::new(v)))
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("eig requires square 2D array"))
}

/// Register linalg submodule.
pub fn register_submodule(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let linalg_module = PyModule::new(parent.py(), "linalg")?;
    linalg_module.add_function(wrap_pyfunction!(solve, &linalg_module)?)?;
    linalg_module.add_function(wrap_pyfunction!(qr, &linalg_module)?)?;
    linalg_module.add_function(wrap_pyfunction!(svd, &linalg_module)?)?;
    linalg_module.add_function(wrap_pyfunction!(inv, &linalg_module)?)?;
    linalg_module.add_function(wrap_pyfunction!(eigh, &linalg_module)?)?;
    linalg_module.add_function(wrap_pyfunction!(det, &linalg_module)?)?;
    linalg_module.add_function(wrap_pyfunction!(norm, &linalg_module)?)?;
    linalg_module.add_function(wrap_pyfunction!(cholesky, &linalg_module)?)?;
    linalg_module.add_function(wrap_pyfunction!(slogdet, &linalg_module)?)?;
    linalg_module.add_function(wrap_pyfunction!(cond, &linalg_module)?)?;
    linalg_module.add_function(wrap_pyfunction!(matrix_rank, &linalg_module)?)?;
    linalg_module.add_function(wrap_pyfunction!(pinv, &linalg_module)?)?;
    linalg_module.add_function(wrap_pyfunction!(lstsq, &linalg_module)?)?;
    linalg_module.add_function(wrap_pyfunction!(eigvals, &linalg_module)?)?;
    linalg_module.add_function(wrap_pyfunction!(eig, &linalg_module)?)?;
    parent.add_submodule(&linalg_module)?;
    Ok(())
}
