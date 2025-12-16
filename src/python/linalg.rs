//! Python bindings for linalg submodule (numpy.linalg compatibility).
//!
//! Many functions are exposed at both top-level (np.dot) and submodule (np.linalg.solve).

use pyo3::prelude::*;
use crate::ops::linalg as linalg_ops;
use crate::ops::solve as solve_ops;
use crate::python::PyRumpyArray;

// ============================================================================
// Top-level linear algebra functions (exposed as np.X)
// ============================================================================

/// Matrix multiplication.
#[pyfunction]
pub fn matmul(a: &PyRumpyArray, b: &PyRumpyArray) -> PyResult<PyRumpyArray> {
    crate::ops::matmul::matmul(&a.inner, &b.inner)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("matmul: incompatible shapes"))
}

/// Dot product with numpy semantics.
#[pyfunction]
pub fn dot(a: &PyRumpyArray, b: &PyRumpyArray) -> PyResult<PyRumpyArray> {
    crate::ops::dot::dot(&a.inner, &b.inner)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("dot: incompatible shapes"))
}

/// Inner product of two arrays.
#[pyfunction]
pub fn inner(a: &PyRumpyArray, b: &PyRumpyArray) -> PyResult<PyRumpyArray> {
    crate::ops::inner::inner(&a.inner, &b.inner)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("inner: incompatible shapes"))
}

/// Outer product of two arrays.
#[pyfunction]
pub fn outer(a: &PyRumpyArray, b: &PyRumpyArray) -> PyResult<PyRumpyArray> {
    crate::ops::outer::outer(&a.inner, &b.inner)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("outer: incompatible shapes"))
}

/// Compute trace of a matrix (sum of diagonal elements).
#[pyfunction]
pub fn trace(a: &PyRumpyArray) -> PyResult<f64> {
    linalg_ops::trace(&a.inner)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("trace requires 2D array"))
}

/// Extract diagonal or construct diagonal matrix.
#[pyfunction]
pub fn diag(a: &PyRumpyArray) -> PyResult<PyRumpyArray> {
    linalg_ops::diag(&a.inner)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("diag requires 1D or 2D array"))
}

/// Vector dot product (flattens arrays then computes inner product).
#[pyfunction]
pub fn vdot(a: &PyRumpyArray, b: &PyRumpyArray) -> PyResult<f64> {
    linalg_ops::vdot(&a.inner, &b.inner)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("vdot: arrays must have same total size"))
}

/// Kronecker product of two arrays.
#[pyfunction]
pub fn kron(a: &PyRumpyArray, b: &PyRumpyArray) -> PyResult<PyRumpyArray> {
    linalg_ops::kron(&a.inner, &b.inner)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("kron: unsupported dimensions"))
}

/// Cross product of two 3D vectors.
#[pyfunction]
pub fn cross(a: &PyRumpyArray, b: &PyRumpyArray) -> PyResult<PyRumpyArray> {
    linalg_ops::cross(&a.inner, &b.inner)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("cross: requires 1D arrays of length 3"))
}

/// Tensor dot product over specified axes.
#[pyfunction]
#[pyo3(signature = (a, b, axes=None))]
pub fn tensordot(a: &PyRumpyArray, b: &PyRumpyArray, axes: Option<&Bound<'_, pyo3::PyAny>>) -> PyResult<PyRumpyArray> {
    // Parse axes - can be int or tuple of two lists (default is 2)
    let (a_axes, b_axes) = if let Some(axes) = axes {
        if let Ok(n) = axes.extract::<usize>() {
            // axes=n means last n axes of a and first n axes of b
            let a_axes: Vec<usize> = (a.inner.ndim().saturating_sub(n)..a.inner.ndim()).collect();
            let b_axes: Vec<usize> = (0..n).collect();
            (a_axes, b_axes)
        } else if let Ok((ax_a, ax_b)) = axes.extract::<(Vec<usize>, Vec<usize>)>() {
            (ax_a, ax_b)
        } else if let Ok((ax_a, ax_b)) = axes.extract::<(Vec<i64>, Vec<i64>)>() {
            // Handle negative indices
            let a_axes: Vec<usize> = ax_a.into_iter()
                .map(|x| if x < 0 { (a.inner.ndim() as i64 + x) as usize } else { x as usize })
                .collect();
            let b_axes: Vec<usize> = ax_b.into_iter()
                .map(|x| if x < 0 { (b.inner.ndim() as i64 + x) as usize } else { x as usize })
                .collect();
            (a_axes, b_axes)
        } else {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "axes must be integer or tuple of two lists"
            ));
        }
    } else {
        // Default: axes=2
        let n = 2;
        let a_axes: Vec<usize> = (a.inner.ndim().saturating_sub(n)..a.inner.ndim()).collect();
        let b_axes: Vec<usize> = (0..n).collect();
        (a_axes, b_axes)
    };

    linalg_ops::tensordot(&a.inner, &b.inner, (a_axes, b_axes))
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("tensordot: incompatible dimensions"))
}

// ============================================================================
// Linalg submodule functions (exposed as np.linalg.X)
// ============================================================================

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

/// Eigenvalues only for symmetric/Hermitian matrix.
#[pyfunction]
pub fn eigvalsh(a: &PyRumpyArray) -> PyResult<PyRumpyArray> {
    linalg_ops::eigvalsh(&a.inner)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("eigvalsh requires square 2D array"))
}

/// Singular values only (no U or Vt).
#[pyfunction]
pub fn svdvals(a: &PyRumpyArray) -> PyResult<PyRumpyArray> {
    linalg_ops::svdvals(&a.inner)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("svdvals requires 2D array"))
}

/// Raise a square matrix to an integer power.
#[pyfunction]
pub fn matrix_power(a: &PyRumpyArray, n: i64) -> PyResult<PyRumpyArray> {
    linalg_ops::matrix_power(&a.inner, n)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("matrix_power requires square 2D array"))
}

/// Efficient matrix multiplication of multiple arrays.
#[pyfunction]
pub fn multi_dot(arrays: Vec<PyRef<'_, PyRumpyArray>>) -> PyResult<PyRumpyArray> {
    let refs: Vec<&crate::array::RumpyArray> = arrays.iter().map(|a| &a.inner).collect();
    linalg_ops::multi_dot(&refs)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("multi_dot: incompatible shapes"))
}

/// Compute the inverse of a tensor.
#[pyfunction]
#[pyo3(signature = (a, ind=None))]
pub fn tensorinv(a: &PyRumpyArray, ind: Option<usize>) -> PyResult<PyRumpyArray> {
    linalg_ops::tensorinv(&a.inner, ind)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("tensorinv: invalid dimensions"))
}

/// Solve the tensor equation A x = b for x.
#[pyfunction]
#[pyo3(signature = (a, b, axes=None))]
pub fn tensorsolve(a: &PyRumpyArray, b: &PyRumpyArray, axes: Option<Vec<usize>>) -> PyResult<PyRumpyArray> {
    linalg_ops::tensorsolve(&a.inner, &b.inner, axes.as_deref())
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("tensorsolve: invalid dimensions"))
}

/// Vector norm with flexible ord parameter.
#[pyfunction]
#[pyo3(signature = (a, ord=None))]
pub fn vector_norm(a: &PyRumpyArray, ord: Option<f64>) -> PyResult<f64> {
    linalg_ops::vector_norm(&a.inner, ord)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("vector_norm failed"))
}

/// Matrix norm with flexible ord parameter.
#[pyfunction]
#[pyo3(signature = (a, ord=None))]
pub fn matrix_norm(a: &PyRumpyArray, ord: Option<&str>) -> PyResult<f64> {
    linalg_ops::matrix_norm(&a.inner, ord)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("matrix_norm: unsupported norm type or invalid array"))
}

// LinAlgError exception class for linalg module.
pyo3::create_exception!(rumpy, LinAlgError, pyo3::exceptions::PyException);

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
    linalg_module.add_function(wrap_pyfunction!(eigvalsh, &linalg_module)?)?;
    linalg_module.add_function(wrap_pyfunction!(svdvals, &linalg_module)?)?;
    linalg_module.add_function(wrap_pyfunction!(matrix_power, &linalg_module)?)?;
    linalg_module.add_function(wrap_pyfunction!(multi_dot, &linalg_module)?)?;
    linalg_module.add_function(wrap_pyfunction!(tensorinv, &linalg_module)?)?;
    linalg_module.add_function(wrap_pyfunction!(tensorsolve, &linalg_module)?)?;
    linalg_module.add_function(wrap_pyfunction!(vector_norm, &linalg_module)?)?;
    linalg_module.add_function(wrap_pyfunction!(matrix_norm, &linalg_module)?)?;
    linalg_module.add("LinAlgError", parent.py().get_type::<LinAlgError>())?;
    parent.add_submodule(&linalg_module)?;
    Ok(())
}
