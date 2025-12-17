//! Python bindings for einsum functions.

use pyo3::prelude::*;
use pyo3::types::PyTuple;
use crate::ops::einsum as einsum_ops;
use crate::python::PyRumpyArray;
use crate::python::shape::to_rumpy_array;

/// Einstein summation convention.
///
/// Evaluates the Einstein summation convention on the operands.
///
/// Parameters
/// ----------
/// subscripts : str
///     Subscripts for the contraction (e.g., "ij,jk->ik").
/// operands : array_like
///     Input arrays for the contraction.
/// optimize : bool or str, optional
///     If True, use optimized contraction order. Can be "greedy", "optimal",
///     or False for no optimization. Default is False.
///
/// Returns
/// -------
/// output : ndarray
///     The result of the Einstein summation.
///
/// Examples
/// --------
/// >>> a = np.array([1, 2, 3])
/// >>> b = np.array([4, 5, 6])
/// >>> np.einsum('i,i->', a, b)  # dot product
/// 32
///
/// >>> A = np.array([[1, 2], [3, 4]])
/// >>> B = np.array([[5, 6], [7, 8]])
/// >>> np.einsum('ij,jk->ik', A, B)  # matrix multiplication
#[pyfunction]
#[pyo3(signature = (subscripts, *operands, optimize=false))]
pub fn einsum(
    subscripts: &str,
    operands: &Bound<'_, PyTuple>,
    optimize: bool,
) -> PyResult<PyRumpyArray> {
    let _ = optimize; // TODO: use for path optimization

    // Extract arrays from operands tuple
    let mut inner_arrays: Vec<crate::array::RumpyArray> = Vec::with_capacity(operands.len());
    for item in operands.iter() {
        inner_arrays.push(to_rumpy_array(&item)?);
    }

    let inner_refs: Vec<&crate::array::RumpyArray> = inner_arrays.iter().collect();

    einsum_ops::einsum(subscripts, &inner_refs)
        .map(PyRumpyArray::new)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))
}

/// Evaluates the lowest cost contraction order for an einsum expression.
///
/// Parameters
/// ----------
/// subscripts : str
///     Subscripts for the contraction (e.g., "ij,jk->ik").
/// operands : array_like
///     Input arrays (shapes are used, values not accessed).
/// optimize : str, optional
///     Optimization strategy: "greedy", "optimal", or True. Default is "greedy".
///
/// Returns
/// -------
/// path : list
///     A list with 'einsum_path' as first element, followed by tuples
///     indicating which operands to contract at each step.
/// info : str
///     A string containing information about the contraction.
#[pyfunction]
#[pyo3(signature = (subscripts, *operands, optimize="greedy"))]
pub fn einsum_path(
    py: Python<'_>,
    subscripts: &str,
    operands: &Bound<'_, PyTuple>,
    optimize: &str,
) -> PyResult<(pyo3::PyObject, String)> {
    // Extract shapes from operands
    let mut shapes: Vec<Vec<usize>> = Vec::with_capacity(operands.len());
    for item in operands.iter() {
        let arr = to_rumpy_array(&item)?;
        shapes.push(arr.shape().to_vec());
    }

    let (path, info) = einsum_ops::einsum_path(subscripts, &shapes, optimize)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;

    // Convert to Python format: ['einsum_path', (0, 1), (0, 1), ...]
    let mut result: Vec<pyo3::PyObject> = Vec::with_capacity(path.len() + 1);
    result.push("einsum_path".into_pyobject(py)?.unbind().into_any());
    for (a, b) in path {
        result.push((a, b).into_pyobject(py)?.unbind().into_any());
    }

    let py_list = pyo3::types::PyList::new(py, result)?;
    Ok((py_list.into(), info))
}

/// Register einsum functions to module.
pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(einsum, m)?)?;
    m.add_function(wrap_pyfunction!(einsum_path, m)?)?;
    Ok(())
}
