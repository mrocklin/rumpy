// Python bindings for reduction operations (sum, mean, etc.)

use pyo3::prelude::*;
use pyo3::types::PyTuple;

use super::{pyarray, PyRumpyArray};
use crate::array::RumpyArray;

/// Helper for axis bounds checking.
fn check_axis(axis: usize, ndim: usize) -> PyResult<()> {
    if axis >= ndim {
        Err(pyo3::exceptions::PyValueError::new_err(format!(
            "axis {} is out of bounds for array of dimension {}",
            axis, ndim
        )))
    } else {
        Ok(())
    }
}

/// Resolve a potentially negative axis to a positive index.
pub fn resolve_axis(axis: isize, ndim: usize) -> usize {
    if axis < 0 {
        (ndim as isize + axis) as usize
    } else {
        axis as usize
    }
}

/// Parse axis parameter - can be None, a single int, or a tuple of ints.
/// Returns None if axis is None, or Some(Vec<usize>) with normalized axes.
fn parse_axis(axis: Option<&Bound<'_, pyo3::PyAny>>, ndim: usize) -> PyResult<Option<Vec<usize>>> {
    match axis {
        None => Ok(None),
        Some(obj) => {
            // Try to extract as a single integer
            if let Ok(single) = obj.extract::<isize>() {
                let normalized = resolve_axis(single, ndim);
                check_axis(normalized, ndim)?;
                Ok(Some(vec![normalized]))
            }
            // Try to extract as a tuple
            else if let Ok(tuple) = obj.downcast::<PyTuple>() {
                let mut axes = Vec::new();
                for item in tuple.iter() {
                    let ax = item.extract::<isize>()?;
                    let normalized = resolve_axis(ax, ndim);
                    check_axis(normalized, ndim)?;
                    axes.push(normalized);
                }
                if axes.is_empty() {
                    return Err(pyo3::exceptions::PyValueError::new_err("axis tuple cannot be empty"));
                }
                Ok(Some(axes))
            }
            // Try to extract as a list
            else if let Ok(list) = obj.extract::<Vec<isize>>() {
                let mut axes = Vec::new();
                for ax in list {
                    let normalized = resolve_axis(ax, ndim);
                    check_axis(normalized, ndim)?;
                    axes.push(normalized);
                }
                if axes.is_empty() {
                    return Err(pyo3::exceptions::PyValueError::new_err("axis list cannot be empty"));
                }
                Ok(Some(axes))
            }
            else {
                Err(pyo3::exceptions::PyTypeError::new_err(
                    "axis must be an integer, tuple of integers, or None"
                ))
            }
        }
    }
}

/// Perform sequential reductions over multiple axes.
/// When reducing axis N, all subsequent axes with index > N shift down by 1.
/// To match NumPy, we sort axes in descending order and reduce from highest to lowest.
fn reduce_multi_axis<F>(arr: &RumpyArray, mut axes: Vec<usize>, reduce_fn: F) -> RumpyArray
where
    F: Fn(&RumpyArray, usize) -> RumpyArray,
{
    // Sort axes in descending order to avoid index shifting issues
    axes.sort_by(|a, b| b.cmp(a));

    let mut result = arr.clone();
    for &axis in &axes {
        result = reduce_fn(&result, axis);
    }
    result
}

// ============================================================================
// Basic reductions
// ============================================================================

/// Sum of array elements.
#[pyfunction]
#[pyo3(signature = (x, axis=None))]
pub fn sum(x: &PyRumpyArray, axis: Option<&Bound<'_, pyo3::PyAny>>) -> PyResult<pyarray::ReductionResult> {
    let axes = parse_axis(axis, x.inner.ndim())?;
    match axes {
        None => Ok(pyarray::ReductionResult::Scalar(x.inner.sum())),
        Some(ax_vec) if ax_vec.len() == 1 => {
            Ok(pyarray::ReductionResult::Array(PyRumpyArray::new(x.inner.sum_axis(ax_vec[0]))))
        }
        Some(ax_vec) => {
            let result = reduce_multi_axis(&x.inner, ax_vec, |arr, axis| arr.sum_axis(axis));
            Ok(pyarray::ReductionResult::Array(PyRumpyArray::new(result)))
        }
    }
}

/// Product of array elements.
#[pyfunction]
#[pyo3(signature = (x, axis=None))]
pub fn prod(x: &PyRumpyArray, axis: Option<&Bound<'_, pyo3::PyAny>>) -> PyResult<pyarray::ReductionResult> {
    let axes = parse_axis(axis, x.inner.ndim())?;
    match axes {
        None => Ok(pyarray::ReductionResult::Scalar(x.inner.prod())),
        Some(ax_vec) if ax_vec.len() == 1 => {
            Ok(pyarray::ReductionResult::Array(PyRumpyArray::new(x.inner.prod_axis(ax_vec[0]))))
        }
        Some(ax_vec) => {
            let result = reduce_multi_axis(&x.inner, ax_vec, |arr, axis| arr.prod_axis(axis));
            Ok(pyarray::ReductionResult::Array(PyRumpyArray::new(result)))
        }
    }
}

/// Mean of array elements.
#[pyfunction]
#[pyo3(signature = (x, axis=None))]
pub fn mean(x: &PyRumpyArray, axis: Option<&Bound<'_, pyo3::PyAny>>) -> PyResult<pyarray::ReductionResult> {
    let axes = parse_axis(axis, x.inner.ndim())?;
    match axes {
        None => Ok(pyarray::ReductionResult::Scalar(x.inner.mean())),
        Some(ax_vec) if ax_vec.len() == 1 => {
            Ok(pyarray::ReductionResult::Array(PyRumpyArray::new(x.inner.mean_axis(ax_vec[0]))))
        }
        Some(ax_vec) => {
            let result = reduce_multi_axis(&x.inner, ax_vec, |arr, axis| arr.mean_axis(axis));
            Ok(pyarray::ReductionResult::Array(PyRumpyArray::new(result)))
        }
    }
}

/// Variance of array elements.
#[pyfunction]
#[pyo3(signature = (x, axis=None))]
pub fn var(x: &PyRumpyArray, axis: Option<&Bound<'_, pyo3::PyAny>>) -> PyResult<pyarray::ReductionResult> {
    let axes = parse_axis(axis, x.inner.ndim())?;
    match axes {
        None => Ok(pyarray::ReductionResult::Scalar(x.inner.var())),
        Some(ax_vec) if ax_vec.len() == 1 => {
            Ok(pyarray::ReductionResult::Array(PyRumpyArray::new(x.inner.var_axis(ax_vec[0]))))
        }
        Some(ax_vec) => {
            let result = reduce_multi_axis(&x.inner, ax_vec, |arr, axis| arr.var_axis(axis));
            Ok(pyarray::ReductionResult::Array(PyRumpyArray::new(result)))
        }
    }
}

/// Standard deviation of array elements.
#[pyfunction]
#[pyo3(name = "std", signature = (x, axis=None))]
pub fn std_fn(x: &PyRumpyArray, axis: Option<&Bound<'_, pyo3::PyAny>>) -> PyResult<pyarray::ReductionResult> {
    let axes = parse_axis(axis, x.inner.ndim())?;
    match axes {
        None => Ok(pyarray::ReductionResult::Scalar(x.inner.std())),
        Some(ax_vec) if ax_vec.len() == 1 => {
            Ok(pyarray::ReductionResult::Array(PyRumpyArray::new(x.inner.std_axis(ax_vec[0]))))
        }
        Some(ax_vec) => {
            let result = reduce_multi_axis(&x.inner, ax_vec, |arr, axis| arr.std_axis(axis));
            Ok(pyarray::ReductionResult::Array(PyRumpyArray::new(result)))
        }
    }
}

/// Maximum of array elements.
#[pyfunction]
#[pyo3(signature = (x, axis=None))]
pub fn max(x: &PyRumpyArray, axis: Option<&Bound<'_, pyo3::PyAny>>) -> PyResult<pyarray::ReductionResult> {
    let axes = parse_axis(axis, x.inner.ndim())?;
    match axes {
        None => Ok(pyarray::ReductionResult::Scalar(x.inner.max())),
        Some(ax_vec) if ax_vec.len() == 1 => {
            Ok(pyarray::ReductionResult::Array(PyRumpyArray::new(x.inner.max_axis(ax_vec[0]))))
        }
        Some(ax_vec) => {
            let result = reduce_multi_axis(&x.inner, ax_vec, |arr, axis| arr.max_axis(axis));
            Ok(pyarray::ReductionResult::Array(PyRumpyArray::new(result)))
        }
    }
}

/// Minimum of array elements.
#[pyfunction]
#[pyo3(signature = (x, axis=None))]
pub fn min(x: &PyRumpyArray, axis: Option<&Bound<'_, pyo3::PyAny>>) -> PyResult<pyarray::ReductionResult> {
    let axes = parse_axis(axis, x.inner.ndim())?;
    match axes {
        None => Ok(pyarray::ReductionResult::Scalar(x.inner.min())),
        Some(ax_vec) if ax_vec.len() == 1 => {
            Ok(pyarray::ReductionResult::Array(PyRumpyArray::new(x.inner.min_axis(ax_vec[0]))))
        }
        Some(ax_vec) => {
            let result = reduce_multi_axis(&x.inner, ax_vec, |arr, axis| arr.min_axis(axis));
            Ok(pyarray::ReductionResult::Array(PyRumpyArray::new(result)))
        }
    }
}

/// Index of maximum element.
#[pyfunction]
#[pyo3(signature = (x, axis=None))]
pub fn argmax(x: &PyRumpyArray, axis: Option<&Bound<'_, pyo3::PyAny>>) -> PyResult<pyarray::ReductionResult> {
    let axes = parse_axis(axis, x.inner.ndim())?;
    match axes {
        None => Ok(pyarray::ReductionResult::Scalar(x.inner.argmax() as f64)),
        Some(ax_vec) if ax_vec.len() == 1 => {
            Ok(pyarray::ReductionResult::Array(PyRumpyArray::new(x.inner.argmax_axis(ax_vec[0]))))
        }
        Some(ax_vec) => {
            let result = reduce_multi_axis(&x.inner, ax_vec, |arr, axis| arr.argmax_axis(axis));
            Ok(pyarray::ReductionResult::Array(PyRumpyArray::new(result)))
        }
    }
}

/// Index of minimum element.
#[pyfunction]
#[pyo3(signature = (x, axis=None))]
pub fn argmin(x: &PyRumpyArray, axis: Option<&Bound<'_, pyo3::PyAny>>) -> PyResult<pyarray::ReductionResult> {
    let axes = parse_axis(axis, x.inner.ndim())?;
    match axes {
        None => Ok(pyarray::ReductionResult::Scalar(x.inner.argmin() as f64)),
        Some(ax_vec) if ax_vec.len() == 1 => {
            Ok(pyarray::ReductionResult::Array(PyRumpyArray::new(x.inner.argmin_axis(ax_vec[0]))))
        }
        Some(ax_vec) => {
            let result = reduce_multi_axis(&x.inner, ax_vec, |arr, axis| arr.argmin_axis(axis));
            Ok(pyarray::ReductionResult::Array(PyRumpyArray::new(result)))
        }
    }
}

// ============================================================================
// NaN-aware reductions
// ============================================================================

/// Sum ignoring NaN values.
#[pyfunction]
#[pyo3(signature = (x, axis=None))]
pub fn nansum(x: &PyRumpyArray, axis: Option<&Bound<'_, pyo3::PyAny>>) -> PyResult<pyarray::ReductionResult> {
    let axes = parse_axis(axis, x.inner.ndim())?;
    match axes {
        None => Ok(pyarray::ReductionResult::Scalar(x.inner.nansum())),
        Some(ax_vec) if ax_vec.len() == 1 => {
            Ok(pyarray::ReductionResult::Array(PyRumpyArray::new(x.inner.nansum_axis(ax_vec[0]))))
        }
        Some(ax_vec) => {
            let result = reduce_multi_axis(&x.inner, ax_vec, |arr, axis| arr.nansum_axis(axis));
            Ok(pyarray::ReductionResult::Array(PyRumpyArray::new(result)))
        }
    }
}

/// Product ignoring NaN values.
#[pyfunction]
#[pyo3(signature = (x, axis=None))]
pub fn nanprod(x: &PyRumpyArray, axis: Option<&Bound<'_, pyo3::PyAny>>) -> PyResult<pyarray::ReductionResult> {
    let axes = parse_axis(axis, x.inner.ndim())?;
    match axes {
        None => Ok(pyarray::ReductionResult::Scalar(x.inner.nanprod())),
        Some(ax_vec) if ax_vec.len() == 1 => {
            Ok(pyarray::ReductionResult::Array(PyRumpyArray::new(x.inner.nanprod_axis(ax_vec[0]))))
        }
        Some(ax_vec) => {
            let result = reduce_multi_axis(&x.inner, ax_vec, |arr, axis| arr.nanprod_axis(axis));
            Ok(pyarray::ReductionResult::Array(PyRumpyArray::new(result)))
        }
    }
}

/// Mean ignoring NaN values.
#[pyfunction]
#[pyo3(signature = (x, axis=None))]
pub fn nanmean(x: &PyRumpyArray, axis: Option<&Bound<'_, pyo3::PyAny>>) -> PyResult<pyarray::ReductionResult> {
    let axes = parse_axis(axis, x.inner.ndim())?;
    match axes {
        None => Ok(pyarray::ReductionResult::Scalar(x.inner.nanmean())),
        Some(ax_vec) if ax_vec.len() == 1 => {
            Ok(pyarray::ReductionResult::Array(PyRumpyArray::new(x.inner.nanmean_axis(ax_vec[0]))))
        }
        Some(ax_vec) => {
            let result = reduce_multi_axis(&x.inner, ax_vec, |arr, axis| arr.nanmean_axis(axis));
            Ok(pyarray::ReductionResult::Array(PyRumpyArray::new(result)))
        }
    }
}

/// Variance ignoring NaN values.
#[pyfunction]
#[pyo3(signature = (x, axis=None))]
pub fn nanvar(x: &PyRumpyArray, axis: Option<&Bound<'_, pyo3::PyAny>>) -> PyResult<pyarray::ReductionResult> {
    let axes = parse_axis(axis, x.inner.ndim())?;
    match axes {
        None => Ok(pyarray::ReductionResult::Scalar(x.inner.nanvar())),
        Some(ax_vec) if ax_vec.len() == 1 => {
            Ok(pyarray::ReductionResult::Array(PyRumpyArray::new(x.inner.nanvar_axis(ax_vec[0]))))
        }
        Some(ax_vec) => {
            let result = reduce_multi_axis(&x.inner, ax_vec, |arr, axis| arr.nanvar_axis(axis));
            Ok(pyarray::ReductionResult::Array(PyRumpyArray::new(result)))
        }
    }
}

/// Standard deviation ignoring NaN values.
#[pyfunction]
#[pyo3(signature = (x, axis=None))]
pub fn nanstd(x: &PyRumpyArray, axis: Option<&Bound<'_, pyo3::PyAny>>) -> PyResult<pyarray::ReductionResult> {
    let axes = parse_axis(axis, x.inner.ndim())?;
    match axes {
        None => Ok(pyarray::ReductionResult::Scalar(x.inner.nanstd())),
        Some(ax_vec) if ax_vec.len() == 1 => {
            Ok(pyarray::ReductionResult::Array(PyRumpyArray::new(x.inner.nanstd_axis(ax_vec[0]))))
        }
        Some(ax_vec) => {
            let result = reduce_multi_axis(&x.inner, ax_vec, |arr, axis| arr.nanstd_axis(axis));
            Ok(pyarray::ReductionResult::Array(PyRumpyArray::new(result)))
        }
    }
}

/// Minimum ignoring NaN values.
#[pyfunction]
#[pyo3(signature = (x, axis=None))]
pub fn nanmin(x: &PyRumpyArray, axis: Option<&Bound<'_, pyo3::PyAny>>) -> PyResult<pyarray::ReductionResult> {
    let axes = parse_axis(axis, x.inner.ndim())?;
    match axes {
        None => Ok(pyarray::ReductionResult::Scalar(x.inner.nanmin())),
        Some(ax_vec) if ax_vec.len() == 1 => {
            Ok(pyarray::ReductionResult::Array(PyRumpyArray::new(x.inner.nanmin_axis(ax_vec[0]))))
        }
        Some(ax_vec) => {
            let result = reduce_multi_axis(&x.inner, ax_vec, |arr, axis| arr.nanmin_axis(axis));
            Ok(pyarray::ReductionResult::Array(PyRumpyArray::new(result)))
        }
    }
}

/// Maximum ignoring NaN values.
#[pyfunction]
#[pyo3(signature = (x, axis=None))]
pub fn nanmax(x: &PyRumpyArray, axis: Option<&Bound<'_, pyo3::PyAny>>) -> PyResult<pyarray::ReductionResult> {
    let axes = parse_axis(axis, x.inner.ndim())?;
    match axes {
        None => Ok(pyarray::ReductionResult::Scalar(x.inner.nanmax())),
        Some(ax_vec) if ax_vec.len() == 1 => {
            Ok(pyarray::ReductionResult::Array(PyRumpyArray::new(x.inner.nanmax_axis(ax_vec[0]))))
        }
        Some(ax_vec) => {
            let result = reduce_multi_axis(&x.inner, ax_vec, |arr, axis| arr.nanmax_axis(axis));
            Ok(pyarray::ReductionResult::Array(PyRumpyArray::new(result)))
        }
    }
}

/// Index of minimum ignoring NaN values.
#[pyfunction]
#[pyo3(signature = (x, axis=None))]
pub fn nanargmin(x: &PyRumpyArray, axis: Option<isize>) -> PyResult<pyarray::ReductionResult> {
    match axis {
        None => {
            match x.inner.nanargmin() {
                Some(idx) => Ok(pyarray::ReductionResult::Scalar(idx as f64)),
                None => Err(pyo3::exceptions::PyValueError::new_err(
                    "All-NaN slice encountered"
                )),
            }
        }
        Some(ax) => {
            let ax = resolve_axis(ax, x.inner.ndim());
            check_axis(ax, x.inner.ndim())?;
            Ok(pyarray::ReductionResult::Array(PyRumpyArray::new(x.inner.nanargmin_axis(ax))))
        }
    }
}

/// Index of maximum ignoring NaN values.
#[pyfunction]
#[pyo3(signature = (x, axis=None))]
pub fn nanargmax(x: &PyRumpyArray, axis: Option<isize>) -> PyResult<pyarray::ReductionResult> {
    match axis {
        None => {
            match x.inner.nanargmax() {
                Some(idx) => Ok(pyarray::ReductionResult::Scalar(idx as f64)),
                None => Err(pyo3::exceptions::PyValueError::new_err(
                    "All-NaN slice encountered"
                )),
            }
        }
        Some(ax) => {
            let ax = resolve_axis(ax, x.inner.ndim());
            check_axis(ax, x.inner.ndim())?;
            Ok(pyarray::ReductionResult::Array(PyRumpyArray::new(x.inner.nanargmax_axis(ax))))
        }
    }
}

// ============================================================================
// Boolean reductions
// ============================================================================

/// Test if all elements evaluate to True.
#[pyfunction]
#[pyo3(signature = (x, axis=None))]
pub fn all(x: &PyRumpyArray, axis: Option<&Bound<'_, pyo3::PyAny>>) -> PyResult<pyarray::ReductionResult> {
    let axes = parse_axis(axis, x.inner.ndim())?;
    match axes {
        None => Ok(pyarray::ReductionResult::Scalar(if x.inner.all() { 1.0 } else { 0.0 })),
        Some(ax_vec) if ax_vec.len() == 1 => {
            Ok(pyarray::ReductionResult::Array(PyRumpyArray::new(x.inner.all_axis(ax_vec[0]))))
        }
        Some(ax_vec) => {
            let result = reduce_multi_axis(&x.inner, ax_vec, |arr, axis| arr.all_axis(axis));
            Ok(pyarray::ReductionResult::Array(PyRumpyArray::new(result)))
        }
    }
}

/// Test if any element evaluates to True.
#[pyfunction]
#[pyo3(signature = (x, axis=None))]
pub fn any(x: &PyRumpyArray, axis: Option<&Bound<'_, pyo3::PyAny>>) -> PyResult<pyarray::ReductionResult> {
    let axes = parse_axis(axis, x.inner.ndim())?;
    match axes {
        None => Ok(pyarray::ReductionResult::Scalar(if x.inner.any() { 1.0 } else { 0.0 })),
        Some(ax_vec) if ax_vec.len() == 1 => {
            Ok(pyarray::ReductionResult::Array(PyRumpyArray::new(x.inner.any_axis(ax_vec[0]))))
        }
        Some(ax_vec) => {
            let result = reduce_multi_axis(&x.inner, ax_vec, |arr, axis| arr.any_axis(axis));
            Ok(pyarray::ReductionResult::Array(PyRumpyArray::new(result)))
        }
    }
}

// ============================================================================
// Cumulative operations
// ============================================================================

/// Cumulative sum along axis (or flattened if axis is None).
#[pyfunction]
#[pyo3(signature = (x, axis=None))]
pub fn cumsum(x: &PyRumpyArray, axis: Option<isize>) -> PyResult<PyRumpyArray> {
    let normalized_axis = axis.map(|ax| {
        let ax = resolve_axis(ax, x.inner.ndim());
        check_axis(ax, x.inner.ndim()).map(|_| ax)
    }).transpose()?;
    Ok(PyRumpyArray::new(x.inner.cumsum(normalized_axis)))
}

/// Cumulative product along axis (or flattened if axis is None).
#[pyfunction]
#[pyo3(signature = (x, axis=None))]
pub fn cumprod(x: &PyRumpyArray, axis: Option<isize>) -> PyResult<PyRumpyArray> {
    let normalized_axis = axis.map(|ax| {
        let ax = resolve_axis(ax, x.inner.ndim());
        check_axis(ax, x.inner.ndim()).map(|_| ax)
    }).transpose()?;
    Ok(PyRumpyArray::new(x.inner.cumprod(normalized_axis)))
}

// ============================================================================
// Statistics
// ============================================================================

/// Median of array elements.
#[pyfunction]
#[pyo3(signature = (x, axis=None))]
pub fn median(x: &PyRumpyArray, axis: Option<&Bound<'_, pyo3::PyAny>>) -> PyResult<pyarray::ReductionResult> {
    let axes = parse_axis(axis, x.inner.ndim())?;
    match axes {
        None => Ok(pyarray::ReductionResult::Scalar(x.inner.median())),
        Some(ax_vec) if ax_vec.len() == 1 => {
            Ok(pyarray::ReductionResult::Array(PyRumpyArray::new(x.inner.median_axis(ax_vec[0]))))
        }
        Some(ax_vec) => {
            let result = reduce_multi_axis(&x.inner, ax_vec, |arr, axis| arr.median_axis(axis));
            Ok(pyarray::ReductionResult::Array(PyRumpyArray::new(result)))
        }
    }
}

/// Weighted average of array elements.
#[pyfunction]
#[pyo3(signature = (x, axis=None, weights=None))]
pub fn average(
    x: &PyRumpyArray,
    axis: Option<isize>,
    weights: Option<&PyRumpyArray>,
) -> PyResult<pyarray::ReductionResult> {
    match axis {
        None => {
            let weights_inner = weights.map(|w| &w.inner);
            Ok(pyarray::ReductionResult::Scalar(x.inner.average(weights_inner)))
        }
        Some(ax) => {
            let ax = resolve_axis(ax, x.inner.ndim());
            check_axis(ax, x.inner.ndim())?;
            let weights_inner = weights.map(|w| &w.inner);
            Ok(pyarray::ReductionResult::Array(PyRumpyArray::new(
                x.inner.average_axis(ax, weights_inner),
            )))
        }
    }
}

/// Peak to peak (max - min) of array elements.
#[pyfunction]
#[pyo3(signature = (x, axis=None))]
pub fn ptp(x: &PyRumpyArray, axis: Option<&Bound<'_, pyo3::PyAny>>) -> PyResult<pyarray::ReductionResult> {
    let axes = parse_axis(axis, x.inner.ndim())?;
    match axes {
        None => Ok(pyarray::ReductionResult::Scalar(x.inner.ptp())),
        Some(ax_vec) if ax_vec.len() == 1 => {
            Ok(pyarray::ReductionResult::Array(PyRumpyArray::new(x.inner.ptp_axis(ax_vec[0]))))
        }
        Some(ax_vec) => {
            let result = reduce_multi_axis(&x.inner, ax_vec, |arr, axis| arr.ptp_axis(axis));
            Ok(pyarray::ReductionResult::Array(PyRumpyArray::new(result)))
        }
    }
}

/// Compute histogram.
#[pyfunction]
#[pyo3(signature = (x, bins=10, range=None))]
pub fn histogram(
    x: &PyRumpyArray,
    bins: usize,
    range: Option<(f64, f64)>,
) -> PyResult<(PyRumpyArray, PyRumpyArray)> {
    let (counts, edges) = crate::ops::histogram(&x.inner, bins, range);
    Ok((PyRumpyArray::new(counts), PyRumpyArray::new(edges)))
}

/// Compute covariance matrix.
#[pyfunction]
#[pyo3(name = "cov", signature = (x, ddof=1))]
pub fn cov_fn(x: &PyRumpyArray, ddof: usize) -> PyRumpyArray {
    PyRumpyArray::new(crate::ops::cov(&x.inner, ddof))
}

/// Compute correlation coefficient matrix.
#[pyfunction]
pub fn corrcoef(x: &PyRumpyArray) -> PyRumpyArray {
    PyRumpyArray::new(crate::ops::corrcoef(&x.inner))
}

/// Count occurrences of each non-negative integer value.
#[pyfunction]
#[pyo3(signature = (x, minlength=0))]
pub fn bincount(x: &PyRumpyArray, minlength: usize) -> PyResult<PyRumpyArray> {
    crate::array::bincount(&x.inner, minlength)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("bincount requires 1D non-negative integer array"))
}

/// Compute percentiles of a dataset.
#[pyfunction]
#[pyo3(signature = (a, q, axis=None))]
pub fn percentile(a: &PyRumpyArray, q: &pyo3::Bound<'_, pyo3::PyAny>, axis: Option<usize>) -> PyResult<PyRumpyArray> {
    // Parse q - can be scalar or array-like
    let q_values: Vec<f64> = if let Ok(val) = q.extract::<f64>() {
        vec![val]
    } else if let Ok(list) = q.extract::<Vec<f64>>() {
        list
    } else {
        return Err(pyo3::exceptions::PyTypeError::new_err("q must be a number or list of numbers"));
    };

    // Validate q values are in [0, 100]
    for &qv in &q_values {
        if !(0.0..=100.0).contains(&qv) {
            return Err(pyo3::exceptions::PyValueError::new_err("percentiles must be in range [0, 100]"));
        }
    }

    crate::array::percentile(&a.inner, &q_values, axis)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("percentile computation failed"))
}

/// Compute quantiles of a dataset.
#[pyfunction]
#[pyo3(signature = (a, q, axis=None))]
pub fn quantile(a: &PyRumpyArray, q: &pyo3::Bound<'_, pyo3::PyAny>, axis: Option<usize>) -> PyResult<PyRumpyArray> {
    // Parse q - can be scalar or array-like
    let q_values: Vec<f64> = if let Ok(val) = q.extract::<f64>() {
        vec![val]
    } else if let Ok(list) = q.extract::<Vec<f64>>() {
        list
    } else {
        return Err(pyo3::exceptions::PyTypeError::new_err("q must be a number or list of numbers"));
    };

    // Validate q values are in [0, 1]
    for &qv in &q_values {
        if !(0.0..=1.0).contains(&qv) {
            return Err(pyo3::exceptions::PyValueError::new_err("quantiles must be in range [0, 1]"));
        }
    }

    // Convert to percentiles
    let pct_values: Vec<f64> = q_values.iter().map(|&q| q * 100.0).collect();

    crate::array::percentile(&a.inner, &pct_values, axis)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("quantile computation failed"))
}

#[pyfunction]
pub fn count_nonzero(x: &PyRumpyArray) -> usize {
    x.inner.count_nonzero()
}

#[pyfunction]
#[pyo3(signature = (x, n=1, axis=-1))]
pub fn diff(x: &PyRumpyArray, n: usize, axis: isize) -> PyRumpyArray {
    PyRumpyArray::new(x.inner.diff(n, resolve_axis(axis, x.inner.ndim())))
}
