// Python bindings for reduction operations (sum, mean, etc.)

use pyo3::prelude::*;

use super::{pyarray, PyRumpyArray};

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

// ============================================================================
// Basic reductions
// ============================================================================

/// Sum of array elements.
#[pyfunction]
#[pyo3(signature = (x, axis=None))]
pub fn sum(x: &PyRumpyArray, axis: Option<usize>) -> PyResult<pyarray::ReductionResult> {
    match axis {
        None => Ok(pyarray::ReductionResult::Scalar(x.inner.sum())),
        Some(ax) => {
            check_axis(ax, x.inner.ndim())?;
            Ok(pyarray::ReductionResult::Array(PyRumpyArray::new(x.inner.sum_axis(ax))))
        }
    }
}

/// Product of array elements.
#[pyfunction]
#[pyo3(signature = (x, axis=None))]
pub fn prod(x: &PyRumpyArray, axis: Option<usize>) -> PyResult<pyarray::ReductionResult> {
    match axis {
        None => Ok(pyarray::ReductionResult::Scalar(x.inner.prod())),
        Some(ax) => {
            check_axis(ax, x.inner.ndim())?;
            Ok(pyarray::ReductionResult::Array(PyRumpyArray::new(x.inner.prod_axis(ax))))
        }
    }
}

/// Mean of array elements.
#[pyfunction]
#[pyo3(signature = (x, axis=None))]
pub fn mean(x: &PyRumpyArray, axis: Option<usize>) -> PyResult<pyarray::ReductionResult> {
    match axis {
        None => Ok(pyarray::ReductionResult::Scalar(x.inner.mean())),
        Some(ax) => {
            check_axis(ax, x.inner.ndim())?;
            Ok(pyarray::ReductionResult::Array(PyRumpyArray::new(x.inner.mean_axis(ax))))
        }
    }
}

/// Variance of array elements.
#[pyfunction]
#[pyo3(signature = (x, axis=None))]
pub fn var(x: &PyRumpyArray, axis: Option<usize>) -> PyResult<pyarray::ReductionResult> {
    match axis {
        None => Ok(pyarray::ReductionResult::Scalar(x.inner.var())),
        Some(ax) => {
            check_axis(ax, x.inner.ndim())?;
            Ok(pyarray::ReductionResult::Array(PyRumpyArray::new(x.inner.var_axis(ax))))
        }
    }
}

/// Standard deviation of array elements.
#[pyfunction]
#[pyo3(name = "std", signature = (x, axis=None))]
pub fn std_fn(x: &PyRumpyArray, axis: Option<usize>) -> PyResult<pyarray::ReductionResult> {
    match axis {
        None => Ok(pyarray::ReductionResult::Scalar(x.inner.std())),
        Some(ax) => {
            check_axis(ax, x.inner.ndim())?;
            Ok(pyarray::ReductionResult::Array(PyRumpyArray::new(x.inner.std_axis(ax))))
        }
    }
}

/// Maximum of array elements.
#[pyfunction]
#[pyo3(signature = (x, axis=None))]
pub fn max(x: &PyRumpyArray, axis: Option<usize>) -> PyResult<pyarray::ReductionResult> {
    match axis {
        None => Ok(pyarray::ReductionResult::Scalar(x.inner.max())),
        Some(ax) => {
            check_axis(ax, x.inner.ndim())?;
            Ok(pyarray::ReductionResult::Array(PyRumpyArray::new(x.inner.max_axis(ax))))
        }
    }
}

/// Minimum of array elements.
#[pyfunction]
#[pyo3(signature = (x, axis=None))]
pub fn min(x: &PyRumpyArray, axis: Option<usize>) -> PyResult<pyarray::ReductionResult> {
    match axis {
        None => Ok(pyarray::ReductionResult::Scalar(x.inner.min())),
        Some(ax) => {
            check_axis(ax, x.inner.ndim())?;
            Ok(pyarray::ReductionResult::Array(PyRumpyArray::new(x.inner.min_axis(ax))))
        }
    }
}

/// Index of maximum element.
#[pyfunction]
#[pyo3(signature = (x, axis=None))]
pub fn argmax(x: &PyRumpyArray, axis: Option<usize>) -> PyResult<pyarray::ReductionResult> {
    match axis {
        None => Ok(pyarray::ReductionResult::Scalar(x.inner.argmax() as f64)),
        Some(ax) => {
            check_axis(ax, x.inner.ndim())?;
            Ok(pyarray::ReductionResult::Array(PyRumpyArray::new(x.inner.argmax_axis(ax))))
        }
    }
}

/// Index of minimum element.
#[pyfunction]
#[pyo3(signature = (x, axis=None))]
pub fn argmin(x: &PyRumpyArray, axis: Option<usize>) -> PyResult<pyarray::ReductionResult> {
    match axis {
        None => Ok(pyarray::ReductionResult::Scalar(x.inner.argmin() as f64)),
        Some(ax) => {
            check_axis(ax, x.inner.ndim())?;
            Ok(pyarray::ReductionResult::Array(PyRumpyArray::new(x.inner.argmin_axis(ax))))
        }
    }
}

// ============================================================================
// NaN-aware reductions
// ============================================================================

/// Sum ignoring NaN values.
#[pyfunction]
#[pyo3(signature = (x, axis=None))]
pub fn nansum(x: &PyRumpyArray, axis: Option<usize>) -> PyResult<pyarray::ReductionResult> {
    match axis {
        None => Ok(pyarray::ReductionResult::Scalar(x.inner.nansum())),
        Some(ax) => {
            check_axis(ax, x.inner.ndim())?;
            Ok(pyarray::ReductionResult::Array(PyRumpyArray::new(x.inner.nansum_axis(ax))))
        }
    }
}

/// Product ignoring NaN values.
#[pyfunction]
#[pyo3(signature = (x, axis=None))]
pub fn nanprod(x: &PyRumpyArray, axis: Option<usize>) -> PyResult<pyarray::ReductionResult> {
    match axis {
        None => Ok(pyarray::ReductionResult::Scalar(x.inner.nanprod())),
        Some(ax) => {
            check_axis(ax, x.inner.ndim())?;
            Ok(pyarray::ReductionResult::Array(PyRumpyArray::new(x.inner.nanprod_axis(ax))))
        }
    }
}

/// Mean ignoring NaN values.
#[pyfunction]
#[pyo3(signature = (x, axis=None))]
pub fn nanmean(x: &PyRumpyArray, axis: Option<usize>) -> PyResult<pyarray::ReductionResult> {
    match axis {
        None => Ok(pyarray::ReductionResult::Scalar(x.inner.nanmean())),
        Some(ax) => {
            check_axis(ax, x.inner.ndim())?;
            Ok(pyarray::ReductionResult::Array(PyRumpyArray::new(x.inner.nanmean_axis(ax))))
        }
    }
}

/// Variance ignoring NaN values.
#[pyfunction]
#[pyo3(signature = (x, axis=None))]
pub fn nanvar(x: &PyRumpyArray, axis: Option<usize>) -> PyResult<pyarray::ReductionResult> {
    match axis {
        None => Ok(pyarray::ReductionResult::Scalar(x.inner.nanvar())),
        Some(ax) => {
            check_axis(ax, x.inner.ndim())?;
            Ok(pyarray::ReductionResult::Array(PyRumpyArray::new(x.inner.nanvar_axis(ax))))
        }
    }
}

/// Standard deviation ignoring NaN values.
#[pyfunction]
#[pyo3(signature = (x, axis=None))]
pub fn nanstd(x: &PyRumpyArray, axis: Option<usize>) -> PyResult<pyarray::ReductionResult> {
    match axis {
        None => Ok(pyarray::ReductionResult::Scalar(x.inner.nanstd())),
        Some(ax) => {
            check_axis(ax, x.inner.ndim())?;
            Ok(pyarray::ReductionResult::Array(PyRumpyArray::new(x.inner.nanstd_axis(ax))))
        }
    }
}

/// Minimum ignoring NaN values.
#[pyfunction]
#[pyo3(signature = (x, axis=None))]
pub fn nanmin(x: &PyRumpyArray, axis: Option<usize>) -> PyResult<pyarray::ReductionResult> {
    match axis {
        None => Ok(pyarray::ReductionResult::Scalar(x.inner.nanmin())),
        Some(ax) => {
            check_axis(ax, x.inner.ndim())?;
            Ok(pyarray::ReductionResult::Array(PyRumpyArray::new(x.inner.nanmin_axis(ax))))
        }
    }
}

/// Maximum ignoring NaN values.
#[pyfunction]
#[pyo3(signature = (x, axis=None))]
pub fn nanmax(x: &PyRumpyArray, axis: Option<usize>) -> PyResult<pyarray::ReductionResult> {
    match axis {
        None => Ok(pyarray::ReductionResult::Scalar(x.inner.nanmax())),
        Some(ax) => {
            check_axis(ax, x.inner.ndim())?;
            Ok(pyarray::ReductionResult::Array(PyRumpyArray::new(x.inner.nanmax_axis(ax))))
        }
    }
}

/// Index of minimum ignoring NaN values.
#[pyfunction]
#[pyo3(signature = (x, axis=None))]
pub fn nanargmin(x: &PyRumpyArray, axis: Option<usize>) -> PyResult<pyarray::ReductionResult> {
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
            check_axis(ax, x.inner.ndim())?;
            Ok(pyarray::ReductionResult::Array(PyRumpyArray::new(x.inner.nanargmin_axis(ax))))
        }
    }
}

/// Index of maximum ignoring NaN values.
#[pyfunction]
#[pyo3(signature = (x, axis=None))]
pub fn nanargmax(x: &PyRumpyArray, axis: Option<usize>) -> PyResult<pyarray::ReductionResult> {
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
pub fn all(x: &PyRumpyArray, axis: Option<usize>) -> PyResult<pyarray::ReductionResult> {
    match axis {
        None => Ok(pyarray::ReductionResult::Scalar(if x.inner.all() { 1.0 } else { 0.0 })),
        Some(ax) => {
            check_axis(ax, x.inner.ndim())?;
            Ok(pyarray::ReductionResult::Array(PyRumpyArray::new(x.inner.all_axis(ax))))
        }
    }
}

/// Test if any element evaluates to True.
#[pyfunction]
#[pyo3(signature = (x, axis=None))]
pub fn any(x: &PyRumpyArray, axis: Option<usize>) -> PyResult<pyarray::ReductionResult> {
    match axis {
        None => Ok(pyarray::ReductionResult::Scalar(if x.inner.any() { 1.0 } else { 0.0 })),
        Some(ax) => {
            check_axis(ax, x.inner.ndim())?;
            Ok(pyarray::ReductionResult::Array(PyRumpyArray::new(x.inner.any_axis(ax))))
        }
    }
}

// ============================================================================
// Cumulative operations
// ============================================================================

/// Cumulative sum along axis (or flattened if axis is None).
#[pyfunction]
#[pyo3(signature = (x, axis=None))]
pub fn cumsum(x: &PyRumpyArray, axis: Option<usize>) -> PyResult<PyRumpyArray> {
    if let Some(ax) = axis {
        if ax >= x.inner.ndim() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "axis {} is out of bounds for array of dimension {}",
                ax, x.inner.ndim()
            )));
        }
    }
    Ok(PyRumpyArray::new(x.inner.cumsum(axis)))
}

/// Cumulative product along axis (or flattened if axis is None).
#[pyfunction]
#[pyo3(signature = (x, axis=None))]
pub fn cumprod(x: &PyRumpyArray, axis: Option<usize>) -> PyResult<PyRumpyArray> {
    if let Some(ax) = axis {
        if ax >= x.inner.ndim() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "axis {} is out of bounds for array of dimension {}",
                ax, x.inner.ndim()
            )));
        }
    }
    Ok(PyRumpyArray::new(x.inner.cumprod(axis)))
}

// ============================================================================
// Statistics
// ============================================================================

/// Median of array elements.
#[pyfunction]
#[pyo3(signature = (x, axis=None))]
pub fn median(x: &PyRumpyArray, axis: Option<usize>) -> PyResult<pyarray::ReductionResult> {
    match axis {
        None => Ok(pyarray::ReductionResult::Scalar(x.inner.median())),
        Some(ax) => {
            check_axis(ax, x.inner.ndim())?;
            Ok(pyarray::ReductionResult::Array(PyRumpyArray::new(x.inner.median_axis(ax))))
        }
    }
}

/// Weighted average of array elements.
#[pyfunction]
#[pyo3(signature = (x, axis=None, weights=None))]
pub fn average(
    x: &PyRumpyArray,
    axis: Option<usize>,
    weights: Option<&PyRumpyArray>,
) -> PyResult<pyarray::ReductionResult> {
    match axis {
        None => {
            let weights_inner = weights.map(|w| &w.inner);
            Ok(pyarray::ReductionResult::Scalar(x.inner.average(weights_inner)))
        }
        Some(ax) => {
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
pub fn ptp(x: &PyRumpyArray, axis: Option<usize>) -> PyResult<pyarray::ReductionResult> {
    match axis {
        None => Ok(pyarray::ReductionResult::Scalar(x.inner.ptp())),
        Some(ax) => {
            check_axis(ax, x.inner.ndim())?;
            Ok(pyarray::ReductionResult::Array(PyRumpyArray::new(x.inner.ptp_axis(ax))))
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
