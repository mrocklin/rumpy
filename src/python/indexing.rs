// Python bindings for indexing and selection operations.

use pyo3::prelude::*;
use pyo3::types::PyList;

use crate::array::{DType, RumpyArray};
use super::{PyRumpyArray, creation};

// ============================================================================
// Selection operations
// ============================================================================

/// Take elements from an array along an axis.
#[pyfunction]
#[pyo3(signature = (a, indices, axis=None))]
pub fn take(
    a: &PyRumpyArray,
    indices: &Bound<'_, pyo3::PyAny>,
    axis: Option<usize>,
) -> PyResult<PyRumpyArray> {
    let indices_arr = if let Ok(arr) = indices.extract::<pyo3::PyRef<'_, PyRumpyArray>>() {
        arr.inner.clone()
    } else if let Ok(list) = indices.extract::<Vec<i64>>() {
        let data: Vec<f64> = list.into_iter().map(|x| x as f64).collect();
        RumpyArray::from_vec(data, DType::int64())
    } else {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "indices must be array or list of integers",
        ));
    };

    crate::ops::indexing::take(&a.inner, &indices_arr, axis)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyIndexError::new_err("index out of bounds"))
}

/// Take values from input array by matching 1d index and data slices.
#[pyfunction]
pub fn take_along_axis(
    arr: &PyRumpyArray,
    indices: &PyRumpyArray,
    axis: usize,
) -> PyResult<PyRumpyArray> {
    crate::ops::indexing::take_along_axis(&arr.inner, &indices.inner, axis)
        .map(PyRumpyArray::new)
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "indices array shape doesn't match or axis out of bounds",
            )
        })
}

/// Select elements using a boolean condition along an axis.
#[pyfunction]
#[pyo3(signature = (condition, a, axis=None))]
pub fn compress(
    condition: &Bound<'_, pyo3::PyAny>,
    a: &PyRumpyArray,
    axis: Option<usize>,
) -> PyResult<PyRumpyArray> {
    let cond_vec: Vec<bool> = if let Ok(list) = condition.extract::<Vec<bool>>() {
        list
    } else if let Ok(arr) = condition.extract::<pyo3::PyRef<'_, PyRumpyArray>>() {
        let ptr = arr.inner.data_ptr();
        let dtype = arr.inner.dtype();
        let ops = dtype.ops();
        arr.inner.iter_offsets()
            .map(|offset| unsafe { ops.is_truthy(ptr, offset) })
            .collect()
    } else {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "condition must be array or list of bools",
        ));
    };

    crate::ops::indexing::compress(&cond_vec, &a.inner, axis)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("axis out of bounds"))
}

// ============================================================================
// Search operations
// ============================================================================

/// Find indices where elements should be inserted to maintain order.
#[pyfunction]
#[pyo3(signature = (a, v, side="left"))]
pub fn searchsorted(
    a: &PyRumpyArray,
    v: &Bound<'_, pyo3::PyAny>,
    side: &str,
) -> PyResult<PyRumpyArray> {
    let v_arr = if let Ok(arr) = v.extract::<pyo3::PyRef<'_, PyRumpyArray>>() {
        arr.inner.clone()
    } else if let Ok(scalar) = v.extract::<f64>() {
        RumpyArray::from_vec(vec![scalar], DType::float64())
    } else {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "v must be array or number",
        ));
    };

    crate::ops::indexing::searchsorted(&a.inner, &v_arr, side)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("a must be 1-D"))
}

/// Find indices of elements that are non-zero.
#[pyfunction]
pub fn argwhere(a: &PyRumpyArray) -> PyRumpyArray {
    PyRumpyArray::new(crate::ops::indexing::argwhere(&a.inner))
}

/// Return indices that are non-zero in the flattened version of a.
#[pyfunction]
pub fn flatnonzero(a: &PyRumpyArray) -> PyRumpyArray {
    PyRumpyArray::new(crate::ops::indexing::flatnonzero(&a.inner))
}

// ============================================================================
// Modification operations
// ============================================================================

/// Replace values at specified flat indices.
#[pyfunction]
pub fn put(
    a: &mut PyRumpyArray,
    ind: &Bound<'_, pyo3::PyAny>,
    v: &Bound<'_, pyo3::PyAny>,
) -> PyResult<()> {
    let indices: Vec<i64> = if let Ok(list) = ind.extract::<Vec<i64>>() {
        list
    } else if let Ok(arr) = ind.extract::<pyo3::PyRef<'_, PyRumpyArray>>() {
        let size = arr.inner.size();
        let mut result = Vec::with_capacity(size);
        for offset in arr.inner.iter_offsets() {
            let val = unsafe { arr.inner.dtype().ops().read_f64(arr.inner.data_ptr(), offset) }
                .unwrap_or(0.0);
            result.push(val as i64);
        }
        result
    } else {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "ind must be array or list of integers",
        ));
    };

    let values: Vec<f64> = if let Ok(scalar) = v.extract::<f64>() {
        vec![scalar]
    } else if let Ok(list) = v.extract::<Vec<f64>>() {
        list
    } else if let Ok(arr) = v.extract::<pyo3::PyRef<'_, PyRumpyArray>>() {
        let size = arr.inner.size();
        let mut result = Vec::with_capacity(size);
        for offset in arr.inner.iter_offsets() {
            let val = unsafe { arr.inner.dtype().ops().read_f64(arr.inner.data_ptr(), offset) }
                .unwrap_or(0.0);
            result.push(val);
        }
        result
    } else {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "v must be number, array, or list",
        ));
    };

    crate::ops::indexing::put(&mut a.inner, &indices, &values)
        .ok_or_else(|| pyo3::exceptions::PyIndexError::new_err("index out of bounds"))
}

/// Put values into the destination array by matching 1d index and data slices.
#[pyfunction]
pub fn put_along_axis(
    arr: &mut PyRumpyArray,
    indices: &PyRumpyArray,
    values: &PyRumpyArray,
    axis: usize,
) -> PyResult<()> {
    crate::ops::indexing::put_along_axis(&mut arr.inner, &indices.inner, &values.inner, axis)
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "indices/values shape doesn't match or axis out of bounds",
            )
        })
}

/// Construct an array from an index array and choices.
#[pyfunction]
pub fn choose(a: &PyRumpyArray, choices: &Bound<'_, PyList>) -> PyResult<PyRumpyArray> {
    let choice_arrs: PyResult<Vec<RumpyArray>> = choices
        .iter()
        .map(|item| {
            item.extract::<pyo3::PyRef<'_, PyRumpyArray>>()
                .map(|arr| arr.inner.clone())
        })
        .collect();
    let choice_arrs = choice_arrs?;

    crate::ops::indexing::choose(&a.inner, &choice_arrs)
        .map(PyRumpyArray::new)
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "shapes don't match or index out of bounds",
            )
        })
}

// ============================================================================
// Conditional selection
// ============================================================================

/// Return elements from x or y depending on condition.
#[pyfunction]
#[pyo3(name = "where")]
pub fn where_fn(
    condition: &PyRumpyArray,
    x: &Bound<'_, pyo3::PyAny>,
    y: &Bound<'_, pyo3::PyAny>,
) -> PyResult<PyRumpyArray> {
    let x_arr = if let Ok(arr) = x.extract::<pyo3::PyRef<'_, PyRumpyArray>>() {
        arr.inner.clone()
    } else if let Ok(scalar) = x.extract::<f64>() {
        RumpyArray::full(vec![1], scalar, DType::float64())
    } else {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "x must be ndarray or number",
        ));
    };

    let y_arr = if let Ok(arr) = y.extract::<pyo3::PyRef<'_, PyRumpyArray>>() {
        arr.inner.clone()
    } else if let Ok(scalar) = y.extract::<f64>() {
        RumpyArray::full(vec![1], scalar, DType::float64())
    } else {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "y must be ndarray or number",
        ));
    };

    crate::ops::where_select(&condition.inner, &x_arr, &y_arr)
        .map(PyRumpyArray::new)
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("operands could not be broadcast together")
        })
}

// ============================================================================
// Set operations
// ============================================================================

/// Test whether each element of a is in test_elements.
#[pyfunction]
#[pyo3(signature = (element, test_elements, invert=false))]
pub fn isin(
    element: &PyRumpyArray,
    test_elements: &PyRumpyArray,
    invert: bool,
) -> PyRumpyArray {
    PyRumpyArray::new(crate::ops::isin(&element.inner, &test_elements.inner, invert))
}

/// Test whether each element of a 1-D array is in another.
/// Deprecated: Use isin instead.
#[pyfunction]
#[pyo3(signature = (ar1, ar2, invert=false))]
pub fn in1d(
    py: Python<'_>,
    ar1: &PyRumpyArray,
    ar2: &PyRumpyArray,
    invert: bool,
) -> PyResult<PyRumpyArray> {
    let warnings = py.import("warnings")?;
    warnings.call_method1(
        "warn",
        (
            "`in1d` is deprecated. Use `isin` instead.",
            py.get_type::<pyo3::exceptions::PyDeprecationWarning>(),
        ),
    )?;
    Ok(PyRumpyArray::new(crate::ops::in1d(&ar1.inner, &ar2.inner, invert)))
}

/// Find the intersection of two arrays.
#[pyfunction]
pub fn intersect1d(ar1: &PyRumpyArray, ar2: &PyRumpyArray) -> PyRumpyArray {
    PyRumpyArray::new(crate::ops::intersect1d(&ar1.inner, &ar2.inner))
}

/// Find the union of two arrays.
#[pyfunction]
pub fn union1d(ar1: &PyRumpyArray, ar2: &PyRumpyArray) -> PyRumpyArray {
    PyRumpyArray::new(crate::ops::union1d(&ar1.inner, &ar2.inner))
}

/// Find the set difference of two arrays.
#[pyfunction]
pub fn setdiff1d(ar1: &PyRumpyArray, ar2: &PyRumpyArray) -> PyRumpyArray {
    PyRumpyArray::new(crate::ops::setdiff1d(&ar1.inner, &ar2.inner))
}

/// Find the set exclusive-or of two arrays.
#[pyfunction]
pub fn setxor1d(ar1: &PyRumpyArray, ar2: &PyRumpyArray) -> PyRumpyArray {
    PyRumpyArray::new(crate::ops::setxor1d(&ar1.inner, &ar2.inner))
}

// ============================================================================
// Sorting
// ============================================================================

/// Return a sorted copy of an array.
#[pyfunction]
#[pyo3(signature = (x, axis=-1))]
pub fn sort(x: &PyRumpyArray, axis: Option<isize>) -> PyRumpyArray {
    let resolved_axis = axis.map(|a| super::reductions::resolve_axis(a, x.inner.ndim()));
    PyRumpyArray::new(x.inner.sort(resolved_axis))
}

/// Return indices that would sort an array.
#[pyfunction]
#[pyo3(signature = (x, axis=-1))]
pub fn argsort(x: &PyRumpyArray, axis: Option<isize>) -> PyRumpyArray {
    let resolved_axis = axis.map(|a| super::reductions::resolve_axis(a, x.inner.ndim()));
    PyRumpyArray::new(x.inner.argsort(resolved_axis))
}

/// Return a partitioned copy of an array.
#[pyfunction]
#[pyo3(signature = (x, kth, axis=-1))]
pub fn partition(x: &PyRumpyArray, kth: usize, axis: Option<isize>) -> PyRumpyArray {
    let resolved_axis = axis.map(|a| super::reductions::resolve_axis(a, x.inner.ndim()));
    PyRumpyArray::new(x.inner.partition(kth, resolved_axis))
}

/// Return indices that would partition an array.
#[pyfunction]
#[pyo3(signature = (x, kth, axis=-1))]
pub fn argpartition(x: &PyRumpyArray, kth: usize, axis: Option<isize>) -> PyRumpyArray {
    let resolved_axis = axis.map(|a| super::reductions::resolve_axis(a, x.inner.ndim()));
    PyRumpyArray::new(x.inner.argpartition(kth, resolved_axis))
}

/// Perform an indirect stable sort using a sequence of keys.
#[pyfunction]
pub fn lexsort(py: Python<'_>, keys: &Bound<'_, pyo3::types::PyTuple>) -> PyResult<PyRumpyArray> {
    let key_arrays: Vec<crate::array::RumpyArray> = keys
        .iter()
        .map(|item| {
            if let Ok(arr) = item.extract::<pyo3::PyRef<'_, PyRumpyArray>>() {
                Ok(arr.inner.clone())
            } else {
                let arr = creation::array_impl(py, &item, None)?;
                Ok(arr.inner.clone())
            }
        })
        .collect::<PyResult<Vec<_>>>()?;

    let key_refs: Vec<&crate::array::RumpyArray> = key_arrays.iter().collect();

    crate::ops::lexsort(&key_refs)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err(
            "lexsort requires 1-D arrays of the same length"
        ))
}
