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
            let val = unsafe { arr.inner.dtype().ops().read_i64(arr.inner.data_ptr(), offset) }
                .unwrap_or(0);
            result.push(val);
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

// ============================================================================
// Index utilities (Stream 26)
// ============================================================================

/// Convert flat indices into a tuple of coordinate arrays.
#[pyfunction]
#[pyo3(signature = (indices, shape, order="C"))]
pub fn unravel_index(
    py: Python<'_>,
    indices: &Bound<'_, pyo3::PyAny>,
    shape: Vec<usize>,
    order: &str,
) -> PyResult<pyo3::PyObject> {
    // Check if input is scalar
    let is_scalar = indices.extract::<i64>().is_ok();

    // Extract indices as Vec<i64>
    let idx_vec: Vec<i64> = if let Ok(scalar) = indices.extract::<i64>() {
        vec![scalar]
    } else if let Ok(list) = indices.extract::<Vec<i64>>() {
        list
    } else if let Ok(arr) = indices.extract::<pyo3::PyRef<'_, PyRumpyArray>>() {
        let size = arr.inner.size();
        let ptr = arr.inner.data_ptr();
        let dtype = arr.inner.dtype();
        let ops = dtype.ops();
        (0..size)
            .map(|i| unsafe { ops.read_i64(ptr, (i * dtype.itemsize()) as isize) }.unwrap_or(0))
            .collect()
    } else {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "indices must be integer, list, or array",
        ));
    };

    let order_char = if order.to_uppercase() == "F" { 'F' } else { 'C' };

    let result = crate::ops::indexing::unravel_index(&idx_vec, &shape, order_char)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("index out of bounds"))?;

    if is_scalar {
        // Return tuple of scalars
        let scalars: Vec<i64> = result.iter().map(|arr| {
            let ptr = arr.data_ptr() as *const i64;
            unsafe { *ptr }
        }).collect();

        let tuple = pyo3::types::PyTuple::new(
            py,
            scalars.into_iter().map(|v| v.into_pyobject(py).unwrap()),
        )?;
        Ok(tuple.into())
    } else {
        // Return tuple of arrays
        let tuple = pyo3::types::PyTuple::new(
            py,
            result.into_iter().map(|arr| {
                pyo3::Py::new(py, PyRumpyArray::new(arr)).unwrap()
            }),
        )?;
        Ok(tuple.into())
    }
}

/// Convert a tuple of coordinate arrays to an array of flat indices.
#[pyfunction]
#[pyo3(signature = (multi_index, dims, mode="raise", order="C"))]
pub fn ravel_multi_index(
    multi_index: &Bound<'_, pyo3::PyAny>,
    dims: Vec<usize>,
    mode: &str,
    order: &str,
) -> PyResult<PyRumpyArray> {
    // Extract multi_index as Vec<Vec<i64>>
    let multi_idx: Vec<Vec<i64>> = if let Ok(list) = multi_index.extract::<Vec<Vec<i64>>>() {
        list
    } else if let Ok(arr) = multi_index.extract::<pyo3::PyRef<'_, PyRumpyArray>>() {
        // Handle 2D array input
        let shape = arr.inner.shape();
        if arr.inner.ndim() != 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "multi_index must be a sequence of arrays or 2D array",
            ));
        }
        let n_dims = shape[0];
        let n_elements = shape[1];
        let ptr = arr.inner.data_ptr();
        let dtype = arr.inner.dtype();
        let ops = dtype.ops();
        let strides = arr.inner.strides();

        (0..n_dims)
            .map(|d| {
                (0..n_elements)
                    .map(|e| {
                        let offset = (d as isize) * strides[0] + (e as isize) * strides[1];
                        unsafe { ops.read_i64(ptr, offset) }.unwrap_or(0)
                    })
                    .collect()
            })
            .collect()
    } else {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "multi_index must be array or list of lists",
        ));
    };

    let order_char = if order.to_uppercase() == "F" { 'F' } else { 'C' };

    crate::ops::indexing::ravel_multi_index(&multi_idx, &dims, mode, order_char)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("index out of bounds"))
}

/// Return the indices for the main diagonal of an array.
#[pyfunction]
#[pyo3(signature = (n, ndim=2))]
pub fn diag_indices(py: Python<'_>, n: usize, ndim: usize) -> PyResult<pyo3::PyObject> {
    let result = crate::ops::indexing::diag_indices(n, ndim);

    let tuple = pyo3::types::PyTuple::new(
        py,
        result.into_iter().map(|arr| {
            pyo3::Py::new(py, PyRumpyArray::new(arr)).unwrap()
        }),
    )?;
    Ok(tuple.into())
}

/// Return the indices for the main diagonal of the given array.
#[pyfunction]
pub fn diag_indices_from(py: Python<'_>, arr: &PyRumpyArray) -> PyResult<pyo3::PyObject> {
    let shape = arr.inner.shape();
    let ndim = arr.inner.ndim();

    if ndim < 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "input array must be at least 2-d",
        ));
    }

    // All dimensions must be equal
    let n = shape[0];
    for &dim in &shape[1..] {
        if dim != n {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "all dimensions must be of equal length",
            ));
        }
    }

    let result = crate::ops::indexing::diag_indices(n, ndim);

    let tuple = pyo3::types::PyTuple::new(
        py,
        result.into_iter().map(|arr| {
            pyo3::Py::new(py, PyRumpyArray::new(arr)).unwrap()
        }),
    )?;
    Ok(tuple.into())
}

/// Return the indices for the lower-triangle of an (n, m) array.
#[pyfunction]
#[pyo3(signature = (n, k=0, m=None))]
pub fn tril_indices(py: Python<'_>, n: usize, k: i64, m: Option<usize>) -> PyResult<pyo3::PyObject> {
    let m = m.unwrap_or(n);
    let (rows, cols) = crate::ops::indexing::tril_indices(n, k, m);

    let tuple = pyo3::types::PyTuple::new(
        py,
        [
            pyo3::Py::new(py, PyRumpyArray::new(rows))?,
            pyo3::Py::new(py, PyRumpyArray::new(cols))?,
        ],
    )?;
    Ok(tuple.into())
}

/// Return the indices for the upper-triangle of an (n, m) array.
#[pyfunction]
#[pyo3(signature = (n, k=0, m=None))]
pub fn triu_indices(py: Python<'_>, n: usize, k: i64, m: Option<usize>) -> PyResult<pyo3::PyObject> {
    let m = m.unwrap_or(n);
    let (rows, cols) = crate::ops::indexing::triu_indices(n, k, m);

    let tuple = pyo3::types::PyTuple::new(
        py,
        [
            pyo3::Py::new(py, PyRumpyArray::new(rows))?,
            pyo3::Py::new(py, PyRumpyArray::new(cols))?,
        ],
    )?;
    Ok(tuple.into())
}

/// Return the indices for the lower-triangle of arr.
#[pyfunction]
#[pyo3(signature = (arr, k=0))]
pub fn tril_indices_from(py: Python<'_>, arr: &PyRumpyArray, k: i64) -> PyResult<pyo3::PyObject> {
    let shape = arr.inner.shape();
    if arr.inner.ndim() != 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "input array must be 2-d",
        ));
    }

    let n = shape[0];
    let m = shape[1];
    let (rows, cols) = crate::ops::indexing::tril_indices(n, k, m);

    let tuple = pyo3::types::PyTuple::new(
        py,
        [
            pyo3::Py::new(py, PyRumpyArray::new(rows))?,
            pyo3::Py::new(py, PyRumpyArray::new(cols))?,
        ],
    )?;
    Ok(tuple.into())
}

/// Return the indices for the upper-triangle of arr.
#[pyfunction]
#[pyo3(signature = (arr, k=0))]
pub fn triu_indices_from(py: Python<'_>, arr: &PyRumpyArray, k: i64) -> PyResult<pyo3::PyObject> {
    let shape = arr.inner.shape();
    if arr.inner.ndim() != 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "input array must be 2-d",
        ));
    }

    let n = shape[0];
    let m = shape[1];
    let (rows, cols) = crate::ops::indexing::triu_indices(n, k, m);

    let tuple = pyo3::types::PyTuple::new(
        py,
        [
            pyo3::Py::new(py, PyRumpyArray::new(rows))?,
            pyo3::Py::new(py, PyRumpyArray::new(cols))?,
        ],
    )?;
    Ok(tuple.into())
}

/// Return the indices to access (n, n) arrays using a mask function.
#[pyfunction]
#[pyo3(signature = (n, mask_func, k=0))]
pub fn mask_indices(
    py: Python<'_>,
    n: usize,
    mask_func: &Bound<'_, pyo3::PyAny>,
    k: i64,
) -> PyResult<pyo3::PyObject> {
    // Create an n x n array of ones
    let ones = crate::array::RumpyArray::full(vec![n, n], 1.0, crate::array::DType::float64());
    let py_ones = pyo3::Py::new(py, PyRumpyArray::new(ones))?;

    // Call the mask function with optional k parameter
    let masked_result = if k != 0 {
        mask_func.call1((py_ones, k))?
    } else {
        mask_func.call1((py_ones,))?
    };
    let masked: pyo3::PyRef<'_, PyRumpyArray> = masked_result.extract()?;

    // Get indices where mask is true (nonzero)
    let result = crate::ops::indexing::argwhere(&masked.inner);

    // Convert to tuple of arrays (row_indices, col_indices)
    let shape = result.shape();
    if shape[0] == 0 {
        let empty1 = crate::array::RumpyArray::zeros(vec![0], crate::array::DType::int64());
        let empty2 = crate::array::RumpyArray::zeros(vec![0], crate::array::DType::int64());
        let tuple = pyo3::types::PyTuple::new(
            py,
            [
                pyo3::Py::new(py, PyRumpyArray::new(empty1))?,
                pyo3::Py::new(py, PyRumpyArray::new(empty2))?,
            ],
        )?;
        return Ok(tuple.into());
    }

    // Extract row and column indices
    let n_indices = shape[0];
    let ptr = result.data_ptr() as *const i64;

    let mut rows: Vec<f64> = Vec::with_capacity(n_indices);
    let mut cols: Vec<f64> = Vec::with_capacity(n_indices);

    for i in 0..n_indices {
        unsafe {
            rows.push(*ptr.add(i * 2) as f64);
            cols.push(*ptr.add(i * 2 + 1) as f64);
        }
    }

    let row_arr = crate::array::RumpyArray::from_vec(rows, crate::array::DType::int64());
    let col_arr = crate::array::RumpyArray::from_vec(cols, crate::array::DType::int64());

    let tuple = pyo3::types::PyTuple::new(
        py,
        [
            pyo3::Py::new(py, PyRumpyArray::new(row_arr))?,
            pyo3::Py::new(py, PyRumpyArray::new(col_arr))?,
        ],
    )?;
    Ok(tuple.into())
}

/// Return the indices of the bins to which each value in input belongs.
#[pyfunction]
#[pyo3(signature = (x, bins, right=false))]
pub fn digitize(x: &PyRumpyArray, bins: &PyRumpyArray, right: bool) -> PyRumpyArray {
    PyRumpyArray::new(crate::ops::indexing::digitize(&x.inner, &bins.inner, right))
}

/// Pack binary-valued array into uint8 array.
#[pyfunction]
#[pyo3(signature = (a, axis=None, bitorder="big"))]
pub fn packbits(
    a: &PyRumpyArray,
    axis: Option<isize>,
    bitorder: &str,
) -> PyRumpyArray {
    let resolved_axis = axis.map(|ax| {
        if ax < 0 {
            (a.inner.ndim() as isize + ax) as usize
        } else {
            ax as usize
        }
    });

    // axis=None means flatten first (use default axis)
    let actual_axis = if axis.is_none() {
        None
    } else {
        resolved_axis
    };

    PyRumpyArray::new(crate::ops::indexing::packbits(&a.inner, actual_axis, bitorder))
}

/// Unpack elements of uint8 array into binary-valued output.
#[pyfunction]
#[pyo3(signature = (a, axis=None, count=None, bitorder="big"))]
pub fn unpackbits(
    a: &PyRumpyArray,
    axis: Option<isize>,
    count: Option<usize>,
    bitorder: &str,
) -> PyRumpyArray {
    let resolved_axis = axis.map(|ax| {
        if ax < 0 {
            (a.inner.ndim() as isize + ax) as usize
        } else {
            ax as usize
        }
    });

    PyRumpyArray::new(crate::ops::indexing::unpackbits(&a.inner, resolved_axis, count, bitorder))
}
