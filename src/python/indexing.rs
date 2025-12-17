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

// ============================================================================
// Index builders (Stream 34)
// ============================================================================

/// Construct an open mesh from multiple sequences.
///
/// Returns arrays that can be used to select from an array using
/// advanced indexing. The arrays returned are broadcastable to the
/// shape `(len(x1), len(x2), ...)`.
#[pyfunction]
#[pyo3(signature = (*args))]
pub fn ix_(_py: Python<'_>, args: &Bound<'_, pyo3::types::PyTuple>) -> PyResult<Vec<PyRumpyArray>> {
    let n = args.len();
    if n == 0 {
        return Ok(vec![]);
    }

    let mut result = Vec::with_capacity(n);

    for (i, arg) in args.iter().enumerate() {
        let arr: pyo3::PyRef<'_, PyRumpyArray> = arg.extract()?;

        // Each array should be 1D
        if arr.inner.ndim() != 1 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "input arrays must be 1-dimensional",
            ));
        }

        // Build the shape: (1, 1, ..., len, ..., 1)
        let mut new_shape = vec![1usize; n];
        new_shape[i] = arr.inner.size();

        // Reshape the array
        let reshaped = arr.inner.reshape(new_shape).expect("reshape failed");
        result.push(PyRumpyArray::new(reshaped));
    }

    Ok(result)
}

/// Fill the main diagonal of the given array in-place.
#[pyfunction]
#[pyo3(signature = (a, val, wrap=false))]
pub fn fill_diagonal(
    a: &mut PyRumpyArray,
    val: &Bound<'_, pyo3::PyAny>,
    wrap: bool,
) -> PyResult<()> {
    let shape = a.inner.shape().to_vec();
    let ndim = a.inner.ndim();

    if ndim < 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "array must be at least 2-d",
        ));
    }

    // Convert val to a RumpyArray to handle all dtypes uniformly
    let val_arr: RumpyArray = if let Ok(arr) = val.extract::<pyo3::PyRef<'_, PyRumpyArray>>() {
        // Cast to target dtype if needed
        if arr.inner.dtype().ops().name() != a.inner.dtype().ops().name() {
            arr.inner.astype(a.inner.dtype())
        } else {
            arr.inner.clone()
        }
    } else if let Ok(scalar) = val.extract::<f64>() {
        RumpyArray::from_vec(vec![scalar], a.inner.dtype().clone())
    } else if let Ok(list) = val.extract::<Vec<f64>>() {
        RumpyArray::from_vec(list, a.inner.dtype().clone())
    } else {
        // Try extracting as complex (Python complex type)
        if let Ok(real_attr) = val.getattr("real") {
            if let (Ok(real), Ok(imag)) = (real_attr.extract::<f64>(), val.getattr("imag").and_then(|i| i.extract::<f64>())) {
                // Create a complex array using full() which handles complex dtypes
                let target_dtype = a.inner.dtype();
                if target_dtype.ops().name().contains("complex") {
                    // Use full_complex which handles complex dtypes properly
                    RumpyArray::full_complex(vec![1], real, imag, target_dtype.clone())
                } else {
                    // For non-complex target, just use the real part
                    RumpyArray::from_vec(vec![real], target_dtype.clone())
                }
            } else {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "val must be scalar, list, or array",
                ));
            }
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "val must be scalar, list, or array",
            ));
        }
    };

    let val_size = val_arr.size();
    let val_ptr = val_arr.data_ptr();
    let val_itemsize = val_arr.dtype().itemsize();

    // Compute diagonal length
    let min_dim = shape.iter().copied().min().unwrap_or(0);

    // Get mutable access to buffer
    let ptr = a.inner.data_ptr_mut();
    let dtype = a.inner.dtype();
    let strides = a.inner.strides();
    let itemsize = dtype.itemsize();

    if ndim == 2 {
        // 2D case: fill diagonal with optional wrapping
        let nrows = shape[0];
        let ncols = shape[1];

        // Compute step for wrapping (the row offset between diagonals)
        // When wrap=True and nrows > ncols, we wrap every ncols+1 rows
        let wrap_step = ncols + 1;

        let mut val_idx = 0;
        let mut row = 0;

        while row < nrows {
            let col = row % wrap_step;

            // Only fill if within column bounds
            if col < ncols {
                let byte_offset = row as isize * strides[0] + col as isize * strides[1];
                // Read from val array, write to target array
                let src_offset = (val_idx % val_size) * val_itemsize;
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        val_ptr.add(src_offset),
                        ptr.offset(byte_offset),
                        itemsize
                    );
                }
                val_idx += 1;
            }

            row += 1;

            // If not wrapping, stop after first diagonal
            if !wrap && row >= min_dim {
                break;
            }
        }
    } else {
        // nD case: fill diagonal at [i, i, i, ...]
        for idx in 0..min_dim {
            let byte_offset: isize = strides.iter().map(|&s| (idx as isize) * s).sum();
            let src_offset = (idx % val_size) * val_itemsize;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    val_ptr.add(src_offset),
                    ptr.offset(byte_offset),
                    itemsize
                );
            }
        }
    }

    Ok(())
}

/// Multidimensional index iterator.
///
/// Returns an iterator yielding pairs of (index, value) for each element
/// in the array.
#[pyclass]
pub struct NdEnumerate {
    arr: RumpyArray,
    shape: Vec<usize>,
    indices: Vec<usize>,
    flat_idx: usize,
    size: usize,
}

#[pymethods]
impl NdEnumerate {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__<'py>(mut slf: PyRefMut<'py, Self>) -> Option<(pyo3::PyObject, f64)> {
        if slf.flat_idx >= slf.size {
            return None;
        }

        let indices = slf.indices.clone();
        let ptr = slf.arr.data_ptr();
        let dtype = slf.arr.dtype();
        let ops = dtype.ops();

        // Calculate offset from indices
        let strides = slf.arr.strides();
        let offset: isize = slf.indices.iter().zip(strides.iter())
            .map(|(&i, &s)| (i as isize) * s)
            .sum();

        let val = unsafe { ops.read_f64(ptr, offset) }.unwrap_or(0.0);

        // Increment indices (clone shape to avoid borrow conflict)
        slf.flat_idx += 1;
        let shape = slf.shape.clone();
        crate::array::increment_indices(&mut slf.indices, &shape);

        // Convert indices to Python tuple
        let py = slf.py();
        let tuple = pyo3::types::PyTuple::new(py, indices).ok()?.into();

        Some((tuple, val))
    }
}

/// Return an iterator yielding (index, value) pairs.
#[pyfunction]
pub fn ndenumerate(arr: &PyRumpyArray) -> NdEnumerate {
    let shape = arr.inner.shape().to_vec();
    let size = arr.inner.size();
    NdEnumerate {
        arr: arr.inner.clone(),
        shape: shape.clone(),
        indices: vec![0; shape.len()],
        flat_idx: 0,
        size,
    }
}

/// N-dimensional index iterator.
///
/// An iterator yielding tuples of indices for iterating over an array
/// of the given shape.
#[pyclass]
pub struct NdIndex {
    shape: Vec<usize>,
    indices: Vec<usize>,
    flat_idx: usize,
    size: usize,
}

#[pymethods]
impl NdIndex {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__<'py>(mut slf: PyRefMut<'py, Self>) -> Option<pyo3::PyObject> {
        if slf.flat_idx >= slf.size {
            return None;
        }

        let indices = slf.indices.clone();

        // Increment indices (clone shape to avoid borrow conflict)
        slf.flat_idx += 1;
        let shape = slf.shape.clone();
        crate::array::increment_indices(&mut slf.indices, &shape);

        // Convert indices to Python tuple
        let py = slf.py();
        let tuple = pyo3::types::PyTuple::new(py, indices).ok()?.into();

        Some(tuple)
    }
}

/// Return an iterator yielding index tuples.
#[pyfunction]
#[pyo3(signature = (*args))]
pub fn ndindex(args: &Bound<'_, pyo3::types::PyTuple>) -> PyResult<NdIndex> {
    // Parse shape from args - can be ndindex(2, 3) or ndindex((2, 3))
    let shape: Vec<usize> = if args.len() == 1 {
        // Check if first arg is a tuple
        if let Ok(tuple) = args.get_item(0)?.extract::<Vec<usize>>() {
            tuple
        } else if let Ok(val) = args.get_item(0)?.extract::<usize>() {
            vec![val]
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "shape must be integers or tuple of integers",
            ));
        }
    } else if args.is_empty() {
        vec![]
    } else {
        args.iter()
            .map(|item| item.extract::<usize>())
            .collect::<PyResult<Vec<_>>>()?
    };

    let size: usize = shape.iter().product();
    let ndim = shape.len();

    // Handle empty shape case - yields single empty tuple
    let (size, indices) = if ndim == 0 {
        (1, vec![])
    } else {
        (size, vec![0; ndim])
    };

    Ok(NdIndex {
        shape,
        indices,
        flat_idx: 0,
        size,
    })
}

/// Open grid class (sparse meshgrid).
///
/// Use: `ogrid[0:5, 0:3]` returns a list of arrays with shapes
/// `[(5, 1), (1, 3)]`.
#[pyclass]
pub struct OGridClass;

#[pymethods]
impl OGridClass {
    fn __getitem__(&self, py: Python<'_>, key: &Bound<'_, pyo3::PyAny>) -> PyResult<pyo3::PyObject> {
        parse_grid_key(py, key, false)
    }
}

/// Dense grid class (dense meshgrid).
///
/// Use: `mgrid[0:5, 0:3]` returns a list of arrays with shapes
/// `[(5, 3), (5, 3)]`.
#[pyclass]
pub struct MGridClass;

#[pymethods]
impl MGridClass {
    fn __getitem__(&self, py: Python<'_>, key: &Bound<'_, pyo3::PyAny>) -> PyResult<pyo3::PyObject> {
        parse_grid_key(py, key, true)
    }
}

/// Parse a grid key (slice or tuple of slices) and return arrays.
fn parse_grid_key(py: Python<'_>, key: &Bound<'_, pyo3::PyAny>, dense: bool) -> PyResult<pyo3::PyObject> {
    // Parse slices from key
    let slices: Vec<GridSlice> = if let Ok(tuple) = key.downcast::<pyo3::types::PyTuple>() {
        tuple.iter()
            .map(|item| parse_single_slice(&item))
            .collect::<PyResult<Vec<_>>>()?
    } else {
        // Single slice
        vec![parse_single_slice(key)?]
    };

    if slices.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err("empty slice"));
    }

    // Generate arrays for each dimension
    let mut arrays: Vec<RumpyArray> = slices.iter()
        .map(|s| generate_range_array(s))
        .collect();

    // If single slice, return single array
    if arrays.len() == 1 {
        let arr = PyRumpyArray::new(arrays.remove(0));
        return Ok(pyo3::Py::new(py, arr)?.into_any().into());
    }

    let n = arrays.len();
    let shapes: Vec<usize> = arrays.iter().map(|a| a.size()).collect();

    // Build result arrays - reshape each to have 1s in other dimensions
    let mut result = Vec::with_capacity(n);
    for (i, arr) in arrays.into_iter().enumerate() {
        let mut new_shape = vec![1usize; n];
        new_shape[i] = shapes[i];
        let reshaped = arr.reshape(new_shape).expect("reshape failed");

        // Dense grid: broadcast each array to full shape
        let final_arr = if dense {
            reshaped.broadcast_to(&shapes).expect("broadcast failed")
        } else {
            reshaped
        };
        result.push(PyRumpyArray::new(final_arr));
    }

    let list = pyo3::types::PyList::new(py, result.into_iter().map(|arr| {
        pyo3::Py::new(py, arr).unwrap()
    }))?;
    Ok(list.into())
}

/// Represents a parsed slice: start:stop:step
struct GridSlice {
    start: f64,
    stop: f64,
    step: GridStep,
}

enum GridStep {
    Integer(i64),      // Regular step (e.g., 0:10:2)
    Complex(f64),      // Number of points (e.g., 0:1:5j means linspace)
}

fn parse_single_slice(item: &Bound<'_, pyo3::PyAny>) -> PyResult<GridSlice> {
    let slice = item.downcast::<pyo3::types::PySlice>()?;

    // Get start, stop
    let start: f64 = slice.getattr("start")?.extract().unwrap_or(0.0);
    let stop: f64 = slice.getattr("stop")?.extract()?;

    // Get step - could be int, float, or complex
    let step_obj = slice.getattr("step")?;
    let step = if step_obj.is_none() {
        GridStep::Integer(1)
    } else {
        // Check if it's a complex number (has .imag attribute)
        if let Ok(imag_attr) = step_obj.getattr("imag") {
            if let Ok(imag) = imag_attr.extract::<f64>() {
                if imag != 0.0 {
                    // Complex step means linspace-like behavior (use imaginary part)
                    return Ok(GridSlice { start, stop, step: GridStep::Complex(imag) });
                }
            }
        }
        // Try integer or float
        if let Ok(i) = step_obj.extract::<i64>() {
            GridStep::Integer(i)
        } else if let Ok(f) = step_obj.extract::<f64>() {
            GridStep::Integer(f as i64)
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "step must be integer or complex",
            ));
        }
    };

    Ok(GridSlice { start, stop, step })
}

fn generate_range_array(slice: &GridSlice) -> RumpyArray {
    match slice.step {
        GridStep::Integer(step) => {
            // arange-like behavior
            let mut values = Vec::new();
            let mut val = slice.start;
            if step > 0 {
                while val < slice.stop {
                    values.push(val);
                    val += step as f64;
                }
            } else if step < 0 {
                while val > slice.stop {
                    values.push(val);
                    val += step as f64;
                }
            }

            // Use int64 dtype for integer ranges, float64 otherwise
            let dtype = if slice.start.fract() == 0.0 && slice.stop.fract() == 0.0 {
                DType::int64()
            } else {
                DType::float64()
            };
            RumpyArray::from_vec(values, dtype)
        }
        GridStep::Complex(num_points) => {
            // linspace-like behavior
            let n = num_points as usize;
            if n == 0 {
                return RumpyArray::zeros(vec![0], DType::float64());
            }
            if n == 1 {
                return RumpyArray::from_vec(vec![slice.start], DType::float64());
            }

            let step = (slice.stop - slice.start) / (n - 1) as f64;
            let values: Vec<f64> = (0..n)
                .map(|i| slice.start + i as f64 * step)
                .collect();
            RumpyArray::from_vec(values, DType::float64())
        }
    }
}
