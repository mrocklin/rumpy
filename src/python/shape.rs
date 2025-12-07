// Python bindings for shape manipulation functions.

use pyo3::prelude::*;
use pyo3::types::PyList;

use crate::array::{DType, RumpyArray};
use super::{PyRumpyArray, creation, reductions::resolve_axis};

/// Convert a Python object to RumpyArray (for concatenate, etc.)
pub fn to_rumpy_array(obj: &Bound<'_, pyo3::PyAny>) -> PyResult<RumpyArray> {
    // Try PyRumpyArray first
    if let Ok(arr) = obj.extract::<pyo3::PyRef<'_, PyRumpyArray>>() {
        return Ok(arr.inner.clone());
    }

    // Try Python list
    if let Ok(list) = obj.downcast::<PyList>() {
        let arr = creation::from_list(list, None)?;
        return Ok(arr.inner);
    }

    // Try scalar
    if let Ok(val) = obj.extract::<f64>() {
        return Ok(RumpyArray::full(vec![1], val, DType::float64()));
    }

    Err(pyo3::exceptions::PyTypeError::new_err(
        "cannot convert to array"
    ))
}

// ============================================================================
// Basic shape operations
// ============================================================================

/// Give a new shape to an array.
#[pyfunction]
pub fn reshape(a: &PyRumpyArray, newshape: Vec<isize>) -> PyResult<PyRumpyArray> {
    // Handle -1 in shape (infer dimension)
    let total = a.inner.size();
    let neg_one_count = newshape.iter().filter(|&&x| x == -1).count();
    if neg_one_count > 1 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "can only specify one unknown dimension"
        ));
    }

    let shape: Vec<usize> = if neg_one_count == 1 {
        let known_product: isize = newshape.iter().filter(|&&x| x != -1).product();
        if known_product <= 0 || total % (known_product as usize) != 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "cannot reshape array of size into shape"
            ));
        }
        let inferred = (total as isize) / known_product;
        newshape.into_iter()
            .map(|x| if x == -1 { inferred as usize } else { x as usize })
            .collect()
    } else {
        newshape.into_iter().map(|x| x as usize).collect()
    };

    a.inner.reshape(shape.clone())
        .map(PyRumpyArray::new)
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("cannot reshape array of given size into shape")
        })
}

/// Return a contiguous flattened array.
#[pyfunction]
pub fn ravel(a: &PyRumpyArray) -> PyRumpyArray {
    // ravel returns a view if possible, otherwise a copy
    if let Some(arr) = a.inner.reshape(vec![a.inner.size()]) {
        PyRumpyArray::new(arr)
    } else {
        // If reshape fails, make a contiguous copy then reshape
        PyRumpyArray::new(a.inner.copy().reshape(vec![a.inner.size()]).unwrap())
    }
}

/// Return a copy of the array collapsed into one dimension.
#[pyfunction]
pub fn flatten(a: &PyRumpyArray) -> PyRumpyArray {
    // flatten always returns a copy
    PyRumpyArray::new(a.inner.copy().reshape(vec![a.inner.size()]).unwrap())
}

/// Permute the dimensions of an array.
#[pyfunction]
#[pyo3(signature = (a, axes=None))]
pub fn transpose(a: &PyRumpyArray, axes: Option<Vec<usize>>) -> PyRumpyArray {
    match axes {
        Some(ax) => PyRumpyArray::new(a.inner.transpose_axes(&ax)),
        None => PyRumpyArray::new(a.inner.transpose()),
    }
}

/// Interchange two axes of an array.
#[pyfunction]
pub fn swapaxes(x: &PyRumpyArray, axis1: usize, axis2: usize) -> PyRumpyArray {
    PyRumpyArray::new(x.inner.swapaxes(axis1, axis2))
}

/// Move axes to new positions.
#[pyfunction]
pub fn moveaxis(a: &PyRumpyArray, source: Vec<isize>, destination: Vec<isize>) -> PyResult<PyRumpyArray> {
    let ndim = a.inner.ndim();

    // Normalize axes
    let src: Vec<usize> = source.iter()
        .map(|&x| if x < 0 { (ndim as isize + x) as usize } else { x as usize })
        .collect();
    let dst: Vec<usize> = destination.iter()
        .map(|&x| if x < 0 { (ndim as isize + x) as usize } else { x as usize })
        .collect();

    if src.len() != dst.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "source and destination must have same length"
        ));
    }

    // Build the permutation
    // Remove source axes and insert at destination positions
    let mut order: Vec<usize> = (0..ndim).filter(|x| !src.contains(x)).collect();

    // Insert source axes at destination positions
    for (&s, &d) in src.iter().zip(dst.iter()) {
        order.insert(d.min(order.len()), s);
    }

    Ok(PyRumpyArray::new(a.inner.transpose_axes(&order)))
}

/// Roll the specified axis backwards.
#[pyfunction]
#[pyo3(signature = (a, axis, start=0))]
pub fn rollaxis(a: &PyRumpyArray, axis: isize, start: isize) -> PyResult<PyRumpyArray> {
    let ndim = a.inner.ndim();
    let axis = resolve_axis(axis, ndim);
    let start = if start < 0 {
        (ndim as isize + start) as usize
    } else {
        (start as usize).min(ndim)
    };

    if axis >= ndim {
        return Err(pyo3::exceptions::PyValueError::new_err("axis out of bounds"));
    }

    // Build permutation: move axis to start position
    let mut order: Vec<usize> = (0..ndim).collect();
    order.remove(axis);
    let insert_pos = if axis < start { start - 1 } else { start };
    order.insert(insert_pos, axis);

    Ok(PyRumpyArray::new(a.inner.transpose_axes(&order)))
}

// ============================================================================
// Dimension manipulation
// ============================================================================

/// Convert input to array with at least one dimension.
#[pyfunction]
pub fn atleast_1d(a: &PyRumpyArray) -> PyRumpyArray {
    if a.inner.ndim() == 0 {
        PyRumpyArray::new(a.inner.reshape(vec![1]).unwrap_or_else(|| a.inner.clone()))
    } else {
        PyRumpyArray::new(a.inner.clone())
    }
}

/// Convert input to array with at least two dimensions.
#[pyfunction]
pub fn atleast_2d(a: &PyRumpyArray) -> PyRumpyArray {
    match a.inner.ndim() {
        0 => PyRumpyArray::new(a.inner.reshape(vec![1, 1]).unwrap_or_else(|| a.inner.clone())),
        1 => PyRumpyArray::new(a.inner.reshape(vec![1, a.inner.size()]).unwrap_or_else(|| a.inner.clone())),
        _ => PyRumpyArray::new(a.inner.clone()),
    }
}

/// Convert input to array with at least three dimensions.
#[pyfunction]
pub fn atleast_3d(a: &PyRumpyArray) -> PyRumpyArray {
    match a.inner.ndim() {
        0 => PyRumpyArray::new(a.inner.reshape(vec![1, 1, 1]).unwrap_or_else(|| a.inner.clone())),
        1 => {
            let n = a.inner.size();
            PyRumpyArray::new(a.inner.reshape(vec![1, n, 1]).unwrap_or_else(|| a.inner.clone()))
        }
        2 => {
            let shape = a.inner.shape();
            PyRumpyArray::new(a.inner.reshape(vec![shape[0], shape[1], 1]).unwrap_or_else(|| a.inner.clone()))
        }
        _ => PyRumpyArray::new(a.inner.clone()),
    }
}

/// Expand array dimensions at specified axis.
#[pyfunction]
pub fn expand_dims(arr: &PyRumpyArray, axis: isize) -> PyResult<PyRumpyArray> {
    let ndim = arr.inner.ndim();
    let axis = if axis < 0 {
        (ndim as isize + axis + 1) as usize
    } else {
        axis as usize
    };

    arr.inner.expand_dims(axis)
        .map(PyRumpyArray::new)
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "axis {} is out of bounds for array of dimension {}",
                axis, ndim
            ))
        })
}

/// Remove single-dimensional entries from array shape.
#[pyfunction]
pub fn squeeze(arr: &PyRumpyArray) -> PyRumpyArray {
    PyRumpyArray::new(arr.inner.squeeze())
}

// ============================================================================
// Broadcasting
// ============================================================================

/// Broadcast an array to a new shape.
#[pyfunction]
pub fn broadcast_to(arr: &PyRumpyArray, shape: Vec<usize>) -> PyResult<PyRumpyArray> {
    arr.inner.broadcast_to(&shape)
        .map(PyRumpyArray::new)
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("cannot broadcast to shape")
        })
}

/// Broadcast arrays to a common shape.
#[pyfunction]
pub fn broadcast_arrays(arrays: &Bound<'_, PyList>) -> PyResult<Vec<PyRumpyArray>> {
    if arrays.is_empty() {
        return Ok(vec![]);
    }

    // Collect all arrays
    let mut arrs: Vec<RumpyArray> = Vec::with_capacity(arrays.len());
    for item in arrays.iter() {
        arrs.push(to_rumpy_array(&item)?);
    }

    // Find common shape
    let mut common_shape = arrs[0].shape().to_vec();
    for arr in arrs.iter().skip(1) {
        common_shape = crate::array::broadcast_shapes(&common_shape, arr.shape())
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("shapes cannot be broadcast together")
            })?;
    }

    // Broadcast each array
    let result: PyResult<Vec<PyRumpyArray>> = arrs.iter()
        .map(|arr| {
            arr.broadcast_to(&common_shape)
                .map(PyRumpyArray::new)
                .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("broadcast failed"))
        })
        .collect();

    result
}

// ============================================================================
// Flipping operations
// ============================================================================

/// Reverse the order of elements along given axis.
#[pyfunction]
#[pyo3(signature = (arr, axis=None))]
pub fn flip(arr: &PyRumpyArray, axis: Option<isize>) -> PyResult<PyRumpyArray> {
    match axis {
        None => {
            // Flip all axes
            let mut result = arr.inner.clone();
            for ax in 0..result.ndim() {
                result = result.flip(ax).unwrap();
            }
            Ok(PyRumpyArray::new(result))
        }
        Some(ax) => {
            let ndim = arr.inner.ndim();
            let axis = if ax < 0 {
                (ndim as isize + ax) as usize
            } else {
                ax as usize
            };
            arr.inner.flip(axis)
                .map(PyRumpyArray::new)
                .ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "axis {} is out of bounds for array of dimension {}",
                        axis, ndim
                    ))
                })
        }
    }
}

/// Flip array vertically (axis=0).
#[pyfunction]
pub fn flipud(arr: &PyRumpyArray) -> PyResult<PyRumpyArray> {
    if arr.inner.ndim() < 1 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "flipud requires array with at least 1 dimension"
        ));
    }
    Ok(PyRumpyArray::new(arr.inner.flip(0).unwrap()))
}

/// Flip array horizontally (axis=1).
#[pyfunction]
pub fn fliplr(arr: &PyRumpyArray) -> PyResult<PyRumpyArray> {
    if arr.inner.ndim() < 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "fliplr requires array with at least 2 dimensions"
        ));
    }
    Ok(PyRumpyArray::new(arr.inner.flip(1).unwrap()))
}

// ============================================================================
// Concatenation and stacking
// ============================================================================

/// Concatenate arrays along an axis.
#[pyfunction]
#[pyo3(signature = (arrays, axis=0))]
pub fn concatenate(arrays: &Bound<'_, PyList>, axis: usize) -> PyResult<PyRumpyArray> {
    let mut inner_arrays: Vec<RumpyArray> = Vec::with_capacity(arrays.len());
    for item in arrays.iter() {
        inner_arrays.push(to_rumpy_array(&item)?);
    }

    crate::array::concatenate(&inner_arrays, axis)
        .map(PyRumpyArray::new)
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "arrays must have same shape except in concatenation axis"
            )
        })
}

/// Stack arrays along a new axis.
#[pyfunction]
#[pyo3(signature = (arrays, axis=0))]
pub fn stack(arrays: &Bound<'_, PyList>, axis: usize) -> PyResult<PyRumpyArray> {
    let mut inner_arrays: Vec<RumpyArray> = Vec::with_capacity(arrays.len());
    for item in arrays.iter() {
        inner_arrays.push(to_rumpy_array(&item)?);
    }
    crate::array::stack(&inner_arrays, axis)
        .map(PyRumpyArray::new)
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("arrays must have same shape for stack")
        })
}

/// Stack arrays vertically (row-wise).
#[pyfunction]
pub fn vstack(arrays: &Bound<'_, PyList>) -> PyResult<PyRumpyArray> {
    if arrays.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err("need at least one array"));
    }

    // For 1D arrays, reshape to (1, N) first
    let mut inner_arrays: Vec<RumpyArray> = Vec::with_capacity(arrays.len());
    for item in arrays.iter() {
        let arr = to_rumpy_array(&item)?;
        if arr.ndim() == 1 {
            inner_arrays.push(arr.reshape(vec![1, arr.size()]).unwrap_or_else(|| arr.clone()));
        } else {
            inner_arrays.push(arr);
        }
    }

    crate::array::concatenate(&inner_arrays, 0)
        .map(PyRumpyArray::new)
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("arrays must have same shape for vstack")
        })
}

/// Stack arrays horizontally (column-wise).
#[pyfunction]
pub fn hstack(arrays: &Bound<'_, PyList>) -> PyResult<PyRumpyArray> {
    if arrays.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err("need at least one array"));
    }

    let mut inner_arrays: Vec<RumpyArray> = Vec::with_capacity(arrays.len());
    for item in arrays.iter() {
        inner_arrays.push(to_rumpy_array(&item)?);
    }

    let first = &inner_arrays[0];
    let axis = if first.ndim() == 1 { 0 } else { 1 };

    crate::array::concatenate(&inner_arrays, axis)
        .map(PyRumpyArray::new)
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("arrays must have same shape for hstack")
        })
}

/// Stack 1D arrays as columns into a 2D array.
#[pyfunction]
pub fn column_stack(arrays: &Bound<'_, PyList>) -> PyResult<PyRumpyArray> {
    if arrays.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err("need at least one array"));
    }

    let mut inner_arrays: Vec<RumpyArray> = Vec::with_capacity(arrays.len());
    for item in arrays.iter() {
        let arr = to_rumpy_array(&item)?;
        // If 1D, reshape to column vector
        let arr = if arr.ndim() == 1 {
            arr.reshape(vec![arr.size(), 1]).unwrap_or(arr)
        } else {
            arr
        };
        inner_arrays.push(arr);
    }

    crate::array::concatenate(&inner_arrays, 1)
        .map(PyRumpyArray::new)
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("arrays must have compatible shapes for column_stack")
        })
}

/// Stack arrays row-wise (alias for vstack).
/// Deprecated: Use vstack instead.
#[pyfunction]
pub fn row_stack(py: Python<'_>, arrays: &Bound<'_, PyList>) -> PyResult<PyRumpyArray> {
    let warnings = py.import("warnings")?;
    warnings.call_method1(
        "warn",
        (
            "`row_stack` is deprecated. Use `vstack` instead.",
            py.get_type::<pyo3::exceptions::PyDeprecationWarning>(),
        ),
    )?;
    vstack(arrays)
}

/// Stack arrays depth-wise (along axis 2).
#[pyfunction]
pub fn dstack(arrays: &Bound<'_, PyList>) -> PyResult<PyRumpyArray> {
    if arrays.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err("need at least one array"));
    }

    let mut inner_arrays: Vec<RumpyArray> = Vec::with_capacity(arrays.len());
    for item in arrays.iter() {
        let arr = to_rumpy_array(&item)?;
        // Promote to at least 3D
        let arr = match arr.ndim() {
            1 => {
                let n = arr.size();
                arr.reshape(vec![1, n, 1]).unwrap_or(arr)
            }
            2 => {
                let shape = arr.shape().to_vec();
                arr.reshape(vec![shape[0], shape[1], 1]).unwrap_or(arr)
            }
            _ => arr,
        };
        inner_arrays.push(arr);
    }

    crate::array::concatenate(&inner_arrays, 2)
        .map(PyRumpyArray::new)
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("arrays must have compatible shapes for dstack")
        })
}

// ============================================================================
// Splitting
// ============================================================================

/// Split array into equal parts.
#[pyfunction]
#[pyo3(signature = (arr, num_sections, axis=0))]
pub fn split(arr: &PyRumpyArray, num_sections: usize, axis: usize) -> PyResult<Vec<PyRumpyArray>> {
    crate::array::split(&arr.inner, num_sections, axis)
        .map(|sections| sections.into_iter().map(PyRumpyArray::new).collect())
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "array split does not result in an equal division"
            )
        })
}

/// Split array into sections, allowing unequal sizes.
#[pyfunction]
#[pyo3(signature = (arr, num_sections, axis=0))]
pub fn array_split(arr: &PyRumpyArray, num_sections: usize, axis: usize) -> PyResult<Vec<PyRumpyArray>> {
    crate::array::array_split(&arr.inner, num_sections, axis)
        .map(|sections| sections.into_iter().map(PyRumpyArray::new).collect())
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("invalid split parameters")
        })
}

/// Split array horizontally (column-wise) along axis 1.
#[pyfunction]
pub fn hsplit(arr: &PyRumpyArray, num_sections: usize) -> PyResult<Vec<PyRumpyArray>> {
    let axis = if arr.inner.ndim() == 1 { 0 } else { 1 };
    crate::array::split(&arr.inner, num_sections, axis)
        .map(|sections| sections.into_iter().map(PyRumpyArray::new).collect())
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("array split does not result in an equal division")
        })
}

/// Split array vertically (row-wise) along axis 0.
#[pyfunction]
pub fn vsplit(arr: &PyRumpyArray, num_sections: usize) -> PyResult<Vec<PyRumpyArray>> {
    if arr.inner.ndim() < 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "vsplit requires array with at least 2 dimensions"
        ));
    }
    crate::array::split(&arr.inner, num_sections, 0)
        .map(|sections| sections.into_iter().map(PyRumpyArray::new).collect())
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("array split does not result in an equal division")
        })
}

/// Split array depth-wise (along axis 2).
#[pyfunction]
pub fn dsplit(arr: &PyRumpyArray, num_sections: usize) -> PyResult<Vec<PyRumpyArray>> {
    if arr.inner.ndim() < 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "dsplit requires array with at least 3 dimensions"
        ));
    }
    crate::array::split(&arr.inner, num_sections, 2)
        .map(|sections| sections.into_iter().map(PyRumpyArray::new).collect())
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("array split does not result in an equal division")
        })
}

// ============================================================================
// Repetition and tiling
// ============================================================================

/// Repeat elements of an array.
#[pyfunction]
#[pyo3(signature = (arr, repeats, axis=None))]
pub fn repeat(arr: &PyRumpyArray, repeats: usize, axis: Option<isize>) -> PyResult<PyRumpyArray> {
    crate::array::repeat(&arr.inner, repeats, axis)
        .map(PyRumpyArray::new)
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("invalid axis for repeat")
        })
}

/// Construct array by repeating input the given number of times.
#[pyfunction]
pub fn tile(arr: &PyRumpyArray, reps: &Bound<'_, pyo3::PyAny>) -> PyResult<PyRumpyArray> {
    let reps_vec: Vec<usize> = if let Ok(n) = reps.extract::<usize>() {
        vec![n]
    } else if let Ok(tuple) = reps.downcast::<pyo3::types::PyTuple>() {
        tuple.iter().map(|x| x.extract::<usize>()).collect::<PyResult<Vec<_>>>()?
    } else if let Ok(list) = reps.downcast::<pyo3::types::PyList>() {
        list.iter().map(|x| x.extract::<usize>()).collect::<PyResult<Vec<_>>>()?
    } else {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "reps must be an integer or sequence of integers"
        ));
    };

    Ok(PyRumpyArray::new(crate::array::tile(&arr.inner, &reps_vec)))
}

// ============================================================================
// Modification
// ============================================================================

/// Append values to the end of an array.
#[pyfunction]
#[pyo3(signature = (arr, values, axis=None))]
pub fn append(arr: &PyRumpyArray, values: &PyRumpyArray, axis: Option<isize>) -> PyResult<PyRumpyArray> {
    crate::array::append(&arr.inner, &values.inner, axis)
        .map(PyRumpyArray::new)
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("incompatible shapes for append")
        })
}

/// Insert values into an array at specified index.
#[pyfunction]
#[pyo3(signature = (arr, index, values, axis=None))]
pub fn insert(arr: &PyRumpyArray, index: isize, values: &Bound<'_, pyo3::PyAny>, axis: Option<isize>) -> PyResult<PyRumpyArray> {
    // Parse values - can be scalar or array
    let values_arr = if let Ok(scalar) = values.extract::<f64>() {
        RumpyArray::full(vec![1], scalar, arr.inner.dtype())
    } else if let Ok(arr_ref) = values.extract::<pyo3::PyRef<'_, PyRumpyArray>>() {
        arr_ref.inner.clone()
    } else if let Ok(list) = values.downcast::<pyo3::types::PyList>() {
        let data: Vec<f64> = list.iter().map(|x| x.extract::<f64>()).collect::<PyResult<Vec<_>>>()?;
        RumpyArray::from_vec(data, arr.inner.dtype())
    } else {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "values must be scalar, array, or list"
        ));
    };

    crate::array::insert(&arr.inner, index, &values_arr, axis)
        .map(PyRumpyArray::new)
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("invalid parameters for insert")
        })
}

/// Delete elements from an array at specified index.
#[pyfunction]
#[pyo3(signature = (arr, index, axis=None))]
pub fn delete(arr: &PyRumpyArray, index: isize, axis: Option<isize>) -> PyResult<PyRumpyArray> {
    crate::array::delete(&arr.inner, index, axis)
        .map(PyRumpyArray::new)
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("invalid parameters for delete")
        })
}

/// Pad an array.
#[pyfunction]
#[pyo3(signature = (arr, pad_width, mode="constant", constant_values=0.0))]
pub fn pad(
    arr: &PyRumpyArray,
    pad_width: &Bound<'_, pyo3::PyAny>,
    mode: &str,
    constant_values: f64,
) -> PyResult<PyRumpyArray> {
    let pad_spec: Vec<(usize, usize)> = if let Ok(n) = pad_width.extract::<usize>() {
        vec![(n, n); arr.inner.ndim()]
    } else if let Ok(tuple) = pad_width.downcast::<pyo3::types::PyTuple>() {
        if tuple.len() == 2 {
            let before = tuple.get_item(0)?.extract::<usize>()?;
            let after = tuple.get_item(1)?.extract::<usize>()?;
            vec![(before, after); arr.inner.ndim()]
        } else {
            tuple.iter().map(|item| {
                if let Ok(n) = item.extract::<usize>() {
                    Ok((n, n))
                } else if let Ok(inner) = item.downcast::<pyo3::types::PyTuple>() {
                    let before = inner.get_item(0)?.extract::<usize>()?;
                    let after = inner.get_item(1)?.extract::<usize>()?;
                    Ok((before, after))
                } else {
                    Err(pyo3::exceptions::PyTypeError::new_err("invalid pad_width format"))
                }
            }).collect::<PyResult<Vec<_>>>()?
        }
    } else {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "pad_width must be int or tuple"
        ));
    };

    crate::array::pad(&arr.inner, &pad_spec, mode, constant_values)
        .map(PyRumpyArray::new)
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("unsupported pad mode")
        })
}

/// Roll array elements along a given axis.
#[pyfunction]
#[pyo3(signature = (arr, shift, axis=None))]
pub fn roll(arr: &PyRumpyArray, shift: isize, axis: Option<isize>) -> PyResult<PyRumpyArray> {
    Ok(PyRumpyArray::new(crate::array::roll(&arr.inner, shift, axis)))
}

/// Rotate an array by 90 degrees in the plane specified by axes.
#[pyfunction]
#[pyo3(signature = (arr, k=1, axes=(0, 1)))]
pub fn rot90(arr: &PyRumpyArray, k: isize, axes: (isize, isize)) -> PyResult<PyRumpyArray> {
    let ndim = arr.inner.ndim();
    if ndim < 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "rot90 requires array with at least 2 dimensions"
        ));
    }

    let ax0 = if axes.0 < 0 { (ndim as isize + axes.0) as usize } else { axes.0 as usize };
    let ax1 = if axes.1 < 0 { (ndim as isize + axes.1) as usize } else { axes.1 as usize };

    if ax0 >= ndim || ax1 >= ndim {
        return Err(pyo3::exceptions::PyValueError::new_err("axes out of range"));
    }
    if ax0 == ax1 {
        return Err(pyo3::exceptions::PyValueError::new_err("axes must be different"));
    }

    Ok(PyRumpyArray::new(crate::array::rot90(&arr.inner, k, ax0, ax1)))
}

// ============================================================================
// Misc
// ============================================================================

/// Return unique sorted values.
#[pyfunction]
pub fn unique(arr: &PyRumpyArray) -> PyRumpyArray {
    PyRumpyArray::new(arr.inner.unique())
}

/// Return indices of non-zero elements.
#[pyfunction]
pub fn nonzero(arr: &PyRumpyArray) -> Vec<PyRumpyArray> {
    arr.inner.nonzero().into_iter().map(PyRumpyArray::new).collect()
}

/// Extract diagonal.
#[pyfunction]
pub fn diagonal(x: &PyRumpyArray) -> PyRumpyArray {
    PyRumpyArray::new(x.inner.diagonal())
}
