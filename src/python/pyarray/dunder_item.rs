//! __getitem__ and __setitem__ implementations for PyRumpyArray.

use pyo3::prelude::*;
use pyo3::types::{PyEllipsis, PyList, PySlice, PyTuple};

use super::{
    expand_ellipsis, extract_slice_indices, fill_view, normalize_index, normalize_slice,
    IndexItem, PyRumpyArray,
};
use crate::array::RumpyArray;

#[pymethods]
impl PyRumpyArray {
    /// Indexing and slicing.
    fn __getitem__<'py>(&self, py: Python<'py>, key: &Bound<'py, PyAny>) -> PyResult<PyObject> {
        // Handle array indexing (boolean or integer)
        if let Ok(idx_arr) = key.extract::<PyRef<'_, PyRumpyArray>>() {
            if idx_arr.inner.dtype().kind() == crate::array::dtype::DTypeKind::Bool {
                // Boolean indexing
                return self
                    .inner
                    .select_by_mask(&idx_arr.inner)
                    .map(|arr| {
                        Self::new(arr)
                            .into_pyobject(py)
                            .unwrap()
                            .into_any()
                            .unbind()
                    })
                    .ok_or_else(|| {
                        pyo3::exceptions::PyIndexError::new_err(
                            "boolean index shape must match array shape",
                        )
                    });
            } else {
                // Fancy indexing (integer array)
                return self
                    .inner
                    .select_by_indices(&idx_arr.inner)
                    .map(|arr| {
                        Self::new(arr)
                            .into_pyobject(py)
                            .unwrap()
                            .into_any()
                            .unbind()
                    })
                    .ok_or_else(|| pyo3::exceptions::PyIndexError::new_err("index out of bounds"));
            }
        }

        // Handle Python list indexing (convert to array for fancy indexing)
        if let Ok(list) = key.downcast::<PyList>() {
            let indices: Vec<i64> = list
                .iter()
                .map(|x| x.extract::<i64>())
                .collect::<PyResult<Vec<_>>>()?;
            let idx_arr = RumpyArray::from_slice_i64(&indices);
            return self
                .inner
                .select_by_indices(&idx_arr)
                .map(|arr| {
                    Self::new(arr)
                        .into_pyobject(py)
                        .unwrap()
                        .into_any()
                        .unbind()
                })
                .ok_or_else(|| pyo3::exceptions::PyIndexError::new_err("index out of bounds"));
        }

        // Handle single None (newaxis)
        if key.is_none() {
            let result = self.inner.expand_dims(0).ok_or_else(|| {
                pyo3::exceptions::PyIndexError::new_err("cannot expand dims")
            })?;
            return Ok(Self::new(result).into_pyobject(py)?.into_any().unbind());
        }

        // Handle single ellipsis
        if key.downcast::<PyEllipsis>().is_ok() {
            return Ok(Self::new(self.inner.clone())
                .into_pyobject(py)?
                .into_any()
                .unbind());
        }

        // Handle tuple for multi-dimensional indexing
        if let Ok(tuple) = key.downcast::<PyTuple>() {
            // Pre-process: expand ellipsis and parse all items
            let items = expand_ellipsis(tuple, self.inner.ndim())?;

            // Fast path: all integers and matches ndim -> scalar access
            if items.len() == self.inner.ndim()
                && items.iter().all(|i| matches!(i, IndexItem::Int(_)))
            {
                let indices: Vec<usize> = items
                    .iter()
                    .enumerate()
                    .map(|(axis, item)| {
                        if let IndexItem::Int(idx) = item {
                            normalize_index(*idx, self.inner.shape()[axis])
                        } else {
                            unreachable!()
                        }
                    })
                    .collect();
                let val = self.inner.get_element(&indices);
                return Ok(val.into_pyobject(py)?.into_any().unbind());
            }

            // Process indices, tracking which axes had integer indices (to squeeze)
            let mut result = self.inner.clone();
            let mut axis = 0usize;
            let mut squeeze_axes: Vec<usize> = Vec::new();

            for item in items.iter() {
                match item {
                    IndexItem::NewAxis => {
                        result = result.expand_dims(axis).ok_or_else(|| {
                            pyo3::exceptions::PyIndexError::new_err("cannot expand dims")
                        })?;
                        axis += 1;
                    }
                    IndexItem::Int(idx) => {
                        if axis >= result.ndim() {
                            return Err(pyo3::exceptions::PyIndexError::new_err(
                                "too many indices for array",
                            ));
                        }
                        let idx = normalize_index(*idx, result.shape()[axis]);
                        result = result.slice_axis(axis, idx as isize, idx as isize + 1, 1);
                        squeeze_axes.push(axis);
                        axis += 1;
                    }
                    IndexItem::Slice(start, stop, step) => {
                        if axis >= result.ndim() {
                            return Err(pyo3::exceptions::PyIndexError::new_err(
                                "too many indices for array",
                            ));
                        }
                        let len = result.shape()[axis] as isize;
                        let (norm_start, norm_stop) = normalize_slice(*start, *stop, *step, len);
                        result = result.slice_axis(axis, norm_start, norm_stop, *step);
                        axis += 1;
                    }
                }
            }

            // Squeeze out axes that had integer indices (in reverse order to preserve indices)
            for &ax in squeeze_axes.iter().rev() {
                result = result.squeeze_axis(ax);
            }

            return Ok(Self::new(result).into_pyobject(py)?.into_any().unbind());
        }

        // Handle single integer index
        if let Ok(idx) = key.extract::<isize>() {
            let idx = normalize_index(idx, self.inner.shape()[0]);
            let result = self.inner.slice_axis(0, idx as isize, idx as isize + 1, 1);
            let result = result.squeeze_axis(0);
            return Ok(Self::new(result).into_pyobject(py)?.into_any().unbind());
        }

        // Handle single slice
        if let Ok(slice) = key.downcast::<PySlice>() {
            let (start, stop, step) = extract_slice_indices(slice, self.inner.shape()[0])?;
            let result = self.inner.slice_axis(0, start, stop, step);
            return Ok(Self::new(result).into_pyobject(py)?.into_any().unbind());
        }

        Err(pyo3::exceptions::PyTypeError::new_err(
            "indices must be integers, slices, or None",
        ))
    }

    /// Item assignment (slice or element).
    fn __setitem__(&mut self, key: &Bound<'_, PyAny>, value: &Bound<'_, PyAny>) -> PyResult<()> {
        // Extract value as scalar or array
        let value_scalar = value.extract::<f64>().ok();
        let value_array = value.extract::<PyRef<'_, PyRumpyArray>>().ok();

        // Handle tuple for multi-dimensional indexing
        if let Ok(tuple) = key.downcast::<PyTuple>() {
            // Check if all indices are integers (single element assignment)
            let all_ints = tuple.iter().all(|item| item.extract::<isize>().is_ok());

            if all_ints && tuple.len() == self.inner.ndim() {
                // Single element assignment
                let indices: Vec<usize> = tuple
                    .iter()
                    .enumerate()
                    .map(|(axis, item)| {
                        let idx: isize = item.extract().unwrap();
                        normalize_index(idx, self.inner.shape()[axis])
                    })
                    .collect();

                if let Some(scalar) = value_scalar {
                    self.inner.set_element(&indices, scalar);
                    return Ok(());
                } else {
                    return Err(pyo3::exceptions::PyTypeError::new_err(
                        "can only assign scalar to single element",
                    ));
                }
            }

            // Slice assignment - get view and fill
            let mut view = self.inner.clone();
            let mut axis = 0usize;

            for item in tuple.iter() {
                if axis >= view.ndim() {
                    return Err(pyo3::exceptions::PyIndexError::new_err(
                        "too many indices for array",
                    ));
                }
                if let Ok(idx) = item.extract::<isize>() {
                    let idx = normalize_index(idx, view.shape()[axis]);
                    view = view.slice_axis(axis, idx as isize, idx as isize + 1, 1);
                    axis += 1;
                } else if let Ok(slice) = item.downcast::<PySlice>() {
                    let (start, stop, step) = extract_slice_indices(slice, view.shape()[axis])?;
                    view = view.slice_axis(axis, start, stop, step);
                    axis += 1;
                } else {
                    return Err(pyo3::exceptions::PyTypeError::new_err(
                        "indices must be integers or slices",
                    ));
                }
            }

            return fill_view(&mut view, value_scalar, value_array.as_deref());
        }

        // Handle single integer index (row assignment for 2D+, or single element for 1D)
        if let Ok(idx) = key.extract::<isize>() {
            let idx = normalize_index(idx, self.inner.shape()[0]);

            if self.inner.ndim() == 1 {
                // Single element in 1D array
                if let Some(scalar) = value_scalar {
                    self.inner.set_element(&[idx], scalar);
                    return Ok(());
                } else {
                    return Err(pyo3::exceptions::PyTypeError::new_err(
                        "can only assign scalar to single element",
                    ));
                }
            } else {
                // Row/slice assignment
                let mut view = self.inner.slice_axis(0, idx as isize, idx as isize + 1, 1);
                return fill_view(&mut view, value_scalar, value_array.as_deref());
            }
        }

        // Handle single slice
        if let Ok(slice) = key.downcast::<PySlice>() {
            let (start, stop, step) = extract_slice_indices(slice, self.inner.shape()[0])?;
            let mut view = self.inner.slice_axis(0, start, stop, step);
            return fill_view(&mut view, value_scalar, value_array.as_deref());
        }

        Err(pyo3::exceptions::PyTypeError::new_err(
            "indices must be integers or slices",
        ))
    }
}
