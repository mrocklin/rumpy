// Python bindings for functional programming operations.
// apply_along_axis, apply_over_axes, vectorize, frompyfunc

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::array::{increment_indices, DType, RumpyArray};
use super::{PyRumpyArray, creation, parse_dtype};

/// Helper to compute indices from flat index
fn flat_to_indices(flat_idx: usize, shape: &[usize]) -> Vec<usize> {
    let mut indices = vec![0usize; shape.len()];
    let mut remaining = flat_idx;
    for i in (0..shape.len()).rev() {
        indices[i] = remaining % shape[i];
        remaining /= shape[i];
    }
    indices
}

/// Apply a function along a given axis.
///
/// Execute `func1d(a, *args, **kwargs)` where `func1d` operates on 1-D arrays
/// and `a` is a 1-D slice of `arr` along `axis`.
#[pyfunction]
#[pyo3(signature = (func1d, axis, arr, *args, **kwargs))]
pub fn apply_along_axis(
    py: Python<'_>,
    func1d: &Bound<'_, PyAny>,
    axis: isize,
    arr: &PyRumpyArray,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
) -> PyResult<PyRumpyArray> {
    let arr = &arr.inner;
    let ndim = arr.ndim();

    // Handle negative axis
    let axis = if axis < 0 {
        (ndim as isize + axis) as usize
    } else {
        axis as usize
    };

    if axis >= ndim {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "axis {} is out of bounds for array of dimension {}",
            axis, ndim
        )));
    }

    let shape = arr.shape();
    let axis_len = shape[axis];

    // Build output shape (all dims except axis)
    let out_shape: Vec<usize> = shape.iter().enumerate()
        .filter(|(i, _)| *i != axis)
        .map(|(_, &s)| s)
        .collect();

    let out_size: usize = out_shape.iter().product();

    if out_size == 0 {
        // Handle empty output
        return Ok(PyRumpyArray::new(RumpyArray::zeros(out_shape, arr.dtype())));
    }

    // Extract 1D slices and apply function
    let mut results: Vec<PyObject> = Vec::with_capacity(out_size);
    let mut out_indices = vec![0usize; out_shape.len()];

    for _ in 0..out_size {
        // Extract 1D slice along axis using efficient stride-based access
        let slice_data = arr.get_slice_along_axis(axis, &out_indices);

        // Create rumpy array from slice
        let slice_arr = RumpyArray::from_vec(slice_data, arr.dtype());
        let py_slice = PyRumpyArray::new(slice_arr);

        // Build args tuple: (slice, *args)
        let call_args = if args.is_empty() {
            PyTuple::new(py, [py_slice.into_pyobject(py)?.into_any()])?
        } else {
            let mut all_args: Vec<Bound<'_, PyAny>> = vec![py_slice.into_pyobject(py)?.into_any()];
            for arg in args.iter() {
                all_args.push(arg.clone());
            }
            PyTuple::new(py, all_args)?
        };

        // Call function
        let result = func1d.call(&call_args, kwargs)?;
        results.push(result.unbind());

        increment_indices(&mut out_indices, &out_shape);
    }

    // Check first result to determine output shape
    let first = results[0].bind(py);

    // If result is scalar, output shape is out_shape
    // If result is array, output shape is out_shape + result_shape
    if let Ok(scalar) = first.extract::<f64>() {
        // Scalar output - build array from scalars
        let mut out_data: Vec<f64> = Vec::with_capacity(out_size);
        out_data.push(scalar);
        for r in results.iter().skip(1) {
            out_data.push(r.bind(py).extract::<f64>()?);
        }
        let out = RumpyArray::from_vec(out_data, arr.dtype());
        let out = out.reshape(out_shape).ok_or_else(||
            pyo3::exceptions::PyValueError::new_err("reshape failed")
        )?;
        Ok(PyRumpyArray::new(out))
    } else if let Ok(arr_result) = first.extract::<PyRef<'_, PyRumpyArray>>() {
        // Rumpy array output
        let result_shape = arr_result.inner.shape().to_vec();
        let result_size: usize = result_shape.iter().product();

        // Final shape is out_shape + result_shape
        let mut final_shape = out_shape.clone();
        final_shape.extend(&result_shape);

        let mut out_data: Vec<f64> = Vec::with_capacity(out_size * result_size);
        for r in results.iter() {
            let arr_r: PyRef<'_, PyRumpyArray> = r.bind(py).extract()?;
            out_data.extend(arr_r.inner.to_vec());
        }

        let out = RumpyArray::from_vec(out_data, arr.dtype());
        let out = out.reshape(final_shape).ok_or_else(||
            pyo3::exceptions::PyValueError::new_err("reshape failed")
        )?;
        Ok(PyRumpyArray::new(out))
    } else {
        // Try to convert through asarray (handles numpy arrays, lists, etc.)
        let first_arr = creation::asarray(py, &first, None)?;
        let result_shape = first_arr.inner.shape().to_vec();
        let result_size: usize = result_shape.iter().product();

        // Final shape is out_shape + result_shape
        let mut final_shape = out_shape.clone();
        final_shape.extend(&result_shape);

        let mut out_data: Vec<f64> = Vec::with_capacity(out_size * result_size);
        out_data.extend(first_arr.inner.to_vec());

        for r in results.iter().skip(1) {
            let arr_r = creation::asarray(py, r.bind(py), None)?;
            out_data.extend(arr_r.inner.to_vec());
        }

        let out = RumpyArray::from_vec(out_data, arr.dtype());
        let out = out.reshape(final_shape).ok_or_else(||
            pyo3::exceptions::PyValueError::new_err("reshape failed")
        )?;
        Ok(PyRumpyArray::new(out))
    }
}

/// Apply a function repeatedly over multiple axes.
///
/// `func` is called as `func(a, axis)` where axis is each of the axes
/// in turn.
#[pyfunction]
#[pyo3(signature = (func, a, axes))]
pub fn apply_over_axes(
    py: Python<'_>,
    func: &Bound<'_, PyAny>,
    a: &PyRumpyArray,
    axes: Vec<isize>,
) -> PyResult<PyRumpyArray> {
    let mut current = PyRumpyArray::new(a.inner.clone());
    let ndim = a.inner.ndim();

    for ax in axes {
        // Normalize axis
        let axis = if ax < 0 {
            (ndim as isize + ax) as usize
        } else {
            ax as usize
        };

        if axis >= ndim {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "axis {} is out of bounds for array of dimension {}",
                ax, ndim
            )));
        }

        // Call func(current, axis) - func should reduce and keep dims
        let args = PyTuple::new(py, [
            current.into_pyobject(py)?.into_any(),
            axis.into_pyobject(py)?.into_any(),
        ])?;

        let result = func.call1(&args)?;

        // Result should have same ndim (dims kept)
        if let Ok(arr_result) = result.extract::<PyRef<'_, PyRumpyArray>>() {
            let result_ndim = arr_result.inner.ndim();
            if result_ndim != ndim {
                // Try to expand dims to match
                let mut new_shape = arr_result.inner.shape().to_vec();
                while new_shape.len() < ndim {
                    // Insert 1 at the reduced axis position
                    if axis < new_shape.len() {
                        new_shape.insert(axis, 1);
                    } else {
                        new_shape.push(1);
                    }
                }
                let reshaped = arr_result.inner.reshape(new_shape).ok_or_else(||
                    pyo3::exceptions::PyValueError::new_err("reshape failed")
                )?;
                current = PyRumpyArray::new(reshaped);
            } else {
                current = PyRumpyArray::new(arr_result.inner.clone());
            }
        } else {
            // Try to convert from numpy via asarray
            current = creation::asarray(py, &result, None)?;

            // Ensure dims are kept
            let result_ndim = current.inner.ndim();
            if result_ndim != ndim {
                let mut new_shape = current.inner.shape().to_vec();
                while new_shape.len() < ndim {
                    if axis < new_shape.len() {
                        new_shape.insert(axis, 1);
                    } else {
                        new_shape.push(1);
                    }
                }
                let reshaped = current.inner.reshape(new_shape).ok_or_else(||
                    pyo3::exceptions::PyValueError::new_err("reshape failed")
                )?;
                current = PyRumpyArray::new(reshaped);
            }
        }
    }

    Ok(current)
}

/// Vectorized function class.
///
/// Takes a scalar function and makes it work on arrays element-wise.
#[pyclass(name = "vectorize")]
pub struct PyVectorize {
    pyfunc: PyObject,
    otypes: Option<Vec<DType>>,
    #[allow(dead_code)]
    excluded: Vec<String>,
    signature: Option<String>,
}

#[pymethods]
impl PyVectorize {
    #[new]
    #[pyo3(signature = (pyfunc, otypes=None, excluded=None, signature=None))]
    fn new(
        _py: Python<'_>,
        pyfunc: &Bound<'_, PyAny>,
        otypes: Option<&Bound<'_, PyAny>>,
        excluded: Option<&Bound<'_, PyAny>>,
        signature: Option<String>,
    ) -> PyResult<Self> {
        let otypes = if let Some(otypes_obj) = otypes {
            if let Ok(list) = otypes_obj.downcast::<PyList>() {
                let mut dtypes = Vec::new();
                for item in list.iter() {
                    let dtype_str = if let Ok(s) = item.extract::<String>() {
                        s
                    } else {
                        // Try to get dtype from numpy dtype object
                        item.getattr("name")?.extract::<String>()?
                    };
                    dtypes.push(parse_dtype(&dtype_str)?);
                }
                Some(dtypes)
            } else {
                None
            }
        } else {
            None
        };

        let excluded = if let Some(excl) = excluded {
            if let Ok(list) = excl.downcast::<PyList>() {
                list.iter().filter_map(|item| item.extract::<String>().ok()).collect()
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };

        Ok(PyVectorize {
            pyfunc: pyfunc.clone().unbind(),
            otypes,
            excluded,
            signature,
        })
    }

    #[pyo3(signature = (*args, **kwargs))]
    fn __call__(
        &self,
        py: Python<'_>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyObject> {
        // If signature is provided, use generalized ufunc logic
        if self.signature.is_some() {
            return self.call_with_signature(py, args, kwargs);
        }

        // Get first array argument to determine shape
        let mut arrays: Vec<PyRumpyArray> = Vec::new();
        let mut scalars: Vec<(usize, PyObject)> = Vec::new();

        for (i, arg) in args.iter().enumerate() {
            if let Ok(arr) = arg.extract::<PyRef<'_, PyRumpyArray>>() {
                arrays.push(PyRumpyArray::new(arr.inner.clone()));
            } else if let Ok(arr) = creation::asarray(py, &arg, None) {
                // Try to convert to array
                if arr.inner.ndim() == 0 {
                    // Scalar
                    scalars.push((i, arg.unbind()));
                } else {
                    arrays.push(arr);
                }
            } else {
                scalars.push((i, arg.unbind()));
            }
        }

        if arrays.is_empty() {
            // All scalars - just call function directly
            return self.pyfunc.bind(py).call(&args, kwargs).map(|r| r.unbind());
        }

        // Broadcast all arrays to common shape
        let shapes: Vec<Vec<usize>> = arrays.iter().map(|a| a.inner.shape().to_vec()).collect();
        let broadcast_shape = broadcast_shapes(&shapes)?;
        let size: usize = broadcast_shape.iter().product();

        // Broadcast arrays
        let broadcasted: Vec<RumpyArray> = arrays
            .iter()
            .map(|a| a.inner.broadcast_to(&broadcast_shape))
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("broadcast failed"))?;

        // Apply function element-wise
        let mut results: Vec<PyObject> = Vec::with_capacity(size);

        for i in 0..size {
            // Build args for this element
            let mut call_args: Vec<PyObject> = Vec::with_capacity(args.len());
            let mut arr_idx = 0;
            let mut scalar_idx = 0;

            // Compute indices from flat index
            let indices = flat_to_indices(i, &broadcast_shape);

            for j in 0..args.len() {
                // Check if this position is a scalar
                if scalar_idx < scalars.len() && scalars[scalar_idx].0 == j {
                    call_args.push(scalars[scalar_idx].1.clone_ref(py));
                    scalar_idx += 1;
                } else if arr_idx < broadcasted.len() {
                    // Get element from broadcasted array
                    let val = broadcasted[arr_idx].get_element(&indices);
                    call_args.push(val.into_pyobject(py)?.unbind().into_any());
                    arr_idx += 1;
                }
            }

            let call_tuple = PyTuple::new(py, call_args)?;
            let result = self.pyfunc.bind(py).call(&call_tuple, kwargs)?;
            results.push(result.unbind());
        }

        // Build output array
        let dtype = if let Some(ref otypes) = self.otypes {
            otypes[0].clone()
        } else {
            // Infer from first result
            DType::float64()
        };

        let mut out_data: Vec<f64> = Vec::with_capacity(size);
        for r in results {
            out_data.push(r.bind(py).extract::<f64>()?);
        }

        let out = RumpyArray::from_vec(out_data, dtype);
        let out = out.reshape(broadcast_shape).ok_or_else(||
            pyo3::exceptions::PyValueError::new_err("reshape failed")
        )?;
        Ok(PyRumpyArray::new(out).into_pyobject(py)?.unbind().into_any())
    }
}

impl PyVectorize {
    fn call_with_signature(
        &self,
        py: Python<'_>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyObject> {
        // Parse signature like "(n),(n)->()"
        let sig = self.signature.as_ref().unwrap();
        let parts: Vec<&str> = sig.split("->").collect();
        if parts.len() != 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Invalid signature format"
            ));
        }

        let input_sig = parts[0].trim();
        let output_sig = parts[1].trim();

        // Parse input dimensions
        let input_dims: Vec<&str> = input_sig
            .split(',')
            .map(|s| s.trim().trim_matches(|c| c == '(' || c == ')'))
            .collect();

        // For now, support simple case: "(n),(n)->()" meaning inner product
        if output_sig == "()" {
            // Output is scalar per outer iteration
            let arrays: Vec<PyRumpyArray> = args
                .iter()
                .map(|arg| {
                    if let Ok(arr) = arg.extract::<PyRef<'_, PyRumpyArray>>() {
                        Ok(PyRumpyArray::new(arr.inner.clone()))
                    } else {
                        creation::asarray(py, &arg, None)
                    }
                })
                .collect::<PyResult<Vec<_>>>()?;

            if arrays.is_empty() {
                return self.pyfunc.bind(py).call(&args, kwargs).map(|r| r.unbind());
            }

            // Assume core dimensions are the last dimension
            let core_len = if !input_dims.is_empty() && !input_dims[0].is_empty() {
                arrays[0].inner.shape().last().copied().unwrap_or(1)
            } else {
                1
            };

            // Output shape is shape without core dimensions
            let out_shape: Vec<usize> = if arrays[0].inner.ndim() > 1 {
                arrays[0].inner.shape()[..arrays[0].inner.ndim() - 1].to_vec()
            } else {
                vec![]
            };

            let out_size: usize = out_shape.iter().product::<usize>().max(1);

            let mut results: Vec<f64> = Vec::with_capacity(out_size);

            for i in 0..out_size {
                // Extract 1D slices for each array along last axis (core dimension)
                let mut call_args: Vec<Bound<'_, PyAny>> = Vec::with_capacity(arrays.len());

                // Indices for all dims except last (the core dimension)
                let other_indices = if out_shape.is_empty() {
                    vec![]
                } else {
                    flat_to_indices(i, &out_shape)
                };

                for arr in &arrays {
                    let last_axis = arr.inner.ndim().saturating_sub(1);
                    let slice_data = if other_indices.is_empty() {
                        // 1D array - just convert to vec
                        arr.inner.to_vec()
                    } else {
                        arr.inner.get_slice_along_axis(last_axis, &other_indices)
                    };
                    let slice_arr = RumpyArray::from_vec(slice_data, arr.inner.dtype());
                    call_args.push(PyRumpyArray::new(slice_arr).into_pyobject(py)?.into_any());
                }

                let call_tuple = PyTuple::new(py, call_args)?;
                let result = self.pyfunc.bind(py).call(&call_tuple, kwargs)?;
                results.push(result.extract::<f64>()?);
            }

            let dtype = if let Some(ref otypes) = self.otypes {
                otypes[0].clone()
            } else {
                DType::float64()
            };

            if !out_shape.is_empty() {
                let out = RumpyArray::from_vec(results, dtype);
                let out = out.reshape(out_shape).ok_or_else(||
                    pyo3::exceptions::PyValueError::new_err("reshape failed")
                )?;
                Ok(PyRumpyArray::new(out).into_pyobject(py)?.unbind().into_any())
            } else {
                // Return scalar
                let scalar = results[0];
                Ok(scalar.into_pyobject(py)?.unbind().into_any())
            }
        } else {
            Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "Only scalar output signatures are supported"
            ))
        }
    }
}

/// Create a generalized universal function from a Python function.
#[pyclass(name = "frompyfunc")]
pub struct PyFrompyfunc {
    pyfunc: PyObject,
    nin: usize,
    nout: usize,
}

#[pymethods]
impl PyFrompyfunc {
    #[new]
    fn new(_py: Python<'_>, pyfunc: &Bound<'_, PyAny>, nin: usize, nout: usize) -> PyResult<Self> {
        Ok(PyFrompyfunc {
            pyfunc: pyfunc.clone().unbind(),
            nin,
            nout,
        })
    }

    #[pyo3(signature = (*args))]
    fn __call__(&self, py: Python<'_>, args: &Bound<'_, PyTuple>) -> PyResult<PyObject> {
        if args.len() != self.nin {
            return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "frompyfunc expected {} arguments, got {}",
                self.nin,
                args.len()
            )));
        }

        // Try to convert all args to arrays
        let mut arrays: Vec<Option<PyRumpyArray>> = Vec::new();
        let mut any_array = false;

        for arg in args.iter() {
            if let Ok(arr) = arg.extract::<PyRef<'_, PyRumpyArray>>() {
                arrays.push(Some(PyRumpyArray::new(arr.inner.clone())));
                any_array = true;
            } else if let Ok(arr) = creation::asarray(py, &arg, None) {
                if arr.inner.ndim() > 0 || arr.inner.size() > 1 {
                    arrays.push(Some(arr));
                    any_array = true;
                } else {
                    arrays.push(None);
                }
            } else {
                arrays.push(None);
            }
        }

        if !any_array {
            // All scalars - call directly
            let result = self.pyfunc.bind(py).call1(args)?;
            return Ok(result.unbind());
        }

        // Get broadcast shape
        let shapes: Vec<Vec<usize>> = arrays
            .iter()
            .filter_map(|a| a.as_ref().map(|arr| arr.inner.shape().to_vec()))
            .collect();

        let broadcast_shape = if shapes.is_empty() {
            vec![1]
        } else {
            broadcast_shapes(&shapes)?
        };

        let size: usize = broadcast_shape.iter().product();

        // Broadcast arrays
        let broadcasted: Vec<Option<RumpyArray>> = arrays
            .iter()
            .map(|a| {
                match a {
                    Some(arr) => arr.inner.broadcast_to(&broadcast_shape),
                    None => None,
                }
            })
            .collect();

        // Apply element-wise
        let mut all_results: Vec<Vec<PyObject>> = (0..self.nout).map(|_| Vec::with_capacity(size)).collect();

        for i in 0..size {
            let mut call_args: Vec<PyObject> = Vec::with_capacity(self.nin);
            let indices = flat_to_indices(i, &broadcast_shape);

            for (j, maybe_arr) in broadcasted.iter().enumerate() {
                if let Some(arr) = maybe_arr {
                    let val = arr.get_element(&indices);
                    call_args.push(val.into_pyobject(py)?.unbind().into_any());
                } else {
                    // Use original scalar
                    call_args.push(args.get_item(j)?.unbind());
                }
            }

            let call_tuple = PyTuple::new(py, call_args)?;
            let result = self.pyfunc.bind(py).call1(&call_tuple)?;

            if self.nout == 1 {
                all_results[0].push(result.unbind());
            } else {
                // Multiple outputs - unpack tuple
                let result_tuple = result.downcast::<PyTuple>()?;
                for (k, item) in result_tuple.iter().enumerate() {
                    if k < self.nout {
                        all_results[k].push(item.unbind());
                    }
                }
            }
        }

        // Build output arrays (object dtype like numpy)
        if self.nout == 1 {
            let out_list = PyList::new(py, all_results[0].iter().map(|r| r.bind(py)))?;
            let numpy = py.import("numpy")?;
            let np_arr = numpy.call_method1("array", (out_list,))?;
            if broadcast_shape.len() > 1 {
                let np_reshaped = np_arr.call_method1("reshape", (broadcast_shape.clone(),))?;
                Ok(np_reshaped.unbind())
            } else {
                Ok(np_arr.unbind())
            }
        } else {
            // Multiple outputs - return tuple of arrays
            let mut out_arrays: Vec<Bound<'_, PyAny>> = Vec::with_capacity(self.nout);
            for results in all_results {
                let out_list = PyList::new(py, results.iter().map(|r| r.bind(py)))?;
                let numpy = py.import("numpy")?;
                let np_arr = numpy.call_method1("array", (out_list,))?;
                if broadcast_shape.len() > 1 {
                    let np_reshaped = np_arr.call_method1("reshape", (broadcast_shape.clone(),))?;
                    out_arrays.push(np_reshaped);
                } else {
                    out_arrays.push(np_arr);
                }
            }
            Ok(PyTuple::new(py, out_arrays)?.into_any().unbind())
        }
    }
}

/// Convenience function to create vectorize object.
#[pyfunction]
#[pyo3(signature = (pyfunc, otypes=None, excluded=None, signature=None))]
pub fn vectorize(
    py: Python<'_>,
    pyfunc: &Bound<'_, PyAny>,
    otypes: Option<&Bound<'_, PyAny>>,
    excluded: Option<&Bound<'_, PyAny>>,
    signature: Option<String>,
) -> PyResult<PyVectorize> {
    PyVectorize::new(py, pyfunc, otypes, excluded, signature)
}

/// Convenience function to create frompyfunc object.
#[pyfunction]
pub fn frompyfunc(py: Python<'_>, pyfunc: &Bound<'_, PyAny>, nin: usize, nout: usize) -> PyResult<PyFrompyfunc> {
    PyFrompyfunc::new(py, pyfunc, nin, nout)
}

/// Broadcast shapes together.
fn broadcast_shapes(shapes: &[Vec<usize>]) -> PyResult<Vec<usize>> {
    if shapes.is_empty() {
        return Ok(vec![]);
    }

    let max_ndim = shapes.iter().map(|s| s.len()).max().unwrap_or(0);
    let mut result = vec![1usize; max_ndim];

    for shape in shapes {
        let offset = max_ndim - shape.len();
        for (i, &dim) in shape.iter().enumerate() {
            let j = i + offset;
            if result[j] == 1 {
                result[j] = dim;
            } else if dim != 1 && dim != result[j] {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "shape mismatch: objects cannot be broadcast to a single shape"
                ));
            }
        }
    }

    Ok(result)
}
