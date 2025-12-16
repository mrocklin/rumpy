// Python bindings for miscellaneous functions (Stream 32).
//
// Functions:
// - resize: resize array with repetition
// - unstack: split along axis into tuple
// - block: assemble from nested blocks
// - trim_zeros: trim leading/trailing zeros
// - extract: extract elements where condition
// - place: place values where condition (cycles)
// - putmask: put values using mask (broadcasts)
// - select: select from choicelist by conditions
// - piecewise: piecewise function evaluation
// - ediff1d: differences with prepend/append
// - unwrap: unwrap phase angles
// - angle: phase angle of complex numbers
// - real_if_close: convert to real if imaginary is small

use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};
use std::sync::Arc;

use crate::array::{DType, RumpyArray};
use crate::array::dtype::DTypeKind;
use super::{PyRumpyArray, shape::to_rumpy_array};


/// Resolve a potentially negative axis to a positive index.
fn resolve_axis(axis: isize, ndim: usize) -> PyResult<usize> {
    let resolved = if axis < 0 {
        (ndim as isize + axis) as usize
    } else {
        axis as usize
    };
    if resolved >= ndim {
        Err(pyo3::exceptions::PyValueError::new_err(format!(
            "axis {} is out of bounds for array of dimension {}",
            axis, ndim
        )))
    } else {
        Ok(resolved)
    }
}

// ============================================================================
// resize - resize array with repetition
// ============================================================================

/// Return a new array with the specified shape.
/// If new shape is larger, array is repeated to fill.
/// If smaller, array is truncated.
#[pyfunction]
pub fn resize(a: &PyRumpyArray, new_shape: &Bound<'_, pyo3::PyAny>) -> PyResult<PyRumpyArray> {
    // Parse new_shape
    let shape: Vec<usize> = if let Ok(n) = new_shape.extract::<usize>() {
        vec![n]
    } else if let Ok(tuple) = new_shape.downcast::<PyTuple>() {
        tuple.iter().map(|x| x.extract::<usize>()).collect::<PyResult<Vec<_>>>()?
    } else if let Ok(list) = new_shape.downcast::<PyList>() {
        list.iter().map(|x| x.extract::<usize>()).collect::<PyResult<Vec<_>>>()?
    } else {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "new_shape must be int, tuple, or list"
        ));
    };

    let inner = &a.inner;
    let src_size = inner.size();
    let dst_size: usize = shape.iter().product();
    let dtype = inner.dtype();

    if dst_size == 0 || src_size == 0 {
        return Ok(PyRumpyArray::new(RumpyArray::zeros(shape, dtype)));
    }

    // Dispatch to typed implementation for direct byte copying
    let itemsize = inner.itemsize();
    let mut result = RumpyArray::zeros(shape, dtype);
    let buffer = Arc::get_mut(&mut result.buffer).expect("unique");
    let dst_ptr = buffer.as_mut_ptr();
    let src_ptr = inner.data_ptr();

    // For contiguous source, we can use fast byte copying
    if inner.is_c_contiguous() {
        // Copy full tiles
        let full_tiles = dst_size / src_size;
        let remainder = dst_size % src_size;
        let src_bytes = src_size * itemsize;

        for tile in 0..full_tiles {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    src_ptr,
                    dst_ptr.add(tile * src_bytes),
                    src_bytes
                );
            }
        }
        // Copy remainder
        if remainder > 0 {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    src_ptr,
                    dst_ptr.add(full_tiles * src_bytes),
                    remainder * itemsize
                );
            }
        }
    } else {
        // Strided source - copy element by element using byte offsets
        let src_offsets: Vec<isize> = inner.iter_offsets().collect();
        for i in 0..dst_size {
            let src_offset = src_offsets[i % src_size];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    src_ptr.offset(src_offset),
                    dst_ptr.add(i * itemsize),
                    itemsize
                );
            }
        }
    }

    Ok(PyRumpyArray::new(result))
}

// ============================================================================
// unstack - split along axis into tuple
// ============================================================================

/// Unstack an array along an axis into a tuple of arrays.
#[pyfunction]
#[pyo3(signature = (x, axis=0))]
pub fn unstack(py: Python<'_>, x: &PyRumpyArray, axis: isize) -> PyResult<Py<PyTuple>> {
    let inner = &x.inner;
    let ndim = inner.ndim();

    if ndim == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "cannot unstack a 0-d array"
        ));
    }

    let axis = resolve_axis(axis, ndim)?;
    let axis_len = inner.shape()[axis];

    let mut result: Vec<Py<PyRumpyArray>> = Vec::with_capacity(axis_len);

    for i in 0..axis_len {
        // Take slice along axis at index i, then squeeze that axis
        let slice = inner.slice_axis(axis, i as isize, (i + 1) as isize, 1);
        let squeezed = slice.squeeze_axis(axis);
        result.push(Py::new(py, PyRumpyArray::new(squeezed))?);
    }

    Ok(PyTuple::new(py, result)?.into())
}

// ============================================================================
// block - assemble from nested blocks
// ============================================================================

/// Assemble an array from nested lists of blocks.
#[pyfunction]
pub fn block(arrays: &Bound<'_, PyAny>) -> PyResult<PyRumpyArray> {
    // Recursively process the nested list structure
    let result = process_block(arrays, 0)?;
    Ok(PyRumpyArray::new(result))
}

fn process_block(obj: &Bound<'_, PyAny>, depth: usize) -> PyResult<RumpyArray> {
    // If it's a RumpyArray, return it
    if let Ok(arr) = obj.extract::<pyo3::PyRef<'_, PyRumpyArray>>() {
        return Ok(arr.inner.clone());
    }

    // If it's a list, process recursively
    if let Ok(list) = obj.downcast::<PyList>() {
        if list.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "empty block list"
            ));
        }

        // Process each element
        let blocks: Vec<RumpyArray> = list.iter()
            .map(|item| process_block(&item, depth + 1))
            .collect::<PyResult<Vec<_>>>()?;

        // Concatenate along the appropriate axis
        // depth 0 = concatenate along last axis (horizontal)
        // depth 1 = concatenate along first axis (vertical)
        // etc.

        // For a simple 2D block, we have depth=0 for rows, depth=1 for cols
        // numpy's block concatenates at depth according to nesting level

        if blocks.len() == 1 {
            return Ok(blocks.into_iter().next().unwrap());
        }

        // Determine axis: for the outermost list it's axis 0, inner is axis 1, etc.
        let max_ndim = blocks.iter().map(|b| b.ndim()).max().unwrap_or(1);
        let axis = if depth < max_ndim { depth } else { max_ndim - 1 };

        crate::array::manipulation::concatenate(&blocks, axis)
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err(
                "block arrays have incompatible shapes"
            ))
    } else {
        // Try to convert to array
        to_rumpy_array(obj)
    }
}

// ============================================================================
// trim_zeros - trim leading/trailing zeros
// ============================================================================

/// Trim leading and/or trailing zeros from a 1-D array.
#[pyfunction]
#[pyo3(signature = (filt, trim="fb"))]
pub fn trim_zeros(filt: &PyRumpyArray, trim: &str) -> PyResult<PyRumpyArray> {
    let inner = &filt.inner;

    if inner.ndim() != 1 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "trim_zeros requires 1-D array"
        ));
    }

    let size = inner.size();
    if size == 0 {
        return Ok(PyRumpyArray::new(inner.clone()));
    }

    let dtype = inner.dtype();

    // Typed dispatch for finding first/last non-zero
    macro_rules! find_bounds_typed {
        ($ty:ty, $zero:expr) => {{
            let ptr = inner.data_ptr() as *const $ty;

            let mut first = 0usize;
            if trim.contains('f') || trim.contains('F') {
                first = size; // Assume all zeros
                if inner.is_c_contiguous() {
                    for i in 0..size {
                        if unsafe { *ptr.add(i) } != $zero {
                            first = i;
                            break;
                        }
                    }
                } else {
                    let offsets: Vec<isize> = inner.iter_offsets().collect();
                    for i in 0..size {
                        let val = unsafe { *(inner.data_ptr().offset(offsets[i]) as *const $ty) };
                        if val != $zero {
                            first = i;
                            break;
                        }
                    }
                }
            }

            let mut last = size;
            if trim.contains('b') || trim.contains('B') {
                last = first; // Assume all zeros after first
                if inner.is_c_contiguous() {
                    for i in (0..size).rev() {
                        if unsafe { *ptr.add(i) } != $zero {
                            last = i + 1;
                            break;
                        }
                    }
                } else {
                    let offsets: Vec<isize> = inner.iter_offsets().collect();
                    for i in (0..size).rev() {
                        let val = unsafe { *(inner.data_ptr().offset(offsets[i]) as *const $ty) };
                        if val != $zero {
                            last = i + 1;
                            break;
                        }
                    }
                }
            }
            (first, last)
        }};
    }

    let (first, last) = match dtype.kind() {
        DTypeKind::Float32 => find_bounds_typed!(f32, 0.0f32),
        DTypeKind::Float64 => find_bounds_typed!(f64, 0.0f64),
        DTypeKind::Int8 => find_bounds_typed!(i8, 0i8),
        DTypeKind::Int16 => find_bounds_typed!(i16, 0i16),
        DTypeKind::Int32 => find_bounds_typed!(i32, 0i32),
        DTypeKind::Int64 => find_bounds_typed!(i64, 0i64),
        DTypeKind::Uint8 => find_bounds_typed!(u8, 0u8),
        DTypeKind::Uint16 => find_bounds_typed!(u16, 0u16),
        DTypeKind::Uint32 => find_bounds_typed!(u32, 0u32),
        DTypeKind::Uint64 => find_bounds_typed!(u64, 0u64),
        DTypeKind::Bool => find_bounds_typed!(u8, 0u8),
        _ => {
            // Complex types: check if both real and imag are zero
            let ops = dtype.ops();
            let offsets: Vec<isize> = inner.iter_offsets().collect();
            let ptr = inner.data_ptr();

            let mut first = 0usize;
            if trim.contains('f') || trim.contains('F') {
                first = size;
                for i in 0..size {
                    let val = unsafe { ops.read_f64(ptr, offsets[i]) }.unwrap_or(0.0);
                    if val != 0.0 {
                        first = i;
                        break;
                    }
                }
            }

            let mut last = size;
            if trim.contains('b') || trim.contains('B') {
                last = first;
                for i in (0..size).rev() {
                    let val = unsafe { ops.read_f64(ptr, offsets[i]) }.unwrap_or(0.0);
                    if val != 0.0 {
                        last = i + 1;
                        break;
                    }
                }
            }
            (first, last)
        }
    };

    if first >= last {
        return Ok(PyRumpyArray::new(RumpyArray::zeros(vec![0], dtype)));
    }

    let sliced = inner.slice_axis(0, first as isize, last as isize, 1);
    Ok(PyRumpyArray::new(sliced.copy()))
}

// ============================================================================
// extract - extract elements where condition is true
// ============================================================================

/// Return elements of an array that satisfy a condition.
#[pyfunction]
pub fn extract(condition: &PyRumpyArray, arr: &PyRumpyArray) -> PyResult<PyRumpyArray> {
    let cond = &condition.inner;
    let inner = &arr.inner;

    // Both must have same total size when flattened
    if cond.size() != inner.size() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "condition and array must have same size"
        ));
    }

    let dtype = inner.dtype();
    let size = inner.size();

    // Fast path: both contiguous boolean condition and typed source
    if cond.is_c_contiguous() && inner.is_c_contiguous() {
        let cond_ptr = cond.data_ptr() as *const u8;  // bool is stored as u8

        // Count true values
        let mut count = 0usize;
        for i in 0..size {
            if unsafe { *cond_ptr.add(i) } != 0 {
                count += 1;
            }
        }

        let mut result = RumpyArray::zeros(vec![count], dtype.clone());
        if count == 0 {
            return Ok(PyRumpyArray::new(result));
        }

        let buffer = Arc::get_mut(&mut result.buffer).expect("unique");
        let dst_ptr = buffer.as_mut_ptr();
        let src_ptr = inner.data_ptr();

        // Typed dispatch for extraction
        macro_rules! extract_typed {
            ($ty:ty) => {{
                let src = src_ptr as *const $ty;
                let dst = dst_ptr as *mut $ty;
                let mut dst_idx = 0usize;
                for i in 0..size {
                    if unsafe { *cond_ptr.add(i) } != 0 {
                        unsafe { *dst.add(dst_idx) = *src.add(i); }
                        dst_idx += 1;
                    }
                }
            }};
        }

        match dtype.kind() {
            DTypeKind::Float32 => extract_typed!(f32),
            DTypeKind::Float64 => extract_typed!(f64),
            DTypeKind::Int32 => extract_typed!(i32),
            DTypeKind::Int64 => extract_typed!(i64),
            DTypeKind::Uint32 => extract_typed!(u32),
            DTypeKind::Uint64 => extract_typed!(u64),
            DTypeKind::Int8 => extract_typed!(i8),
            DTypeKind::Int16 => extract_typed!(i16),
            DTypeKind::Uint8 => extract_typed!(u8),
            DTypeKind::Uint16 => extract_typed!(u16),
            DTypeKind::Bool => extract_typed!(u8),
            _ => {
                // Fallback for complex types
                let ops = dtype.ops();
                let mut dst_idx = 0usize;
                for i in 0..size {
                    if unsafe { *cond_ptr.add(i) } != 0 {
                        let val = unsafe { ops.read_f64(src_ptr, (i * inner.itemsize()) as isize) }.unwrap_or(0.0);
                        unsafe { ops.write_f64(dst_ptr, dst_idx, val); }
                        dst_idx += 1;
                    }
                }
            }
        }

        return Ok(PyRumpyArray::new(result));
    }

    // Strided path with typed dispatch
    let cond_ptr = cond.data_ptr();
    let cond_offsets: Vec<isize> = cond.iter_offsets().collect();

    // Count true values (condition is always bool = u8)
    let mut count = 0usize;
    for &offset in &cond_offsets {
        if unsafe { *(cond_ptr.offset(offset) as *const u8) } != 0 {
            count += 1;
        }
    }

    let mut result = RumpyArray::zeros(vec![count], dtype.clone());
    if count == 0 {
        return Ok(PyRumpyArray::new(result));
    }

    let src_ptr = inner.data_ptr();
    let src_offsets: Vec<isize> = inner.iter_offsets().collect();
    let buffer = Arc::get_mut(&mut result.buffer).expect("unique");
    let dst_ptr = buffer.as_mut_ptr();

    macro_rules! extract_strided_typed {
        ($ty:ty) => {{
            let dst = dst_ptr as *mut $ty;
            let mut dst_idx = 0usize;
            for i in 0..size {
                if unsafe { *(cond_ptr.offset(cond_offsets[i]) as *const u8) } != 0 {
                    let val = unsafe { *(src_ptr.offset(src_offsets[i]) as *const $ty) };
                    unsafe { *dst.add(dst_idx) = val; }
                    dst_idx += 1;
                }
            }
        }};
    }

    match dtype.kind() {
        DTypeKind::Float32 => extract_strided_typed!(f32),
        DTypeKind::Float64 => extract_strided_typed!(f64),
        DTypeKind::Int32 => extract_strided_typed!(i32),
        DTypeKind::Int64 => extract_strided_typed!(i64),
        DTypeKind::Uint32 => extract_strided_typed!(u32),
        DTypeKind::Uint64 => extract_strided_typed!(u64),
        DTypeKind::Int8 => extract_strided_typed!(i8),
        DTypeKind::Int16 => extract_strided_typed!(i16),
        DTypeKind::Uint8 => extract_strided_typed!(u8),
        DTypeKind::Uint16 => extract_strided_typed!(u16),
        DTypeKind::Bool => extract_strided_typed!(u8),
        _ => {
            // Complex types fallback
            let ops = dtype.ops();
            let mut dst_idx = 0usize;
            for i in 0..size {
                if unsafe { *(cond_ptr.offset(cond_offsets[i]) as *const u8) } != 0 {
                    let val = unsafe { ops.read_f64(src_ptr, src_offsets[i]) }.unwrap_or(0.0);
                    unsafe { ops.write_f64(dst_ptr, dst_idx as usize, val); }
                    dst_idx += 1;
                }
            }
        }
    }

    Ok(PyRumpyArray::new(result))
}

// ============================================================================
// place - place values where condition is true (cycles values)
// ============================================================================

/// Change elements of an array based on condition and values.
/// Values are cycled through.
#[pyfunction]
pub fn place(arr: &PyRumpyArray, mask: &PyRumpyArray, vals: &PyRumpyArray) -> PyResult<()> {
    let inner = &arr.inner;
    let mask_arr = &mask.inner;
    let vals_arr = &vals.inner;

    if mask_arr.size() != inner.size() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "mask and array must have same size"
        ));
    }

    if vals_arr.size() == 0 {
        return Ok(());
    }

    let arr_dtype = inner.dtype();
    let size = inner.size();
    let vals_size = vals_arr.size();

    // Fast path: all contiguous, same dtype, bool mask
    if mask_arr.is_c_contiguous() && inner.is_c_contiguous() && vals_arr.is_c_contiguous()
        && arr_dtype.kind() == vals_arr.dtype().kind()
    {
        let mask_ptr = mask_arr.data_ptr() as *const u8;
        let arr_ptr = inner.data_ptr();
        let vals_ptr = vals_arr.data_ptr();

        macro_rules! place_typed {
            ($ty:ty) => {{
                let dst = arr_ptr as *mut $ty;
                let src = vals_ptr as *const $ty;
                let mut val_idx = 0usize;
                for i in 0..size {
                    if unsafe { *mask_ptr.add(i) } != 0 {
                        unsafe { *dst.add(i) = *src.add(val_idx % vals_size); }
                        val_idx += 1;
                    }
                }
            }};
        }

        match arr_dtype.kind() {
            DTypeKind::Float32 => place_typed!(f32),
            DTypeKind::Float64 => place_typed!(f64),
            DTypeKind::Int32 => place_typed!(i32),
            DTypeKind::Int64 => place_typed!(i64),
            DTypeKind::Uint32 => place_typed!(u32),
            DTypeKind::Uint64 => place_typed!(u64),
            DTypeKind::Int8 => place_typed!(i8),
            DTypeKind::Int16 => place_typed!(i16),
            DTypeKind::Uint8 => place_typed!(u8),
            DTypeKind::Uint16 => place_typed!(u16),
            DTypeKind::Bool => place_typed!(u8),
            _ => {} // Fall through to generic path
        }
        return Ok(());
    }

    // Strided path with typed dispatch
    let mask_ptr = mask_arr.data_ptr();
    let mask_offsets: Vec<isize> = mask_arr.iter_offsets().collect();
    let vals_ptr = vals_arr.data_ptr();
    let vals_offsets: Vec<isize> = vals_arr.iter_offsets().collect();
    let arr_ptr = inner.data_ptr();
    let arr_offsets: Vec<isize> = inner.iter_offsets().collect();

    macro_rules! place_strided_typed {
        ($ty:ty) => {{
            let mut val_idx = 0usize;
            for i in 0..size {
                if unsafe { *(mask_ptr.offset(mask_offsets[i]) as *const u8) } != 0 {
                    let val = unsafe { *(vals_ptr.offset(vals_offsets[val_idx % vals_size]) as *const $ty) };
                    unsafe { *(arr_ptr.offset(arr_offsets[i]) as *mut $ty) = val; }
                    val_idx += 1;
                }
            }
        }};
    }

    match arr_dtype.kind() {
        DTypeKind::Float32 => place_strided_typed!(f32),
        DTypeKind::Float64 => place_strided_typed!(f64),
        DTypeKind::Int32 => place_strided_typed!(i32),
        DTypeKind::Int64 => place_strided_typed!(i64),
        DTypeKind::Uint32 => place_strided_typed!(u32),
        DTypeKind::Uint64 => place_strided_typed!(u64),
        DTypeKind::Int8 => place_strided_typed!(i8),
        DTypeKind::Int16 => place_strided_typed!(i16),
        DTypeKind::Uint8 => place_strided_typed!(u8),
        DTypeKind::Uint16 => place_strided_typed!(u16),
        DTypeKind::Bool => place_strided_typed!(u8),
        _ => {
            // Complex types fallback
            let arr_ops = arr_dtype.ops();
            let vals_dtype = vals_arr.dtype();
            let vals_ops = vals_dtype.ops();
            let mut val_idx = 0usize;
            for i in 0..size {
                if unsafe { *(mask_ptr.offset(mask_offsets[i]) as *const u8) } != 0 {
                    let val = unsafe { vals_ops.read_f64(vals_ptr, vals_offsets[val_idx % vals_size]) }.unwrap_or(0.0);
                    unsafe { arr_ops.write_f64_at_byte_offset(arr_ptr as *mut u8, arr_offsets[i], val); }
                    val_idx += 1;
                }
            }
        }
    }

    Ok(())
}

// ============================================================================
// putmask - put values using mask (broadcasts values)
// ============================================================================

/// Changes elements of an array based on mask and values.
/// Values are broadcast to match mask positions.
#[pyfunction]
pub fn putmask(a: &PyRumpyArray, mask: &PyRumpyArray, values: &PyRumpyArray) -> PyResult<()> {
    let inner = &a.inner;
    let mask_arr = &mask.inner;
    let vals_arr = &values.inner;

    if mask_arr.size() != inner.size() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "mask and array must have same size"
        ));
    }

    if vals_arr.size() == 0 {
        return Ok(());
    }

    let arr_dtype = inner.dtype();
    let size = inner.size();
    let vals_size = vals_arr.size();

    // Fast path: all contiguous, same dtype, bool mask
    if mask_arr.is_c_contiguous() && inner.is_c_contiguous() && vals_arr.is_c_contiguous()
        && arr_dtype.kind() == vals_arr.dtype().kind()
    {
        let mask_ptr = mask_arr.data_ptr() as *const u8;
        let arr_ptr = inner.data_ptr();
        let vals_ptr = vals_arr.data_ptr();

        macro_rules! putmask_typed {
            ($ty:ty) => {{
                let dst = arr_ptr as *mut $ty;
                let src = vals_ptr as *const $ty;
                for i in 0..size {
                    if unsafe { *mask_ptr.add(i) } != 0 {
                        unsafe { *dst.add(i) = *src.add(i % vals_size); }
                    }
                }
            }};
        }

        match arr_dtype.kind() {
            DTypeKind::Float32 => putmask_typed!(f32),
            DTypeKind::Float64 => putmask_typed!(f64),
            DTypeKind::Int32 => putmask_typed!(i32),
            DTypeKind::Int64 => putmask_typed!(i64),
            DTypeKind::Uint32 => putmask_typed!(u32),
            DTypeKind::Uint64 => putmask_typed!(u64),
            DTypeKind::Int8 => putmask_typed!(i8),
            DTypeKind::Int16 => putmask_typed!(i16),
            DTypeKind::Uint8 => putmask_typed!(u8),
            DTypeKind::Uint16 => putmask_typed!(u16),
            DTypeKind::Bool => putmask_typed!(u8),
            _ => {} // Fall through to generic path
        }
        return Ok(());
    }

    // Strided path with typed dispatch
    let mask_ptr = mask_arr.data_ptr();
    let mask_offsets: Vec<isize> = mask_arr.iter_offsets().collect();
    let vals_ptr = vals_arr.data_ptr();
    let vals_offsets: Vec<isize> = vals_arr.iter_offsets().collect();
    let arr_ptr = inner.data_ptr();
    let arr_offsets: Vec<isize> = inner.iter_offsets().collect();

    macro_rules! putmask_strided_typed {
        ($ty:ty) => {{
            for i in 0..size {
                if unsafe { *(mask_ptr.offset(mask_offsets[i]) as *const u8) } != 0 {
                    let val = unsafe { *(vals_ptr.offset(vals_offsets[i % vals_size]) as *const $ty) };
                    unsafe { *(arr_ptr.offset(arr_offsets[i]) as *mut $ty) = val; }
                }
            }
        }};
    }

    match arr_dtype.kind() {
        DTypeKind::Float32 => putmask_strided_typed!(f32),
        DTypeKind::Float64 => putmask_strided_typed!(f64),
        DTypeKind::Int32 => putmask_strided_typed!(i32),
        DTypeKind::Int64 => putmask_strided_typed!(i64),
        DTypeKind::Uint32 => putmask_strided_typed!(u32),
        DTypeKind::Uint64 => putmask_strided_typed!(u64),
        DTypeKind::Int8 => putmask_strided_typed!(i8),
        DTypeKind::Int16 => putmask_strided_typed!(i16),
        DTypeKind::Uint8 => putmask_strided_typed!(u8),
        DTypeKind::Uint16 => putmask_strided_typed!(u16),
        DTypeKind::Bool => putmask_strided_typed!(u8),
        _ => {
            // Complex types fallback
            let arr_ops = arr_dtype.ops();
            let vals_dtype = vals_arr.dtype();
            let vals_ops = vals_dtype.ops();
            for i in 0..size {
                if unsafe { *(mask_ptr.offset(mask_offsets[i]) as *const u8) } != 0 {
                    let val = unsafe { vals_ops.read_f64(vals_ptr, vals_offsets[i % vals_size]) }.unwrap_or(0.0);
                    unsafe { arr_ops.write_f64_at_byte_offset(arr_ptr as *mut u8, arr_offsets[i], val); }
                }
            }
        }
    }

    Ok(())
}

// ============================================================================
// select - select from choicelist by conditions
// ============================================================================

/// Return array from choicelist based on conditions.
#[pyfunction]
#[pyo3(signature = (condlist, choicelist, default=0.0))]
pub fn select(
    condlist: Vec<pyo3::PyRef<'_, PyRumpyArray>>,
    choicelist: Vec<pyo3::PyRef<'_, PyRumpyArray>>,
    default: f64,
) -> PyResult<PyRumpyArray> {
    if condlist.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "select requires at least one condition"
        ));
    }

    if condlist.len() != choicelist.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "condlist and choicelist must have same length"
        ));
    }

    let shape = condlist[0].inner.shape().to_vec();
    let size = condlist[0].inner.size();
    let out_dtype = choicelist[0].inner.dtype();

    let mut result = RumpyArray::full(shape, default, out_dtype.clone());
    let buffer = Arc::get_mut(&mut result.buffer).expect("unique");
    let dst_ptr = buffer.as_mut_ptr();

    // Collect offsets
    let cond_offsets: Vec<Vec<isize>> = condlist.iter()
        .map(|c| c.inner.iter_offsets().collect())
        .collect();
    let choice_offsets: Vec<Vec<isize>> = choicelist.iter()
        .map(|c| c.inner.iter_offsets().collect())
        .collect();

    // Collect raw pointers
    let cond_ptrs: Vec<*const u8> = condlist.iter()
        .map(|c| c.inner.data_ptr())
        .collect();
    let choice_ptrs: Vec<*const u8> = choicelist.iter()
        .map(|c| c.inner.data_ptr())
        .collect();

    macro_rules! select_typed {
        ($ty:ty) => {{
            let dst = dst_ptr as *mut $ty;
            for i in 0..size {
                for cond_idx in 0..condlist.len() {
                    // Condition is bool = u8
                    let cond_val = unsafe { *(cond_ptrs[cond_idx].offset(cond_offsets[cond_idx][i]) as *const u8) };
                    if cond_val != 0 {
                        let val = unsafe { *(choice_ptrs[cond_idx].offset(choice_offsets[cond_idx][i]) as *const $ty) };
                        unsafe { *dst.add(i) = val; }
                        break;
                    }
                }
            }
        }};
    }

    match out_dtype.kind() {
        DTypeKind::Float32 => select_typed!(f32),
        DTypeKind::Float64 => select_typed!(f64),
        DTypeKind::Int32 => select_typed!(i32),
        DTypeKind::Int64 => select_typed!(i64),
        DTypeKind::Uint32 => select_typed!(u32),
        DTypeKind::Uint64 => select_typed!(u64),
        DTypeKind::Int8 => select_typed!(i8),
        DTypeKind::Int16 => select_typed!(i16),
        DTypeKind::Uint8 => select_typed!(u8),
        DTypeKind::Uint16 => select_typed!(u16),
        DTypeKind::Bool => select_typed!(u8),
        _ => {
            // Complex types fallback
            let dst_ops = out_dtype.ops();
            for i in 0..size {
                for cond_idx in 0..condlist.len() {
                    let cond_val = unsafe { *(cond_ptrs[cond_idx].offset(cond_offsets[cond_idx][i]) as *const u8) };
                    if cond_val != 0 {
                        let choice = &choicelist[cond_idx].inner;
                        let choice_dtype = choice.dtype();
                        let val = unsafe { choice_dtype.ops().read_f64(choice_ptrs[cond_idx], choice_offsets[cond_idx][i]) }.unwrap_or(0.0);
                        unsafe { dst_ops.write_f64(dst_ptr, i, val); }
                        break;
                    }
                }
            }
        }
    }

    Ok(PyRumpyArray::new(result))
}

// ============================================================================
// piecewise - piecewise function evaluation
// ============================================================================

/// Evaluate a piecewise-defined function.
/// Note: This function uses f64 for Python callable interop, which is necessary
/// for passing values to/from Python functions. Constants are written via typed dispatch.
#[pyfunction]
pub fn piecewise(
    py: Python<'_>,
    x: &PyRumpyArray,
    condlist: Vec<pyo3::PyRef<'_, PyRumpyArray>>,
    funclist: &Bound<'_, PyList>,
) -> PyResult<PyRumpyArray> {
    let inner = &x.inner;
    let size = inner.size();
    let shape = inner.shape().to_vec();
    let dtype = inner.dtype();

    if condlist.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "piecewise requires at least one condition"
        ));
    }

    let has_default = funclist.len() == condlist.len() + 1;
    if !has_default && funclist.len() != condlist.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "funclist must have same length as condlist or one more"
        ));
    }

    let mut result = RumpyArray::zeros(shape, dtype.clone());
    let buffer = Arc::get_mut(&mut result.buffer).expect("unique");
    let dst_ptr = buffer.as_mut_ptr();

    let src_ptr = inner.data_ptr();
    let src_offsets: Vec<isize> = inner.iter_offsets().collect();

    // Collect condition pointers and offsets
    let cond_ptrs: Vec<*const u8> = condlist.iter()
        .map(|c| c.inner.data_ptr())
        .collect();
    let cond_offsets: Vec<Vec<isize>> = condlist.iter()
        .map(|c| c.inner.iter_offsets().collect())
        .collect();

    let mut assigned = vec![false; size];

    // Helper macro for writing typed values with dtype dispatch
    macro_rules! write_val {
        ($idx:expr, $val:expr) => {{
            match dtype.kind() {
                DTypeKind::Float32 => unsafe { *(dst_ptr as *mut f32).add($idx) = $val as f32; },
                DTypeKind::Float64 => unsafe { *(dst_ptr as *mut f64).add($idx) = $val as f64; },
                DTypeKind::Int32 => unsafe { *(dst_ptr as *mut i32).add($idx) = $val as i32; },
                DTypeKind::Int64 => unsafe { *(dst_ptr as *mut i64).add($idx) = $val as i64; },
                DTypeKind::Uint32 => unsafe { *(dst_ptr as *mut u32).add($idx) = $val as u32; },
                DTypeKind::Uint64 => unsafe { *(dst_ptr as *mut u64).add($idx) = $val as u64; },
                DTypeKind::Int8 => unsafe { *(dst_ptr as *mut i8).add($idx) = $val as i8; },
                DTypeKind::Int16 => unsafe { *(dst_ptr as *mut i16).add($idx) = $val as i16; },
                DTypeKind::Uint8 => unsafe { *(dst_ptr as *mut u8).add($idx) = $val as u8; },
                DTypeKind::Uint16 => unsafe { *(dst_ptr as *mut u16).add($idx) = $val as u16; },
                DTypeKind::Bool => unsafe { *(dst_ptr as *mut u8).add($idx) = $val as u8; },
                _ => unsafe { dtype.ops().write_f64(dst_ptr, $idx, $val); }
            }
        }};
    }

    // Helper to call a Python function and get result as f64
    let call_func = |func: &Bound<'_, pyo3::PyAny>, i: usize| -> PyResult<f64> {
        let x_val = unsafe { dtype.ops().read_f64(src_ptr, src_offsets[i]) }.unwrap_or(0.0);
        let x_arr = RumpyArray::full(vec![1], x_val, dtype.clone());
        let x_py = Py::new(py, PyRumpyArray::new(x_arr))?;
        let result_py = func.call1((x_py,))?;
        if let Ok(arr) = result_py.extract::<pyo3::PyRef<'_, PyRumpyArray>>() {
            let arr_dtype = arr.inner.dtype();
            Ok(unsafe { arr_dtype.ops().read_f64(arr.inner.data_ptr(), 0) }.unwrap_or(0.0))
        } else {
            result_py.extract::<f64>()
        }
    };

    // Process each condition/function pair
    for cond_idx in 0..condlist.len() {
        let func_item = funclist.get_item(cond_idx)?;
        let constant_val: Option<f64> = if !func_item.is_callable() {
            Some(func_item.extract::<f64>()?)
        } else {
            None
        };

        for i in 0..size {
            if assigned[i] { continue; }

            let cond_val = unsafe { *(cond_ptrs[cond_idx].offset(cond_offsets[cond_idx][i]) as *const u8) };
            if cond_val != 0 {
                let val = constant_val.map_or_else(|| call_func(&func_item, i), Ok)?;
                write_val!(i, val);
                assigned[i] = true;
            }
        }
    }

    // Handle default if present
    if has_default {
        let default_item = funclist.get_item(condlist.len())?;
        let constant_val: Option<f64> = if !default_item.is_callable() {
            Some(default_item.extract::<f64>()?)
        } else {
            None
        };

        for i in 0..size {
            if !assigned[i] {
                let val = constant_val.map_or_else(|| call_func(&default_item, i), Ok)?;
                write_val!(i, val);
            }
        }
    }

    Ok(PyRumpyArray::new(result))
}

// ============================================================================
// ediff1d - differences with prepend/append
// ============================================================================

/// Differences between consecutive elements with optional prepend/append.
#[pyfunction]
#[pyo3(signature = (ary, to_end=None, to_begin=None))]
pub fn ediff1d(
    ary: &PyRumpyArray,
    to_end: Option<&Bound<'_, pyo3::PyAny>>,
    to_begin: Option<&Bound<'_, pyo3::PyAny>>,
) -> PyResult<PyRumpyArray> {
    let inner = &ary.inner;
    let size = inner.size();
    let dtype = inner.dtype();

    // Get begin/end array references or scalars
    let begin_arr: Option<pyo3::PyRef<'_, PyRumpyArray>> = to_begin.and_then(|v| v.extract().ok());
    let end_arr: Option<pyo3::PyRef<'_, PyRumpyArray>> = to_end.and_then(|v| v.extract().ok());
    let begin_scalar: Option<f64> = if begin_arr.is_none() { to_begin.and_then(|v| v.extract().ok()) } else { None };
    let end_scalar: Option<f64> = if end_arr.is_none() { to_end.and_then(|v| v.extract().ok()) } else { None };

    let begin_len = begin_arr.as_ref().map(|a| a.inner.size()).unwrap_or(if begin_scalar.is_some() { 1 } else { 0 });
    let end_len = end_arr.as_ref().map(|a| a.inner.size()).unwrap_or(if end_scalar.is_some() { 1 } else { 0 });

    let diff_len = if size > 0 { size - 1 } else { 0 };
    let out_len = begin_len + diff_len + end_len;

    let mut result = RumpyArray::zeros(vec![out_len], dtype.clone());
    if out_len == 0 {
        return Ok(PyRumpyArray::new(result));
    }

    let buffer = Arc::get_mut(&mut result.buffer).expect("unique");
    let dst_ptr = buffer.as_mut_ptr();

    macro_rules! ediff1d_typed {
        ($ty:ty) => {{
            let dst = dst_ptr as *mut $ty;
            let mut idx = 0usize;

            // Write begin values
            if let Some(ref arr) = begin_arr {
                let offsets: Vec<isize> = arr.inner.iter_offsets().collect();
                for j in 0..begin_len {
                    unsafe { *dst.add(idx) = *(arr.inner.data_ptr().offset(offsets[j]) as *const $ty); }
                    idx += 1;
                }
            } else if let Some(scalar) = begin_scalar {
                unsafe { *dst.add(idx) = scalar as $ty; }
                idx += 1;
            }

            // Write differences
            if size > 1 {
                let src = inner.data_ptr() as *const $ty;
                if inner.is_c_contiguous() {
                    for i in 1..size {
                        unsafe { *dst.add(idx) = *src.add(i) - *src.add(i - 1); }
                        idx += 1;
                    }
                } else {
                    let offsets: Vec<isize> = inner.iter_offsets().collect();
                    for i in 1..size {
                        let curr = unsafe { *(inner.data_ptr().offset(offsets[i]) as *const $ty) };
                        let prev = unsafe { *(inner.data_ptr().offset(offsets[i - 1]) as *const $ty) };
                        unsafe { *dst.add(idx) = curr - prev; }
                        idx += 1;
                    }
                }
            }

            // Write end values
            if let Some(ref arr) = end_arr {
                let offsets: Vec<isize> = arr.inner.iter_offsets().collect();
                for j in 0..end_len {
                    unsafe { *dst.add(idx) = *(arr.inner.data_ptr().offset(offsets[j]) as *const $ty); }
                    idx += 1;
                }
            } else if let Some(scalar) = end_scalar {
                unsafe { *dst.add(idx) = scalar as $ty; }
            }
        }};
    }

    match dtype.kind() {
        DTypeKind::Float32 => ediff1d_typed!(f32),
        DTypeKind::Float64 => ediff1d_typed!(f64),
        DTypeKind::Int32 => ediff1d_typed!(i32),
        DTypeKind::Int64 => ediff1d_typed!(i64),
        DTypeKind::Uint32 => ediff1d_typed!(u32),
        DTypeKind::Uint64 => ediff1d_typed!(u64),
        DTypeKind::Int8 => ediff1d_typed!(i8),
        DTypeKind::Int16 => ediff1d_typed!(i16),
        DTypeKind::Uint8 => ediff1d_typed!(u8),
        DTypeKind::Uint16 => ediff1d_typed!(u16),
        _ => {
            // Complex types fallback
            let ops = dtype.ops();
            let mut idx = 0usize;

            // Write begin values
            if let Some(ref arr) = begin_arr {
                let arr_dtype = arr.inner.dtype();
                let arr_ops = arr_dtype.ops();
                let offsets: Vec<isize> = arr.inner.iter_offsets().collect();
                for j in 0..begin_len {
                    let val = unsafe { arr_ops.read_f64(arr.inner.data_ptr(), offsets[j]) }.unwrap_or(0.0);
                    unsafe { ops.write_f64(dst_ptr, idx, val); }
                    idx += 1;
                }
            } else if let Some(scalar) = begin_scalar {
                unsafe { ops.write_f64(dst_ptr, idx, scalar); }
                idx += 1;
            }

            // Write differences
            if size > 1 {
                let offsets: Vec<isize> = inner.iter_offsets().collect();
                for i in 1..size {
                    let curr = unsafe { ops.read_f64(inner.data_ptr(), offsets[i]) }.unwrap_or(0.0);
                    let prev = unsafe { ops.read_f64(inner.data_ptr(), offsets[i - 1]) }.unwrap_or(0.0);
                    unsafe { ops.write_f64(dst_ptr, idx, curr - prev); }
                    idx += 1;
                }
            }

            // Write end values
            if let Some(ref arr) = end_arr {
                let arr_dtype = arr.inner.dtype();
                let arr_ops = arr_dtype.ops();
                let offsets: Vec<isize> = arr.inner.iter_offsets().collect();
                for j in 0..end_len {
                    let val = unsafe { arr_ops.read_f64(arr.inner.data_ptr(), offsets[j]) }.unwrap_or(0.0);
                    unsafe { ops.write_f64(dst_ptr, idx, val); }
                    idx += 1;
                }
            } else if let Some(scalar) = end_scalar {
                unsafe { ops.write_f64(dst_ptr, idx, scalar); }
            }
        }
    }

    Ok(PyRumpyArray::new(result))
}

// ============================================================================
// unwrap - unwrap phase angles
// ============================================================================

/// Unwrap by changing deltas between values to 2*pi complement.
#[pyfunction]
#[pyo3(signature = (p, discont=None, axis=-1, period=None))]
pub fn unwrap(
    p: &PyRumpyArray,
    discont: Option<f64>,
    axis: isize,
    period: Option<f64>,
) -> PyResult<PyRumpyArray> {
    let inner = &p.inner;
    let ndim = inner.ndim();

    if ndim == 0 {
        return Ok(PyRumpyArray::new(inner.clone()));
    }

    let axis = resolve_axis(axis, ndim)?;
    let period = period.unwrap_or(2.0 * std::f64::consts::PI);
    let discont = discont.unwrap_or(period / 2.0);

    let shape = inner.shape().to_vec();
    let axis_len = shape[axis];

    if axis_len <= 1 {
        return Ok(PyRumpyArray::new(inner.copy()));
    }

    // unwrap always returns float64 (like numpy)
    let float_dtype = DType::float64();
    let src_values = inner.to_vec();
    let mut result = RumpyArray::zeros(inner.shape().to_vec(), float_dtype.clone());
    {
        let buffer = Arc::get_mut(&mut result.buffer).expect("unique");
        let ptr = buffer.as_mut_ptr() as *mut f64;
        for (i, &v) in src_values.iter().enumerate() {
            unsafe { *ptr.add(i) = v; }
        }
    }
    let buffer = Arc::get_mut(&mut result.buffer).expect("unique");
    let ptr = buffer.as_mut_ptr();
    let ops = float_dtype.ops();

    let axis_stride = result.strides()[axis];

    // Process each 1D slice along the axis
    for base_offset in result.axis_offsets(axis) {
        let mut cumulative_correction = 0.0;
        let prev_val = unsafe { ops.read_f64(ptr.offset(base_offset), 0) }.unwrap_or(0.0);
        let mut prev = prev_val;

        for i in 1..axis_len {
            let curr_ptr = unsafe { ptr.offset(base_offset + (i as isize) * axis_stride) };
            let curr_val = unsafe { ops.read_f64(curr_ptr, 0) }.unwrap_or(0.0);

            let diff = curr_val - prev;

            // Check for discontinuity
            if diff.abs() > discont {
                // Calculate correction to bring diff into [-period/2, period/2]
                let correction = ((diff + period / 2.0) / period).floor() * period;
                cumulative_correction -= correction;
            }

            let corrected = curr_val + cumulative_correction;
            unsafe { ops.write_f64(curr_ptr, 0, corrected); }
            prev = curr_val;
        }
    }

    Ok(PyRumpyArray::new(result))
}

// ============================================================================
// angle - phase angle of complex numbers
// ============================================================================

/// Return the angle of a complex argument.
#[pyfunction]
#[pyo3(signature = (z, deg=false))]
pub fn angle(z: &PyRumpyArray, deg: bool) -> PyResult<PyRumpyArray> {
    let inner = &z.inner;

    let mut result = RumpyArray::zeros(inner.shape().to_vec(), DType::float64());
    let buffer = Arc::get_mut(&mut result.buffer).expect("unique");
    let dst_ptr = buffer.as_mut_ptr() as *mut f64;

    // Check if input is complex
    let kind = inner.dtype().kind();
    let is_complex = matches!(kind, DTypeKind::Complex64 | DTypeKind::Complex128);

    if is_complex {
        // For complex, extract real and imag and compute atan2(imag, real)
        let ptr = inner.data_ptr();
        let offsets: Vec<isize> = inner.iter_offsets().collect();
        let itemsize = inner.itemsize();

        for (i, &offset) in offsets.iter().enumerate() {
            let elem_ptr = unsafe { ptr.offset(offset) };
            let (re, im) = if itemsize == 8 {
                // complex64: two f32
                let re = unsafe { *(elem_ptr as *const f32) } as f64;
                let im = unsafe { *(elem_ptr.add(4) as *const f32) } as f64;
                (re, im)
            } else {
                // complex128: two f64
                let re = unsafe { *(elem_ptr as *const f64) };
                let im = unsafe { *(elem_ptr.add(8) as *const f64) };
                (re, im)
            };

            let mut ang = im.atan2(re);
            if deg {
                ang = ang.to_degrees();
            }
            unsafe { *dst_ptr.add(i) = ang; }
        }
    } else {
        // For real, angle is 0 for positive, pi for negative
        let ptr = inner.data_ptr();
        let dtype = inner.dtype();
        let ops = dtype.ops();
        let offsets: Vec<isize> = inner.iter_offsets().collect();

        for (i, &offset) in offsets.iter().enumerate() {
            let val = unsafe { ops.read_f64(ptr, offset) }.unwrap_or(0.0);
            let mut ang = if val < 0.0 { std::f64::consts::PI } else { 0.0 };
            if deg {
                ang = ang.to_degrees();
            }
            unsafe { *dst_ptr.add(i) = ang; }
        }
    }

    Ok(PyRumpyArray::new(result))
}

// ============================================================================
// real_if_close - convert to real if imaginary is small
// ============================================================================

/// If complex input has zero or nearly-zero imaginary parts, return real.
#[pyfunction]
#[pyo3(signature = (a, tol=100.0))]
pub fn real_if_close(a: &PyRumpyArray, tol: f64) -> PyResult<PyRumpyArray> {
    let inner = &a.inner;

    let kind = inner.dtype().kind();
    let is_complex = matches!(kind, DTypeKind::Complex64 | DTypeKind::Complex128);
    if !is_complex {
        // Already real
        return Ok(PyRumpyArray::new(inner.clone()));
    }

    let ptr = inner.data_ptr();
    let itemsize = inner.itemsize();
    let offsets: Vec<isize> = inner.iter_offsets().collect();

    // Check if all imaginary parts are small
    // tol is multiplied by machine epsilon for the type
    let eps = if itemsize == 8 { f32::EPSILON as f64 } else { f64::EPSILON };
    let threshold = tol * eps;

    let mut all_close = true;
    for &offset in &offsets {
        let elem_ptr = unsafe { ptr.offset(offset) };
        let im = if itemsize == 8 {
            (unsafe { *(elem_ptr.add(4) as *const f32) }).abs() as f64
        } else {
            (unsafe { *(elem_ptr.add(8) as *const f64) }).abs()
        };

        if im > threshold {
            all_close = false;
            break;
        }
    }

    if all_close {
        // Convert to real - extract real parts
        let out_dtype = if itemsize == 8 { DType::float32() } else { DType::float64() };
        let mut result = RumpyArray::zeros(inner.shape().to_vec(), out_dtype);
        let buffer = Arc::get_mut(&mut result.buffer).expect("unique");
        let dst_ptr = buffer.as_mut_ptr();

        if itemsize == 8 {
            let dst = dst_ptr as *mut f32;
            for (i, &offset) in offsets.iter().enumerate() {
                let elem_ptr = unsafe { ptr.offset(offset) };
                let re = unsafe { *(elem_ptr as *const f32) };
                unsafe { *dst.add(i) = re; }
            }
        } else {
            let dst = dst_ptr as *mut f64;
            for (i, &offset) in offsets.iter().enumerate() {
                let elem_ptr = unsafe { ptr.offset(offset) };
                let re = unsafe { *(elem_ptr as *const f64) };
                unsafe { *dst.add(i) = re; }
            }
        }

        Ok(PyRumpyArray::new(result))
    } else {
        // Return original complex array
        Ok(PyRumpyArray::new(inner.clone()))
    }
}
