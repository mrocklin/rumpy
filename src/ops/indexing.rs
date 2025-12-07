//! Indexing operations: take, take_along_axis, compress, searchsorted, etc.

use std::sync::Arc;
use crate::array::{DType, RumpyArray};

/// Normalize a potentially negative index.
#[inline]
fn normalize_index(idx: isize, len: usize) -> Option<usize> {
    let normalized = if idx < 0 {
        (len as isize + idx) as usize
    } else {
        idx as usize
    };
    if normalized < len { Some(normalized) } else { None }
}

/// Take elements from an array along an axis.
pub fn take(arr: &RumpyArray, indices: &RumpyArray, axis: Option<usize>) -> Option<RumpyArray> {
    let axis = axis.unwrap_or(0);
    if axis >= arr.ndim() {
        return None;
    }

    let num_indices = indices.size();
    let axis_len = arr.shape()[axis];

    // Build output shape
    let mut out_shape = arr.shape().to_vec();
    out_shape[axis] = num_indices;

    let result = RumpyArray::zeros(out_shape.clone(), arr.dtype());
    if num_indices == 0 {
        return Some(result);
    }

    // Pre-collect and validate indices
    let idx_ptr = indices.data_ptr();
    let idx_dtype = indices.dtype();
    let idx_ops = idx_dtype.ops();
    let mut index_values: Vec<usize> = Vec::with_capacity(num_indices);
    for offset in indices.iter_offsets() {
        let val = unsafe { idx_ops.read_f64(idx_ptr, offset) }.unwrap_or(0.0) as isize;
        index_values.push(normalize_index(val, axis_len)?);
    }

    // Fast path: 1D contiguous case
    if arr.ndim() == 1 && arr.is_c_contiguous() {
        return take_1d_contiguous(arr, &index_values, result);
    }

    // Fast path: take along axis 0 with contiguous source
    if axis == 0 && arr.is_c_contiguous() {
        return take_axis0_contiguous(arr, &index_values, result);
    }

    // General path
    take_general(arr, &index_values, axis, result)
}

/// Fast path for 1D contiguous take - direct pointer indexing.
fn take_1d_contiguous(arr: &RumpyArray, indices: &[usize], mut result: RumpyArray) -> Option<RumpyArray> {
    let result_buffer = result.buffer_mut();
    let buffer = Arc::get_mut(result_buffer)?;
    let result_ptr = buffer.as_mut_ptr();
    let src_ptr = arr.data_ptr();
    let itemsize = arr.dtype().itemsize();

    // Direct memcpy per element - no stride computation
    for (i, &idx) in indices.iter().enumerate() {
        unsafe {
            std::ptr::copy_nonoverlapping(
                src_ptr.add(idx * itemsize),
                result_ptr.add(i * itemsize),
                itemsize,
            );
        }
    }
    Some(result)
}

/// Fast path for axis=0 take on contiguous array - copy entire rows.
fn take_axis0_contiguous(arr: &RumpyArray, indices: &[usize], mut result: RumpyArray) -> Option<RumpyArray> {
    let result_buffer = result.buffer_mut();
    let buffer = Arc::get_mut(result_buffer)?;
    let result_ptr = buffer.as_mut_ptr();
    let src_ptr = arr.data_ptr();

    // Each "row" is stride[0] bytes (for contiguous, this is product of remaining dims * itemsize)
    let row_bytes = arr.strides()[0] as usize;

    for (i, &idx) in indices.iter().enumerate() {
        unsafe {
            std::ptr::copy_nonoverlapping(
                src_ptr.add(idx * row_bytes),
                result_ptr.add(i * row_bytes),
                row_bytes,
            );
        }
    }
    Some(result)
}

/// General take implementation for non-contiguous or non-axis-0 cases.
fn take_general(arr: &RumpyArray, index_values: &[usize], axis: usize, mut result: RumpyArray) -> Option<RumpyArray> {
    let out_shape = result.shape().to_vec();
    let result_size: usize = out_shape.iter().product();

    let result_buffer = result.buffer_mut();
    let buffer = Arc::get_mut(result_buffer)?;
    let result_ptr = buffer.as_mut_ptr();
    let result_dtype = result.dtype();
    let ops = result_dtype.ops();
    let src_ptr = arr.data_ptr();

    // Pre-compute strides for manual offset calculation (avoids allocation per iteration)
    let arr_strides = arr.strides();
    let ndim = arr.ndim();

    let mut out_indices = vec![0usize; ndim];
    let mut src_indices = vec![0usize; ndim];

    for out_idx in 0..result_size {
        // Copy indices and substitute axis
        src_indices.copy_from_slice(&out_indices);
        src_indices[axis] = index_values[out_indices[axis]];

        // Compute offset inline (avoid byte_offset_for call)
        let src_offset: isize = src_indices.iter()
            .zip(arr_strides.iter())
            .map(|(&i, &s)| i as isize * s)
            .sum();

        unsafe { ops.copy_element(src_ptr, src_offset, result_ptr, out_idx); }

        // Increment indices (inline for speed)
        for d in (0..ndim).rev() {
            out_indices[d] += 1;
            if out_indices[d] < out_shape[d] {
                break;
            }
            out_indices[d] = 0;
        }
    }

    Some(result)
}

/// Take values along an axis using an index array.
pub fn take_along_axis(arr: &RumpyArray, indices: &RumpyArray, axis: usize) -> Option<RumpyArray> {
    if axis >= arr.ndim() || indices.ndim() != arr.ndim() {
        return None;
    }

    // Validate shape compatibility
    for i in 0..arr.ndim() {
        if i != axis && arr.shape()[i] != indices.shape()[i] {
            return None;
        }
    }

    let axis_len = arr.shape()[axis];
    let out_shape = indices.shape().to_vec();
    let mut result = RumpyArray::zeros(out_shape.clone(), arr.dtype());

    let result_size: usize = out_shape.iter().product();
    if result_size == 0 {
        return Some(result);
    }

    let result_buffer = result.buffer_mut();
    let buffer = Arc::get_mut(result_buffer)?;
    let result_ptr = buffer.as_mut_ptr();
    let result_dtype = result.dtype();
    let ops = result_dtype.ops();
    let src_ptr = arr.data_ptr();
    let idx_ptr = indices.data_ptr();
    let idx_dtype = indices.dtype();
    let idx_ops = idx_dtype.ops();

    let arr_strides = arr.strides();
    let idx_strides = indices.strides();
    let ndim = arr.ndim();

    let mut out_indices = vec![0usize; ndim];
    let mut src_indices = vec![0usize; ndim];

    for out_idx in 0..result_size {
        // Compute index array offset
        let idx_offset: isize = out_indices.iter()
            .zip(idx_strides.iter())
            .map(|(&i, &s)| i as isize * s)
            .sum();

        let idx_val = unsafe { idx_ops.read_f64(idx_ptr, idx_offset) }.unwrap_or(0.0) as isize;
        let normalized_idx = normalize_index(idx_val, axis_len)?;

        // Build source indices
        src_indices.copy_from_slice(&out_indices);
        src_indices[axis] = normalized_idx;

        let src_offset: isize = src_indices.iter()
            .zip(arr_strides.iter())
            .map(|(&i, &s)| i as isize * s)
            .sum();

        unsafe { ops.copy_element(src_ptr, src_offset, result_ptr, out_idx); }

        // Increment indices
        for d in (0..ndim).rev() {
            out_indices[d] += 1;
            if out_indices[d] < out_shape[d] {
                break;
            }
            out_indices[d] = 0;
        }
    }

    Some(result)
}

/// Select elements using a boolean condition along an axis.
pub fn compress(condition: &[bool], arr: &RumpyArray, axis: Option<usize>) -> Option<RumpyArray> {
    match axis {
        None => compress_flat(condition, arr),
        Some(ax) => compress_axis(condition, arr, ax),
    }
}

fn compress_flat(condition: &[bool], arr: &RumpyArray) -> Option<RumpyArray> {
    let size = arr.size();
    let cond_len = condition.len().min(size);
    let count = condition.iter().take(cond_len).filter(|&&b| b).count();

    let mut result = RumpyArray::zeros(vec![count], arr.dtype());
    if count == 0 {
        return Some(result);
    }

    // Fast path for contiguous 1D
    if arr.ndim() == 1 && arr.is_c_contiguous() {
        let result_buffer = result.buffer_mut();
        let buffer = Arc::get_mut(result_buffer)?;
        let result_ptr = buffer.as_mut_ptr();
        let src_ptr = arr.data_ptr();
        let itemsize = arr.dtype().itemsize();

        let mut dst_idx = 0;
        for (i, &cond) in condition.iter().take(cond_len).enumerate() {
            if cond {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        src_ptr.add(i * itemsize),
                        result_ptr.add(dst_idx * itemsize),
                        itemsize,
                    );
                }
                dst_idx += 1;
            }
        }
        return Some(result);
    }

    // General path
    let result_buffer = result.buffer_mut();
    let buffer = Arc::get_mut(result_buffer)?;
    let result_ptr = buffer.as_mut_ptr();
    let result_dtype = result.dtype();
    let ops = result_dtype.ops();
    let src_ptr = arr.data_ptr();

    let mut dst_idx = 0;
    for (i, offset) in arr.iter_offsets().take(cond_len).enumerate() {
        if condition[i] {
            unsafe { ops.copy_element(src_ptr, offset, result_ptr, dst_idx); }
            dst_idx += 1;
        }
    }

    Some(result)
}

fn compress_axis(condition: &[bool], arr: &RumpyArray, axis: usize) -> Option<RumpyArray> {
    if axis >= arr.ndim() {
        return None;
    }

    let axis_len = arr.shape()[axis];
    let cond_len = condition.len().min(axis_len);
    let count = condition.iter().take(cond_len).filter(|&&b| b).count();

    let mut out_shape = arr.shape().to_vec();
    out_shape[axis] = count;

    let mut result = RumpyArray::zeros(out_shape.clone(), arr.dtype());
    if count == 0 {
        return Some(result);
    }

    // Build index mapping once
    let cond_to_src: Vec<usize> = condition.iter()
        .take(cond_len)
        .enumerate()
        .filter_map(|(i, &b)| if b { Some(i) } else { None })
        .collect();

    let result_buffer = result.buffer_mut();
    let buffer = Arc::get_mut(result_buffer)?;
    let result_ptr = buffer.as_mut_ptr();
    let result_dtype = result.dtype();
    let ops = result_dtype.ops();
    let src_ptr = arr.data_ptr();
    let arr_strides = arr.strides();
    let ndim = arr.ndim();

    let result_size: usize = out_shape.iter().product();
    let mut out_indices = vec![0usize; ndim];
    let mut src_indices = vec![0usize; ndim];

    for out_idx in 0..result_size {
        src_indices.copy_from_slice(&out_indices);
        src_indices[axis] = cond_to_src[out_indices[axis]];

        let src_offset: isize = src_indices.iter()
            .zip(arr_strides.iter())
            .map(|(&i, &s)| i as isize * s)
            .sum();

        unsafe { ops.copy_element(src_ptr, src_offset, result_ptr, out_idx); }

        for d in (0..ndim).rev() {
            out_indices[d] += 1;
            if out_indices[d] < out_shape[d] {
                break;
            }
            out_indices[d] = 0;
        }
    }

    Some(result)
}

/// Find indices where elements should be inserted to maintain order.
pub fn searchsorted(a: &RumpyArray, v: &RumpyArray, side: &str) -> Option<RumpyArray> {
    if a.ndim() != 1 {
        return None;
    }

    let a_len = a.size();
    let v_size = v.size();

    // Collect sorted array values (required for binary search)
    let a_ptr = a.data_ptr();
    let a_dtype = a.dtype();
    let a_ops = a_dtype.ops();

    // Fast path for contiguous sorted array
    let a_vals: Vec<f64> = if a.is_c_contiguous() {
        (0..a_len)
            .map(|i| unsafe { a_ops.read_f64(a_ptr, (i * a_dtype.itemsize()) as isize) }.unwrap_or(0.0))
            .collect()
    } else {
        a.iter_offsets()
            .map(|offset| unsafe { a_ops.read_f64(a_ptr, offset) }.unwrap_or(0.0))
            .collect()
    };

    let mut result = RumpyArray::zeros(v.shape().to_vec(), DType::int64());
    if v_size == 0 {
        return Some(result);
    }

    let result_buffer = result.buffer_mut();
    let buffer = Arc::get_mut(result_buffer)?;
    let result_ptr = buffer.as_mut_ptr() as *mut i64;

    let v_ptr = v.data_ptr();
    let v_dtype = v.dtype();
    let v_ops = v_dtype.ops();
    let use_left = side != "right";

    for (i, offset) in v.iter_offsets().enumerate() {
        let val = unsafe { v_ops.read_f64(v_ptr, offset) }.unwrap_or(0.0);
        let idx = if use_left {
            a_vals.partition_point(|&x| x < val)
        } else {
            a_vals.partition_point(|&x| x <= val)
        };
        unsafe { *result_ptr.add(i) = idx as i64; }
    }

    Some(result)
}

/// Return indices of elements that are non-zero.
pub fn argwhere(arr: &RumpyArray) -> RumpyArray {
    let size = arr.size();
    let ndim = arr.ndim();

    if size == 0 {
        return RumpyArray::zeros(vec![0, ndim.max(1)], DType::int64());
    }

    let ptr = arr.data_ptr();
    let dtype = arr.dtype();
    let ops = dtype.ops();

    // First pass: count nonzero elements
    let count = arr.iter_offsets()
        .filter(|&offset| unsafe { ops.is_truthy(ptr, offset) })
        .count();

    if count == 0 {
        return RumpyArray::zeros(vec![0, ndim.max(1)], DType::int64());
    }

    let out_ndim = ndim.max(1);
    let mut result = RumpyArray::zeros(vec![count, out_ndim], DType::int64());
    let result_buffer = result.buffer_mut();
    let buffer = Arc::get_mut(result_buffer).expect("buffer must be unique");
    let result_ptr = buffer.as_mut_ptr() as *mut i64;

    // Second pass: collect indices
    let shape = arr.shape();
    let mut indices = vec![0usize; ndim];
    let mut result_row = 0;

    for _ in 0..size {
        let offset = arr.byte_offset_for(&indices);
        if unsafe { ops.is_truthy(ptr, offset) } {
            for (j, &idx) in indices.iter().enumerate() {
                unsafe { *result_ptr.add(result_row * out_ndim + j) = idx as i64; }
            }
            result_row += 1;
        }

        // Increment indices
        for d in (0..ndim).rev() {
            indices[d] += 1;
            if indices[d] < shape[d] {
                break;
            }
            indices[d] = 0;
        }
    }

    result
}

/// Return indices that are non-zero in the flattened array.
pub fn flatnonzero(arr: &RumpyArray) -> RumpyArray {
    let size = arr.size();
    if size == 0 {
        return RumpyArray::zeros(vec![0], DType::int64());
    }

    let ptr = arr.data_ptr();
    let dtype = arr.dtype();
    let ops = dtype.ops();

    // Two-pass approach: count then fill (avoids Vec<usize> allocation)
    let count = arr.iter_offsets()
        .filter(|&offset| unsafe { ops.is_truthy(ptr, offset) })
        .count();

    let mut result = RumpyArray::zeros(vec![count], DType::int64());
    if count == 0 {
        return result;
    }

    let result_buffer = result.buffer_mut();
    let buffer = Arc::get_mut(result_buffer).expect("buffer must be unique");
    let result_ptr = buffer.as_mut_ptr() as *mut i64;

    let mut result_idx = 0;
    for (i, offset) in arr.iter_offsets().enumerate() {
        if unsafe { ops.is_truthy(ptr, offset) } {
            unsafe { *result_ptr.add(result_idx) = i as i64; }
            result_idx += 1;
        }
    }

    result
}

/// Replace elements at given flat indices with given values.
pub fn put(arr: &mut RumpyArray, indices: &[i64], values: &[f64]) -> Option<()> {
    let size = arr.size();
    if indices.is_empty() {
        return Some(());
    }

    // Normalize and validate all indices first
    let normalized: Vec<usize> = indices.iter()
        .map(|&idx| normalize_index(idx as isize, size))
        .collect::<Option<Vec<_>>>()?;

    let ndim = arr.ndim();
    let shape = arr.shape().to_vec();
    let values_len = values.len();

    for (i, &flat_idx) in normalized.iter().enumerate() {
        let val = values[i % values_len];

        // Convert flat index to nd indices
        let mut indices_nd = vec![0usize; ndim];
        let mut remaining = flat_idx;
        for d in (0..ndim).rev() {
            indices_nd[d] = remaining % shape[d];
            remaining /= shape[d];
        }

        arr.set_element(&indices_nd, val);
    }

    Some(())
}

/// Put values into array along an axis using indices.
pub fn put_along_axis(arr: &mut RumpyArray, indices: &RumpyArray, values: &RumpyArray, axis: usize) -> Option<()> {
    if axis >= arr.ndim() || indices.shape() != values.shape() || indices.ndim() != arr.ndim() {
        return None;
    }

    for i in 0..arr.ndim() {
        if i != axis && arr.shape()[i] != indices.shape()[i] {
            return None;
        }
    }

    let axis_len = arr.shape()[axis];
    let idx_ptr = indices.data_ptr();
    let idx_dtype = indices.dtype();
    let idx_ops = idx_dtype.ops();
    let val_ptr = values.data_ptr();
    let val_dtype = values.dtype();
    let val_ops = val_dtype.ops();

    let idx_strides = indices.strides();
    let val_strides = values.strides();
    let ndim = indices.ndim();
    let shape = indices.shape().to_vec();
    let idx_size = indices.size();

    let mut idx_indices = vec![0usize; ndim];
    let mut target_indices = vec![0usize; ndim];

    for _ in 0..idx_size {
        // Compute offsets inline
        let idx_offset: isize = idx_indices.iter()
            .zip(idx_strides.iter())
            .map(|(&i, &s)| i as isize * s)
            .sum();
        let val_offset: isize = idx_indices.iter()
            .zip(val_strides.iter())
            .map(|(&i, &s)| i as isize * s)
            .sum();

        let idx_val = unsafe { idx_ops.read_f64(idx_ptr, idx_offset) }.unwrap_or(0.0) as isize;
        let normalized_idx = normalize_index(idx_val, axis_len)?;

        let val = unsafe { val_ops.read_f64(val_ptr, val_offset) }.unwrap_or(0.0);

        target_indices.copy_from_slice(&idx_indices);
        target_indices[axis] = normalized_idx;

        arr.set_element(&target_indices, val);

        for d in (0..ndim).rev() {
            idx_indices[d] += 1;
            if idx_indices[d] < shape[d] {
                break;
            }
            idx_indices[d] = 0;
        }
    }

    Some(())
}

/// Construct an array from index array and a list of arrays to choose from.
pub fn choose(indices: &RumpyArray, choices: &[RumpyArray]) -> Option<RumpyArray> {
    if choices.is_empty() {
        return None;
    }

    let num_choices = choices.len();
    let out_shape = indices.shape().to_vec();
    let out_dtype = choices[0].dtype();

    for choice in choices {
        if choice.shape() != indices.shape() {
            return None;
        }
    }

    let mut result = RumpyArray::zeros(out_shape.clone(), out_dtype);
    let result_size: usize = out_shape.iter().product();
    if result_size == 0 {
        return Some(result);
    }

    let result_buffer = result.buffer_mut();
    let buffer = Arc::get_mut(result_buffer)?;
    let result_ptr = buffer.as_mut_ptr();
    let result_dtype = result.dtype();
    let ops = result_dtype.ops();

    let idx_ptr = indices.data_ptr();
    let idx_dtype = indices.dtype();
    let idx_ops = idx_dtype.ops();
    let idx_strides = indices.strides();
    let ndim = indices.ndim();

    let mut out_indices = vec![0usize; ndim];

    for out_idx in 0..result_size {
        let idx_offset: isize = out_indices.iter()
            .zip(idx_strides.iter())
            .map(|(&i, &s)| i as isize * s)
            .sum();

        let choice_idx = unsafe { idx_ops.read_f64(idx_ptr, idx_offset) }.unwrap_or(0.0) as usize;
        if choice_idx >= num_choices {
            return None;
        }

        let choice_arr = &choices[choice_idx];
        let choice_ptr = choice_arr.data_ptr();
        let choice_strides = choice_arr.strides();

        let choice_offset: isize = out_indices.iter()
            .zip(choice_strides.iter())
            .map(|(&i, &s)| i as isize * s)
            .sum();

        unsafe { ops.copy_element(choice_ptr, choice_offset, result_ptr, out_idx); }

        for d in (0..ndim).rev() {
            out_indices[d] += 1;
            if out_indices[d] < out_shape[d] {
                break;
            }
            out_indices[d] = 0;
        }
    }

    Some(result)
}
