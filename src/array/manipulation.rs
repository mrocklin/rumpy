//! Array manipulation functions: concatenate, stack, split, tile, repeat, pad, roll, etc.

use std::sync::Arc;

use super::dtype::DType;
use super::{increment_indices, RumpyArray};

/// Concatenate arrays along an axis.
/// All arrays must have same shape except in the concatenation axis.
pub fn concatenate(arrays: &[RumpyArray], axis: usize) -> Option<RumpyArray> {
    if arrays.is_empty() {
        return None;
    }

    let first = &arrays[0];
    let ndim = first.ndim();

    if axis >= ndim {
        return None;
    }

    // Check all arrays have same shape except for concat axis
    for arr in arrays.iter().skip(1) {
        if arr.ndim() != ndim {
            return None;
        }
        for (i, (&a, &b)) in first.shape().iter().zip(arr.shape().iter()).enumerate() {
            if i != axis && a != b {
                return None;
            }
        }
    }

    // Compute output shape
    let mut out_shape = first.shape().to_vec();
    out_shape[axis] = arrays.iter().map(|a| a.shape()[axis]).sum();

    // Use dtype of first array (TODO: dtype promotion)
    let dtype = first.dtype();
    let mut result = RumpyArray::zeros(out_shape, dtype);

    if result.size() == 0 {
        return Some(result);
    }

    let buffer = Arc::get_mut(&mut result.buffer).expect("buffer must be unique");
    let result_ptr = buffer.as_mut_ptr();
    let ops = result.dtype.ops();

    // Copy data from each array
    let mut axis_offset = 0usize;
    for arr in arrays {
        let size = arr.size();
        let src_ptr = arr.data_ptr();
        let mut src_indices = vec![0usize; ndim];
        for _ in 0..size {
            // Compute destination indices
            let mut dst_indices = src_indices.clone();
            dst_indices[axis] += axis_offset;

            // Compute linear index in result
            let mut dst_idx = 0;
            let mut stride = 1;
            for i in (0..ndim).rev() {
                dst_idx += dst_indices[i] * stride;
                stride *= result.shape()[i];
            }

            let src_offset = arr.byte_offset_for(&src_indices);
            unsafe { ops.copy_element(src_ptr, src_offset, result_ptr, dst_idx); }

            increment_indices(&mut src_indices, arr.shape());
        }
        axis_offset += arr.shape()[axis];
    }

    Some(result)
}

/// Stack arrays along a new axis.
/// All arrays must have the same shape.
pub fn stack(arrays: &[RumpyArray], axis: usize) -> Option<RumpyArray> {
    if arrays.is_empty() {
        return None;
    }

    let first = &arrays[0];

    // axis can be at most ndim (to add at end)
    if axis > first.ndim() {
        return None;
    }

    // Check all arrays have same shape
    for arr in arrays.iter().skip(1) {
        if arr.shape() != first.shape() {
            return None;
        }
    }

    // Expand dims on each array, then concatenate
    let expanded: Vec<RumpyArray> = arrays
        .iter()
        .filter_map(|a| a.expand_dims(axis))
        .collect();

    if expanded.len() != arrays.len() {
        return None;
    }

    concatenate(&expanded, axis)
}

/// Split array into equal parts along axis.
/// Returns None if array cannot be split evenly.
pub fn split(arr: &RumpyArray, num_sections: usize, axis: usize) -> Option<Vec<RumpyArray>> {
    if num_sections == 0 || axis >= arr.ndim() {
        return None;
    }

    let axis_len = arr.shape()[axis];
    if !axis_len.is_multiple_of(num_sections) {
        return None; // Must divide evenly
    }

    let section_size = axis_len / num_sections;
    Some(split_into_sizes(arr, &vec![section_size; num_sections], axis))
}

/// Split array into sections, allowing unequal sizes.
/// For `n % num_sections` leftover elements, first sections get one extra.
pub fn array_split(arr: &RumpyArray, num_sections: usize, axis: usize) -> Option<Vec<RumpyArray>> {
    if num_sections == 0 || axis >= arr.ndim() {
        return None;
    }

    let axis_len = arr.shape()[axis];
    let base_size = axis_len / num_sections;
    let remainder = axis_len % num_sections;

    // First `remainder` sections get base_size + 1, rest get base_size
    let sizes: Vec<usize> = (0..num_sections)
        .map(|i| if i < remainder { base_size + 1 } else { base_size })
        .collect();

    Some(split_into_sizes(arr, &sizes, axis))
}

/// Helper to split array into sections with given sizes along axis.
/// Returns views (no copy) like numpy.
fn split_into_sizes(arr: &RumpyArray, sizes: &[usize], axis: usize) -> Vec<RumpyArray> {
    let mut result = Vec::with_capacity(sizes.len());
    let mut start = 0isize;

    for &size in sizes {
        let section = arr.slice_axis(axis, start, start + size as isize, 1);
        result.push(section); // Return view, not copy
        start += size as isize;
    }

    result
}

/// Compute broadcast shape from two input shapes.
/// Returns None if shapes are incompatible.
pub fn broadcast_shapes(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
    let max_ndim = a.len().max(b.len());
    let mut result = vec![0usize; max_ndim];

    // Align from the right
    for i in 0..max_ndim {
        let a_dim = if i < a.len() { a[a.len() - 1 - i] } else { 1 };
        let b_dim = if i < b.len() { b[b.len() - 1 - i] } else { 1 };

        if a_dim == b_dim {
            result[max_ndim - 1 - i] = a_dim;
        } else if a_dim == 1 {
            result[max_ndim - 1 - i] = b_dim;
        } else if b_dim == 1 {
            result[max_ndim - 1 - i] = a_dim;
        } else {
            return None; // Incompatible
        }
    }
    Some(result)
}

/// Count occurrences of each value in an array of non-negative integers.
/// Returns an array of counts where result[i] = number of times i appears in x.
pub fn bincount(x: &RumpyArray, minlength: usize) -> Option<RumpyArray> {
    // Must be 1D
    if x.ndim() != 1 {
        return None;
    }

    let size = x.size();
    if size == 0 {
        return Some(RumpyArray::zeros(vec![minlength.max(0)], DType::int64()));
    }

    let ptr = x.data_ptr();
    let dtype = x.dtype();
    let ops = dtype.ops();

    // Find max value to determine output size
    let mut max_val: i64 = 0;
    for offset in x.iter_offsets() {
        let val = unsafe { ops.read_f64(ptr, offset) }.unwrap_or(0.0);
        let ival = val as i64;
        if ival < 0 {
            return None; // Negative values not allowed
        }
        if ival > max_val {
            max_val = ival;
        }
    }

    let out_len = (max_val as usize + 1).max(minlength);
    let mut counts = vec![0i64; out_len];

    // Count occurrences
    for offset in x.iter_offsets() {
        let val = unsafe { ops.read_f64(ptr, offset) }.unwrap_or(0.0);
        let idx = val as usize;
        counts[idx] += 1;
    }

    // Convert to RumpyArray
    let mut result = RumpyArray::zeros(vec![out_len], DType::int64());
    let buffer = result.buffer_mut();
    let result_buffer = std::sync::Arc::get_mut(buffer).expect("unique");
    let result_ptr = result_buffer.as_mut_ptr() as *mut i64;
    for (i, &count) in counts.iter().enumerate() {
        unsafe { *result_ptr.add(i) = count; }
    }

    Some(result)
}

/// Compute the q-th percentile(s) of the data.
/// q values should be in [0, 100].
/// Uses linear interpolation (numpy default method).
pub fn percentile(arr: &RumpyArray, q: &[f64], axis: Option<usize>) -> Option<RumpyArray> {
    match axis {
        None => percentile_flat(arr, q),
        Some(ax) => percentile_axis(arr, q, ax),
    }
}

/// Percentile over flattened array.
fn percentile_flat(arr: &RumpyArray, q: &[f64]) -> Option<RumpyArray> {
    let size = arr.size();
    if size == 0 {
        return Some(RumpyArray::zeros(vec![q.len()], DType::float64()));
    }

    // Collect and sort values
    let ptr = arr.data_ptr();
    let dtype = arr.dtype();
    let ops = dtype.ops();

    let mut values: Vec<f64> = Vec::with_capacity(size);
    for offset in arr.iter_offsets() {
        values.push(unsafe { ops.read_f64(ptr, offset) }.unwrap_or(0.0));
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Compute percentiles using linear interpolation
    let mut result = RumpyArray::zeros(vec![q.len()], DType::float64());
    let buffer = result.buffer_mut();
    let result_buffer = std::sync::Arc::get_mut(buffer).expect("unique");
    let result_ptr = result_buffer.as_mut_ptr() as *mut f64;

    for (i, &pct) in q.iter().enumerate() {
        let val = interpolate_percentile(&values, pct);
        unsafe { *result_ptr.add(i) = val; }
    }

    Some(result)
}

/// Percentile along axis.
fn percentile_axis(arr: &RumpyArray, q: &[f64], axis: usize) -> Option<RumpyArray> {
    if axis >= arr.ndim() {
        return None;
    }

    let shape = arr.shape();
    let axis_len = shape[axis];
    let axis_stride = arr.strides()[axis];

    // Output shape: remove axis, prepend q.len() if multiple percentiles
    let mut reduced_shape: Vec<usize> = shape[..axis].to_vec();
    reduced_shape.extend_from_slice(&shape[axis + 1..]);
    if reduced_shape.is_empty() {
        reduced_shape = vec![1];
    }

    let out_shape = if q.len() == 1 {
        reduced_shape.clone()
    } else {
        let mut s = vec![q.len()];
        s.extend(&reduced_shape);
        s
    };

    let reduced_size: usize = reduced_shape.iter().product();
    if axis_len == 0 || reduced_size == 0 {
        return Some(RumpyArray::zeros(out_shape, DType::float64()));
    }

    let mut result = RumpyArray::zeros(out_shape, DType::float64());
    let buffer = result.buffer_mut();
    let result_buffer = std::sync::Arc::get_mut(buffer).expect("unique");
    let result_ptr = result_buffer.as_mut_ptr() as *mut f64;

    let src_ptr = arr.data_ptr();
    let dtype = arr.dtype();
    let ops = dtype.ops();

    // Use axis_offsets iterator (same pattern as reduce_axis_op)
    for (i, base_offset) in arr.axis_offsets(axis).enumerate() {
        // Collect values along axis from this base offset
        let mut values: Vec<f64> = Vec::with_capacity(axis_len);
        let mut ptr = unsafe { src_ptr.offset(base_offset) };
        for _ in 0..axis_len {
            values.push(unsafe { ops.read_f64(ptr, 0) }.unwrap_or(0.0));
            ptr = unsafe { ptr.offset(axis_stride) };
        }
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Compute percentiles
        if q.len() == 1 {
            unsafe { *result_ptr.add(i) = interpolate_percentile(&values, q[0]); }
        } else {
            for (qi, &pct) in q.iter().enumerate() {
                unsafe { *result_ptr.add(qi * reduced_size + i) = interpolate_percentile(&values, pct); }
            }
        }
    }

    Some(result)
}

/// Linear interpolation for percentile.
fn interpolate_percentile(sorted: &[f64], pct: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    if n == 1 {
        return sorted[0];
    }

    // Convert percentile to index (numpy's linear interpolation method)
    let idx = (pct / 100.0) * (n - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;

    if lo == hi || hi >= n {
        sorted[lo.min(n - 1)]
    } else {
        let frac = idx - lo as f64;
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }
}

// ============================================================================
// NaN-aware percentile/quantile
// ============================================================================

/// Compute the q-th percentile(s) of the data, ignoring NaN values.
/// q values should be in [0, 100].
pub fn nanpercentile(arr: &RumpyArray, q: &[f64], axis: Option<usize>) -> Option<RumpyArray> {
    match axis {
        None => nanpercentile_flat(arr, q),
        Some(ax) => nanpercentile_axis(arr, q, ax),
    }
}

/// Nanpercentile over flattened array.
fn nanpercentile_flat(arr: &RumpyArray, q: &[f64]) -> Option<RumpyArray> {
    // Collect non-NaN values
    let mut values: Vec<f64> = arr.to_vec().into_iter().filter(|v| !v.is_nan()).collect();

    if values.is_empty() {
        // All NaN - return NaN for each percentile
        // Single q -> scalar-like output; multiple q -> 1D array
        let shape = if q.len() == 1 { vec![1] } else { vec![q.len()] };
        return Some(RumpyArray::full(shape, f64::NAN, DType::float64()));
    }

    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Compute percentiles using linear interpolation
    // Match numpy behavior: single q returns scalar-like, multiple q returns 1D array
    let out_shape = if q.len() == 1 { vec![1] } else { vec![q.len()] };
    let mut result = RumpyArray::zeros(out_shape, DType::float64());
    let buffer = result.buffer_mut();
    let result_buffer = std::sync::Arc::get_mut(buffer).expect("unique");
    let result_ptr = result_buffer.as_mut_ptr() as *mut f64;

    for (i, &pct) in q.iter().enumerate() {
        let val = interpolate_percentile(&values, pct);
        unsafe { *result_ptr.add(i) = val; }
    }

    Some(result)
}

/// Nanpercentile along axis.
fn nanpercentile_axis(arr: &RumpyArray, q: &[f64], axis: usize) -> Option<RumpyArray> {
    if axis >= arr.ndim() {
        return None;
    }

    let shape = arr.shape();
    let axis_len = shape[axis];
    let axis_stride = arr.strides()[axis];

    // Output shape: remove axis, prepend q.len() if multiple percentiles
    let mut reduced_shape: Vec<usize> = shape[..axis].to_vec();
    reduced_shape.extend_from_slice(&shape[axis + 1..]);
    if reduced_shape.is_empty() {
        reduced_shape = vec![1];
    }

    let out_shape = if q.len() == 1 {
        reduced_shape.clone()
    } else {
        let mut s = vec![q.len()];
        s.extend(&reduced_shape);
        s
    };

    let reduced_size: usize = reduced_shape.iter().product();
    if axis_len == 0 || reduced_size == 0 {
        return Some(RumpyArray::zeros(out_shape, DType::float64()));
    }

    let mut result = RumpyArray::zeros(out_shape, DType::float64());
    let buffer = result.buffer_mut();
    let result_buffer = std::sync::Arc::get_mut(buffer).expect("unique");
    let result_ptr = result_buffer.as_mut_ptr() as *mut f64;

    let src_ptr = arr.data_ptr();
    let dtype = arr.dtype();
    let ops = dtype.ops();

    // Use axis_offsets iterator
    for (i, base_offset) in arr.axis_offsets(axis).enumerate() {
        // Collect non-NaN values along axis from this base offset
        let mut values: Vec<f64> = Vec::with_capacity(axis_len);
        let mut ptr = unsafe { src_ptr.offset(base_offset) };
        for _ in 0..axis_len {
            let v = unsafe { ops.read_f64(ptr, 0) }.unwrap_or(0.0);
            if !v.is_nan() {
                values.push(v);
            }
            ptr = unsafe { ptr.offset(axis_stride) };
        }
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Compute percentiles (NaN if all-NaN slice)
        if q.len() == 1 {
            let val = if values.is_empty() { f64::NAN } else { interpolate_percentile(&values, q[0]) };
            unsafe { *result_ptr.add(i) = val; }
        } else {
            for (qi, &pct) in q.iter().enumerate() {
                let val = if values.is_empty() { f64::NAN } else { interpolate_percentile(&values, pct) };
                unsafe { *result_ptr.add(qi * reduced_size + i) = val; }
            }
        }
    }

    Some(result)
}

/// Create logarithmically spaced array.
/// Returns base^linspace(start, stop, num).
pub fn logspace(start: f64, stop: f64, num: usize, base: f64, dtype: DType) -> RumpyArray {
    let mut arr = RumpyArray::zeros(vec![num], dtype);
    if num == 0 {
        return arr;
    }

    let buffer = Arc::get_mut(&mut arr.buffer).expect("buffer must be unique");
    let ptr = buffer.as_mut_ptr();
    let ops = arr.dtype.ops();

    let step = if num > 1 { (stop - start) / (num - 1) as f64 } else { 0.0 };

    for i in 0..num {
        let exponent = start + (i as f64) * step;
        let val = base.powf(exponent);
        unsafe { ops.write_f64(ptr, i, val); }
    }
    arr
}

/// Create geometrically spaced array.
/// Returns numbers spaced evenly on a log scale between start and stop.
pub fn geomspace(start: f64, stop: f64, num: usize, dtype: DType) -> Option<RumpyArray> {
    // start and stop must have same sign (and not zero)
    if start * stop <= 0.0 {
        return None;
    }

    let mut arr = RumpyArray::zeros(vec![num], dtype);
    if num == 0 {
        return Some(arr);
    }

    let buffer = Arc::get_mut(&mut arr.buffer).expect("buffer must be unique");
    let ptr = buffer.as_mut_ptr();
    let ops = arr.dtype.ops();

    // Handle negative values by using absolute values and restoring sign
    let sign = if start < 0.0 { -1.0 } else { 1.0 };
    let log_start = start.abs().ln();
    let log_stop = stop.abs().ln();
    let step = if num > 1 { (log_stop - log_start) / (num - 1) as f64 } else { 0.0 };

    for i in 0..num {
        let val = sign * (log_start + (i as f64) * step).exp();
        unsafe { ops.write_f64(ptr, i, val); }
    }
    Some(arr)
}

/// Create a triangular matrix of ones.
/// k: diagonal offset (0 = main diagonal, positive = above, negative = below)
pub fn tri(n: usize, m: usize, k: isize, dtype: DType) -> RumpyArray {
    let mut arr = RumpyArray::zeros(vec![n, m], dtype);
    if n == 0 || m == 0 {
        return arr;
    }

    let buffer = Arc::get_mut(&mut arr.buffer).expect("buffer must be unique");
    let ptr = buffer.as_mut_ptr();
    let ops = arr.dtype.ops();

    for i in 0..n {
        for j in 0..m {
            // Element is 1 if j <= i + k
            if (j as isize) <= (i as isize) + k {
                let idx = i * m + j;
                unsafe { ops.write_one(ptr, idx); }
            }
        }
    }
    arr
}

/// Helper for tril/triu - extract triangle from 2D array.
fn extract_triangle(arr: &RumpyArray, k: isize, lower: bool) -> Option<RumpyArray> {
    if arr.ndim() != 2 {
        return None;
    }

    let shape = arr.shape();
    let n = shape[0];
    let m = shape[1];
    let mut result = RumpyArray::zeros(shape.to_vec(), arr.dtype());

    if n == 0 || m == 0 {
        return Some(result);
    }

    let buffer = Arc::get_mut(&mut result.buffer).expect("buffer must be unique");
    let result_ptr = buffer.as_mut_ptr();
    let itemsize = arr.itemsize();

    // Fast path: contiguous source array - use memcpy for row segments
    if arr.is_c_contiguous() {
        let src_ptr = arr.data_ptr();
        for i in 0..n {
            // Calculate range of columns to copy for this row
            let (start_col, end_col) = if lower {
                // Lower triangle: copy columns 0..min(i+k+1, m)
                let end = ((i as isize + k + 1).max(0) as usize).min(m);
                (0, end)
            } else {
                // Upper triangle: copy columns max(i+k, 0)..m
                let start = ((i as isize + k).max(0) as usize).min(m);
                (start, m)
            };

            if end_col > start_col {
                let src_row = unsafe { src_ptr.add(i * m * itemsize + start_col * itemsize) };
                let dst_row = unsafe { result_ptr.add(i * m * itemsize + start_col * itemsize) };
                let copy_bytes = (end_col - start_col) * itemsize;
                unsafe { std::ptr::copy_nonoverlapping(src_row, dst_row, copy_bytes); }
            }
        }
        return Some(result);
    }

    // Slow path: strided array - element by element
    let ops = result.dtype.ops();
    let src_ptr = arr.data_ptr();

    for i in 0..n {
        for j in 0..m {
            let include = if lower {
                (j as isize) <= (i as isize) + k
            } else {
                (j as isize) >= (i as isize) + k
            };
            if include {
                let offset = arr.byte_offset_for(&[i, j]);
                let idx = i * m + j;
                unsafe { ops.copy_element(src_ptr, offset, result_ptr, idx); }
            }
        }
    }
    Some(result)
}

/// Return lower triangle of an array.
pub fn tril(arr: &RumpyArray, k: isize) -> Option<RumpyArray> {
    extract_triangle(arr, k, true)
}

/// Return upper triangle of an array.
pub fn triu(arr: &RumpyArray, k: isize) -> Option<RumpyArray> {
    extract_triangle(arr, k, false)
}

/// Create a 2D array with flattened input as diagonal.
/// k: diagonal offset (0 = main diagonal)
pub fn diagflat(v: &RumpyArray, k: isize) -> RumpyArray {
    // Flatten input
    let size = v.size();
    let n = size + k.unsigned_abs();

    let mut result = RumpyArray::zeros(vec![n, n], v.dtype());
    if size == 0 {
        return result;
    }

    let buffer = Arc::get_mut(&mut result.buffer).expect("buffer must be unique");
    let result_ptr = buffer.as_mut_ptr();
    let ops = result.dtype.ops();
    let src_ptr = v.data_ptr();

    for (i, offset) in v.iter_offsets().enumerate() {
        let (row, col) = if k >= 0 {
            (i, i + k as usize)
        } else {
            (i + (-k) as usize, i)
        };
        let idx = row * n + col;
        unsafe { ops.copy_element(src_ptr, offset, result_ptr, idx); }
    }
    result
}

/// Return coordinate matrices from coordinate vectors.
/// indexing: "xy" (Cartesian) or "ij" (matrix)
pub fn meshgrid(arrays: &[RumpyArray], indexing: &str) -> Option<Vec<RumpyArray>> {
    if arrays.is_empty() {
        return Some(Vec::new());
    }

    // All inputs must be 1D
    for arr in arrays {
        if arr.ndim() != 1 {
            return None;
        }
    }

    let ndim = arrays.len();
    let sizes: Vec<usize> = arrays.iter().map(|a| a.size()).collect();

    // Output shape depends on indexing mode
    let output_shape: Vec<usize> = if indexing == "xy" && ndim >= 2 {
        let mut shape = sizes.clone();
        shape.swap(0, 1);
        shape
    } else {
        sizes.clone()
    };

    let mut result = Vec::with_capacity(ndim);

    for (dim, arr) in arrays.iter().enumerate() {
        // The dimension this array varies along in output
        let vary_dim = if indexing == "xy" && ndim >= 2 {
            if dim == 0 { 1 } else if dim == 1 { 0 } else { dim }
        } else {
            dim
        };

        // Create broadcast shape: all 1s except for vary_dim
        let mut bc_shape = vec![1usize; ndim];
        bc_shape[vary_dim] = arr.size();

        // Reshape to broadcast shape, then broadcast to output shape
        let reshaped = arr.reshape(bc_shape)?;
        let broadcast = reshaped.broadcast_to(&output_shape)?;

        // Copy to contiguous
        result.push(broadcast.copy());
    }

    Some(result)
}

/// Return an array representing indices of a grid.
/// Result shape is (len(dimensions),) + dimensions.
pub fn indices(dimensions: &[usize], dtype: DType) -> RumpyArray {
    let ndim = dimensions.len();
    if ndim == 0 {
        return RumpyArray::zeros(vec![0], dtype);
    }

    let mut shape = vec![ndim];
    shape.extend_from_slice(dimensions);

    let mut result = RumpyArray::zeros(shape.clone(), dtype.clone());
    let total_per_dim: usize = dimensions.iter().product();

    if total_per_dim == 0 {
        return result;
    }

    let buffer = Arc::get_mut(&mut result.buffer).expect("buffer must be unique");
    let ptr = buffer.as_mut_ptr();

    // Fast path for int64 (default dtype for indices)
    if dtype == DType::int64() {
        let dst = ptr as *mut i64;

        for dim in 0..ndim {
            let base = dim * total_per_dim;
            // For dimension dim, the index repeats with period = product of dims after it
            // and increments after every (product of dims after it) elements
            let stride: usize = dimensions[dim + 1..].iter().product();
            let repeat: usize = dimensions[..dim].iter().product();
            let dim_size = dimensions[dim];

            for r in 0..repeat {
                let r_base = r * dim_size * stride;
                for idx in 0..dim_size {
                    let val = idx as i64;
                    for s in 0..stride {
                        unsafe { *dst.add(base + r_base + idx * stride + s) = val; }
                    }
                }
            }
        }
    } else if dtype == DType::float64() {
        let dst = ptr as *mut f64;

        for dim in 0..ndim {
            let base = dim * total_per_dim;
            let stride: usize = dimensions[dim + 1..].iter().product();
            let repeat: usize = dimensions[..dim].iter().product();
            let dim_size = dimensions[dim];

            for r in 0..repeat {
                let r_base = r * dim_size * stride;
                for idx in 0..dim_size {
                    let val = idx as f64;
                    for s in 0..stride {
                        unsafe { *dst.add(base + r_base + idx * stride + s) = val; }
                    }
                }
            }
        }
    } else {
        // Fallback for other dtypes
        let ops = result.dtype.ops();
        for dim in 0..ndim {
            let base_offset = dim * total_per_dim;
            let mut idx_vec = vec![0usize; ndim];

            for i in 0..total_per_dim {
                let val = idx_vec[dim] as f64;
                unsafe { ops.write_f64(ptr, base_offset + i, val); }
                increment_indices(&mut idx_vec, dimensions);
            }
        }
    }

    result
}

/// 1D discrete convolution.
/// mode: "full", "same", or "valid"
///
/// Uses direct O(n*m) method for small inputs, FFT O(n log n) for large.
pub fn convolve(a: &RumpyArray, v: &RumpyArray, mode: &str) -> Option<RumpyArray> {
    // Both must be 1D
    if a.ndim() != 1 || v.ndim() != 1 {
        return None;
    }

    let n = a.size();
    let m = v.size();

    if n == 0 || m == 0 {
        let out_len = match mode {
            "full" => n + m - 1,
            "same" => n,
            "valid" => n.saturating_sub(m - 1),
            _ => return None,
        };
        return Some(RumpyArray::zeros(vec![out_len.max(0)], DType::float64()));
    }

    // Collect values (needed for both methods)
    let a_ptr = a.data_ptr();
    let a_dtype = a.dtype();
    let a_ops = a_dtype.ops();
    let v_ptr = v.data_ptr();
    let v_dtype = v.dtype();
    let v_ops = v_dtype.ops();

    let mut a_vals: Vec<f64> = Vec::with_capacity(n);
    for offset in a.iter_offsets() {
        a_vals.push(unsafe { a_ops.read_f64(a_ptr, offset) }.unwrap_or(0.0));
    }

    let mut v_vals: Vec<f64> = Vec::with_capacity(m);
    for offset in v.iter_offsets() {
        v_vals.push(unsafe { v_ops.read_f64(v_ptr, offset) }.unwrap_or(0.0));
    }

    // Choose algorithm: FFT is O((n+m) log(n+m)), direct is O(n*m)
    // FFT wins when n*m is large enough to offset FFT overhead
    // Empirically, crossover vs numpy is around n*m = 2M
    // We use FFT when it's faster than our own direct method (~500k)
    let use_fft = n * m > 500_000;

    let full_result = if use_fft {
        convolve_fft(&a_vals, &v_vals)
    } else {
        convolve_direct(&a_vals, &v_vals)
    };

    // Extract the portion based on mode
    let full_len = n + m - 1;
    let (out_len, start) = match mode {
        "full" => (full_len, 0),
        "same" => (n, (m - 1) / 2),
        "valid" => (n.saturating_sub(m - 1).max(1), m - 1),
        _ => return None,
    };

    let mut result = RumpyArray::zeros(vec![out_len], DType::float64());
    let buffer = result.buffer_mut();
    let result_buffer = std::sync::Arc::get_mut(buffer).expect("unique");
    let result_ptr = result_buffer.as_mut_ptr() as *mut f64;

    for i in 0..out_len {
        unsafe { *result_ptr.add(i) = full_result[start + i]; }
    }

    Some(result)
}

/// Direct O(n*m) convolution helper.
fn convolve_direct(a: &[f64], v: &[f64]) -> Vec<f64> {
    let (n, m) = (a.len(), v.len());
    let out_len = n + m - 1;
    let mut result = vec![0.0; out_len];
    for k in 0..out_len {
        result[k] = (k.saturating_sub(n - 1)..m.min(k + 1))
            .map(|j| a[k - j] * v[j]).sum();
    }
    result
}

/// FFT-based O(n log n) convolution helper.
fn convolve_fft(a: &[f64], v: &[f64]) -> Vec<f64> {
    use rustfft::{FftPlanner, num_complex::Complex64};
    let out_len = a.len() + v.len() - 1;
    let fft_len = out_len.next_power_of_two();

    let mut a_c: Vec<_> = a.iter().map(|&x| Complex64::new(x, 0.0)).collect();
    a_c.resize(fft_len, Complex64::new(0.0, 0.0));
    let mut v_c: Vec<_> = v.iter().map(|&x| Complex64::new(x, 0.0)).collect();
    v_c.resize(fft_len, Complex64::new(0.0, 0.0));

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(fft_len);
    fft.process(&mut a_c);
    fft.process(&mut v_c);

    for i in 0..fft_len { a_c[i] *= v_c[i]; }

    planner.plan_fft_inverse(fft_len).process(&mut a_c);
    a_c[..out_len].iter().map(|c| c.re / fft_len as f64).collect()
}

/// Helper: compute linear index from multi-dimensional indices for C-contiguous array.
fn linear_index(indices: &[usize], shape: &[usize]) -> usize {
    let mut idx = 0;
    let mut stride = 1;
    for i in (0..indices.len()).rev() {
        idx += indices[i] * stride;
        stride *= shape[i];
    }
    idx
}

/// Repeat elements of an array.
/// If axis is None, flatten and repeat. Otherwise repeat along axis.
pub fn repeat(arr: &RumpyArray, repeats: usize, axis: Option<isize>) -> Option<RumpyArray> {
    if repeats == 0 {
        return Some(RumpyArray::zeros(vec![0], arr.dtype()));
    }

    match axis {
        None => {
            // Flatten and repeat each element
            let size = arr.size();
            let new_size = size * repeats;
            let src_dtype = arr.dtype();
            let mut result = RumpyArray::zeros(vec![new_size], src_dtype.clone());

            let src_ptr = arr.data_ptr();
            let src_ops = src_dtype.ops();
            let dst_dtype = result.dtype();
            let buffer = Arc::get_mut(result.buffer_mut()).expect("unique");
            let dst_ptr = buffer.as_mut_ptr();
            let dst_ops = dst_dtype.ops();

            let mut dst_idx = 0;
            for offset in arr.iter_offsets() {
                let val = unsafe { src_ops.read_f64(src_ptr, offset) }.unwrap_or(0.0);
                for _ in 0..repeats {
                    unsafe { dst_ops.write_f64(dst_ptr, dst_idx, val); }
                    dst_idx += 1;
                }
            }
            Some(result)
        }
        Some(ax) => {
            let ndim = arr.ndim();
            let axis = if ax < 0 { (ndim as isize + ax) as usize } else { ax as usize };
            if axis >= ndim {
                return None;
            }

            // New shape with axis dimension multiplied by repeats
            let mut new_shape = arr.shape().to_vec();
            new_shape[axis] *= repeats;
            let src_dtype = arr.dtype();
            let mut result = RumpyArray::zeros(new_shape.clone(), src_dtype.clone());

            let src_ptr = arr.data_ptr();
            let src_ops = src_dtype.ops();
            let dst_dtype = result.dtype();
            let buffer = Arc::get_mut(result.buffer_mut()).expect("unique");
            let dst_ptr = buffer.as_mut_ptr();
            let dst_ops = dst_dtype.ops();

            // Iterate through source indices and write repeated values
            let src_shape = arr.shape();
            let mut src_indices = vec![0usize; ndim];
            let mut dst_indices = vec![0usize; ndim];

            let size = arr.size();
            for _ in 0..size {
                let src_offset = arr.byte_offset_for(&src_indices);
                let val = unsafe { src_ops.read_f64(src_ptr, src_offset) }.unwrap_or(0.0);

                // Write this value `repeats` times along the axis
                dst_indices.copy_from_slice(&src_indices);
                let base_axis_idx = src_indices[axis] * repeats;
                for r in 0..repeats {
                    dst_indices[axis] = base_axis_idx + r;
                    let dst_offset = linear_index(&dst_indices, &new_shape);
                    unsafe { dst_ops.write_f64(dst_ptr, dst_offset, val); }
                }

                increment_indices(&mut src_indices, src_shape);
            }

            Some(result)
        }
    }
}

/// Construct array by repeating input array by given reps.
pub fn tile(arr: &RumpyArray, reps: &[usize]) -> RumpyArray {
    if reps.is_empty() || reps.iter().all(|&r| r == 1) {
        return arr.copy();
    }

    // Extend reps to match dimensions if needed
    let arr_ndim = arr.ndim();
    let reps_len = reps.len();
    let max_ndim = arr_ndim.max(reps_len);

    // Pad reps with 1s on the left
    let mut full_reps = vec![1usize; max_ndim];
    for (i, &r) in reps.iter().enumerate() {
        full_reps[max_ndim - reps_len + i] = r;
    }

    // Pad shape with 1s on the left
    let src_shape = arr.shape();
    let mut padded_shape = vec![1usize; max_ndim];
    for (i, &s) in src_shape.iter().enumerate() {
        padded_shape[max_ndim - arr_ndim + i] = s;
    }

    // New shape
    let new_shape: Vec<usize> = padded_shape.iter().zip(full_reps.iter()).map(|(&s, &r)| s * r).collect();
    let src_dtype = arr.dtype();
    let mut result = RumpyArray::zeros(new_shape.clone(), src_dtype.clone());

    // Fast path: 1D contiguous array - use memcpy
    if arr_ndim == 1 && reps_len == 1 && arr.is_c_contiguous() {
        let src_size = arr.size();
        let itemsize = arr.itemsize();
        let nbytes = src_size * itemsize;
        let src_ptr = arr.data_ptr();
        let buffer = Arc::get_mut(result.buffer_mut()).expect("unique");
        let dst_ptr = buffer.as_mut_ptr();

        for i in 0..reps[0] {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    src_ptr,
                    dst_ptr.add(i * nbytes),
                    nbytes
                );
            }
        }
        return result;
    }

    let src_ptr = arr.data_ptr();
    let src_ops = src_dtype.ops();
    let dst_dtype = result.dtype();
    let buffer = Arc::get_mut(result.buffer_mut()).expect("unique");
    let dst_ptr = buffer.as_mut_ptr();
    let dst_ops = dst_dtype.ops();

    // Iterate through destination indices
    let mut dst_indices = vec![0usize; max_ndim];
    let dst_size: usize = new_shape.iter().product();

    for i in 0..dst_size {
        // Map destination index to source index (modulo)
        let src_indices: Vec<usize> = dst_indices.iter()
            .zip(padded_shape.iter())
            .map(|(&di, &ps)| di % ps)
            .collect();

        // Get source offset (for original array shape)
        let actual_src_indices: Vec<usize> = if arr_ndim < max_ndim {
            // Use only the rightmost dimensions
            src_indices[max_ndim - arr_ndim..].to_vec()
        } else {
            src_indices
        };
        let src_byte_offset = arr.byte_offset_for(&actual_src_indices);
        let val = unsafe { src_ops.read_f64(src_ptr, src_byte_offset) }.unwrap_or(0.0);

        // Destination offset (result is C-contiguous, so use linear index)
        let dst_offset = linear_index(&dst_indices, &new_shape);
        unsafe { dst_ops.write_f64(dst_ptr, dst_offset, val); }

        // Increment destination indices
        if i + 1 < dst_size {
            increment_indices(&mut dst_indices, &new_shape);
        }
    }

    result
}

/// Helper: flatten array to 1D (always copies).
fn ravel_arr(arr: &RumpyArray) -> RumpyArray {
    let size = arr.size();
    if let Some(view) = arr.reshape(vec![size]) {
        view
    } else {
        arr.copy().reshape(vec![size]).unwrap()
    }
}

/// Append values to end of array.
/// If axis is None, both arrays are flattened before concatenation.
pub fn append(arr: &RumpyArray, values: &RumpyArray, axis: Option<isize>) -> Option<RumpyArray> {
    match axis {
        None => {
            // Flatten both and concatenate
            let flat_arr = ravel_arr(arr);
            let flat_vals = ravel_arr(values);
            concatenate(&[flat_arr, flat_vals], 0)
        }
        Some(ax) => {
            let ndim = arr.ndim();
            let axis = if ax < 0 { (ndim as isize + ax) as usize } else { ax as usize };
            concatenate(&[arr.clone(), values.clone()], axis)
        }
    }
}

/// Insert values into array at given index.
/// If axis is None, flatten first.
pub fn insert(arr: &RumpyArray, index: isize, values: &RumpyArray, axis: Option<isize>) -> Option<RumpyArray> {
    match axis {
        None => {
            // Flatten
            let flat = ravel_arr(arr);
            let size = flat.size();
            let idx = if index < 0 { (size as isize + index + 1).max(0) as usize } else { index as usize };
            let idx = idx.min(size);

            // Split at index, insert, concatenate
            let before = if idx > 0 { flat.slice_axis(0, 0, idx as isize, 1) } else { RumpyArray::zeros(vec![0], arr.dtype()) };
            let after = if idx < size { flat.slice_axis(0, idx as isize, size as isize, 1) } else { RumpyArray::zeros(vec![0], arr.dtype()) };
            let flat_vals = ravel_arr(values);

            concatenate(&[before, flat_vals, after], 0)
        }
        Some(ax) => {
            let ndim = arr.ndim();
            let axis = if ax < 0 { (ndim as isize + ax) as usize } else { ax as usize };
            if axis >= ndim {
                return None;
            }

            let axis_len = arr.shape()[axis];
            let idx = if index < 0 { (axis_len as isize + index + 1).max(0) as usize } else { index as usize };
            let idx = idx.min(axis_len);

            let before = arr.slice_axis(axis, 0, idx as isize, 1);
            let after = arr.slice_axis(axis, idx as isize, axis_len as isize, 1);

            // Values need to be reshaped to match
            let mut val_shape = arr.shape().to_vec();
            val_shape[axis] = values.size() / (arr.size() / axis_len).max(1);
            let vals_reshaped = values.reshape(val_shape).unwrap_or_else(|| values.clone());

            concatenate(&[before, vals_reshaped, after], axis)
        }
    }
}

/// Delete elements from array at given index.
/// If axis is None, flatten first.
pub fn delete(arr: &RumpyArray, index: isize, axis: Option<isize>) -> Option<RumpyArray> {
    match axis {
        None => {
            let flat = ravel_arr(arr);
            let size = flat.size();
            if size == 0 {
                return Some(flat);
            }
            let idx = if index < 0 { (size as isize + index).max(0) as usize } else { index as usize };
            if idx >= size {
                return Some(flat);
            }

            let before = if idx > 0 { flat.slice_axis(0, 0, idx as isize, 1) } else { RumpyArray::zeros(vec![0], arr.dtype()) };
            let after = if idx + 1 < size { flat.slice_axis(0, (idx + 1) as isize, size as isize, 1) } else { RumpyArray::zeros(vec![0], arr.dtype()) };

            concatenate(&[before, after], 0)
        }
        Some(ax) => {
            let ndim = arr.ndim();
            let axis = if ax < 0 { (ndim as isize + ax) as usize } else { ax as usize };
            if axis >= ndim {
                return None;
            }

            let axis_len = arr.shape()[axis];
            if axis_len == 0 {
                return Some(arr.clone());
            }
            let idx = if index < 0 { (axis_len as isize + index).max(0) as usize } else { index as usize };
            if idx >= axis_len {
                return Some(arr.clone());
            }

            let before = arr.slice_axis(axis, 0, idx as isize, 1);
            let after = arr.slice_axis(axis, (idx + 1) as isize, axis_len as isize, 1);

            concatenate(&[before, after], axis)
        }
    }
}

/// Pad an array.
/// pad_width is a slice of (before, after) tuples for each dimension.
pub fn pad(arr: &RumpyArray, pad_width: &[(usize, usize)], mode: &str, constant_value: f64) -> Option<RumpyArray> {
    let ndim = arr.ndim();
    if pad_width.len() != ndim {
        return None;
    }

    // Calculate new shape
    let new_shape: Vec<usize> = arr.shape().iter()
        .zip(pad_width.iter())
        .map(|(&s, &(b, a))| s + b + a)
        .collect();

    match mode {
        "constant" => {
            let mut result = RumpyArray::full(new_shape.clone(), constant_value, arr.dtype());
            copy_region(arr, &mut result, pad_width, &new_shape);
            Some(result)
        }
        "edge" => {
            let src_dtype = arr.dtype();
            let mut result = RumpyArray::zeros(new_shape.clone(), src_dtype.clone());
            let src_ops = src_dtype.ops();
            let dst_dtype = result.dtype();
            let buffer = Arc::get_mut(result.buffer_mut()).expect("unique");
            let dst_ptr = buffer.as_mut_ptr();
            let dst_ops = dst_dtype.ops();

            let mut dst_indices = vec![0usize; ndim];
            let dst_size: usize = new_shape.iter().product();
            let src_ptr = arr.data_ptr();

            // Fill all cells with edge-clamped values
            for _ in 0..dst_size {
                let src_indices: Vec<usize> = dst_indices.iter()
                    .zip(pad_width.iter())
                    .zip(arr.shape().iter())
                    .map(|((&di, &(b, _)), &s)| {
                        if di < b { 0 }
                        else if di >= b + s { s.saturating_sub(1) }
                        else { di - b }
                    })
                    .collect();

                let src_byte_offset = arr.byte_offset_for(&src_indices);
                let val = unsafe { src_ops.read_f64(src_ptr, src_byte_offset) }.unwrap_or(0.0);
                let dst_offset = linear_index(&dst_indices, &new_shape);
                unsafe { dst_ops.write_f64(dst_ptr, dst_offset, val); }

                increment_indices(&mut dst_indices, &new_shape);
            }

            Some(result)
        }
        _ => None,
    }
}

/// Helper to copy source array into a padded destination.
fn copy_region(src: &RumpyArray, dst: &mut RumpyArray, pad_width: &[(usize, usize)], dst_shape: &[usize]) {
    let ndim = src.ndim();
    let src_shape = src.shape();

    let src_ptr = src.data_ptr();
    let src_dtype = src.dtype();
    let src_ops = src_dtype.ops();
    let dst_dtype = dst.dtype();
    let buffer = Arc::get_mut(dst.buffer_mut()).expect("unique");
    let dst_ptr = buffer.as_mut_ptr();
    let dst_ops = dst_dtype.ops();

    let mut src_indices = vec![0usize; ndim];
    let src_size = src.size();

    for _ in 0..src_size {
        // Calculate destination indices by adding padding
        let dst_indices: Vec<usize> = src_indices.iter()
            .zip(pad_width.iter())
            .map(|(&si, &(b, _))| si + b)
            .collect();

        let src_byte_offset = src.byte_offset_for(&src_indices);
        let val = unsafe { src_ops.read_f64(src_ptr, src_byte_offset) }.unwrap_or(0.0);

        let dst_offset = linear_index(&dst_indices, dst_shape);
        unsafe { dst_ops.write_f64(dst_ptr, dst_offset, val); }

        increment_indices(&mut src_indices, src_shape);
    }
}

/// Roll array elements along given axis.
/// If axis is None, rolls over flattened array but preserves original shape.
pub fn roll(arr: &RumpyArray, shift: isize, axis: Option<isize>) -> RumpyArray {
    match axis {
        None => {
            // Flatten, roll, reshape back to original shape
            let original_shape = arr.shape().to_vec();
            let flat = ravel_arr(arr);
            let size = flat.size();
            if size == 0 {
                return arr.copy();
            }

            // Normalize shift
            let shift = ((shift % size as isize) + size as isize) as usize % size;
            if shift == 0 {
                return arr.copy();
            }

            let src_dtype = flat.dtype();
            let mut result = RumpyArray::zeros(original_shape.clone(), src_dtype.clone());
            let src_ptr = flat.data_ptr();
            let buffer = Arc::get_mut(result.buffer_mut()).expect("unique");
            let dst_ptr = buffer.as_mut_ptr();

            // Fast path: use memcpy for the two parts
            // Result: [src[size-shift:], src[:size-shift]]
            let itemsize = src_dtype.itemsize();
            let split_point = size - shift; // Elements from this index go to the front
            let bytes_part1 = shift * itemsize;       // Elements shift..size go to dst[0..shift]
            let bytes_part2 = split_point * itemsize; // Elements 0..split_point go to dst[shift..]

            unsafe {
                // Copy elements [split_point..] to [0..shift]
                std::ptr::copy_nonoverlapping(
                    src_ptr.add(split_point * itemsize),
                    dst_ptr,
                    bytes_part1,
                );
                // Copy elements [0..split_point] to [shift..]
                std::ptr::copy_nonoverlapping(
                    src_ptr,
                    dst_ptr.add(bytes_part1),
                    bytes_part2,
                );
            }

            result
        }
        Some(ax) => {
            let ndim = arr.ndim();
            let axis = if ax < 0 { (ndim as isize + ax) as usize } else { ax as usize };
            if axis >= ndim {
                return arr.copy();
            }

            let axis_len = arr.shape()[axis];
            if axis_len == 0 {
                return arr.copy();
            }

            // Normalize shift
            let shift = ((shift % axis_len as isize) + axis_len as isize) as usize % axis_len;
            if shift == 0 {
                return arr.copy();
            }

            let result_shape = arr.shape().to_vec();
            let src_dtype = arr.dtype();
            let mut result = RumpyArray::zeros(result_shape.clone(), src_dtype.clone());
            let src_ptr = arr.data_ptr();
            let src_ops = src_dtype.ops();
            let dst_dtype = result.dtype();
            let buffer = Arc::get_mut(result.buffer_mut()).expect("unique");
            let dst_ptr = buffer.as_mut_ptr();
            let dst_ops = dst_dtype.ops();

            let mut dst_indices = vec![0usize; ndim];
            let size = arr.size();

            for _ in 0..size {
                // Source index: roll along axis
                let mut src_indices = dst_indices.clone();
                src_indices[axis] = (dst_indices[axis] + axis_len - shift) % axis_len;

                let src_byte_offset = arr.byte_offset_for(&src_indices);
                let val = unsafe { src_ops.read_f64(src_ptr, src_byte_offset) }.unwrap_or(0.0);

                let dst_offset = linear_index(&dst_indices, &result_shape);
                unsafe { dst_ops.write_f64(dst_ptr, dst_offset, val); }

                increment_indices(&mut dst_indices, arr.shape());
            }

            result
        }
    }
}

/// Rotate array 90 degrees k times in the plane specified by axes.
/// Returns a view (no copy) like numpy.
pub fn rot90(arr: &RumpyArray, k: isize, axis0: usize, axis1: usize) -> RumpyArray {
    // Normalize k to 0-3
    let k = ((k % 4) + 4) as usize % 4;

    let ndim = arr.ndim();
    let mut axes: Vec<usize> = (0..ndim).collect();
    axes.swap(axis0, axis1);

    match k {
        0 => arr.clone(),
        1 => {
            // k=1: flip axis1, then transpose
            let flipped = arr.flip(axis1).unwrap();
            flipped.transpose_axes(&axes)
        }
        2 => {
            // k=2: flip both axes (180 degree rotation), no transpose
            let f1 = arr.flip(axis0).unwrap();
            f1.flip(axis1).unwrap()
        }
        3 => {
            // k=3 = k=-1: flip axis0, then transpose
            let flipped = arr.flip(axis0).unwrap();
            flipped.transpose_axes(&axes)
        }
        _ => unreachable!()
    }
}
