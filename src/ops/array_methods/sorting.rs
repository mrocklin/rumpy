//! Sorting operations on RumpyArray (sort, argsort, partition, argpartition, lexsort).

use crate::array::{increment_indices, DType, RumpyArray};
use crate::array::dtype::DTypeKind;
use std::sync::Arc;

impl RumpyArray {
    /// Collect all elements into a Vec (flattened, row-major order).
    pub(crate) fn to_vec(&self) -> Vec<f64> {
        let size = self.size();
        if size == 0 {
            return Vec::new();
        }

        // Fast path for contiguous f64 arrays
        if self.is_c_contiguous() && self.dtype().kind() == DTypeKind::Float64 {
            let ptr = self.data_ptr() as *const f64;
            let slice = unsafe { std::slice::from_raw_parts(ptr, size) };
            return slice.to_vec();
        }

        // Slow path for non-contiguous or non-f64
        let ptr = self.data_ptr();
        let dtype = self.dtype();
        let ops = dtype.ops();
        let mut values = Vec::with_capacity(size);
        for offset in self.iter_offsets() {
            values.push(unsafe { ops.read_f64(ptr, offset) }.unwrap_or(0.0));
        }
        values
    }

    /// Return unique sorted values.
    pub fn unique(&self) -> RumpyArray {
        let mut values = self.to_vec();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        values.dedup_by(|a, b| (*a - *b).abs() < f64::EPSILON);
        RumpyArray::from_vec(values, self.dtype())
    }

    /// Sort array along an axis.
    pub fn sort(&self, axis: Option<usize>) -> RumpyArray {
        // If axis is None, flatten and sort
        let axis = match axis {
            None => {
                let flat = self.reshape(vec![self.size()]).expect("flatten should work");
                return flat.sort(Some(0));
            }
            Some(ax) => ax,
        };
        assert!(axis < self.ndim(), "axis out of bounds");

        let shape = self.shape();
        let axis_len = shape[axis];
        let dtype = self.dtype().clone();

        let mut result = self.copy();
        if axis_len <= 1 {
            return result;
        }

        // For each position along other axes, sort the elements along the target axis
        let mut out_shape: Vec<usize> = shape[..axis].to_vec();
        out_shape.extend_from_slice(&shape[axis + 1..]);
        if out_shape.is_empty() {
            out_shape = vec![1];
        }
        let out_size: usize = out_shape.iter().product();

        let ops = dtype.ops();
        let result_buffer = Arc::get_mut(result.buffer_mut()).expect("unique");
        let result_ptr = result_buffer.as_mut_ptr();

        let mut out_indices = vec![0usize; out_shape.len()];
        for _ in 0..out_size {
            // Collect values along axis
            let mut values: Vec<(f64, usize)> = Vec::with_capacity(axis_len);
            for k in 0..axis_len {
                let mut in_indices: Vec<usize> = out_indices[..axis.min(out_indices.len())].to_vec();
                in_indices.push(k);
                if axis < self.ndim() - 1 {
                    in_indices.extend_from_slice(&out_indices[axis..]);
                }
                values.push((self.get_element(&in_indices), k));
            }

            // Sort by value
            values.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

            // Write sorted values back
            for (k, (val, _)) in values.iter().enumerate() {
                let mut in_indices: Vec<usize> = out_indices[..axis.min(out_indices.len())].to_vec();
                in_indices.push(k);
                if axis < self.ndim() - 1 {
                    in_indices.extend_from_slice(&out_indices[axis..]);
                }
                let byte_offset = result.byte_offset_for(&in_indices);
                let linear_idx = byte_offset as usize / dtype.itemsize();
                unsafe { ops.write_f64(result_ptr, linear_idx, *val); }
            }

            increment_indices(&mut out_indices, &out_shape);
        }
        result
    }

    /// Return indices that would sort the array along an axis.
    pub fn argsort(&self, axis: Option<usize>) -> RumpyArray {
        // If axis is None, flatten and argsort
        let axis = match axis {
            None => {
                let flat = self.reshape(vec![self.size()]).expect("flatten should work");
                return flat.argsort(Some(0));
            }
            Some(ax) => ax,
        };
        assert!(axis < self.ndim(), "axis out of bounds");

        let shape = self.shape();
        let axis_len = shape[axis];

        let mut result = RumpyArray::zeros(shape.to_vec(), DType::int64());
        if axis_len <= 1 {
            // For single element, index is 0
            return result;
        }

        let mut out_shape: Vec<usize> = shape[..axis].to_vec();
        out_shape.extend_from_slice(&shape[axis + 1..]);
        if out_shape.is_empty() {
            out_shape = vec![1];
        }
        let out_size: usize = out_shape.iter().product();

        let result_buffer = Arc::get_mut(result.buffer_mut()).expect("unique");
        let result_ptr = result_buffer.as_mut_ptr() as *mut i64;

        let mut out_indices = vec![0usize; out_shape.len()];
        for _ in 0..out_size {
            // Collect values along axis with their indices
            let mut values: Vec<(f64, usize)> = Vec::with_capacity(axis_len);
            for k in 0..axis_len {
                let mut in_indices: Vec<usize> = out_indices[..axis.min(out_indices.len())].to_vec();
                in_indices.push(k);
                if axis < self.ndim() - 1 {
                    in_indices.extend_from_slice(&out_indices[axis..]);
                }
                values.push((self.get_element(&in_indices), k));
            }

            // Sort by value, keeping track of original indices
            values.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

            // Write sorted indices back
            for (k, (_, orig_idx)) in values.iter().enumerate() {
                let mut in_indices: Vec<usize> = out_indices[..axis.min(out_indices.len())].to_vec();
                in_indices.push(k);
                if axis < self.ndim() - 1 {
                    in_indices.extend_from_slice(&out_indices[axis..]);
                }
                let byte_offset = result.byte_offset_for(&in_indices);
                let linear_idx = byte_offset as usize / 8; // i64 is 8 bytes
                unsafe { *result_ptr.add(linear_idx) = *orig_idx as i64; }
            }

            increment_indices(&mut out_indices, &out_shape);
        }
        result
    }

    /// Partition array along an axis using partial sort.
    /// The kth element will be in its sorted position, with smaller elements before
    /// and larger elements after (but not necessarily sorted).
    pub fn partition(&self, kth: usize, axis: Option<usize>) -> RumpyArray {
        // If axis is None, flatten and partition
        let axis = match axis {
            None => {
                let flat = self.reshape(vec![self.size()]).expect("flatten should work");
                return flat.partition(kth, Some(0));
            }
            Some(ax) => ax,
        };
        assert!(axis < self.ndim(), "axis out of bounds");

        let shape = self.shape();
        let axis_len = shape[axis];
        assert!(kth < axis_len, "kth must be less than axis length");

        if axis_len <= 1 {
            return self.copy();
        }

        // Copy input to result, then partition in-place
        let mut result = self.copy();
        partition_inplace(&mut result, kth, axis);
        result
    }

    /// Return indices that would partition the array along an axis.
    /// The kth index will be in its sorted position.
    pub fn argpartition(&self, kth: usize, axis: Option<usize>) -> RumpyArray {
        // If axis is None, flatten and argpartition
        let axis = match axis {
            None => {
                let flat = self.reshape(vec![self.size()]).expect("flatten should work");
                return flat.argpartition(kth, Some(0));
            }
            Some(ax) => ax,
        };
        assert!(axis < self.ndim(), "axis out of bounds");

        let shape = self.shape();
        let axis_len = shape[axis];
        assert!(kth < axis_len, "kth must be less than axis length");

        if axis_len <= 1 {
            return RumpyArray::zeros(shape.to_vec(), DType::int64());
        }

        // Compute partitioned indices using helper
        argpartition_impl(self, kth, axis)
    }
}

/// Partition array in-place along an axis.
/// Uses type-specific operations for best performance.
fn partition_inplace(arr: &mut RumpyArray, kth: usize, axis: usize) {
    // Copy metadata before borrowing mutably
    let shape: Vec<usize> = arr.shape().to_vec();
    let strides: Vec<isize> = arr.strides().to_vec();
    let axis_len = shape[axis];
    let dtype = arr.dtype().clone();
    let itemsize = dtype.itemsize();
    let ndim = arr.ndim();
    let is_contiguous = arr.is_c_contiguous();

    // For 1D contiguous arrays, use optimized typed partition
    if ndim == 1 && is_contiguous {
        let buffer = Arc::get_mut(arr.buffer_mut()).expect("unique");
        let ptr = buffer.as_mut_ptr();

        match dtype.kind() {
            DTypeKind::Float64 => {
                let slice = unsafe { std::slice::from_raw_parts_mut(ptr as *mut f64, axis_len) };
                slice.select_nth_unstable_by(kth, |a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            }
            DTypeKind::Float32 => {
                let slice = unsafe { std::slice::from_raw_parts_mut(ptr as *mut f32, axis_len) };
                slice.select_nth_unstable_by(kth, |a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            }
            DTypeKind::Int64 => {
                let slice = unsafe { std::slice::from_raw_parts_mut(ptr as *mut i64, axis_len) };
                slice.select_nth_unstable(kth);
            }
            DTypeKind::Int32 => {
                let slice = unsafe { std::slice::from_raw_parts_mut(ptr as *mut i32, axis_len) };
                slice.select_nth_unstable(kth);
            }
            DTypeKind::Int16 => {
                let slice = unsafe { std::slice::from_raw_parts_mut(ptr as *mut i16, axis_len) };
                slice.select_nth_unstable(kth);
            }
            DTypeKind::Uint64 => {
                let slice = unsafe { std::slice::from_raw_parts_mut(ptr as *mut u64, axis_len) };
                slice.select_nth_unstable(kth);
            }
            DTypeKind::Uint32 => {
                let slice = unsafe { std::slice::from_raw_parts_mut(ptr as *mut u32, axis_len) };
                slice.select_nth_unstable(kth);
            }
            DTypeKind::Uint16 => {
                let slice = unsafe { std::slice::from_raw_parts_mut(ptr as *mut u16, axis_len) };
                slice.select_nth_unstable(kth);
            }
            DTypeKind::Uint8 => {
                let slice = unsafe { std::slice::from_raw_parts_mut(ptr, axis_len) };
                slice.select_nth_unstable(kth);
            }
            _ => {
                // Fallback: use indices and DTypeOps comparison via partition_lane_typed
                partition_lane_typed(ptr, 0, axis_len, itemsize as isize, kth, &dtype, itemsize);
            }
        }
        return;
    }

    // Multi-dimensional case: partition each lane along axis
    let mut out_shape: Vec<usize> = shape[..axis].to_vec();
    out_shape.extend_from_slice(&shape[axis + 1..]);
    if out_shape.is_empty() {
        out_shape = vec![1];
    }
    let out_size: usize = out_shape.iter().product();
    let axis_stride = strides[axis];

    let buffer = Arc::get_mut(arr.buffer_mut()).expect("unique");
    let ptr = buffer.as_mut_ptr();

    let mut out_indices = vec![0usize; out_shape.len()];
    for _ in 0..out_size {
        // Calculate base offset for this lane
        let mut base_offset: isize = 0;
        let mut idx_pos = 0;
        for (d, _) in shape.iter().enumerate() {
            if d == axis {
                continue;
            }
            base_offset += (out_indices[idx_pos] as isize) * strides[d];
            idx_pos += 1;
        }

        // Partition this lane using typed operations
        partition_lane_typed(ptr, base_offset, axis_len, axis_stride, kth, &dtype, itemsize);

        increment_indices(&mut out_indices, &out_shape);
    }
}

/// Partition a single lane (1D slice) in-place using typed operations.
fn partition_lane_typed(
    ptr: *mut u8,
    base_offset: isize,
    len: usize,
    stride: isize,
    kth: usize,
    dtype: &DType,
    itemsize: usize,
) {
    // For strided access, we need to work with indices
    let mut indices: Vec<usize> = (0..len).collect();
    let ops = dtype.ops();

    // Partition indices based on values
    indices.select_nth_unstable_by(kth, |&a, &b| {
        let a_off = base_offset + (a as isize) * stride;
        let b_off = base_offset + (b as isize) * stride;
        unsafe { ops.compare_elements(ptr, a_off, ptr, b_off) }
    });

    // Now reorder data according to partitioned indices
    // Use a temporary buffer to avoid in-place permutation complexity
    let mut temp = vec![0u8; len * itemsize];
    for (new_pos, &old_pos) in indices.iter().enumerate() {
        let src_off = base_offset + (old_pos as isize) * stride;
        unsafe {
            std::ptr::copy_nonoverlapping(
                ptr.offset(src_off),
                temp.as_mut_ptr().add(new_pos * itemsize),
                itemsize,
            );
        }
    }
    // Copy back
    for i in 0..len {
        let dst_off = base_offset + (i as isize) * stride;
        unsafe {
            std::ptr::copy_nonoverlapping(
                temp.as_ptr().add(i * itemsize),
                ptr.offset(dst_off),
                itemsize,
            );
        }
    }
}

/// Compute argpartition: return indices that would partition the array.
fn argpartition_impl(arr: &RumpyArray, kth: usize, axis: usize) -> RumpyArray {
    let shape = arr.shape();
    let strides = arr.strides();
    let axis_len = shape[axis];
    let dtype = arr.dtype().clone();
    let ndim = arr.ndim();

    // Output shape is same as input
    let out_shape = shape.to_vec();
    let out_size: usize = out_shape.iter().product();

    // Create output array of indices (int64)
    let mut result = RumpyArray::zeros(out_shape.clone(), DType::int64());

    if out_size == 0 {
        return result;
    }

    let src_ptr = arr.data_ptr();
    let result_buffer = Arc::get_mut(result.buffer_mut()).expect("unique");
    let result_ptr = result_buffer.as_mut_ptr() as *mut i64;

    // For 1D contiguous arrays, use optimized typed comparison
    if ndim == 1 && arr.is_c_contiguous() {
        let mut indices: Vec<i64> = (0..axis_len as i64).collect();

        match dtype.kind() {
            DTypeKind::Float64 => {
                let slice = unsafe { std::slice::from_raw_parts(src_ptr as *const f64, axis_len) };
                indices.select_nth_unstable_by(kth, |&a, &b| {
                    let va = slice[a as usize];
                    let vb = slice[b as usize];
                    va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            DTypeKind::Float32 => {
                let slice = unsafe { std::slice::from_raw_parts(src_ptr as *const f32, axis_len) };
                indices.select_nth_unstable_by(kth, |&a, &b| {
                    let va = slice[a as usize];
                    let vb = slice[b as usize];
                    va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            DTypeKind::Int64 => {
                let slice = unsafe { std::slice::from_raw_parts(src_ptr as *const i64, axis_len) };
                indices.select_nth_unstable_by(kth, |&a, &b| {
                    slice[a as usize].cmp(&slice[b as usize])
                });
            }
            DTypeKind::Int32 => {
                let slice = unsafe { std::slice::from_raw_parts(src_ptr as *const i32, axis_len) };
                indices.select_nth_unstable_by(kth, |&a, &b| {
                    slice[a as usize].cmp(&slice[b as usize])
                });
            }
            DTypeKind::Int16 => {
                let slice = unsafe { std::slice::from_raw_parts(src_ptr as *const i16, axis_len) };
                indices.select_nth_unstable_by(kth, |&a, &b| {
                    slice[a as usize].cmp(&slice[b as usize])
                });
            }
            DTypeKind::Uint64 => {
                let slice = unsafe { std::slice::from_raw_parts(src_ptr as *const u64, axis_len) };
                indices.select_nth_unstable_by(kth, |&a, &b| {
                    slice[a as usize].cmp(&slice[b as usize])
                });
            }
            DTypeKind::Uint32 => {
                let slice = unsafe { std::slice::from_raw_parts(src_ptr as *const u32, axis_len) };
                indices.select_nth_unstable_by(kth, |&a, &b| {
                    slice[a as usize].cmp(&slice[b as usize])
                });
            }
            DTypeKind::Uint16 => {
                let slice = unsafe { std::slice::from_raw_parts(src_ptr as *const u16, axis_len) };
                indices.select_nth_unstable_by(kth, |&a, &b| {
                    slice[a as usize].cmp(&slice[b as usize])
                });
            }
            DTypeKind::Uint8 => {
                let slice = unsafe { std::slice::from_raw_parts(src_ptr, axis_len) };
                indices.select_nth_unstable_by(kth, |&a, &b| {
                    slice[a as usize].cmp(&slice[b as usize])
                });
            }
            _ => {
                // Generic fallback using DTypeOps
                let ops = dtype.ops();
                indices.select_nth_unstable_by(kth, |&a, &b| {
                    let a_off = (a as isize) * strides[0];
                    let b_off = (b as isize) * strides[0];
                    unsafe { ops.compare_elements(src_ptr, a_off, src_ptr, b_off) }
                });
            }
        }

        // Copy indices to result
        unsafe {
            std::ptr::copy_nonoverlapping(indices.as_ptr(), result_ptr, axis_len);
        }
        return result;
    }

    // Multi-dimensional case: argpartition each lane along axis
    let axis_stride = strides[axis];
    let ops = dtype.ops();

    // Iterate over all positions except the axis dimension
    let mut lane_shape: Vec<usize> = shape[..axis].to_vec();
    lane_shape.extend_from_slice(&shape[axis + 1..]);
    if lane_shape.is_empty() {
        lane_shape = vec![1];
    }
    let lane_count: usize = lane_shape.iter().product();

    let result_strides = result.strides().to_vec();
    let result_axis_stride = result_strides[axis];

    let mut lane_indices = vec![0usize; lane_shape.len()];
    for _ in 0..lane_count {
        // Calculate base offset for this lane in source and result
        let mut src_base: isize = 0;
        let mut result_base: isize = 0;
        let mut idx_pos = 0;
        for (d, _) in shape.iter().enumerate() {
            if d == axis {
                continue;
            }
            src_base += (lane_indices[idx_pos] as isize) * strides[d];
            result_base += (lane_indices[idx_pos] as isize) * result_strides[d];
            idx_pos += 1;
        }

        // Create indices for this lane and argpartition
        let mut indices: Vec<i64> = (0..axis_len as i64).collect();
        indices.select_nth_unstable_by(kth, |&a, &b| {
            let a_off = src_base + (a as isize) * axis_stride;
            let b_off = src_base + (b as isize) * axis_stride;
            unsafe { ops.compare_elements(src_ptr, a_off, src_ptr, b_off) }
        });

        // Write indices to result
        for (i, &idx) in indices.iter().enumerate() {
            let result_off = result_base + (i as isize) * result_axis_stride;
            unsafe {
                *result_ptr.offset(result_off / 8) = idx;
            }
        }

        increment_indices(&mut lane_indices, &lane_shape);
    }

    result
}

/// Perform indirect sort using multiple keys.
/// Keys are sorted in reverse order: last key is primary, second-to-last is secondary, etc.
/// All keys must be 1-D arrays of the same length.
/// Returns indices that would sort the keys.
pub fn lexsort(keys: &[&RumpyArray]) -> Option<RumpyArray> {
    if keys.is_empty() {
        return None;
    }

    // All keys must be 1-D
    for key in keys {
        if key.ndim() != 1 {
            return None;
        }
    }

    // All keys must have the same length
    let n = keys[0].size();
    for key in &keys[1..] {
        if key.size() != n {
            return None;
        }
    }

    if n == 0 {
        return Some(RumpyArray::zeros(vec![0], DType::int64()));
    }

    // Collect indices and sort by key comparisons
    // Last key is primary (highest priority)
    let mut indices: Vec<usize> = (0..n).collect();

    // Store key metadata for comparison - use pointers, dtypes, strides
    struct KeyInfo {
        ptr: *const u8,
        dtype: DType,
        stride: isize,
        is_contiguous: bool,
    }

    let key_infos: Vec<KeyInfo> = keys
        .iter()
        .map(|key| KeyInfo {
            ptr: key.data_ptr(),
            dtype: key.dtype().clone(),
            stride: key.strides()[0],
            is_contiguous: key.is_c_contiguous(),
        })
        .collect();

    indices.sort_by(|&a, &b| {
        // Compare from last key (primary) to first key (least significant)
        for info in key_infos.iter().rev() {
            let ord = if info.is_contiguous {
                match info.dtype.kind() {
                    DTypeKind::Float64 => {
                        let slice = unsafe { std::slice::from_raw_parts(info.ptr as *const f64, n) };
                        slice[a].partial_cmp(&slice[b]).unwrap_or(std::cmp::Ordering::Equal)
                    }
                    DTypeKind::Float32 => {
                        let slice = unsafe { std::slice::from_raw_parts(info.ptr as *const f32, n) };
                        slice[a].partial_cmp(&slice[b]).unwrap_or(std::cmp::Ordering::Equal)
                    }
                    DTypeKind::Int64 => {
                        let slice = unsafe { std::slice::from_raw_parts(info.ptr as *const i64, n) };
                        slice[a].cmp(&slice[b])
                    }
                    DTypeKind::Int32 => {
                        let slice = unsafe { std::slice::from_raw_parts(info.ptr as *const i32, n) };
                        slice[a].cmp(&slice[b])
                    }
                    DTypeKind::Int16 => {
                        let slice = unsafe { std::slice::from_raw_parts(info.ptr as *const i16, n) };
                        slice[a].cmp(&slice[b])
                    }
                    DTypeKind::Uint64 => {
                        let slice = unsafe { std::slice::from_raw_parts(info.ptr as *const u64, n) };
                        slice[a].cmp(&slice[b])
                    }
                    DTypeKind::Uint32 => {
                        let slice = unsafe { std::slice::from_raw_parts(info.ptr as *const u32, n) };
                        slice[a].cmp(&slice[b])
                    }
                    DTypeKind::Uint16 => {
                        let slice = unsafe { std::slice::from_raw_parts(info.ptr as *const u16, n) };
                        slice[a].cmp(&slice[b])
                    }
                    DTypeKind::Uint8 => {
                        let slice = unsafe { std::slice::from_raw_parts(info.ptr, n) };
                        slice[a].cmp(&slice[b])
                    }
                    _ => {
                        let ops = info.dtype.ops();
                        let itemsize = info.dtype.itemsize() as isize;
                        let a_off = (a as isize) * itemsize;
                        let b_off = (b as isize) * itemsize;
                        unsafe { ops.compare_elements(info.ptr, a_off, info.ptr, b_off) }
                    }
                }
            } else {
                // Non-contiguous: use DTypeOps with actual strides
                let ops = info.dtype.ops();
                let a_off = (a as isize) * info.stride;
                let b_off = (b as isize) * info.stride;
                unsafe { ops.compare_elements(info.ptr, a_off, info.ptr, b_off) }
            };

            if ord != std::cmp::Ordering::Equal {
                return ord;
            }
        }
        std::cmp::Ordering::Equal
    });

    // Create result array
    let mut result = RumpyArray::zeros(vec![n], DType::int64());
    let result_buffer = Arc::get_mut(result.buffer_mut()).expect("unique");
    let result_ptr = result_buffer.as_mut_ptr() as *mut i64;
    for (i, &idx) in indices.iter().enumerate() {
        unsafe { *result_ptr.add(i) = idx as i64; }
    }

    Some(result)
}
