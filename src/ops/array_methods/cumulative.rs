//! Cumulative operations on RumpyArray (cumsum, cumprod, diff).

use crate::array::dtype::DTypeKind;
use crate::array::{increment_indices, DType, RumpyArray};
use crate::ops::kernels::arithmetic::Sub;
use crate::ops::loops;
use num_complex::Complex;
use std::sync::Arc;

// ============================================================================
// diff helper functions
// ============================================================================

/// Fast 1D contiguous diff using kernel/loop system.
/// result[i] = src[i+1] - src[i]
#[inline]
fn diff_1d_contiguous(src_ptr: *const u8, result_ptr: *mut u8, n: usize, dtype: &DType) {
    // Use kernel/loop system for subtraction
    macro_rules! diff_typed {
        ($T:ty) => {{
            let a_slice = unsafe { std::slice::from_raw_parts((src_ptr as *const $T).add(1), n) };
            let b_slice = unsafe { std::slice::from_raw_parts(src_ptr as *const $T, n) };
            let out_slice = unsafe { std::slice::from_raw_parts_mut(result_ptr as *mut $T, n) };
            loops::map_binary(a_slice, b_slice, out_slice, Sub);
        }};
    }

    match dtype.kind() {
        DTypeKind::Float64 => diff_typed!(f64),
        DTypeKind::Float32 => diff_typed!(f32),
        DTypeKind::Int64 => diff_typed!(i64),
        DTypeKind::Int32 => diff_typed!(i32),
        DTypeKind::Int16 => diff_typed!(i16),
        DTypeKind::Uint64 => diff_typed!(u64),
        DTypeKind::Uint32 => diff_typed!(u32),
        DTypeKind::Uint16 => diff_typed!(u16),
        DTypeKind::Uint8 => diff_typed!(u8),
        DTypeKind::Complex128 => diff_typed!(Complex<f64>),
        DTypeKind::Complex64 => diff_typed!(Complex<f32>),
        _ => {
            // Fallback for unsupported dtypes (Float16, Bool, DateTime64)
            let itemsize = dtype.itemsize() as isize;
            let ops = dtype.ops();
            for i in 0..n {
                unsafe {
                    let v1 = ops.read_f64(src_ptr, (i as isize) * itemsize).unwrap_or(0.0);
                    let v2 = ops.read_f64(src_ptr, (i as isize + 1) * itemsize).unwrap_or(0.0);
                    ops.write_f64(result_ptr, i, v2 - v1);
                }
            }
        }
    }
}

/// Strided diff for N-D arrays along arbitrary axis.
/// Element-by-element via DTypeOps (no vectorization benefit for strided access).
fn diff_strided(
    src: &RumpyArray,
    result: &RumpyArray,
    axis_stride: isize,
    src_ptr: *const u8,
    result_ptr: *mut u8,
    dtype: &DType,
) {
    let out_shape = result.shape();
    let out_size: usize = out_shape.iter().product();
    let ops = dtype.ops();

    let mut out_indices = vec![0usize; src.ndim()];
    for i in 0..out_size {
        let offset1 = src.byte_offset_for(&out_indices);
        unsafe {
            let v1 = ops.read_f64(src_ptr, offset1).unwrap_or(0.0);
            let v2 = ops.read_f64(src_ptr, offset1 + axis_stride).unwrap_or(0.0);
            ops.write_f64(result_ptr, i, v2 - v1);
        }
        increment_indices(&mut out_indices, out_shape);
    }
}

impl RumpyArray {
    /// Calculate the n-th discrete difference along the given axis.
    pub fn diff(&self, n: usize, axis: usize) -> RumpyArray {
        assert!(axis < self.ndim(), "axis out of bounds");

        if n == 0 {
            return self.copy();
        }

        let axis_len = self.shape()[axis];
        if axis_len <= n {
            // Result has size 0 along axis
            let mut new_shape = self.shape().to_vec();
            new_shape[axis] = 0;
            return RumpyArray::zeros(new_shape, self.dtype().clone());
        }

        // Single diff: result[i] = input[i+1] - input[i]
        let mut new_shape = self.shape().to_vec();
        new_shape[axis] = axis_len - 1;

        let dtype = self.dtype().clone();
        let mut result = RumpyArray::zeros(new_shape.clone(), dtype.clone());

        let out_size: usize = new_shape.iter().product();
        if out_size == 0 {
            return result;
        }

        let result_buffer = Arc::get_mut(result.buffer_mut()).expect("unique");
        let result_ptr = result_buffer.as_mut_ptr();
        let src_ptr = self.data_ptr();
        let axis_stride = self.strides()[axis];

        // Fast path for 1D contiguous case
        if self.ndim() == 1 && self.is_c_contiguous() {
            diff_1d_contiguous(src_ptr, result_ptr, out_size, &dtype);
        } else {
            // General strided case
            diff_strided(self, &result, axis_stride, src_ptr, result_ptr, &dtype);
        }

        // Apply recursively for n > 1
        if n > 1 {
            result.diff(n - 1, axis)
        } else {
            result
        }
    }

    /// Generic cumulative operation along axis (or flattened if axis is None).
    fn cumulative_op<F>(&self, axis: Option<usize>, identity: f64, op: F) -> RumpyArray
    where
        F: Fn(f64, f64) -> f64,
    {
        match axis {
            None => {
                let size = self.size();
                let mut result = RumpyArray::zeros(vec![size], self.dtype());
                if size == 0 {
                    return result;
                }

                let dtype = self.dtype();
                let buffer = result.buffer_mut();
                let result_buffer = Arc::get_mut(buffer).expect("buffer must be unique");
                let result_ptr = result_buffer.as_mut_ptr();
                let ops = dtype.ops();
                let src_ptr = self.data_ptr();

                let mut acc = identity;
                for (i, offset) in self.iter_offsets().enumerate() {
                    let val = unsafe { ops.read_f64(src_ptr, offset) }.unwrap_or(0.0);
                    acc = op(acc, val);
                    unsafe { ops.write_f64(result_ptr, i, acc); }
                }
                result
            }
            Some(axis) => {
                let shape = self.shape().to_vec();
                let dtype = self.dtype();
                let mut result = RumpyArray::zeros(shape.clone(), dtype.clone());
                let size = result.size();
                if size == 0 {
                    return result;
                }

                let buffer = result.buffer_mut();
                let result_buffer = Arc::get_mut(buffer).expect("buffer must be unique");
                let result_ptr = result_buffer.as_mut_ptr();
                let ops = dtype.ops();

                let axis_len = shape[axis];
                let outer_size = size / axis_len;

                let mut outer_shape: Vec<usize> = shape[..axis].to_vec();
                outer_shape.extend_from_slice(&shape[axis + 1..]);
                if outer_shape.is_empty() {
                    outer_shape = vec![1];
                }

                let mut outer_indices = vec![0usize; outer_shape.len()];
                for _ in 0..outer_size {
                    let mut in_indices: Vec<usize> = outer_indices[..axis.min(outer_indices.len())].to_vec();
                    in_indices.push(0);
                    if axis < self.ndim() - 1 && outer_indices.len() > axis {
                        in_indices.extend_from_slice(&outer_indices[axis..]);
                    } else if axis < self.ndim() - 1 {
                        in_indices.extend_from_slice(&outer_indices[..]);
                    }

                    let mut acc = identity;
                    for j in 0..axis_len {
                        in_indices[axis] = j;
                        acc = op(acc, self.get_element(&in_indices));
                        let flat_idx = self.flat_index_for(&in_indices);
                        unsafe { ops.write_f64(result_ptr, flat_idx, acc); }
                    }
                    increment_indices(&mut outer_indices, &outer_shape);
                }
                result
            }
        }
    }

    /// Cumulative sum along axis (or flattened if axis is None).
    pub fn cumsum(&self, axis: Option<usize>) -> RumpyArray {
        // Try typed dispatch first
        if let Some(result) = crate::ops::dispatch::dispatch_cumsum(self, axis) {
            return result;
        }
        // Fallback for Bool, DateTime, Float16
        self.cumulative_op(axis, 0.0, |acc, x| acc + x)
    }

    /// Cumulative product along axis (or flattened if axis is None).
    pub fn cumprod(&self, axis: Option<usize>) -> RumpyArray {
        // Try typed dispatch first
        if let Some(result) = crate::ops::dispatch::dispatch_cumprod(self, axis) {
            return result;
        }
        // Fallback for Bool, DateTime, Float16
        self.cumulative_op(axis, 1.0, |acc, x| acc * x)
    }

    /// Calculate flat index for given n-dimensional indices (C-order).
    pub(crate) fn flat_index_for(&self, indices: &[usize]) -> usize {
        let shape = self.shape();
        let mut flat = 0;
        let mut stride = 1;
        for i in (0..indices.len()).rev() {
            flat += indices[i] * stride;
            stride *= shape[i];
        }
        flat
    }
}
