//! Bitwise operations (dtype-preserving for integers/bool).
//!
//! Provides: bitwise_and, bitwise_or, bitwise_xor, bitwise_not, left_shift, right_shift

use crate::array::{increment_indices, promote_dtype, RumpyArray};
use crate::array::dtype::BitwiseOp;
use crate::ops::registry::{registry, BinaryLoopFn};
use crate::ops::comparison::broadcast_binary_setup;
use std::sync::Arc;

/// Apply a binary bitwise operation element-wise, preserving dtype.
fn map_binary_bitwise(a: &RumpyArray, b: &RumpyArray, op: BitwiseOp) -> Option<RumpyArray> {
    let out_dtype = promote_dtype(&a.dtype(), &b.dtype());
    let (a_bc, b_bc, mut result, out_shape) = broadcast_binary_setup(a, b, out_dtype.clone())?;
    let size = result.size();
    if size == 0 {
        return Some(result);
    }

    let buffer = result.buffer_mut();
    let result_buffer = Arc::get_mut(buffer).expect("buffer must be unique");
    let result_ptr = result_buffer.as_mut_ptr();
    let a_ptr = a_bc.data_ptr();
    let b_ptr = b_bc.data_ptr();
    let itemsize = out_dtype.itemsize() as isize;

    // Try registry first for contiguous same-type operations
    let a_kind = a.dtype().kind();
    let b_kind = b.dtype().kind();
    if a_kind == b_kind {
        let reg = registry().read().unwrap();
        if let Some(loop_fn) = reg.lookup_bitwise_binary(op, a_kind.clone()) {
            let a_contig = a_bc.is_c_contiguous() && a.shape() == out_shape.as_slice();
            let b_contig = b_bc.is_c_contiguous() && b.shape() == out_shape.as_slice();

            if a_contig && b_contig {
                // Fast path: both contiguous, same shape as output
                let strides = (itemsize, itemsize, itemsize);
                unsafe { loop_fn(a_ptr, b_ptr, result_ptr, size, strides); }
                return Some(result);
            }
            // Strided but still use registry loop for vectorization
            drop(reg);
            return map_binary_bitwise_strided(a_bc, b_bc, result, out_shape, op, loop_fn, itemsize);
        }
    }

    // Fallback to trait dispatch
    let ops = out_dtype.ops();
    let mut indices = vec![0usize; out_shape.len()];
    for i in 0..size {
        let a_offset = a_bc.byte_offset_for(&indices);
        let b_offset = b_bc.byte_offset_for(&indices);
        let ok = unsafe { ops.bitwise_op(op, a_ptr, a_offset, b_ptr, b_offset, result_ptr, i) };
        if !ok {
            return None;
        }
        increment_indices(&mut indices, &out_shape);
    }
    Some(result)
}

/// Strided bitwise binary using registry loop.
fn map_binary_bitwise_strided(
    a_bc: RumpyArray,
    b_bc: RumpyArray,
    mut result: RumpyArray,
    out_shape: Vec<usize>,
    _op: BitwiseOp,
    loop_fn: BinaryLoopFn,
    itemsize: isize,
) -> Option<RumpyArray> {
    let size = result.size();
    let buffer = result.buffer_mut();
    let result_buffer = Arc::get_mut(buffer).expect("buffer must be unique");
    let result_ptr = result_buffer.as_mut_ptr();

    // Process in contiguous chunks along the last axis when possible
    let ndim = out_shape.len();
    if ndim > 0 {
        let inner_size = out_shape[ndim - 1];
        let a_inner_contig = a_bc.strides().last().copied().unwrap_or(0) == itemsize;
        let b_inner_contig = b_bc.strides().last().copied().unwrap_or(0) == itemsize;

        if a_inner_contig && b_inner_contig && inner_size > 0 {
            let outer_size = size / inner_size;
            let mut indices = vec![0usize; ndim];
            for _ in 0..outer_size {
                let a_offset = a_bc.byte_offset_for(&indices);
                let b_offset = b_bc.byte_offset_for(&indices);
                let out_offset = (0..ndim).fold(0isize, |acc, d| {
                    acc + (indices[d] as isize) * result.strides()[d]
                });
                unsafe {
                    loop_fn(
                        a_bc.data_ptr().offset(a_offset),
                        b_bc.data_ptr().offset(b_offset),
                        result_ptr.offset(out_offset),
                        inner_size,
                        (itemsize, itemsize, itemsize),
                    );
                }
                // Advance outer indices
                for d in (0..ndim - 1).rev() {
                    indices[d] += 1;
                    if indices[d] < out_shape[d] {
                        break;
                    }
                    indices[d] = 0;
                }
                indices[ndim - 1] = 0;
            }
            return Some(result);
        }
    }

    // Fully strided fallback
    let a_ptr = a_bc.data_ptr();
    let b_ptr = b_bc.data_ptr();
    let mut indices = vec![0usize; ndim];
    for i in 0..size {
        let a_offset = a_bc.byte_offset_for(&indices);
        let b_offset = b_bc.byte_offset_for(&indices);
        unsafe {
            loop_fn(
                a_ptr.offset(a_offset),
                b_ptr.offset(b_offset),
                result_ptr.add(i * itemsize as usize),
                1,
                (itemsize, itemsize, itemsize),
            );
        }
        increment_indices(&mut indices, &out_shape);
    }
    Some(result)
}

/// Apply bitwise NOT element-wise, preserving dtype.
fn map_unary_bitwise_not(a: &RumpyArray) -> Option<RumpyArray> {
    let mut result = RumpyArray::zeros(a.shape().to_vec(), a.dtype().clone());
    let size = result.size();
    if size == 0 {
        return Some(result);
    }

    let out_dtype = result.dtype().clone();
    let buffer = result.buffer_mut();
    let result_buffer = Arc::get_mut(buffer).expect("buffer must be unique");
    let result_ptr = result_buffer.as_mut_ptr();
    let a_ptr = a.data_ptr();
    let itemsize = out_dtype.itemsize() as isize;

    // Try registry first for contiguous operations
    let a_kind = a.dtype().kind();
    let reg = registry().read().unwrap();
    if let Some(loop_fn) = reg.lookup_bitwise_not(a_kind.clone()) {
        if a.is_c_contiguous() {
            // Fast path: contiguous
            let strides = (itemsize, itemsize);
            unsafe { loop_fn(a_ptr, result_ptr, size, strides); }
            return Some(result);
        }
        // Strided
        drop(reg);
        let mut indices = vec![0usize; a.shape().len()];
        for i in 0..size {
            let a_offset = a.byte_offset_for(&indices);
            unsafe {
                loop_fn(
                    a_ptr.offset(a_offset),
                    result_ptr.add(i * itemsize as usize),
                    1,
                    (itemsize, itemsize),
                );
            }
            increment_indices(&mut indices, a.shape());
        }
        return Some(result);
    }
    drop(reg);

    // Fallback to trait dispatch
    let ops = out_dtype.ops();
    let mut indices = vec![0usize; a.shape().len()];
    for i in 0..size {
        let a_offset = a.byte_offset_for(&indices);
        let ok = unsafe { ops.bitwise_not(a_ptr, a_offset, result_ptr, i) };
        if !ok {
            return None;
        }
        increment_indices(&mut indices, a.shape());
    }
    Some(result)
}

/// Element-wise bitwise AND.
pub fn bitwise_and(a: &RumpyArray, b: &RumpyArray) -> Option<RumpyArray> {
    map_binary_bitwise(a, b, BitwiseOp::And)
}

/// Element-wise bitwise OR.
pub fn bitwise_or(a: &RumpyArray, b: &RumpyArray) -> Option<RumpyArray> {
    map_binary_bitwise(a, b, BitwiseOp::Or)
}

/// Element-wise bitwise XOR.
pub fn bitwise_xor(a: &RumpyArray, b: &RumpyArray) -> Option<RumpyArray> {
    map_binary_bitwise(a, b, BitwiseOp::Xor)
}

/// Element-wise bitwise NOT (invert).
pub fn bitwise_not(a: &RumpyArray) -> Option<RumpyArray> {
    map_unary_bitwise_not(a)
}

/// Element-wise left shift.
pub fn left_shift(a: &RumpyArray, b: &RumpyArray) -> Option<RumpyArray> {
    map_binary_bitwise(a, b, BitwiseOp::LeftShift)
}

/// Element-wise right shift.
pub fn right_shift(a: &RumpyArray, b: &RumpyArray) -> Option<RumpyArray> {
    map_binary_bitwise(a, b, BitwiseOp::RightShift)
}
