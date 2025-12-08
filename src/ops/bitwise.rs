//! Bitwise operations (dtype-preserving for integers/bool).
//!
//! Provides: bitwise_and, bitwise_or, bitwise_xor, bitwise_not, left_shift, right_shift
//!
//! Uses kernel/dispatch system for integer types and bool.
//! Falls back to trait dispatch for other types.

use crate::array::{increment_indices, promote_dtype, RumpyArray};
use crate::array::dtype::BitwiseOp;
use crate::ops::comparison::broadcast_binary_setup;
use crate::ops::dispatch;

/// Apply a binary bitwise operation element-wise, preserving dtype.
fn map_binary_bitwise(a: &RumpyArray, b: &RumpyArray, op: BitwiseOp) -> Option<RumpyArray> {
    let out_dtype = promote_dtype(&a.dtype(), &b.dtype());
    let (a_bc, b_bc, result, out_shape) = broadcast_binary_setup(a, b, out_dtype.clone())?;

    if result.size() == 0 {
        return Some(result);
    }

    // Try kernel/dispatch system first
    let dispatched = match op {
        BitwiseOp::And => dispatch::dispatch_bitwise_and(&a_bc, &b_bc, &out_shape),
        BitwiseOp::Or => dispatch::dispatch_bitwise_or(&a_bc, &b_bc, &out_shape),
        BitwiseOp::Xor => dispatch::dispatch_bitwise_xor(&a_bc, &b_bc, &out_shape),
        BitwiseOp::LeftShift => dispatch::dispatch_left_shift(&a_bc, &b_bc, &out_shape),
        BitwiseOp::RightShift => dispatch::dispatch_right_shift(&a_bc, &b_bc, &out_shape),
    };

    if let Some(result) = dispatched {
        return Some(result);
    }

    // Fallback to trait dispatch
    map_binary_bitwise_trait(&a_bc, &b_bc, out_dtype, out_shape, op)
}

/// Trait-based fallback for bitwise binary operations.
fn map_binary_bitwise_trait(
    a_bc: &RumpyArray,
    b_bc: &RumpyArray,
    out_dtype: crate::array::DType,
    out_shape: Vec<usize>,
    op: BitwiseOp,
) -> Option<RumpyArray> {
    use std::sync::Arc;

    let mut result = RumpyArray::zeros(out_shape.clone(), out_dtype.clone());
    let size = result.size();

    let buffer = result.buffer_mut();
    let result_buffer = Arc::get_mut(buffer).expect("buffer must be unique");
    let result_ptr = result_buffer.as_mut_ptr();
    let a_ptr = a_bc.data_ptr();
    let b_ptr = b_bc.data_ptr();

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

/// Apply bitwise NOT element-wise, preserving dtype.
fn map_unary_bitwise_not(a: &RumpyArray) -> Option<RumpyArray> {
    // Try kernel/dispatch system first
    if let Some(result) = dispatch::dispatch_bitwise_not(a) {
        return Some(result);
    }

    // Fallback to trait dispatch
    map_unary_bitwise_not_trait(a)
}

/// Trait-based fallback for bitwise NOT.
fn map_unary_bitwise_not_trait(a: &RumpyArray) -> Option<RumpyArray> {
    use std::sync::Arc;

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
