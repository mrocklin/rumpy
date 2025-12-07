//! Comparison and logical operations.
//!
//! This module provides:
//! - Element-wise comparisons: equal, less, greater, etc.
//! - Approximate equality: isclose, allclose, array_equal
//! - Logical operations: logical_and, logical_or, logical_not, logical_xor

use crate::array::{broadcast_shapes, increment_indices, DType, RumpyArray};
use crate::ops::ufunc::map_compare_op;
use crate::ops::ComparisonOp;
use std::sync::Arc;

// ============================================================================
// Helper functions
// ============================================================================

/// Set up broadcasting for a binary operation, returning broadcasted arrays and output.
/// Returns (a_broadcasted, b_broadcasted, result_array, output_shape).
pub(crate) fn broadcast_binary_setup(
    a: &RumpyArray,
    b: &RumpyArray,
    out_dtype: DType,
) -> Option<(RumpyArray, RumpyArray, RumpyArray, Vec<usize>)> {
    let out_shape = broadcast_shapes(a.shape(), b.shape())?;
    let a_bc = a.broadcast_to(&out_shape)?;
    let b_bc = b.broadcast_to(&out_shape)?;
    let result = RumpyArray::zeros(out_shape.clone(), out_dtype);
    Some((a_bc, b_bc, result, out_shape))
}

/// Apply a binary function element-wise, writing bool results.
/// The function receives (a_value, b_value) as f64 and returns bool.
pub fn map_binary_to_bool<F>(a: &RumpyArray, b: &RumpyArray, f: F) -> Option<RumpyArray>
where
    F: Fn(f64, f64) -> bool,
{
    let (a_bc, b_bc, mut result, out_shape) = broadcast_binary_setup(a, b, DType::bool())?;
    let size = result.size();
    if size == 0 {
        return Some(result);
    }

    let buffer = result.buffer_mut();
    let result_buffer = Arc::get_mut(buffer).expect("buffer must be unique");
    let result_ptr = result_buffer.as_mut_ptr();

    let mut indices = vec![0usize; out_shape.len()];
    for i in 0..size {
        let a_val = a_bc.get_element(&indices);
        let b_val = b_bc.get_element(&indices);
        unsafe { *result_ptr.add(i) = if f(a_val, b_val) { 1 } else { 0 }; }
        increment_indices(&mut indices, &out_shape);
    }
    Some(result)
}

// ============================================================================
// Logical operations (element-wise boolean logic)
// ============================================================================

/// Apply a binary operation on truthiness values, returning a bool array.
fn map_binary_logical<F>(a: &RumpyArray, b: &RumpyArray, f: F) -> Option<RumpyArray>
where
    F: Fn(bool, bool) -> bool,
{
    let (a_bc, b_bc, mut result, out_shape) = broadcast_binary_setup(a, b, DType::bool())?;
    let size = result.size();
    if size == 0 {
        return Some(result);
    }

    let buffer = result.buffer_mut();
    let result_buffer = Arc::get_mut(buffer).expect("buffer must be unique");
    let result_ptr = result_buffer.as_mut_ptr();
    let a_ptr = a_bc.data_ptr();
    let b_ptr = b_bc.data_ptr();
    let a_dtype = a_bc.dtype();
    let b_dtype = b_bc.dtype();
    let a_ops = a_dtype.ops();
    let b_ops = b_dtype.ops();

    let mut indices = vec![0usize; out_shape.len()];
    for i in 0..size {
        let a_offset = a_bc.byte_offset_for(&indices);
        let b_offset = b_bc.byte_offset_for(&indices);
        let a_truthy = unsafe { a_ops.is_truthy(a_ptr, a_offset) };
        let b_truthy = unsafe { b_ops.is_truthy(b_ptr, b_offset) };
        unsafe { *result_ptr.add(i) = if f(a_truthy, b_truthy) { 1 } else { 0 }; }
        increment_indices(&mut indices, &out_shape);
    }
    Some(result)
}

/// Apply a unary operation on truthiness values, returning a bool array.
fn map_unary_logical<F>(a: &RumpyArray, f: F) -> RumpyArray
where
    F: Fn(bool) -> bool,
{
    let mut result = RumpyArray::zeros(a.shape().to_vec(), DType::bool());
    let size = result.size();
    if size == 0 {
        return result;
    }

    let buffer = result.buffer_mut();
    let result_buffer = Arc::get_mut(buffer).expect("buffer must be unique");
    let result_ptr = result_buffer.as_mut_ptr();
    let a_ptr = a.data_ptr();
    let a_dtype = a.dtype();
    let a_ops = a_dtype.ops();

    for (i, offset) in a.iter_offsets().enumerate() {
        let a_truthy = unsafe { a_ops.is_truthy(a_ptr, offset) };
        unsafe { *result_ptr.add(i) = if f(a_truthy) { 1 } else { 0 }; }
    }
    result
}

/// Element-wise logical AND.
pub fn logical_and(a: &RumpyArray, b: &RumpyArray) -> Option<RumpyArray> {
    map_binary_logical(a, b, |x, y| x && y)
}

/// Element-wise logical OR.
pub fn logical_or(a: &RumpyArray, b: &RumpyArray) -> Option<RumpyArray> {
    map_binary_logical(a, b, |x, y| x || y)
}

/// Element-wise logical XOR.
pub fn logical_xor(a: &RumpyArray, b: &RumpyArray) -> Option<RumpyArray> {
    map_binary_logical(a, b, |x, y| x != y)
}

/// Element-wise logical NOT.
pub fn logical_not(a: &RumpyArray) -> RumpyArray {
    map_unary_logical(a, |x| !x)
}

// ============================================================================
// Comparison functions (element-wise, return bool array)
// ============================================================================

/// Element-wise equality test.
pub fn equal(a: &RumpyArray, b: &RumpyArray) -> Option<RumpyArray> {
    map_compare_op(a, b, ComparisonOp::Eq)
}

/// Element-wise not-equal test.
pub fn not_equal(a: &RumpyArray, b: &RumpyArray) -> Option<RumpyArray> {
    map_compare_op(a, b, ComparisonOp::Ne)
}

/// Element-wise less-than test.
pub fn less(a: &RumpyArray, b: &RumpyArray) -> Option<RumpyArray> {
    map_compare_op(a, b, ComparisonOp::Lt)
}

/// Element-wise less-than-or-equal test.
pub fn less_equal(a: &RumpyArray, b: &RumpyArray) -> Option<RumpyArray> {
    map_compare_op(a, b, ComparisonOp::Le)
}

/// Element-wise greater-than test.
pub fn greater(a: &RumpyArray, b: &RumpyArray) -> Option<RumpyArray> {
    map_compare_op(a, b, ComparisonOp::Gt)
}

/// Element-wise greater-than-or-equal test.
pub fn greater_equal(a: &RumpyArray, b: &RumpyArray) -> Option<RumpyArray> {
    map_compare_op(a, b, ComparisonOp::Ge)
}

/// Element-wise approximate equality test: |a - b| <= atol + rtol * |b|.
/// Handles NaN and infinity according to NumPy semantics.
pub fn isclose(a: &RumpyArray, b: &RumpyArray, rtol: f64, atol: f64, equal_nan: bool) -> Option<RumpyArray> {
    map_binary_to_bool(a, b, |x, y| {
        if x.is_nan() && y.is_nan() {
            equal_nan
        } else if x.is_nan() || y.is_nan() {
            false
        } else if x.is_infinite() || y.is_infinite() {
            x == y  // Both must be same infinity
        } else {
            (x - y).abs() <= atol + rtol * y.abs()
        }
    })
}

/// Test if all elements are approximately equal.
pub fn allclose(a: &RumpyArray, b: &RumpyArray, rtol: f64, atol: f64, equal_nan: bool) -> Option<bool> {
    let close = isclose(a, b, rtol, atol, equal_nan)?;
    Some(close.all())
}

/// Test if two arrays have the same shape and elements.
pub fn array_equal(a: &RumpyArray, b: &RumpyArray) -> bool {
    if a.shape() != b.shape() {
        return false;
    }
    equal(a, b).map(|r| r.all()).unwrap_or(false)
}
