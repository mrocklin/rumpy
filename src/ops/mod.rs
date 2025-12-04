//! Element-wise operations (ufunc-style).

use crate::array::{broadcast_shapes, increment_indices, promote_dtype, write_element, RumpyArray};
use std::sync::Arc;

/// Binary operation types.
#[derive(Clone, Copy)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
}

// ============================================================================
// Core ufunc machinery
// ============================================================================

/// Apply a unary function element-wise, returning a new array.
fn map_unary<F>(arr: &RumpyArray, f: F) -> RumpyArray
where
    F: Fn(f64) -> f64,
{
    let mut result = RumpyArray::zeros(arr.shape().to_vec(), arr.dtype());
    let size = arr.size();
    if size == 0 {
        return result;
    }

    let buffer = result.buffer_mut();
    let result_buffer = Arc::get_mut(buffer).expect("buffer must be unique");
    let result_ptr = result_buffer.as_mut_ptr();

    let mut indices = vec![0usize; arr.ndim()];
    for i in 0..size {
        let val = f(arr.get_element(&indices));
        unsafe { write_element(result_ptr, i, val, arr.dtype()); }
        increment_indices(&mut indices, arr.shape());
    }
    result
}

/// Apply a binary function element-wise with broadcasting.
fn map_binary<F>(a: &RumpyArray, b: &RumpyArray, f: F) -> Option<RumpyArray>
where
    F: Fn(f64, f64) -> f64,
{
    let out_shape = broadcast_shapes(a.shape(), b.shape())?;
    let a = a.broadcast_to(&out_shape)?;
    let b = b.broadcast_to(&out_shape)?;

    let result_dtype = promote_dtype(a.dtype(), b.dtype());
    let mut result = RumpyArray::zeros(out_shape.clone(), result_dtype);
    let size = result.size();
    if size == 0 {
        return Some(result);
    }

    let buffer = result.buffer_mut();
    let result_buffer = Arc::get_mut(buffer).expect("buffer must be unique");
    let result_ptr = result_buffer.as_mut_ptr();

    let mut indices = vec![0usize; out_shape.len()];
    for i in 0..size {
        let val = f(a.get_element(&indices), b.get_element(&indices));
        unsafe { write_element(result_ptr, i, val, result_dtype); }
        increment_indices(&mut indices, &out_shape);
    }
    Some(result)
}

/// Reduce array along all axes using a binary function.
fn reduce_all<F>(arr: &RumpyArray, init: f64, f: F) -> f64
where
    F: Fn(f64, f64) -> f64,
{
    let size = arr.size();
    if size == 0 {
        return init;
    }

    let mut acc = init;
    let mut indices = vec![0usize; arr.ndim()];
    for _ in 0..size {
        acc = f(acc, arr.get_element(&indices));
        increment_indices(&mut indices, arr.shape());
    }
    acc
}

/// Reduce array along a specific axis using a binary function.
fn reduce_axis<F>(arr: &RumpyArray, axis: usize, init: f64, f: F) -> RumpyArray
where
    F: Fn(f64, f64) -> f64,
{
    // Output shape: remove the reduction axis
    let mut out_shape: Vec<usize> = arr.shape().to_vec();
    let axis_len = out_shape.remove(axis);

    // Handle edge cases
    if out_shape.is_empty() {
        out_shape = vec![1]; // Scalar result wrapped in 1D array
    }

    let mut result = RumpyArray::zeros(out_shape.clone(), arr.dtype());
    let out_size = result.size();

    if out_size == 0 || axis_len == 0 {
        // Fill with init for empty reduction
        let buffer = result.buffer_mut();
        let result_buffer = Arc::get_mut(buffer).expect("buffer must be unique");
        let result_ptr = result_buffer.as_mut_ptr();
        for i in 0..out_size {
            unsafe { write_element(result_ptr, i, init, arr.dtype()); }
        }
        return result;
    }

    let buffer = result.buffer_mut();
    let result_buffer = Arc::get_mut(buffer).expect("buffer must be unique");
    let result_ptr = result_buffer.as_mut_ptr();

    // Iterate over output positions
    let mut out_indices = vec![0usize; out_shape.len()];
    for i in 0..out_size {
        // Build input indices by inserting axis position
        let mut in_indices: Vec<usize> = out_indices[..axis.min(out_indices.len())].to_vec();
        in_indices.push(0); // placeholder for axis
        if axis < arr.ndim() - 1 {
            in_indices.extend_from_slice(&out_indices[axis..]);
        }

        // Reduce along axis
        let mut acc = init;
        for j in 0..axis_len {
            in_indices[axis] = j;
            acc = f(acc, arr.get_element(&in_indices));
        }

        unsafe { write_element(result_ptr, i, acc, arr.dtype()); }
        increment_indices(&mut out_indices, &out_shape);
    }

    result
}

// Type promotion uses crate::array::promote_dtype

// ============================================================================
// Public API using ufunc machinery
// ============================================================================

impl RumpyArray {
    /// Element-wise binary operation with broadcasting.
    pub fn binary_op(&self, other: &RumpyArray, op: BinaryOp) -> Option<RumpyArray> {
        let f = match op {
            BinaryOp::Add => |a, b| a + b,
            BinaryOp::Sub => |a, b| a - b,
            BinaryOp::Mul => |a, b| a * b,
            BinaryOp::Div => |a, b| a / b,
        };
        map_binary(self, other, f)
    }

    /// Element-wise operation with scalar (arr op scalar).
    pub fn scalar_op(&self, scalar: f64, op: BinaryOp) -> RumpyArray {
        match op {
            BinaryOp::Add => map_unary(self, |a| a + scalar),
            BinaryOp::Sub => map_unary(self, |a| a - scalar),
            BinaryOp::Mul => map_unary(self, |a| a * scalar),
            BinaryOp::Div => map_unary(self, |a| a / scalar),
        }
    }

    /// Scalar on left side (scalar op arr).
    pub fn rscalar_op(&self, scalar: f64, op: BinaryOp) -> RumpyArray {
        match op {
            BinaryOp::Add => map_unary(self, |a| scalar + a),
            BinaryOp::Sub => map_unary(self, |a| scalar - a),
            BinaryOp::Mul => map_unary(self, |a| scalar * a),
            BinaryOp::Div => map_unary(self, |a| scalar / a),
        }
    }

    /// Negate each element.
    pub fn neg(&self) -> RumpyArray {
        map_unary(self, |x| -x)
    }

    /// Absolute value of each element.
    pub fn abs(&self) -> RumpyArray {
        map_unary(self, |x| x.abs())
    }

    /// Sum all elements.
    pub fn sum(&self) -> f64 {
        reduce_all(self, 0.0, |a, b| a + b)
    }

    /// Sum along axis.
    pub fn sum_axis(&self, axis: usize) -> RumpyArray {
        reduce_axis(self, axis, 0.0, |a, b| a + b)
    }

    /// Product of all elements.
    pub fn prod(&self) -> f64 {
        reduce_all(self, 1.0, |a, b| a * b)
    }

    /// Product along axis.
    pub fn prod_axis(&self, axis: usize) -> RumpyArray {
        reduce_axis(self, axis, 1.0, |a, b| a * b)
    }

    /// Maximum element.
    pub fn max(&self) -> f64 {
        reduce_all(self, f64::NEG_INFINITY, |a, b| a.max(b))
    }

    /// Maximum along axis.
    pub fn max_axis(&self, axis: usize) -> RumpyArray {
        reduce_axis(self, axis, f64::NEG_INFINITY, |a, b| a.max(b))
    }

    /// Minimum element.
    pub fn min(&self) -> f64 {
        reduce_all(self, f64::INFINITY, |a, b| a.min(b))
    }

    /// Minimum along axis.
    pub fn min_axis(&self, axis: usize) -> RumpyArray {
        reduce_axis(self, axis, f64::INFINITY, |a, b| a.min(b))
    }

    /// Mean of all elements.
    pub fn mean(&self) -> f64 {
        if self.size() == 0 {
            return f64::NAN;
        }
        self.sum() / self.size() as f64
    }

    /// Mean along axis.
    pub fn mean_axis(&self, axis: usize) -> RumpyArray {
        let sum = self.sum_axis(axis);
        let count = self.shape()[axis] as f64;
        map_unary(&sum, |x| x / count)
    }

    // Math ufuncs

    /// Square root of each element.
    pub fn sqrt(&self) -> RumpyArray {
        map_unary(self, |x| x.sqrt())
    }

    /// Exponential (e^x) of each element.
    pub fn exp(&self) -> RumpyArray {
        map_unary(self, |x| x.exp())
    }

    /// Natural logarithm of each element.
    pub fn log(&self) -> RumpyArray {
        map_unary(self, |x| x.ln())
    }

    /// Sine of each element (radians).
    pub fn sin(&self) -> RumpyArray {
        map_unary(self, |x| x.sin())
    }

    /// Cosine of each element (radians).
    pub fn cos(&self) -> RumpyArray {
        map_unary(self, |x| x.cos())
    }

    /// Tangent of each element (radians).
    pub fn tan(&self) -> RumpyArray {
        map_unary(self, |x| x.tan())
    }
}
