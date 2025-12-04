//! Element-wise operations (ufunc-style).

use crate::array::{broadcast_shapes, DType, RumpyArray};
use std::sync::Arc;

/// Binary operation types.
#[derive(Clone, Copy)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
}

/// Unary operation types.
#[derive(Clone, Copy)]
pub enum UnaryOp {
    Neg,
    Abs,
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

/// Write a value to result buffer at linear index.
#[inline]
unsafe fn write_element(ptr: *mut u8, idx: usize, val: f64, dtype: DType) {
    match dtype {
        DType::Float64 => *(ptr as *mut f64).add(idx) = val,
        DType::Float32 => *(ptr as *mut f32).add(idx) = val as f32,
        DType::Int64 => *(ptr as *mut i64).add(idx) = val as i64,
        DType::Int32 => *(ptr as *mut i32).add(idx) = val as i32,
        DType::Bool => *ptr.add(idx) = (val != 0.0) as u8,
    }
}

/// Increment indices in row-major (C) order.
fn increment_indices(indices: &mut [usize], shape: &[usize]) {
    for i in (0..indices.len()).rev() {
        indices[i] += 1;
        if indices[i] < shape[i] {
            return;
        }
        indices[i] = 0;
    }
}

/// Type promotion for binary ops.
pub fn promote_dtype(a: DType, b: DType) -> DType {
    use DType::*;
    match (a, b) {
        (Float64, _) | (_, Float64) => Float64,
        (Float32, _) | (_, Float32) => Float32,
        (Int64, _) | (_, Int64) => Int64,
        (Int32, _) | (_, Int32) => Int32,
        (Bool, Bool) => Bool,
    }
}

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

    /// Unary operation.
    pub fn unary_op(&self, op: UnaryOp) -> RumpyArray {
        let f: fn(f64) -> f64 = match op {
            UnaryOp::Neg => |a| -a,
            UnaryOp::Abs => |a| a.abs(),
        };
        map_unary(self, f)
    }

    /// Sum all elements.
    pub fn sum(&self) -> f64 {
        reduce_all(self, 0.0, |a, b| a + b)
    }

    /// Product of all elements.
    pub fn prod(&self) -> f64 {
        reduce_all(self, 1.0, |a, b| a * b)
    }

    /// Maximum element.
    pub fn max(&self) -> f64 {
        reduce_all(self, f64::NEG_INFINITY, |a, b| a.max(b))
    }

    /// Minimum element.
    pub fn min(&self) -> f64 {
        reduce_all(self, f64::INFINITY, |a, b| a.min(b))
    }

    /// Mean of all elements.
    pub fn mean(&self) -> f64 {
        if self.size() == 0 {
            return f64::NAN;
        }
        self.sum() / self.size() as f64
    }
}
