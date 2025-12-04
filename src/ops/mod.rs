//! Element-wise operations on arrays.

use crate::array::{DType, RumpyArray};
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

impl RumpyArray {
    /// Element-wise binary operation (same shape required).
    pub fn binary_op(&self, other: &RumpyArray, op: BinaryOp) -> Option<RumpyArray> {
        if self.shape() != other.shape() {
            return None;
        }

        let result_dtype = promote_dtype(self.dtype(), other.dtype());
        let mut result = RumpyArray::zeros(self.shape().to_vec(), result_dtype);

        let size = self.size();
        if size == 0 {
            return Some(result);
        }

        // Get mutable access to result buffer
        let buffer = result.buffer_mut();
        let result_buffer = Arc::get_mut(buffer).expect("result buffer must be unique");
        let result_ptr = result_buffer.as_mut_ptr();

        // Iterate over all elements
        let mut indices = vec![0usize; self.ndim()];
        for i in 0..size {
            let a = self.get_element(&indices);
            let b = other.get_element(&indices);
            let val = apply_binary_op(a, b, op);

            // Write to result (contiguous, so linear index works)
            unsafe {
                match result_dtype {
                    DType::Float64 => {
                        *(result_ptr as *mut f64).add(i) = val;
                    }
                    DType::Float32 => {
                        *(result_ptr as *mut f32).add(i) = val as f32;
                    }
                    DType::Int64 => {
                        *(result_ptr as *mut i64).add(i) = val as i64;
                    }
                    DType::Int32 => {
                        *(result_ptr as *mut i32).add(i) = val as i32;
                    }
                    DType::Bool => {
                        *result_ptr.add(i) = (val != 0.0) as u8;
                    }
                }
            }

            // Increment indices (row-major order)
            increment_indices(&mut indices, self.shape());
        }

        Some(result)
    }

    /// Element-wise operation with scalar.
    pub fn scalar_op(&self, scalar: f64, op: BinaryOp) -> RumpyArray {
        let mut result = RumpyArray::zeros(self.shape().to_vec(), self.dtype());

        let size = self.size();
        if size == 0 {
            return result;
        }

        let buffer = result.buffer_mut();
        let result_buffer = Arc::get_mut(buffer).expect("result buffer must be unique");
        let result_ptr = result_buffer.as_mut_ptr();

        let mut indices = vec![0usize; self.ndim()];
        for i in 0..size {
            let a = self.get_element(&indices);
            let val = apply_binary_op(a, scalar, op);

            unsafe {
                match self.dtype() {
                    DType::Float64 => {
                        *(result_ptr as *mut f64).add(i) = val;
                    }
                    DType::Float32 => {
                        *(result_ptr as *mut f32).add(i) = val as f32;
                    }
                    DType::Int64 => {
                        *(result_ptr as *mut i64).add(i) = val as i64;
                    }
                    DType::Int32 => {
                        *(result_ptr as *mut i32).add(i) = val as i32;
                    }
                    DType::Bool => {
                        *result_ptr.add(i) = (val != 0.0) as u8;
                    }
                }
            }

            increment_indices(&mut indices, self.shape());
        }

        result
    }

    /// Scalar on left side (scalar - arr, scalar / arr).
    pub fn rscalar_op(&self, scalar: f64, op: BinaryOp) -> RumpyArray {
        let mut result = RumpyArray::zeros(self.shape().to_vec(), self.dtype());

        let size = self.size();
        if size == 0 {
            return result;
        }

        let buffer = result.buffer_mut();
        let result_buffer = Arc::get_mut(buffer).expect("result buffer must be unique");
        let result_ptr = result_buffer.as_mut_ptr();

        let mut indices = vec![0usize; self.ndim()];
        for i in 0..size {
            let a = self.get_element(&indices);
            let val = apply_binary_op(scalar, a, op);  // Note: scalar first

            unsafe {
                match self.dtype() {
                    DType::Float64 => {
                        *(result_ptr as *mut f64).add(i) = val;
                    }
                    DType::Float32 => {
                        *(result_ptr as *mut f32).add(i) = val as f32;
                    }
                    DType::Int64 => {
                        *(result_ptr as *mut i64).add(i) = val as i64;
                    }
                    DType::Int32 => {
                        *(result_ptr as *mut i32).add(i) = val as i32;
                    }
                    DType::Bool => {
                        *result_ptr.add(i) = (val != 0.0) as u8;
                    }
                }
            }

            increment_indices(&mut indices, self.shape());
        }

        result
    }

    /// Unary operation.
    pub fn unary_op(&self, op: UnaryOp) -> RumpyArray {
        let mut result = RumpyArray::zeros(self.shape().to_vec(), self.dtype());

        let size = self.size();
        if size == 0 {
            return result;
        }

        let buffer = result.buffer_mut();
        let result_buffer = Arc::get_mut(buffer).expect("result buffer must be unique");
        let result_ptr = result_buffer.as_mut_ptr();

        let mut indices = vec![0usize; self.ndim()];
        for i in 0..size {
            let a = self.get_element(&indices);
            let val = apply_unary_op(a, op);

            unsafe {
                match self.dtype() {
                    DType::Float64 => {
                        *(result_ptr as *mut f64).add(i) = val;
                    }
                    DType::Float32 => {
                        *(result_ptr as *mut f32).add(i) = val as f32;
                    }
                    DType::Int64 => {
                        *(result_ptr as *mut i64).add(i) = val as i64;
                    }
                    DType::Int32 => {
                        *(result_ptr as *mut i32).add(i) = val as i32;
                    }
                    DType::Bool => {
                        *result_ptr.add(i) = (val != 0.0) as u8;
                    }
                }
            }

            increment_indices(&mut indices, self.shape());
        }

        result
    }

}

/// Apply binary operation.
fn apply_binary_op(a: f64, b: f64, op: BinaryOp) -> f64 {
    match op {
        BinaryOp::Add => a + b,
        BinaryOp::Sub => a - b,
        BinaryOp::Mul => a * b,
        BinaryOp::Div => a / b,
    }
}

/// Apply unary operation.
fn apply_unary_op(a: f64, op: UnaryOp) -> f64 {
    match op {
        UnaryOp::Neg => -a,
        UnaryOp::Abs => a.abs(),
    }
}

/// Type promotion: result dtype for binary ops.
fn promote_dtype(a: DType, b: DType) -> DType {
    use DType::*;
    match (a, b) {
        // Float64 dominates
        (Float64, _) | (_, Float64) => Float64,
        // Float32 over integers
        (Float32, _) | (_, Float32) => Float32,
        // Int64 over Int32
        (Int64, _) | (_, Int64) => Int64,
        // Int32 over Bool
        (Int32, _) | (_, Int32) => Int32,
        // Bool + Bool = Bool
        (Bool, Bool) => Bool,
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
