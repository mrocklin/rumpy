//! Element-wise operations (ufunc-style).

pub mod gufunc;
pub mod matmul;

use crate::array::{broadcast_shapes, increment_indices, promote_dtype, write_element, DType, RumpyArray};
use std::sync::Arc;

/// Binary operation types.
#[derive(Clone, Copy)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
}

/// Comparison operation types.
#[derive(Clone, Copy)]
pub enum ComparisonOp {
    Gt,  // >
    Lt,  // <
    Ge,  // >=
    Le,  // <=
    Eq,  // ==
    Ne,  // !=
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
        unsafe { write_element(result_ptr, i, val, &arr.dtype()); }
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

    let result_dtype = promote_dtype(&a.dtype(), &b.dtype());
    let mut result = RumpyArray::zeros(out_shape.clone(), result_dtype);
    let size = result.size();
    if size == 0 {
        return Some(result);
    }

    let dtype = result.dtype();
    let buffer = result.buffer_mut();
    let result_buffer = Arc::get_mut(buffer).expect("buffer must be unique");
    let result_ptr = result_buffer.as_mut_ptr();

    let mut indices = vec![0usize; out_shape.len()];
    for i in 0..size {
        let val = f(a.get_element(&indices), b.get_element(&indices));
        unsafe { write_element(result_ptr, i, val, &dtype); }
        increment_indices(&mut indices, &out_shape);
    }
    Some(result)
}

/// Apply a comparison function element-wise, returning bool array.
fn map_compare<F>(a: &RumpyArray, b: &RumpyArray, f: F) -> Option<RumpyArray>
where
    F: Fn(f64, f64) -> bool,
{
    let out_shape = broadcast_shapes(a.shape(), b.shape())?;
    let a = a.broadcast_to(&out_shape)?;
    let b = b.broadcast_to(&out_shape)?;

    let mut result = RumpyArray::zeros(out_shape.clone(), DType::bool());
    let size = result.size();
    if size == 0 {
        return Some(result);
    }

    let dtype = result.dtype();
    let buffer = result.buffer_mut();
    let result_buffer = Arc::get_mut(buffer).expect("buffer must be unique");
    let result_ptr = result_buffer.as_mut_ptr();

    let mut indices = vec![0usize; out_shape.len()];
    for i in 0..size {
        let val = if f(a.get_element(&indices), b.get_element(&indices)) { 1.0 } else { 0.0 };
        unsafe { write_element(result_ptr, i, val, &dtype); }
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
            unsafe { write_element(result_ptr, i, init, &arr.dtype()); }
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

        unsafe { write_element(result_ptr, i, acc, &arr.dtype()); }
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

    /// Element-wise comparison with broadcasting.
    pub fn compare(&self, other: &RumpyArray, op: ComparisonOp) -> Option<RumpyArray> {
        let f: fn(f64, f64) -> bool = match op {
            ComparisonOp::Gt => |a, b| a > b,
            ComparisonOp::Lt => |a, b| a < b,
            ComparisonOp::Ge => |a, b| a >= b,
            ComparisonOp::Le => |a, b| a <= b,
            ComparisonOp::Eq => |a, b| a == b,
            ComparisonOp::Ne => |a, b| a != b,
        };
        map_compare(self, other, f)
    }

    /// Scalar comparison (arr op scalar).
    pub fn compare_scalar(&self, scalar: f64, op: ComparisonOp) -> RumpyArray {
        // Create scalar array and use broadcasting
        let scalar_arr = RumpyArray::full(vec![1], scalar, DType::float64());
        self.compare(&scalar_arr, op).expect("scalar broadcast always succeeds")
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

    /// Variance of all elements.
    pub fn var(&self) -> f64 {
        let size = self.size();
        if size == 0 {
            return f64::NAN;
        }
        let mean = self.mean();
        let mut sum_sq = 0.0;
        let mut indices = vec![0usize; self.ndim()];
        for _ in 0..size {
            let diff = self.get_element(&indices) - mean;
            sum_sq += diff * diff;
            increment_indices(&mut indices, self.shape());
        }
        sum_sq / size as f64
    }

    /// Variance along axis.
    pub fn var_axis(&self, axis: usize) -> RumpyArray {
        let mean = self.mean_axis(axis);
        // Expand mean to add back the reduced axis for broadcasting
        let mean_expanded = mean.expand_dims(axis).expect("expand_dims succeeds");
        let mean_broadcast = mean_expanded.broadcast_to(self.shape()).expect("broadcast succeeds");

        // Sum of squared differences
        let mut out_shape: Vec<usize> = self.shape().to_vec();
        let axis_len = out_shape.remove(axis);
        if out_shape.is_empty() {
            out_shape = vec![1];
        }

        let mut result = RumpyArray::zeros(out_shape.clone(), self.dtype());
        let out_size = result.size();
        if out_size == 0 || axis_len == 0 {
            return result;
        }

        let buffer = result.buffer_mut();
        let result_buffer = Arc::get_mut(buffer).expect("buffer must be unique");
        let result_ptr = result_buffer.as_mut_ptr();

        let mut out_indices = vec![0usize; out_shape.len()];
        for i in 0..out_size {
            let mut in_indices: Vec<usize> = out_indices[..axis.min(out_indices.len())].to_vec();
            in_indices.push(0);
            if axis < self.ndim() - 1 {
                in_indices.extend_from_slice(&out_indices[axis..]);
            }

            let mut sum_sq = 0.0;
            for j in 0..axis_len {
                in_indices[axis] = j;
                let diff = self.get_element(&in_indices) - mean_broadcast.get_element(&in_indices);
                sum_sq += diff * diff;
            }

            unsafe { write_element(result_ptr, i, sum_sq / axis_len as f64, &self.dtype()); }
            increment_indices(&mut out_indices, &out_shape);
        }

        result
    }

    /// Standard deviation of all elements.
    pub fn std(&self) -> f64 {
        self.var().sqrt()
    }

    /// Standard deviation along axis.
    pub fn std_axis(&self, axis: usize) -> RumpyArray {
        map_unary(&self.var_axis(axis), |x| x.sqrt())
    }

    /// Index of maximum element (flattened).
    pub fn argmax(&self) -> usize {
        let size = self.size();
        if size == 0 {
            return 0;
        }
        let mut max_val = f64::NEG_INFINITY;
        let mut max_idx = 0;
        let mut indices = vec![0usize; self.ndim()];
        for i in 0..size {
            let val = self.get_element(&indices);
            if val > max_val {
                max_val = val;
                max_idx = i;
            }
            increment_indices(&mut indices, self.shape());
        }
        max_idx
    }

    /// Index of minimum element (flattened).
    pub fn argmin(&self) -> usize {
        let size = self.size();
        if size == 0 {
            return 0;
        }
        let mut min_val = f64::INFINITY;
        let mut min_idx = 0;
        let mut indices = vec![0usize; self.ndim()];
        for i in 0..size {
            let val = self.get_element(&indices);
            if val < min_val {
                min_val = val;
                min_idx = i;
            }
            increment_indices(&mut indices, self.shape());
        }
        min_idx
    }

    /// Collect all elements into a Vec (flattened, row-major order).
    fn to_vec(&self) -> Vec<f64> {
        let size = self.size();
        let mut values = Vec::with_capacity(size);
        let mut indices = vec![0usize; self.ndim()];
        for _ in 0..size {
            values.push(self.get_element(&indices));
            increment_indices(&mut indices, self.shape());
        }
        values
    }

    /// Sort array (flattened, returns new 1D array).
    pub fn sort(&self) -> RumpyArray {
        let mut values = self.to_vec();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        RumpyArray::from_vec(values, self.dtype())
    }

    /// Return indices that would sort the array (flattened).
    pub fn argsort(&self) -> RumpyArray {
        let values = self.to_vec();
        let mut indexed: Vec<(usize, f64)> = values.into_iter().enumerate().map(|(i, v)| (i, v)).collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let result: Vec<f64> = indexed.into_iter().map(|(i, _)| i as f64).collect();
        RumpyArray::from_vec(result, DType::int64())
    }

    /// Return unique sorted values.
    pub fn unique(&self) -> RumpyArray {
        let mut values = self.to_vec();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        values.dedup_by(|a, b| (*a - *b).abs() < f64::EPSILON);
        RumpyArray::from_vec(values, self.dtype())
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

/// Conditional selection: where(condition, x, y).
/// Returns elements from x where condition is true, else from y.
/// All three arrays are broadcast together.
pub fn where_select(condition: &RumpyArray, x: &RumpyArray, y: &RumpyArray) -> Option<RumpyArray> {
    // Broadcast all three shapes together
    let shape_cx = broadcast_shapes(condition.shape(), x.shape())?;
    let out_shape = broadcast_shapes(&shape_cx, y.shape())?;

    let cond = condition.broadcast_to(&out_shape)?;
    let x = x.broadcast_to(&out_shape)?;
    let y = y.broadcast_to(&out_shape)?;

    // Result dtype is promoted from x and y
    let result_dtype = promote_dtype(&x.dtype(), &y.dtype());
    let mut result = RumpyArray::zeros(out_shape.clone(), result_dtype);
    let size = result.size();

    if size == 0 {
        return Some(result);
    }

    let dtype = result.dtype();
    let buffer = result.buffer_mut();
    let result_buffer = Arc::get_mut(buffer).expect("buffer must be unique");
    let result_ptr = result_buffer.as_mut_ptr();

    let mut indices = vec![0usize; out_shape.len()];
    for i in 0..size {
        let cond_val = cond.get_element(&indices);
        let val = if cond_val != 0.0 {
            x.get_element(&indices)
        } else {
            y.get_element(&indices)
        };
        unsafe { write_element(result_ptr, i, val, &dtype); }
        increment_indices(&mut indices, &out_shape);
    }

    Some(result)
}
