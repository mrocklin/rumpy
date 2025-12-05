//! Element-wise operations (ufunc-style).

pub mod dot;
pub mod gufunc;
pub mod inner;
pub mod linalg;
pub mod matmul;
pub mod outer;
pub mod registry;
pub mod solve;

use crate::array::{broadcast_shapes, increment_indices, promote_dtype, DType, RumpyArray};
use crate::array::dtype::{UnaryOp, ReduceOp};
use registry::registry;
use std::sync::Arc;

// Re-export BinaryOp from dtype module
pub use crate::array::dtype::BinaryOp;

/// Error type for binary operations.
#[derive(Debug, Clone)]
pub enum BinaryOpError {
    /// Shapes cannot be broadcast together
    ShapeMismatch,
    /// Operation not supported for these dtypes
    UnsupportedDtype,
}

/// Error type for unary operations.
#[derive(Debug, Clone)]
pub enum UnaryOpError {
    /// Operation not supported for this dtype
    UnsupportedDtype,
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

/// Apply a unary operation element-wise, returning a new array.
fn map_unary_op(arr: &RumpyArray, op: UnaryOp) -> Result<RumpyArray, UnaryOpError> {
    use crate::array::dtype::DTypeKind;

    let dtype = arr.dtype();
    let kind = dtype.kind();

    // Validate unsupported operations
    if matches!(kind, DTypeKind::Complex128) {
        match op {
            UnaryOp::Floor | UnaryOp::Ceil => return Err(UnaryOpError::UnsupportedDtype),
            _ => {}
        }
    }

    let mut result = RumpyArray::zeros(arr.shape().to_vec(), dtype.clone());
    let size = arr.size();
    if size == 0 {
        return Ok(result);
    }

    let buffer = result.buffer_mut();
    let result_buffer = Arc::get_mut(buffer).expect("buffer must be unique");
    let result_ptr = result_buffer.as_mut_ptr();
    let src_ptr = arr.data_ptr();
    let itemsize = dtype.itemsize() as isize;

    // Try registry first
    {
        let reg = registry().read().unwrap();
        if let Some((loop_fn, _)) = reg.lookup_unary(op, kind.clone()) {
            if arr.is_c_contiguous() {
                // Fast path: call loop once for entire array
                let strides = (itemsize, itemsize);
                unsafe { loop_fn(src_ptr, result_ptr, size, strides); }
            } else {
                // Strided path: iterate with stride info
                let mut indices = vec![0usize; arr.ndim()];
                for i in 0..size {
                    let src_offset = arr.byte_offset_for(&indices);
                    // Call loop for single element
                    unsafe {
                        loop_fn(
                            src_ptr.offset(src_offset),
                            result_ptr.offset(i as isize * itemsize),
                            1,
                            (itemsize, itemsize),  // doesn't matter for n=1
                        );
                    }
                    increment_indices(&mut indices, arr.shape());
                }
            }
            return Ok(result);
        }
    }

    // Fallback: trait-based dispatch
    let ops = dtype.ops();
    let mut indices = vec![0usize; arr.ndim()];
    for i in 0..size {
        let src_offset = arr.byte_offset_for(&indices);
        unsafe { ops.unary_op(op, src_ptr, src_offset, result_ptr, i); }
        increment_indices(&mut indices, arr.shape());
    }
    Ok(result)
}

/// Apply a binary operation element-wise with broadcasting.
fn map_binary_op(a: &RumpyArray, b: &RumpyArray, op: BinaryOp) -> Result<RumpyArray, BinaryOpError> {
    use crate::array::dtype::DTypeKind;

    let out_shape = broadcast_shapes(a.shape(), b.shape()).ok_or(BinaryOpError::ShapeMismatch)?;
    let a_bc = a.broadcast_to(&out_shape).ok_or(BinaryOpError::ShapeMismatch)?;
    let b_bc = b.broadcast_to(&out_shape).ok_or(BinaryOpError::ShapeMismatch)?;

    // Validate datetime operations
    let a_is_datetime = matches!(a_bc.dtype().kind(), DTypeKind::DateTime64(_));
    let b_is_datetime = matches!(b_bc.dtype().kind(), DTypeKind::DateTime64(_));
    if a_is_datetime || b_is_datetime {
        // datetime + datetime is invalid (only sub is valid between datetimes)
        // datetime * anything and datetime / anything are also invalid
        match op {
            BinaryOp::Add if a_is_datetime && b_is_datetime => return Err(BinaryOpError::UnsupportedDtype),
            BinaryOp::Mul | BinaryOp::Div => return Err(BinaryOpError::UnsupportedDtype),
            _ => {}
        }
    }

    let result_dtype = promote_dtype(&a_bc.dtype(), &b_bc.dtype());
    let mut result = RumpyArray::zeros(out_shape.clone(), result_dtype.clone());
    let size = result.size();
    if size == 0 {
        return Ok(result);
    }

    let buffer = result.buffer_mut();
    let result_buffer = Arc::get_mut(buffer).expect("buffer must be unique");
    let result_ptr = result_buffer.as_mut_ptr();
    let a_ptr = a_bc.data_ptr();
    let b_ptr = b_bc.data_ptr();
    let result_dtype_ref = result.dtype();
    let result_ops = result_dtype_ref.ops();

    let a_kind = a_bc.dtype().kind();
    let b_kind = b_bc.dtype().kind();

    let mut indices = vec![0usize; out_shape.len()];

    let itemsize = result_dtype_ref.itemsize() as isize;

    // Try registry first for same-type operations
    if a_kind == b_kind {
        let reg = registry().read().unwrap();
        if let Some((loop_fn, _)) = reg.lookup_binary(op, a_kind.clone(), b_kind) {
            // Check if we can use the fast contiguous path
            // (both inputs and output are C-contiguous with same shape)
            let a_contig = a_bc.is_c_contiguous() && a.shape() == out_shape.as_slice();
            let b_contig = b_bc.is_c_contiguous() && b.shape() == out_shape.as_slice();

            if a_contig && b_contig {
                // Fast path: call loop once for entire array
                let strides = (itemsize, itemsize, itemsize);
                unsafe { loop_fn(a_ptr, b_ptr, result_ptr, size, strides); }
            } else {
                // Strided path: iterate with stride info
                for i in 0..size {
                    let a_offset = a_bc.byte_offset_for(&indices);
                    let b_offset = b_bc.byte_offset_for(&indices);
                    // Call loop for single element
                    unsafe {
                        loop_fn(
                            a_ptr.offset(a_offset),
                            b_ptr.offset(b_offset),
                            result_ptr.offset(i as isize * itemsize),
                            1,
                            (itemsize, itemsize, itemsize),  // doesn't matter for n=1
                        );
                    }
                    increment_indices(&mut indices, &out_shape);
                }
            }
            return Ok(result);
        }
    }

    // Fallback: trait-based dispatch
    let same_dtype = a_bc.dtype() == b_bc.dtype() && a_bc.dtype() == result.dtype();

    if same_dtype {
        for i in 0..size {
            let a_offset = a_bc.byte_offset_for(&indices);
            let b_offset = b_bc.byte_offset_for(&indices);
            unsafe { result_ops.binary_op(op, a_ptr, a_offset, b_ptr, b_offset, result_ptr, i); }
            increment_indices(&mut indices, &out_shape);
        }
    } else {
        // Different dtypes: use complex path if result is complex, else f64
        let a_dtype = a_bc.dtype();
        let a_ops = a_dtype.ops();
        let b_dtype = b_bc.dtype();
        let b_ops = b_dtype.ops();
        let result_is_complex = result.dtype().kind() == crate::array::dtype::DTypeKind::Complex128;

        if result_is_complex {
            // Complex result: read as complex, operate, write as complex
            for i in 0..size {
                let a_offset = a_bc.byte_offset_for(&indices);
                let b_offset = b_bc.byte_offset_for(&indices);

                let av = unsafe { a_ops.read_complex(a_ptr, a_offset).unwrap_or((0.0, 0.0)) };
                let bv = unsafe { b_ops.read_complex(b_ptr, b_offset).unwrap_or((0.0, 0.0)) };

                let result_val = match op {
                    BinaryOp::Add => (av.0 + bv.0, av.1 + bv.1),
                    BinaryOp::Sub => (av.0 - bv.0, av.1 - bv.1),
                    BinaryOp::Mul => (av.0 * bv.0 - av.1 * bv.1, av.0 * bv.1 + av.1 * bv.0),
                    BinaryOp::Div => {
                        let denom = bv.0 * bv.0 + bv.1 * bv.1;
                        if denom != 0.0 {
                            ((av.0 * bv.0 + av.1 * bv.1) / denom, (av.1 * bv.0 - av.0 * bv.1) / denom)
                        } else {
                            (f64::NAN, f64::NAN)
                        }
                    }
                };

                unsafe { result_ops.write_complex(result_ptr, i, result_val.0, result_val.1); }
                increment_indices(&mut indices, &out_shape);
            }
        } else {
            // Non-complex result: use f64 path
            for i in 0..size {
                let a_offset = a_bc.byte_offset_for(&indices);
                let b_offset = b_bc.byte_offset_for(&indices);

                let av = unsafe { a_ops.read_f64(a_ptr, a_offset).unwrap_or(0.0) };
                let bv = unsafe { b_ops.read_f64(b_ptr, b_offset).unwrap_or(0.0) };

                let result_val = match op {
                    BinaryOp::Add => av + bv,
                    BinaryOp::Sub => av - bv,
                    BinaryOp::Mul => av * bv,
                    BinaryOp::Div => if bv != 0.0 { av / bv } else { f64::NAN },
                };

                unsafe { result_ops.write_f64(result_ptr, i, result_val); }
                increment_indices(&mut indices, &out_shape);
            }
        }
    }
    Ok(result)
}

/// Apply a comparison function element-wise, returning bool array.
/// Note: comparison still uses f64 for now, since ordering on complex is tricky.
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

    let buffer = result.buffer_mut();
    let result_buffer = Arc::get_mut(buffer).expect("buffer must be unique");
    let result_ptr = result_buffer.as_mut_ptr();
    let dtype = result.dtype();
    let ops = dtype.ops();

    let mut indices = vec![0usize; out_shape.len()];
    for i in 0..size {
        let val = if f(a.get_element(&indices), b.get_element(&indices)) { 1.0 } else { 0.0 };
        unsafe { ops.write_f64(result_ptr, i, val); }
        increment_indices(&mut indices, &out_shape);
    }
    Some(result)
}

/// Reduce array along all axes, returning a 0-d array.
fn reduce_all_op(arr: &RumpyArray, op: ReduceOp) -> RumpyArray {
    let mut result = RumpyArray::zeros(vec![1], arr.dtype());
    let size = arr.size();

    let buffer = result.buffer_mut();
    let result_buffer = Arc::get_mut(buffer).expect("buffer must be unique");
    let result_ptr = result_buffer.as_mut_ptr();
    let dtype = arr.dtype();
    let kind = dtype.kind();

    // Try registry first
    {
        let reg = registry().read().unwrap();
        if let Some((init_fn, acc_fn, _)) = reg.lookup_reduce(op, kind.clone()) {
            // Initialize
            unsafe { init_fn(result_ptr, 0); }

            if size == 0 {
                return result;
            }

            let src_ptr = arr.data_ptr();
            let mut indices = vec![0usize; arr.ndim()];
            for _ in 0..size {
                let src_offset = arr.byte_offset_for(&indices);
                unsafe { acc_fn(result_ptr, 0, src_ptr, src_offset); }
                increment_indices(&mut indices, arr.shape());
            }
            return result;
        }
    }

    // Fallback: trait-based dispatch
    let ops = dtype.ops();

    // Initialize accumulator
    unsafe { ops.reduce_init(op, result_ptr, 0); }

    if size == 0 {
        return result;
    }

    let src_ptr = arr.data_ptr();
    let mut indices = vec![0usize; arr.ndim()];
    for _ in 0..size {
        let src_offset = arr.byte_offset_for(&indices);
        unsafe { ops.reduce_acc(op, result_ptr, 0, src_ptr, src_offset); }
        increment_indices(&mut indices, arr.shape());
    }
    result
}

/// Get reduction result as f64 (for backwards compatibility).
fn reduce_all_f64(arr: &RumpyArray, op: ReduceOp) -> f64 {
    let result = reduce_all_op(arr, op);
    result.get_element(&[0])
}

/// Reduce array along a specific axis.
fn reduce_axis_op(arr: &RumpyArray, axis: usize, op: ReduceOp) -> RumpyArray {
    // Output shape: remove the reduction axis
    let mut out_shape: Vec<usize> = arr.shape().to_vec();
    let axis_len = out_shape.remove(axis);

    // Handle edge cases
    if out_shape.is_empty() {
        out_shape = vec![1]; // Scalar result wrapped in 1D array
    }

    let mut result = RumpyArray::zeros(out_shape.clone(), arr.dtype());
    let out_size = result.size();

    let buffer = result.buffer_mut();
    let result_buffer = Arc::get_mut(buffer).expect("buffer must be unique");
    let result_ptr = result_buffer.as_mut_ptr();
    let dtype = arr.dtype();
    let kind = dtype.kind();

    // Try registry first
    {
        let reg = registry().read().unwrap();
        if let Some((init_fn, acc_fn, _)) = reg.lookup_reduce(op, kind.clone()) {
            // Initialize all accumulators
            for i in 0..out_size {
                unsafe { init_fn(result_ptr, i); }
            }

            if out_size == 0 || axis_len == 0 {
                return result;
            }

            let src_ptr = arr.data_ptr();

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
                for j in 0..axis_len {
                    in_indices[axis] = j;
                    let src_offset = arr.byte_offset_for(&in_indices);
                    unsafe { acc_fn(result_ptr, i, src_ptr, src_offset); }
                }

                increment_indices(&mut out_indices, &out_shape);
            }

            return result;
        }
    }

    // Fallback: trait-based dispatch
    let ops = dtype.ops();

    // Initialize all accumulators
    for i in 0..out_size {
        unsafe { ops.reduce_init(op, result_ptr, i); }
    }

    if out_size == 0 || axis_len == 0 {
        return result;
    }

    let src_ptr = arr.data_ptr();

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
        for j in 0..axis_len {
            in_indices[axis] = j;
            let src_offset = arr.byte_offset_for(&in_indices);
            unsafe { ops.reduce_acc(op, result_ptr, i, src_ptr, src_offset); }
        }

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
    pub fn binary_op(&self, other: &RumpyArray, op: BinaryOp) -> Result<RumpyArray, BinaryOpError> {
        map_binary_op(self, other, op)
    }

    /// Element-wise operation with scalar (arr op scalar).
    pub fn scalar_op(&self, scalar: f64, op: BinaryOp) -> RumpyArray {
        // Create a scalar array and use binary_op with broadcasting
        let scalar_arr = RumpyArray::full(vec![1], scalar, self.dtype());
        self.binary_op(&scalar_arr, op).expect("scalar broadcast always works")
    }

    /// Scalar on left side (scalar op arr).
    pub fn rscalar_op(&self, scalar: f64, op: BinaryOp) -> RumpyArray {
        let scalar_arr = RumpyArray::full(vec![1], scalar, self.dtype());
        scalar_arr.binary_op(self, op).expect("scalar broadcast always works")
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
    pub fn neg(&self) -> Result<RumpyArray, UnaryOpError> {
        map_unary_op(self, UnaryOp::Neg)
    }

    /// Absolute value of each element.
    pub fn abs(&self) -> Result<RumpyArray, UnaryOpError> {
        map_unary_op(self, UnaryOp::Abs)
    }

    /// Sum all elements.
    pub fn sum(&self) -> f64 {
        reduce_all_f64(self, ReduceOp::Sum)
    }

    /// Sum along axis.
    pub fn sum_axis(&self, axis: usize) -> RumpyArray {
        reduce_axis_op(self, axis, ReduceOp::Sum)
    }

    /// Product of all elements.
    pub fn prod(&self) -> f64 {
        reduce_all_f64(self, ReduceOp::Prod)
    }

    /// Product along axis.
    pub fn prod_axis(&self, axis: usize) -> RumpyArray {
        reduce_axis_op(self, axis, ReduceOp::Prod)
    }

    /// Maximum element.
    pub fn max(&self) -> f64 {
        reduce_all_f64(self, ReduceOp::Max)
    }

    /// Maximum along axis.
    pub fn max_axis(&self, axis: usize) -> RumpyArray {
        reduce_axis_op(self, axis, ReduceOp::Max)
    }

    /// Minimum element.
    pub fn min(&self) -> f64 {
        reduce_all_f64(self, ReduceOp::Min)
    }

    /// Minimum along axis.
    pub fn min_axis(&self, axis: usize) -> RumpyArray {
        reduce_axis_op(self, axis, ReduceOp::Min)
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
        // Divide each element by count
        let count_arr = RumpyArray::full(vec![1], count, sum.dtype());
        sum.binary_op(&count_arr, BinaryOp::Div).expect("broadcast works")
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
        let dtype = self.dtype();
        let ops = dtype.ops();

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

            unsafe { ops.write_f64(result_ptr, i, sum_sq / axis_len as f64); }
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
        map_unary_op(&self.var_axis(axis), UnaryOp::Sqrt).expect("sqrt always succeeds on numeric types")
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
    pub fn sqrt(&self) -> Result<RumpyArray, UnaryOpError> {
        map_unary_op(self, UnaryOp::Sqrt)
    }

    /// Exponential (e^x) of each element.
    pub fn exp(&self) -> Result<RumpyArray, UnaryOpError> {
        map_unary_op(self, UnaryOp::Exp)
    }

    /// Natural logarithm of each element.
    pub fn log(&self) -> Result<RumpyArray, UnaryOpError> {
        map_unary_op(self, UnaryOp::Log)
    }

    /// Sine of each element (radians).
    pub fn sin(&self) -> Result<RumpyArray, UnaryOpError> {
        map_unary_op(self, UnaryOp::Sin)
    }

    /// Cosine of each element (radians).
    pub fn cos(&self) -> Result<RumpyArray, UnaryOpError> {
        map_unary_op(self, UnaryOp::Cos)
    }

    /// Tangent of each element (radians).
    pub fn tan(&self) -> Result<RumpyArray, UnaryOpError> {
        map_unary_op(self, UnaryOp::Tan)
    }

    /// Floor of each element.
    pub fn floor(&self) -> Result<RumpyArray, UnaryOpError> {
        map_unary_op(self, UnaryOp::Floor)
    }

    /// Ceiling of each element.
    pub fn ceil(&self) -> Result<RumpyArray, UnaryOpError> {
        map_unary_op(self, UnaryOp::Ceil)
    }

    /// Inverse sine (arcsine) of each element.
    pub fn arcsin(&self) -> Result<RumpyArray, UnaryOpError> {
        map_unary_op(self, UnaryOp::Arcsin)
    }

    /// Inverse cosine (arccosine) of each element.
    pub fn arccos(&self) -> Result<RumpyArray, UnaryOpError> {
        map_unary_op(self, UnaryOp::Arccos)
    }

    /// Inverse tangent (arctangent) of each element.
    pub fn arctan(&self) -> Result<RumpyArray, UnaryOpError> {
        map_unary_op(self, UnaryOp::Arctan)
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
    let x_bc = x.broadcast_to(&out_shape)?;
    let y_bc = y.broadcast_to(&out_shape)?;

    // Result dtype is promoted from x and y
    let result_dtype = promote_dtype(&x_bc.dtype(), &y_bc.dtype());
    let mut result = RumpyArray::zeros(out_shape.clone(), result_dtype);
    let size = result.size();

    if size == 0 {
        return Some(result);
    }

    let buffer = result.buffer_mut();
    let result_buffer = Arc::get_mut(buffer).expect("buffer must be unique");
    let result_ptr = result_buffer.as_mut_ptr();
    let result_dtype_ref = result.dtype();
    let result_ops = result_dtype_ref.ops();

    let cond_ptr = cond.data_ptr();
    let cond_dtype = cond.dtype();
    let cond_ops = cond_dtype.ops();
    let x_ptr = x_bc.data_ptr();
    let y_ptr = y_bc.data_ptr();

    // Check if all dtypes match for direct copy
    let same_dtype = x_bc.dtype() == y_bc.dtype() && x_bc.dtype() == result.dtype();

    let mut indices = vec![0usize; out_shape.len()];

    if same_dtype {
        for i in 0..size {
            let cond_offset = cond.byte_offset_for(&indices);
            let is_true = unsafe { cond_ops.is_truthy(cond_ptr, cond_offset) };
            if is_true {
                let x_offset = x_bc.byte_offset_for(&indices);
                unsafe { result_ops.copy_element(x_ptr, x_offset, result_ptr, i); }
            } else {
                let y_offset = y_bc.byte_offset_for(&indices);
                unsafe { result_ops.copy_element(y_ptr, y_offset, result_ptr, i); }
            }
            increment_indices(&mut indices, &out_shape);
        }
    } else {
        // Different dtypes: read as f64, write as f64
        let x_dtype = x_bc.dtype();
        let x_ops = x_dtype.ops();
        let y_dtype = y_bc.dtype();
        let y_ops = y_dtype.ops();

        for i in 0..size {
            let cond_offset = cond.byte_offset_for(&indices);
            let is_true = unsafe { cond_ops.is_truthy(cond_ptr, cond_offset) };
            let val = if is_true {
                let x_offset = x_bc.byte_offset_for(&indices);
                unsafe { x_ops.read_f64(x_ptr, x_offset).unwrap_or(0.0) }
            } else {
                let y_offset = y_bc.byte_offset_for(&indices);
                unsafe { y_ops.read_f64(y_ptr, y_offset).unwrap_or(0.0) }
            };
            unsafe { result_ops.write_f64(result_ptr, i, val); }
            increment_indices(&mut indices, &out_shape);
        }
    }

    Some(result)
}
