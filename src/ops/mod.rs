//! Element-wise operations (ufunc-style).

#![allow(clippy::new_without_default)]

pub mod array_methods;
pub mod bitwise;
pub mod comparison;
pub mod dispatch;
pub mod dot;
pub mod fft;
pub mod gufunc;
pub mod indexing;
pub mod inner;
pub mod io;
pub mod kernels;
pub mod linalg;
pub mod loops;
pub mod matmul;
pub mod numerical;
pub mod outer;
pub mod poly;
pub mod registry;
pub mod set_ops;
pub mod solve;
pub mod special;
pub mod statistics;
pub mod ufunc;

// Re-export from new modules
pub use ufunc::{
    map_unary_op, map_binary_op, map_binary_op_inplace, map_compare_op,
    reduce_all_op, reduce_all_f64, reduce_axis_op,
};
pub use comparison::{
    logical_and, logical_or, logical_xor, logical_not,
    equal, not_equal, less, less_equal, greater, greater_equal,
    isclose, allclose, array_equal, map_binary_to_bool,
};
pub use bitwise::{
    bitwise_and, bitwise_or, bitwise_xor, bitwise_not,
    left_shift, right_shift,
};
pub use statistics::{histogram, cov, corrcoef};
pub use set_ops::{isin, in1d, intersect1d, union1d, setdiff1d, setxor1d};
pub use poly::{polyfit, polyval, polyder, polyint, roots};
pub use numerical::{gradient, gradient_with_coords, trapezoid, interp, correlate};
pub use array_methods::lexsort;
pub use special::{sinc, i0, spacing, modf, frexp, ldexp, heaviside, gcd, lcm};

use crate::array::{broadcast_shapes, increment_indices, promote_dtype, DType, RumpyArray};
use crate::array::dtype::UnaryOp;
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
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum ComparisonOp {
    Gt,  // >
    Lt,  // <
    Ge,  // >=
    Le,  // <=
    Eq,  // ==
    Ne,  // !=
}

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
        map_compare_op(self, other, op)
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

    // Reductions (sum, mean, var, std, argmax, argmin, etc.) are in array_methods/reductions.rs
    // to_vec and unique are in array_methods/sorting.rs

    /// Return indices of non-zero elements (tuple of arrays, one per dimension).
    pub fn nonzero(&self) -> Vec<RumpyArray> {
        let ndim = self.ndim();
        let mut indices_per_dim: Vec<Vec<i64>> = vec![Vec::new(); ndim];

        let mut indices = vec![0usize; ndim];
        let size = self.size();
        for _ in 0..size {
            let val = self.get_element(&indices);
            if val != 0.0 {
                for (dim, &idx) in indices.iter().enumerate() {
                    indices_per_dim[dim].push(idx as i64);
                }
            }
            increment_indices(&mut indices, self.shape());
        }

        indices_per_dim
            .into_iter()
            .map(|v| {
                let data: Vec<f64> = v.into_iter().map(|x| x as f64).collect();
                RumpyArray::from_vec(data, DType::int64())
            })
            .collect()
    }

    // Math ufuncs are in array_methods/unary.rs

    /// Extract diagonal from a 2D array.
    pub fn diagonal(&self) -> RumpyArray {
        assert!(self.ndim() >= 2, "diagonal requires at least 2D array");
        let shape = self.shape();
        let n = shape[0].min(shape[1]);
        let dtype = self.dtype().clone();

        let mut result = RumpyArray::zeros(vec![n], dtype.clone());
        if n == 0 {
            return result;
        }

        let result_buffer = Arc::get_mut(result.buffer_mut()).expect("unique");
        let result_ptr = result_buffer.as_mut_ptr();
        let ops = dtype.ops();

        for i in 0..n {
            let mut indices = vec![0usize; self.ndim()];
            indices[0] = i;
            indices[1] = i;
            let byte_offset = self.byte_offset_for(&indices);
            unsafe { ops.copy_element(self.data_ptr(), byte_offset, result_ptr, i); }
        }
        result
    }

    /// Return sum of diagonal elements (trace).
    pub fn trace(&self) -> f64 {
        assert!(self.ndim() >= 2, "trace requires at least 2D array");
        let shape = self.shape();
        let n = shape[0].min(shape[1]);

        let mut sum = 0.0;
        for i in 0..n {
            let mut indices = vec![0usize; self.ndim()];
            indices[0] = i;
            indices[1] = i;
            sum += self.get_element(&indices);
        }
        sum
    }

    /// Swap two axes of the array.
    pub fn swapaxes(&self, axis1: usize, axis2: usize) -> RumpyArray {
        let ndim = self.ndim();
        assert!(axis1 < ndim, "axis1 out of bounds");
        assert!(axis2 < ndim, "axis2 out of bounds");

        if axis1 == axis2 {
            return self.copy();
        }

        // Build permutation: swap axis1 and axis2
        let mut axes: Vec<usize> = (0..ndim).collect();
        axes.swap(axis1, axis2);

        self.transpose_axes(&axes)
    }

    // sort, argsort, partition, argpartition are in array_methods/sorting.rs
    // count_nonzero, all, any are in array_methods/logical.rs
    // diff, cumsum, cumprod are in array_methods/cumulative.rs

    /// Clip values to a range.
    pub fn clip(&self, a_min: Option<f64>, a_max: Option<f64>) -> RumpyArray {
        // Try typed dispatch first
        if let Some(result) = crate::ops::dispatch::dispatch_clip(self, a_min, a_max) {
            return result;
        }

        // Fallback for complex, bool, datetime
        let dtype = self.dtype();
        let mut result = RumpyArray::zeros(self.shape().to_vec(), dtype.clone());
        let size = result.size();
        if size == 0 {
            return result;
        }

        let buffer = result.buffer_mut();
        let result_buffer = Arc::get_mut(buffer).expect("buffer must be unique");
        let result_ptr = result_buffer.as_mut_ptr();
        let ops = dtype.ops();
        let src_ptr = self.data_ptr();

        for (i, offset) in self.iter_offsets().enumerate() {
            let mut val = unsafe { ops.read_f64(src_ptr, offset) }.unwrap_or(0.0);
            if let Some(min) = a_min {
                if val < min {
                    val = min;
                }
            }
            if let Some(max) = a_max {
                if val > max {
                    val = max;
                }
            }
            unsafe { ops.write_f64(result_ptr, i, val); }
        }

        result
    }

    /// Round to the given number of decimals.
    pub fn round(&self, decimals: i32) -> RumpyArray {
        // Try typed dispatch first
        if let Some(result) = crate::ops::dispatch::dispatch_round(self, decimals) {
            return result;
        }

        // Fallback for complex, bool, datetime
        let dtype = self.dtype();
        let mut result = RumpyArray::zeros(self.shape().to_vec(), dtype.clone());
        let size = result.size();
        if size == 0 {
            return result;
        }

        let scale = 10.0_f64.powi(decimals);
        let buffer = result.buffer_mut();
        let result_buffer = Arc::get_mut(buffer).expect("buffer must be unique");
        let result_ptr = result_buffer.as_mut_ptr();
        let ops = dtype.ops();
        let src_ptr = self.data_ptr();

        for (i, offset) in self.iter_offsets().enumerate() {
            let val = unsafe { ops.read_f64(src_ptr, offset) }.unwrap_or(0.0);
            let rounded = (val * scale).round() / scale;
            unsafe { ops.write_f64(result_ptr, i, rounded); }
        }

        result
    }

    /// Convert array to nested Python lists.
    pub fn to_pylist(&self, py: pyo3::Python<'_>) -> pyo3::PyResult<pyo3::PyObject> {
        use pyo3::types::PyList;

        fn build_list(
            arr: &RumpyArray,
            py: pyo3::Python<'_>,
            depth: usize,
            indices: &mut Vec<usize>,
            use_int: bool,
        ) -> pyo3::PyResult<pyo3::PyObject> {
            use pyo3::IntoPyObject;
            if depth == arr.ndim() {
                // Base case: return scalar of appropriate type
                let offset = arr.byte_offset_for(indices);
                let ptr = arr.data_ptr();
                let dtype = arr.dtype();
                let ops = dtype.ops();
                if use_int {
                    let val = unsafe { ops.read_i64(ptr, offset) }.unwrap_or(0);
                    return Ok(val.into_pyobject(py)?.into_any().unbind());
                } else {
                    let val = unsafe { ops.read_f64(ptr, offset) }.unwrap_or(0.0);
                    return Ok(val.into_pyobject(py)?.into_any().unbind());
                }
            }

            // Build list for this dimension
            let dim_size = arr.shape()[depth];
            let mut items = Vec::with_capacity(dim_size);
            for i in 0..dim_size {
                indices[depth] = i;
                items.push(build_list(arr, py, depth + 1, indices, use_int)?);
            }
            let list = PyList::new(py, items)?;
            Ok(list.into_pyobject(py)?.into_any().unbind())
        }

        let mut indices = vec![0usize; self.ndim()];
        let use_int = self.dtype().is_integer();
        build_list(self, py, 0, &mut indices, use_int)
    }
}

// partition_inplace, partition_lane_typed, argpartition_impl, lexsort
// are in array_methods/sorting.rs

/// Conditional selection: where(condition, x, y).
/// Returns elements from x where condition is true, else from y.
/// All three arrays are broadcast together.
pub fn where_select(condition: &RumpyArray, x: &RumpyArray, y: &RumpyArray) -> Option<RumpyArray> {
    // Broadcast all three shapes together
    let shape_cx = broadcast_shapes(condition.shape(), x.shape())?;
    let out_shape = broadcast_shapes(&shape_cx, y.shape())?;

    // Result dtype is promoted from x and y
    let result_dtype = promote_dtype(&x.dtype(), &y.dtype());

    // Try typed dispatch first (same-dtype x, y, result)
    if let Some(result) = crate::ops::dispatch::dispatch_where(condition, x, y, &result_dtype, &out_shape) {
        return Some(result);
    }

    // Fallback for mixed dtypes
    let cond = condition.broadcast_to(&out_shape)?;
    let x_bc = x.broadcast_to(&out_shape)?;
    let y_bc = y.broadcast_to(&out_shape)?;

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

    // Check if all dtypes match for direct copy (shouldn't hit this if dispatch worked)
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
