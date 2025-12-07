//! Element-wise operations (ufunc-style).

#![allow(clippy::new_without_default)]

pub mod bitwise;
pub mod comparison;
pub mod dot;
pub mod fft;
pub mod gufunc;
pub mod indexing;
pub mod inner;
pub mod linalg;
pub mod matmul;
pub mod outer;
pub mod poly;
pub mod registry;
pub mod set_ops;
pub mod solve;
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

use crate::array::{broadcast_shapes, increment_indices, promote_dtype, DType, RumpyArray};
use crate::array::dtype::{DTypeKind, UnaryOp, ReduceOp};
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
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum ComparisonOp {
    Gt,  // >
    Lt,  // <
    Ge,  // >=
    Le,  // <=
    Eq,  // ==
    Ne,  // !=
}

// Import helpers from ufunc that are still needed here
use ufunc::variance_f64_contiguous;

// ============================================================================
// diff helper functions
// ============================================================================

/// Fast 1D contiguous diff using registry's Sub loop.
/// result[i] = src[i+1] - src[i]
#[inline]
fn diff_1d_contiguous(src_ptr: *const u8, result_ptr: *mut u8, n: usize, dtype: &DType) {
    let itemsize = dtype.itemsize() as isize;

    // Try registry's vectorized Sub loop
    if let Some((loop_fn, _)) = registry().read().unwrap()
        .lookup_binary(BinaryOp::Sub, dtype.kind(), dtype.kind())
    {
        unsafe {
            loop_fn(
                src_ptr.offset(itemsize),  // a = src[i+1]
                src_ptr,                    // b = src[i]
                result_ptr,
                n,
                (itemsize, itemsize, itemsize),
            );
        }
    } else {
        // Fallback for unsupported dtypes
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
    /// Uses two-pass algorithm (mean then sum of squared deviations) for vectorization.
    pub fn var(&self) -> f64 {
        let size = self.size();
        if size == 0 {
            return f64::NAN;
        }

        // Fast path for contiguous f64
        use crate::array::dtype::DTypeKind;
        if self.is_c_contiguous() && self.dtype().kind() == DTypeKind::Float64 {
            let ptr = self.data_ptr() as *const f64;
            return variance_f64_contiguous(ptr, size);
        }

        // General strided path: two-pass for numerical stability
        let mean = self.mean();
        let ptr = self.data_ptr();
        let dtype = self.dtype();
        let ops = dtype.ops();

        let mut sum_sq = 0.0;
        for offset in self.iter_offsets() {
            let x = unsafe { ops.read_f64(ptr, offset) }.unwrap_or(0.0);
            let diff = x - mean;
            sum_sq += diff * diff;
        }
        sum_sq / size as f64
    }

    /// Variance along axis (second central moment).
    pub fn var_axis(&self, axis: usize) -> RumpyArray {
        self.moment_axis(2, axis)
    }

    /// Central moment of order k for all elements.
    /// moment(k=2) == variance, moment(k=3)/std^3 == skewness, etc.
    pub fn moment(&self, k: usize) -> f64 {
        let size = self.size();
        if size == 0 {
            return f64::NAN;
        }
        let ptr = self.data_ptr();
        let dtype = self.dtype();
        let ops = dtype.ops();

        // Two-pass: compute mean, then sum of (x - mean)^k
        let mean = self.mean();
        let mut sum_mk = 0.0;
        for offset in self.iter_offsets() {
            let x = unsafe { ops.read_f64(ptr, offset) }.unwrap_or(0.0);
            sum_mk += (x - mean).powi(k as i32);
        }
        sum_mk / size as f64
    }

    /// Central moment along axis.
    /// Uses vectorized operations: (x - mean)^k summed along axis.
    pub fn moment_axis(&self, k: usize, axis: usize) -> RumpyArray {
        let axis_len = self.shape()[axis];
        if axis_len == 0 {
            let mut out_shape: Vec<usize> = self.shape().to_vec();
            out_shape.remove(axis);
            if out_shape.is_empty() {
                out_shape = vec![1];
            }
            return RumpyArray::zeros(out_shape, DType::float64());
        }

        // Compute mean and broadcast back to original shape
        let mean = self.mean_axis(axis);
        let mean_expanded = mean.expand_dims(axis).expect("expand_dims succeeds");

        // Compute deviations: x - mean (uses vectorized binary op with broadcasting)
        let diff = self.binary_op(&mean_expanded, BinaryOp::Sub).expect("broadcast succeeds");

        // Raise to power k using vectorized ops
        let powered = if k == 2 {
            // Special case: x^2 = x * x (faster than pow)
            diff.binary_op(&diff, BinaryOp::Mul).expect("same shape")
        } else {
            // General case: use pow
            let k_arr = RumpyArray::full(vec![1], k as f64, diff.dtype());
            diff.binary_op(&k_arr, BinaryOp::Pow).expect("broadcast works")
        };

        // Sum along axis and divide by count (uses efficient reduce_axis_op)
        let sum = powered.sum_axis(axis);
        let count_arr = RumpyArray::full(vec![1], axis_len as f64, sum.dtype());
        sum.binary_op(&count_arr, BinaryOp::Div).expect("broadcast works")
    }

    /// Skewness of all elements (Fisher's definition: m3 / m2^1.5).
    pub fn skew(&self) -> f64 {
        let m2 = self.moment(2);
        let m3 = self.moment(3);
        if m2 == 0.0 {
            return 0.0;
        }
        m3 / m2.powf(1.5)
    }

    /// Skewness along axis.
    pub fn skew_axis(&self, axis: usize) -> RumpyArray {
        let m2 = self.moment_axis(2, axis);
        let m3 = self.moment_axis(3, axis);
        // m3 / m2^1.5
        let m2_pow = map_unary_op(&m2, UnaryOp::Sqrt).expect("sqrt works");
        let m2_pow = m2.binary_op(&m2_pow, BinaryOp::Mul).expect("broadcast works");
        m3.binary_op(&m2_pow, BinaryOp::Div).expect("broadcast works")
    }

    /// Kurtosis of all elements (Fisher's definition: m4 / m2^2 - 3).
    pub fn kurtosis(&self) -> f64 {
        let m2 = self.moment(2);
        let m4 = self.moment(4);
        if m2 == 0.0 {
            return 0.0;
        }
        m4 / (m2 * m2) - 3.0
    }

    /// Kurtosis along axis.
    pub fn kurtosis_axis(&self, axis: usize) -> RumpyArray {
        let m2 = self.moment_axis(2, axis);
        let m4 = self.moment_axis(4, axis);
        // m4 / m2^2 - 3
        let m2_sq = m2.binary_op(&m2, BinaryOp::Mul).expect("broadcast works");
        let ratio = m4.binary_op(&m2_sq, BinaryOp::Div).expect("broadcast works");
        let three = RumpyArray::full(vec![1], 3.0, ratio.dtype());
        ratio.binary_op(&three, BinaryOp::Sub).expect("broadcast works")
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
        let ptr = self.data_ptr();
        let dtype = self.dtype();
        let ops = dtype.ops();
        let mut max_val = f64::NEG_INFINITY;
        let mut max_idx = 0;
        for (i, offset) in self.iter_offsets().enumerate() {
            let val = unsafe { ops.read_f64(ptr, offset) }.unwrap_or(0.0);
            if val > max_val {
                max_val = val;
                max_idx = i;
            }
        }
        max_idx
    }

    /// Index of minimum element (flattened).
    pub fn argmin(&self) -> usize {
        let size = self.size();
        if size == 0 {
            return 0;
        }
        let ptr = self.data_ptr();
        let dtype = self.dtype();
        let ops = dtype.ops();
        let mut min_val = f64::INFINITY;
        let mut min_idx = 0;
        for (i, offset) in self.iter_offsets().enumerate() {
            let val = unsafe { ops.read_f64(ptr, offset) }.unwrap_or(0.0);
            if val < min_val {
                min_val = val;
                min_idx = i;
            }
        }
        min_idx
    }

    /// Index of maximum element along axis.
    pub fn argmax_axis(&self, axis: usize) -> RumpyArray {
        self.arg_reduce_axis(axis, |a, b| a > b)
    }

    /// Index of minimum element along axis.
    pub fn argmin_axis(&self, axis: usize) -> RumpyArray {
        self.arg_reduce_axis(axis, |a, b| a < b)
    }

    /// Helper for argmax_axis/argmin_axis.
    fn arg_reduce_axis<F>(&self, axis: usize, is_better: F) -> RumpyArray
    where
        F: Fn(f64, f64) -> bool,
    {
        let shape = self.shape();
        let axis_len = shape[axis];

        // Output shape: remove the axis dimension
        let mut out_shape: Vec<usize> = shape[..axis].to_vec();
        out_shape.extend_from_slice(&shape[axis + 1..]);
        if out_shape.is_empty() {
            out_shape = vec![1];
        }

        let out_size: usize = out_shape.iter().product();
        let mut result = RumpyArray::zeros(out_shape.clone(), DType::int64());

        if out_size == 0 || axis_len == 0 {
            return result;
        }

        let buffer = result.buffer_mut();
        let result_buffer = Arc::get_mut(buffer).expect("buffer must be unique");
        let result_ptr = result_buffer.as_mut_ptr();
        let int64_dtype = DType::int64();
        let ops = int64_dtype.ops();

        let mut outer_indices = vec![0usize; out_shape.len()];
        for out_i in 0..out_size {
            // Build indices for the input array
            let mut in_indices: Vec<usize> = outer_indices[..axis.min(outer_indices.len())].to_vec();
            in_indices.push(0);
            if axis < self.ndim() - 1 && outer_indices.len() > axis {
                in_indices.extend_from_slice(&outer_indices[axis..]);
            }

            // Find best index along axis
            let mut best_val = self.get_element(&in_indices);
            let mut best_idx: i64 = 0;
            for j in 1..axis_len {
                in_indices[axis] = j;
                let val = self.get_element(&in_indices);
                if is_better(val, best_val) {
                    best_val = val;
                    best_idx = j as i64;
                }
            }

            unsafe { ops.write_f64(result_ptr, out_i, best_idx as f64); }
            increment_indices(&mut outer_indices, &out_shape);
        }
        result
    }

    /// Collect all elements into a Vec (flattened, row-major order).
    fn to_vec(&self) -> Vec<f64> {
        let size = self.size();
        if size == 0 {
            return Vec::new();
        }

        // Fast path for contiguous f64 arrays
        if self.is_c_contiguous() && self.dtype().kind() == DTypeKind::Float64 {
            let ptr = self.data_ptr() as *const f64;
            let slice = unsafe { std::slice::from_raw_parts(ptr, size) };
            return slice.to_vec();
        }

        // Slow path for non-contiguous or non-f64
        let ptr = self.data_ptr();
        let dtype = self.dtype();
        let ops = dtype.ops();
        let mut values = Vec::with_capacity(size);
        for offset in self.iter_offsets() {
            values.push(unsafe { ops.read_f64(ptr, offset) }.unwrap_or(0.0));
        }
        values
    }

    // sort and argsort with axis parameter are defined later

    /// Return unique sorted values.
    pub fn unique(&self) -> RumpyArray {
        let mut values = self.to_vec();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        values.dedup_by(|a, b| (*a - *b).abs() < f64::EPSILON);
        RumpyArray::from_vec(values, self.dtype())
    }

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

    /// Base-10 logarithm of each element.
    pub fn log10(&self) -> Result<RumpyArray, UnaryOpError> {
        map_unary_op(self, UnaryOp::Log10)
    }

    /// Base-2 logarithm of each element.
    pub fn log2(&self) -> Result<RumpyArray, UnaryOpError> {
        map_unary_op(self, UnaryOp::Log2)
    }

    /// Hyperbolic sine of each element.
    pub fn sinh(&self) -> Result<RumpyArray, UnaryOpError> {
        map_unary_op(self, UnaryOp::Sinh)
    }

    /// Hyperbolic cosine of each element.
    pub fn cosh(&self) -> Result<RumpyArray, UnaryOpError> {
        map_unary_op(self, UnaryOp::Cosh)
    }

    /// Hyperbolic tangent of each element.
    pub fn tanh(&self) -> Result<RumpyArray, UnaryOpError> {
        map_unary_op(self, UnaryOp::Tanh)
    }

    /// Element-wise sign indication.
    pub fn sign(&self) -> Result<RumpyArray, UnaryOpError> {
        map_unary_op(self, UnaryOp::Sign)
    }

    /// Test element-wise for NaN.
    pub fn isnan(&self) -> Result<RumpyArray, UnaryOpError> {
        map_unary_op(self, UnaryOp::Isnan)
    }

    /// Test element-wise for infinity.
    pub fn isinf(&self) -> Result<RumpyArray, UnaryOpError> {
        map_unary_op(self, UnaryOp::Isinf)
    }

    /// Test element-wise for finiteness.
    pub fn isfinite(&self) -> Result<RumpyArray, UnaryOpError> {
        map_unary_op(self, UnaryOp::Isfinite)
    }

    /// Square of each element (x^2).
    pub fn square(&self) -> Result<RumpyArray, UnaryOpError> {
        map_unary_op(self, UnaryOp::Square)
    }

    /// Return a copy of the array (positive identity).
    pub fn positive(&self) -> Result<RumpyArray, UnaryOpError> {
        map_unary_op(self, UnaryOp::Positive)
    }

    /// Reciprocal of each element (1/x).
    pub fn reciprocal(&self) -> Result<RumpyArray, UnaryOpError> {
        map_unary_op(self, UnaryOp::Reciprocal)
    }

    /// Base-2 exponential (2^x) of each element.
    pub fn exp2(&self) -> Result<RumpyArray, UnaryOpError> {
        map_unary_op(self, UnaryOp::Exp2)
    }

    /// exp(x) - 1 for each element (more precise for small x).
    pub fn expm1(&self) -> Result<RumpyArray, UnaryOpError> {
        map_unary_op(self, UnaryOp::Expm1)
    }

    /// log(1 + x) for each element (more precise for small x).
    pub fn log1p(&self) -> Result<RumpyArray, UnaryOpError> {
        map_unary_op(self, UnaryOp::Log1p)
    }

    /// Cube root of each element.
    pub fn cbrt(&self) -> Result<RumpyArray, UnaryOpError> {
        map_unary_op(self, UnaryOp::Cbrt)
    }

    /// Truncate each element to integer towards zero.
    pub fn trunc(&self) -> Result<RumpyArray, UnaryOpError> {
        map_unary_op(self, UnaryOp::Trunc)
    }

    /// Round each element to nearest integer.
    pub fn rint(&self) -> Result<RumpyArray, UnaryOpError> {
        map_unary_op(self, UnaryOp::Rint)
    }

    /// Inverse hyperbolic sine of each element.
    pub fn arcsinh(&self) -> Result<RumpyArray, UnaryOpError> {
        map_unary_op(self, UnaryOp::Arcsinh)
    }

    /// Inverse hyperbolic cosine of each element.
    pub fn arccosh(&self) -> Result<RumpyArray, UnaryOpError> {
        map_unary_op(self, UnaryOp::Arccosh)
    }

    /// Inverse hyperbolic tangent of each element.
    pub fn arctanh(&self) -> Result<RumpyArray, UnaryOpError> {
        map_unary_op(self, UnaryOp::Arctanh)
    }

    /// Returns True where the sign bit is set (negative).
    pub fn signbit(&self) -> Result<RumpyArray, UnaryOpError> {
        map_unary_op(self, UnaryOp::Signbit)
    }

    /// Return the real part of the array.
    /// For complex arrays, extracts the real component.
    /// For real arrays, returns a copy.
    pub fn real(&self) -> RumpyArray {
        use crate::array::dtype::DTypeKind;

        let kind = self.dtype().kind();
        let src_ptr = self.data_ptr();
        let dtype = self.dtype();
        let ops = dtype.ops();

        match kind {
            DTypeKind::Complex128 => {
                let mut result = RumpyArray::zeros(self.shape().to_vec(), DType::float64());
                let result_buffer = Arc::get_mut(result.buffer_mut()).expect("unique");
                let result_ptr = result_buffer.as_mut_ptr();
                for (i, offset) in self.iter_offsets().enumerate() {
                    let (re, _im) = unsafe { ops.read_complex(src_ptr, offset) }.unwrap_or((0.0, 0.0));
                    unsafe { *(result_ptr as *mut f64).add(i) = re; }
                }
                result
            }
            DTypeKind::Complex64 => {
                let mut result = RumpyArray::zeros(self.shape().to_vec(), DType::float32());
                let result_buffer = Arc::get_mut(result.buffer_mut()).expect("unique");
                let result_ptr = result_buffer.as_mut_ptr();
                for (i, offset) in self.iter_offsets().enumerate() {
                    let (re, _im) = unsafe { ops.read_complex(src_ptr, offset) }.unwrap_or((0.0, 0.0));
                    unsafe { *(result_ptr as *mut f32).add(i) = re as f32; }
                }
                result
            }
            _ => self.copy()
        }
    }

    /// Return the imaginary part of the array.
    /// For complex arrays, extracts the imaginary component.
    /// For real arrays, returns zeros.
    pub fn imag(&self) -> RumpyArray {
        use crate::array::dtype::DTypeKind;

        let kind = self.dtype().kind();
        let src_ptr = self.data_ptr();
        let dtype = self.dtype();
        let ops = dtype.ops();

        match kind {
            DTypeKind::Complex128 => {
                let mut result = RumpyArray::zeros(self.shape().to_vec(), DType::float64());
                let result_buffer = Arc::get_mut(result.buffer_mut()).expect("unique");
                let result_ptr = result_buffer.as_mut_ptr();
                for (i, offset) in self.iter_offsets().enumerate() {
                    let (_re, im) = unsafe { ops.read_complex(src_ptr, offset) }.unwrap_or((0.0, 0.0));
                    unsafe { *(result_ptr as *mut f64).add(i) = im; }
                }
                result
            }
            DTypeKind::Complex64 => {
                let mut result = RumpyArray::zeros(self.shape().to_vec(), DType::float32());
                let result_buffer = Arc::get_mut(result.buffer_mut()).expect("unique");
                let result_ptr = result_buffer.as_mut_ptr();
                for (i, offset) in self.iter_offsets().enumerate() {
                    let (_re, im) = unsafe { ops.read_complex(src_ptr, offset) }.unwrap_or((0.0, 0.0));
                    unsafe { *(result_ptr as *mut f32).add(i) = im as f32; }
                }
                result
            }
            _ => RumpyArray::zeros(self.shape().to_vec(), self.dtype().clone())
        }
    }

    /// Return the complex conjugate of the array.
    /// For complex arrays, negates the imaginary component.
    /// For real arrays, returns a copy.
    pub fn conj(&self) -> RumpyArray {
        use crate::array::dtype::DTypeKind;

        let kind = self.dtype().kind();
        let src_ptr = self.data_ptr();
        let dtype = self.dtype();
        let ops = dtype.ops();

        match kind {
            DTypeKind::Complex128 => {
                let mut result = RumpyArray::zeros(self.shape().to_vec(), DType::complex128());
                let result_buffer = Arc::get_mut(result.buffer_mut()).expect("unique");
                let result_ptr = result_buffer.as_mut_ptr();
                for (i, offset) in self.iter_offsets().enumerate() {
                    let (re, im) = unsafe { ops.read_complex(src_ptr, offset) }.unwrap_or((0.0, 0.0));
                    unsafe {
                        *(result_ptr as *mut f64).add(i * 2) = re;
                        *(result_ptr as *mut f64).add(i * 2 + 1) = -im;
                    }
                }
                result
            }
            DTypeKind::Complex64 => {
                let mut result = RumpyArray::zeros(self.shape().to_vec(), DType::complex64());
                let result_buffer = Arc::get_mut(result.buffer_mut()).expect("unique");
                let result_ptr = result_buffer.as_mut_ptr();
                for (i, offset) in self.iter_offsets().enumerate() {
                    let (re, im) = unsafe { ops.read_complex(src_ptr, offset) }.unwrap_or((0.0, 0.0));
                    unsafe {
                        *(result_ptr as *mut f32).add(i * 2) = re as f32;
                        *(result_ptr as *mut f32).add(i * 2 + 1) = -im as f32;
                    }
                }
                result
            }
            _ => self.copy()
        }
    }

    /// Replace NaN with zero and infinity with large finite numbers.
    ///
    /// Returns an array with the same shape where:
    /// - NaN is replaced with `nan` (default 0.0)
    /// - positive infinity is replaced with `posinf` (default a large positive number)
    /// - negative infinity is replaced with `neginf` (default a large negative number)
    pub fn nan_to_num(&self, nan: f64, posinf: Option<f64>, neginf: Option<f64>) -> RumpyArray {
        use crate::array::dtype::DTypeKind;

        let kind = self.dtype().kind();
        let src_ptr = self.data_ptr();
        let dtype = self.dtype();
        let ops = dtype.ops();

        macro_rules! nan_to_num_impl {
            ($T:ty, $dtype:expr) => {{
                let max_val = <$T>::MAX;
                let min_val = <$T>::MIN;
                let pos = posinf.map(|v| v as $T).unwrap_or(max_val);
                let neg = neginf.map(|v| v as $T).unwrap_or(min_val);
                let nan_val = nan as $T;

                let mut result = RumpyArray::zeros(self.shape().to_vec(), $dtype);
                let result_buffer = Arc::get_mut(result.buffer_mut()).expect("unique");
                let result_ptr = result_buffer.as_mut_ptr();
                for (i, offset) in self.iter_offsets().enumerate() {
                    let v: $T = unsafe { *(src_ptr.offset(offset) as *const $T) };
                    let out = if v.is_nan() {
                        nan_val
                    } else if v.is_infinite() {
                        if v > 0.0 { pos } else { neg }
                    } else {
                        v
                    };
                    unsafe { *(result_ptr as *mut $T).add(i) = out; }
                }
                result
            }};
        }

        match kind {
            DTypeKind::Float64 => nan_to_num_impl!(f64, DType::float64()),
            DTypeKind::Float32 => nan_to_num_impl!(f32, DType::float32()),
            DTypeKind::Float16 => {
                use half::f16;
                let pos = posinf.map(f16::from_f64).unwrap_or(f16::MAX);
                let neg = neginf.map(f16::from_f64).unwrap_or(f16::MIN);
                let nan_val = f16::from_f64(nan);

                let mut result = RumpyArray::zeros(self.shape().to_vec(), DType::float16());
                let result_buffer = Arc::get_mut(result.buffer_mut()).expect("unique");
                let result_ptr = result_buffer.as_mut_ptr();
                for (i, offset) in self.iter_offsets().enumerate() {
                    let v = unsafe { *(src_ptr.offset(offset) as *const f16) };
                    let out = if v.is_nan() {
                        nan_val
                    } else if v.is_infinite() {
                        if v.to_f32() > 0.0 { pos } else { neg }
                    } else {
                        v
                    };
                    unsafe { *(result_ptr as *mut f16).add(i) = out; }
                }
                result
            }
            DTypeKind::Complex128 => {
                let max_val = f64::MAX;
                let min_val = f64::MIN;
                let pos = posinf.unwrap_or(max_val);
                let neg = neginf.unwrap_or(min_val);

                let mut result = RumpyArray::zeros(self.shape().to_vec(), DType::complex128());
                let result_buffer = Arc::get_mut(result.buffer_mut()).expect("unique");
                let result_ptr = result_buffer.as_mut_ptr();
                for (i, offset) in self.iter_offsets().enumerate() {
                    let (re, im) = unsafe { ops.read_complex(src_ptr, offset) }.unwrap_or((0.0, 0.0));
                    let out_re = if re.is_nan() { nan } else if re.is_infinite() { if re > 0.0 { pos } else { neg } } else { re };
                    let out_im = if im.is_nan() { nan } else if im.is_infinite() { if im > 0.0 { pos } else { neg } } else { im };
                    unsafe {
                        *(result_ptr as *mut f64).add(i * 2) = out_re;
                        *(result_ptr as *mut f64).add(i * 2 + 1) = out_im;
                    }
                }
                result
            }
            DTypeKind::Complex64 => {
                let max_val = f32::MAX;
                let min_val = f32::MIN;
                let pos = posinf.map(|v| v as f32).unwrap_or(max_val);
                let neg = neginf.map(|v| v as f32).unwrap_or(min_val);
                let nan_val = nan as f32;

                let mut result = RumpyArray::zeros(self.shape().to_vec(), DType::complex64());
                let result_buffer = Arc::get_mut(result.buffer_mut()).expect("unique");
                let result_ptr = result_buffer.as_mut_ptr();
                for (i, offset) in self.iter_offsets().enumerate() {
                    let (re, im) = unsafe { ops.read_complex(src_ptr, offset) }.unwrap_or((0.0, 0.0));
                    let (re, im) = (re as f32, im as f32);
                    let out_re = if re.is_nan() { nan_val } else if re.is_infinite() { if re > 0.0 { pos } else { neg } } else { re };
                    let out_im = if im.is_nan() { nan_val } else if im.is_infinite() { if im > 0.0 { pos } else { neg } } else { im };
                    unsafe {
                        *(result_ptr as *mut f32).add(i * 2) = out_re;
                        *(result_ptr as *mut f32).add(i * 2 + 1) = out_im;
                    }
                }
                result
            }
            // For integer types, no NaN or Inf, just return a copy
            _ => self.copy()
        }
    }

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

    /// Sort array along an axis.
    pub fn sort(&self, axis: Option<usize>) -> RumpyArray {
        // If axis is None, flatten and sort
        let axis = match axis {
            None => {
                let flat = self.reshape(vec![self.size()]).expect("flatten should work");
                return flat.sort(Some(0));
            }
            Some(ax) => ax,
        };
        assert!(axis < self.ndim(), "axis out of bounds");

        let shape = self.shape();
        let axis_len = shape[axis];
        let dtype = self.dtype().clone();

        let mut result = self.copy();
        if axis_len <= 1 {
            return result;
        }

        // For each position along other axes, sort the elements along the target axis
        let mut out_shape: Vec<usize> = shape[..axis].to_vec();
        out_shape.extend_from_slice(&shape[axis + 1..]);
        if out_shape.is_empty() {
            out_shape = vec![1];
        }
        let out_size: usize = out_shape.iter().product();

        let ops = dtype.ops();
        let result_buffer = Arc::get_mut(result.buffer_mut()).expect("unique");
        let result_ptr = result_buffer.as_mut_ptr();

        let mut out_indices = vec![0usize; out_shape.len()];
        for _ in 0..out_size {
            // Collect values along axis
            let mut values: Vec<(f64, usize)> = Vec::with_capacity(axis_len);
            for k in 0..axis_len {
                let mut in_indices: Vec<usize> = out_indices[..axis.min(out_indices.len())].to_vec();
                in_indices.push(k);
                if axis < self.ndim() - 1 {
                    in_indices.extend_from_slice(&out_indices[axis..]);
                }
                values.push((self.get_element(&in_indices), k));
            }

            // Sort by value
            values.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

            // Write sorted values back
            for (k, (val, _)) in values.iter().enumerate() {
                let mut in_indices: Vec<usize> = out_indices[..axis.min(out_indices.len())].to_vec();
                in_indices.push(k);
                if axis < self.ndim() - 1 {
                    in_indices.extend_from_slice(&out_indices[axis..]);
                }
                let byte_offset = result.byte_offset_for(&in_indices);
                let linear_idx = byte_offset as usize / dtype.itemsize();
                unsafe { ops.write_f64(result_ptr, linear_idx, *val); }
            }

            increment_indices(&mut out_indices, &out_shape);
        }
        result
    }

    /// Return indices that would sort the array along an axis.
    pub fn argsort(&self, axis: Option<usize>) -> RumpyArray {
        // If axis is None, flatten and argsort
        let axis = match axis {
            None => {
                let flat = self.reshape(vec![self.size()]).expect("flatten should work");
                return flat.argsort(Some(0));
            }
            Some(ax) => ax,
        };
        assert!(axis < self.ndim(), "axis out of bounds");

        let shape = self.shape();
        let axis_len = shape[axis];

        let mut result = RumpyArray::zeros(shape.to_vec(), DType::int64());
        if axis_len <= 1 {
            // For single element, index is 0
            return result;
        }

        let mut out_shape: Vec<usize> = shape[..axis].to_vec();
        out_shape.extend_from_slice(&shape[axis + 1..]);
        if out_shape.is_empty() {
            out_shape = vec![1];
        }
        let out_size: usize = out_shape.iter().product();

        let result_buffer = Arc::get_mut(result.buffer_mut()).expect("unique");
        let result_ptr = result_buffer.as_mut_ptr() as *mut i64;

        let mut out_indices = vec![0usize; out_shape.len()];
        for _ in 0..out_size {
            // Collect values along axis with their indices
            let mut values: Vec<(f64, usize)> = Vec::with_capacity(axis_len);
            for k in 0..axis_len {
                let mut in_indices: Vec<usize> = out_indices[..axis.min(out_indices.len())].to_vec();
                in_indices.push(k);
                if axis < self.ndim() - 1 {
                    in_indices.extend_from_slice(&out_indices[axis..]);
                }
                values.push((self.get_element(&in_indices), k));
            }

            // Sort by value, keeping track of original indices
            values.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

            // Write sorted indices back
            for (k, (_, orig_idx)) in values.iter().enumerate() {
                let mut in_indices: Vec<usize> = out_indices[..axis.min(out_indices.len())].to_vec();
                in_indices.push(k);
                if axis < self.ndim() - 1 {
                    in_indices.extend_from_slice(&out_indices[axis..]);
                }
                let byte_offset = result.byte_offset_for(&in_indices);
                let linear_idx = byte_offset as usize / 8; // i64 is 8 bytes
                unsafe { *result_ptr.add(linear_idx) = *orig_idx as i64; }
            }

            increment_indices(&mut out_indices, &out_shape);
        }
        result
    }

    /// Partition array along an axis using partial sort.
    /// The kth element will be in its sorted position, with smaller elements before
    /// and larger elements after (but not necessarily sorted).
    pub fn partition(&self, kth: usize, axis: Option<usize>) -> RumpyArray {
        // If axis is None, flatten and partition
        let axis = match axis {
            None => {
                let flat = self.reshape(vec![self.size()]).expect("flatten should work");
                return flat.partition(kth, Some(0));
            }
            Some(ax) => ax,
        };
        assert!(axis < self.ndim(), "axis out of bounds");

        let shape = self.shape();
        let axis_len = shape[axis];
        assert!(kth < axis_len, "kth must be less than axis length");

        if axis_len <= 1 {
            return self.copy();
        }

        // Copy input to result, then partition in-place
        let mut result = self.copy();
        partition_inplace(&mut result, kth, axis);
        result
    }

    /// Return indices that would partition the array along an axis.
    /// The kth index will be in its sorted position.
    pub fn argpartition(&self, kth: usize, axis: Option<usize>) -> RumpyArray {
        // If axis is None, flatten and argpartition
        let axis = match axis {
            None => {
                let flat = self.reshape(vec![self.size()]).expect("flatten should work");
                return flat.argpartition(kth, Some(0));
            }
            Some(ax) => ax,
        };
        assert!(axis < self.ndim(), "axis out of bounds");

        let shape = self.shape();
        let axis_len = shape[axis];
        assert!(kth < axis_len, "kth must be less than axis length");

        if axis_len <= 1 {
            return RumpyArray::zeros(shape.to_vec(), DType::int64());
        }

        // Compute partitioned indices using helper
        argpartition_impl(self, kth, axis)
    }

    /// Count number of non-zero elements.
    pub fn count_nonzero(&self) -> usize {
        let size = self.size();
        if size == 0 {
            return 0;
        }
        let ptr = self.data_ptr();
        let dtype = self.dtype();
        let ops = dtype.ops();
        let mut count = 0;
        for offset in self.iter_offsets() {
            if let Some(val) = unsafe { ops.read_f64(ptr, offset) } {
                if val != 0.0 {
                    count += 1;
                }
            }
        }
        count
    }

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

    /// Test if all elements evaluate to True.
    pub fn all(&self) -> bool {
        let size = self.size();
        if size == 0 {
            return true; // numpy convention: empty array is all True
        }
        let ptr = self.data_ptr();
        let dtype = self.dtype();
        let ops = dtype.ops();
        for offset in self.iter_offsets() {
            if let Some(val) = unsafe { ops.read_f64(ptr, offset) } {
                if val == 0.0 {
                    return false;
                }
            }
        }
        true
    }

    /// Test if all elements along axis evaluate to True.
    pub fn all_axis(&self, axis: usize) -> RumpyArray {
        let mut out_shape: Vec<usize> = self.shape().to_vec();
        let axis_len = out_shape.remove(axis);
        if out_shape.is_empty() {
            out_shape = vec![1];
        }

        let mut result = RumpyArray::zeros(out_shape.clone(), DType::bool());
        let out_size = result.size();
        if out_size == 0 || axis_len == 0 {
            return result;
        }

        let bool_dtype = DType::bool();
        let buffer = result.buffer_mut();
        let result_buffer = Arc::get_mut(buffer).expect("buffer must be unique");
        let result_ptr = result_buffer.as_mut_ptr();
        let ops = bool_dtype.ops();

        let mut out_indices = vec![0usize; out_shape.len()];
        for i in 0..out_size {
            let mut in_indices: Vec<usize> = out_indices[..axis.min(out_indices.len())].to_vec();
            in_indices.push(0);
            if axis < self.ndim() - 1 {
                in_indices.extend_from_slice(&out_indices[axis..]);
            }

            let mut result_val = true;
            for j in 0..axis_len {
                in_indices[axis] = j;
                if self.get_element(&in_indices) == 0.0 {
                    result_val = false;
                    break;
                }
            }

            unsafe { ops.write_f64(result_ptr, i, if result_val { 1.0 } else { 0.0 }); }
            increment_indices(&mut out_indices, &out_shape);
        }

        result
    }

    /// Test if any element evaluates to True.
    pub fn any(&self) -> bool {
        let size = self.size();
        if size == 0 {
            return false; // numpy convention: empty array is all False
        }
        let ptr = self.data_ptr();
        let dtype = self.dtype();
        let ops = dtype.ops();
        for offset in self.iter_offsets() {
            if let Some(val) = unsafe { ops.read_f64(ptr, offset) } {
                if val != 0.0 {
                    return true;
                }
            }
        }
        false
    }

    /// Test if any element along axis evaluates to True.
    pub fn any_axis(&self, axis: usize) -> RumpyArray {
        let mut out_shape: Vec<usize> = self.shape().to_vec();
        let axis_len = out_shape.remove(axis);
        if out_shape.is_empty() {
            out_shape = vec![1];
        }

        let mut result = RumpyArray::zeros(out_shape.clone(), DType::bool());
        let out_size = result.size();
        if out_size == 0 || axis_len == 0 {
            return result;
        }

        let bool_dtype = DType::bool();
        let buffer = result.buffer_mut();
        let result_buffer = Arc::get_mut(buffer).expect("buffer must be unique");
        let result_ptr = result_buffer.as_mut_ptr();
        let ops = bool_dtype.ops();

        let mut out_indices = vec![0usize; out_shape.len()];
        for i in 0..out_size {
            let mut in_indices: Vec<usize> = out_indices[..axis.min(out_indices.len())].to_vec();
            in_indices.push(0);
            if axis < self.ndim() - 1 {
                in_indices.extend_from_slice(&out_indices[axis..]);
            }

            let mut result_val = false;
            for j in 0..axis_len {
                in_indices[axis] = j;
                if self.get_element(&in_indices) != 0.0 {
                    result_val = true;
                    break;
                }
            }

            unsafe { ops.write_f64(result_ptr, i, if result_val { 1.0 } else { 0.0 }); }
            increment_indices(&mut out_indices, &out_shape);
        }

        result
    }

    /// Clip values to a range.
    pub fn clip(&self, a_min: Option<f64>, a_max: Option<f64>) -> RumpyArray {
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
        self.cumulative_op(axis, 0.0, |acc, x| acc + x)
    }

    /// Cumulative product along axis (or flattened if axis is None).
    pub fn cumprod(&self, axis: Option<usize>) -> RumpyArray {
        self.cumulative_op(axis, 1.0, |acc, x| acc * x)
    }

    /// Calculate flat index for given n-dimensional indices (C-order).
    fn flat_index_for(&self, indices: &[usize]) -> usize {
        let shape = self.shape();
        let mut flat = 0;
        let mut stride = 1;
        for i in (0..indices.len()).rev() {
            flat += indices[i] * stride;
            stride *= shape[i];
        }
        flat
    }

    // ========================================================================
    // NaN-aware reductions
    // ========================================================================
    //
    // Note: NaN-aware axis reductions still use get_element() which is slower.
    // Non-NaN reductions use registry strided loops via reduce_axis_op().

    /// Helper: iterate non-NaN values, returning (value, was_found) for full reductions.
    fn nan_reduce_full<F>(&self, init: f64, mut f: F) -> (f64, bool)
    where
        F: FnMut(f64, f64) -> f64,
    {
        if self.size() == 0 {
            return (init, false);
        }
        let ptr = self.data_ptr();
        let dtype = self.dtype();
        let ops = dtype.ops();
        let mut acc = init;
        let mut found = false;
        for offset in self.iter_offsets() {
            let val = unsafe { ops.read_f64(ptr, offset) }.unwrap_or(0.0);
            if !val.is_nan() {
                acc = f(acc, val);
                found = true;
            }
        }
        (acc, found)
    }

    /// Helper: axis reduction ignoring NaN values.
    fn nan_reduce_axis<F>(&self, axis: usize, result_dtype: DType, mut reduce_fn: F) -> RumpyArray
    where
        F: FnMut(&mut [usize], usize, usize) -> f64,
    {
        let mut out_shape: Vec<usize> = self.shape().to_vec();
        let axis_len = out_shape.remove(axis);
        if out_shape.is_empty() {
            out_shape = vec![1];
        }

        let mut result = RumpyArray::zeros(out_shape.clone(), result_dtype.clone());
        let out_size = result.size();
        if out_size == 0 || axis_len == 0 {
            return result;
        }

        let buffer = result.buffer_mut();
        let result_buffer = Arc::get_mut(buffer).expect("buffer must be unique");
        let result_ptr = result_buffer.as_mut_ptr();
        let ops = result_dtype.ops();

        let mut out_indices = vec![0usize; out_shape.len()];
        for i in 0..out_size {
            // Build input indices with placeholder for axis
            let mut in_indices: Vec<usize> = out_indices[..axis.min(out_indices.len())].to_vec();
            in_indices.push(0);
            if axis < self.ndim() - 1 {
                in_indices.extend_from_slice(&out_indices[axis..]);
            }

            let val = reduce_fn(&mut in_indices, axis, axis_len);
            unsafe { ops.write_f64(result_ptr, i, val); }
            increment_indices(&mut out_indices, &out_shape);
        }
        result
    }

    /// Sum all elements, ignoring NaN values.
    pub fn nansum(&self) -> f64 {
        self.nan_reduce_full(0.0, |acc, v| acc + v).0
    }

    /// Sum along axis, ignoring NaN values.
    pub fn nansum_axis(&self, axis: usize) -> RumpyArray {
        self.nan_reduce_axis(axis, self.dtype(), |indices, ax, len| {
            let mut sum = 0.0;
            for j in 0..len {
                indices[ax] = j;
                let val = self.get_element(indices);
                if !val.is_nan() {
                    sum += val;
                }
            }
            sum
        })
    }

    /// Product of all elements, ignoring NaN values.
    pub fn nanprod(&self) -> f64 {
        self.nan_reduce_full(1.0, |acc, v| acc * v).0
    }

    /// Product along axis, ignoring NaN values.
    pub fn nanprod_axis(&self, axis: usize) -> RumpyArray {
        self.nan_reduce_axis(axis, self.dtype(), |indices, ax, len| {
            let mut prod = 1.0;
            for j in 0..len {
                indices[ax] = j;
                let val = self.get_element(indices);
                if !val.is_nan() {
                    prod *= val;
                }
            }
            prod
        })
    }

    /// Mean of all elements, ignoring NaN values.
    pub fn nanmean(&self) -> f64 {
        if self.size() == 0 {
            return f64::NAN;
        }
        let ptr = self.data_ptr();
        let dtype = self.dtype();
        let ops = dtype.ops();
        let mut sum = 0.0;
        let mut count = 0usize;
        for offset in self.iter_offsets() {
            let val = unsafe { ops.read_f64(ptr, offset) }.unwrap_or(0.0);
            if !val.is_nan() {
                sum += val;
                count += 1;
            }
        }
        if count == 0 { f64::NAN } else { sum / count as f64 }
    }

    /// Mean along axis, ignoring NaN values.
    pub fn nanmean_axis(&self, axis: usize) -> RumpyArray {
        self.nan_reduce_axis(axis, DType::float64(), |indices, ax, len| {
            let mut sum = 0.0;
            let mut count = 0usize;
            for j in 0..len {
                indices[ax] = j;
                let val = self.get_element(indices);
                if !val.is_nan() {
                    sum += val;
                    count += 1;
                }
            }
            if count == 0 { f64::NAN } else { sum / count as f64 }
        })
    }

    /// Variance of all elements, ignoring NaN values.
    pub fn nanvar(&self) -> f64 {
        let mean = self.nanmean();
        if mean.is_nan() {
            return f64::NAN;
        }
        let ptr = self.data_ptr();
        let dtype = self.dtype();
        let ops = dtype.ops();
        let mut sum_sq = 0.0;
        let mut count = 0usize;
        for offset in self.iter_offsets() {
            let val = unsafe { ops.read_f64(ptr, offset) }.unwrap_or(0.0);
            if !val.is_nan() {
                let diff = val - mean;
                sum_sq += diff * diff;
                count += 1;
            }
        }
        sum_sq / count as f64
    }

    /// Variance along axis, ignoring NaN values.
    pub fn nanvar_axis(&self, axis: usize) -> RumpyArray {
        self.nan_reduce_axis(axis, DType::float64(), |indices, ax, len| {
            // Two-pass: mean then variance
            let mut sum = 0.0;
            let mut count = 0usize;
            for j in 0..len {
                indices[ax] = j;
                let val = self.get_element(indices);
                if !val.is_nan() {
                    sum += val;
                    count += 1;
                }
            }
            if count == 0 {
                return f64::NAN;
            }
            let mean = sum / count as f64;
            let mut sum_sq = 0.0;
            for j in 0..len {
                indices[ax] = j;
                let val = self.get_element(indices);
                if !val.is_nan() {
                    let diff = val - mean;
                    sum_sq += diff * diff;
                }
            }
            sum_sq / count as f64
        })
    }

    /// Standard deviation of all elements, ignoring NaN values.
    pub fn nanstd(&self) -> f64 {
        self.nanvar().sqrt()
    }

    /// Standard deviation along axis, ignoring NaN values.
    pub fn nanstd_axis(&self, axis: usize) -> RumpyArray {
        map_unary_op(&self.nanvar_axis(axis), UnaryOp::Sqrt)
            .expect("sqrt always succeeds on numeric types")
    }

    /// Minimum of all elements, ignoring NaN values.
    pub fn nanmin(&self) -> f64 {
        let (val, found) = self.nan_reduce_full(f64::INFINITY, |acc, v| acc.min(v));
        if found { val } else { f64::NAN }
    }

    /// Minimum along axis, ignoring NaN values.
    pub fn nanmin_axis(&self, axis: usize) -> RumpyArray {
        self.nan_reduce_axis(axis, self.dtype(), |indices, ax, len| {
            let mut min_val = f64::INFINITY;
            let mut found = false;
            for j in 0..len {
                indices[ax] = j;
                let val = self.get_element(indices);
                if !val.is_nan() {
                    min_val = min_val.min(val);
                    found = true;
                }
            }
            if found { min_val } else { f64::NAN }
        })
    }

    /// Maximum of all elements, ignoring NaN values.
    pub fn nanmax(&self) -> f64 {
        let (val, found) = self.nan_reduce_full(f64::NEG_INFINITY, |acc, v| acc.max(v));
        if found { val } else { f64::NAN }
    }

    /// Maximum along axis, ignoring NaN values.
    pub fn nanmax_axis(&self, axis: usize) -> RumpyArray {
        self.nan_reduce_axis(axis, self.dtype(), |indices, ax, len| {
            let mut max_val = f64::NEG_INFINITY;
            let mut found = false;
            for j in 0..len {
                indices[ax] = j;
                let val = self.get_element(indices);
                if !val.is_nan() {
                    max_val = max_val.max(val);
                    found = true;
                }
            }
            if found { max_val } else { f64::NAN }
        })
    }

    /// Index of minimum element, ignoring NaN values.
    /// Returns None if all elements are NaN.
    pub fn nanargmin(&self) -> Option<usize> {
        if self.size() == 0 {
            return None;
        }
        let ptr = self.data_ptr();
        let dtype = self.dtype();
        let ops = dtype.ops();
        let mut min_val = f64::INFINITY;
        let mut min_idx: Option<usize> = None;
        for (i, offset) in self.iter_offsets().enumerate() {
            let val = unsafe { ops.read_f64(ptr, offset) }.unwrap_or(0.0);
            if !val.is_nan() && val < min_val {
                min_val = val;
                min_idx = Some(i);
            }
        }
        min_idx
    }

    /// Index of minimum element along axis, ignoring NaN values.
    pub fn nanargmin_axis(&self, axis: usize) -> RumpyArray {
        self.nan_arg_reduce_axis(axis, |a, b| a < b)
    }

    /// Index of maximum element, ignoring NaN values.
    /// Returns None if all elements are NaN.
    pub fn nanargmax(&self) -> Option<usize> {
        if self.size() == 0 {
            return None;
        }
        let ptr = self.data_ptr();
        let dtype = self.dtype();
        let ops = dtype.ops();
        let mut max_val = f64::NEG_INFINITY;
        let mut max_idx: Option<usize> = None;
        for (i, offset) in self.iter_offsets().enumerate() {
            let val = unsafe { ops.read_f64(ptr, offset) }.unwrap_or(0.0);
            if !val.is_nan() && val > max_val {
                max_val = val;
                max_idx = Some(i);
            }
        }
        max_idx
    }

    /// Index of maximum element along axis, ignoring NaN values.
    pub fn nanargmax_axis(&self, axis: usize) -> RumpyArray {
        self.nan_arg_reduce_axis(axis, |a, b| a > b)
    }

    /// Helper for nanargmax_axis/nanargmin_axis.
    fn nan_arg_reduce_axis<F>(&self, axis: usize, is_better: F) -> RumpyArray
    where
        F: Fn(f64, f64) -> bool,
    {
        self.nan_reduce_axis(axis, DType::int64(), |indices, ax, len| {
            let mut best_val: Option<f64> = None;
            let mut best_idx: i64 = 0;
            for j in 0..len {
                indices[ax] = j;
                let val = self.get_element(indices);
                if !val.is_nan() {
                    match best_val {
                        None => {
                            best_val = Some(val);
                            best_idx = j as i64;
                        }
                        Some(bv) if is_better(val, bv) => {
                            best_val = Some(val);
                            best_idx = j as i64;
                        }
                        _ => {}
                    }
                }
            }
            best_idx as f64
        })
    }

    /// Convert array to nested Python lists.
    pub fn to_pylist(&self, py: pyo3::Python<'_>) -> pyo3::PyResult<pyo3::PyObject> {
        use pyo3::types::PyList;

        fn build_list(
            arr: &RumpyArray,
            py: pyo3::Python<'_>,
            depth: usize,
            indices: &mut Vec<usize>,
        ) -> pyo3::PyResult<pyo3::PyObject> {
            use pyo3::IntoPyObject;
            if depth == arr.ndim() {
                // Base case: return scalar
                let val = arr.get_element(indices);
                return Ok(val.into_pyobject(py)?.into_any().unbind());
            }

            // Build list for this dimension
            let dim_size = arr.shape()[depth];
            let mut items = Vec::with_capacity(dim_size);
            for i in 0..dim_size {
                indices[depth] = i;
                items.push(build_list(arr, py, depth + 1, indices)?);
            }
            let list = PyList::new(py, items)?;
            Ok(list.into_pyobject(py)?.into_any().unbind())
        }

        let mut indices = vec![0usize; self.ndim()];
        build_list(self, py, 0, &mut indices)
    }
}

/// Partition array in-place along an axis.
/// Uses type-specific operations for best performance.
fn partition_inplace(arr: &mut RumpyArray, kth: usize, axis: usize) {
    // Copy metadata before borrowing mutably
    let shape: Vec<usize> = arr.shape().to_vec();
    let strides: Vec<isize> = arr.strides().to_vec();
    let axis_len = shape[axis];
    let dtype = arr.dtype().clone();
    let itemsize = dtype.itemsize();
    let ndim = arr.ndim();
    let is_contiguous = arr.is_c_contiguous();

    // For 1D contiguous arrays, use optimized typed partition
    if ndim == 1 && is_contiguous {
        let buffer = Arc::get_mut(arr.buffer_mut()).expect("unique");
        let ptr = buffer.as_mut_ptr();

        match dtype.kind() {
            DTypeKind::Float64 => {
                let slice = unsafe { std::slice::from_raw_parts_mut(ptr as *mut f64, axis_len) };
                slice.select_nth_unstable_by(kth, |a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            }
            DTypeKind::Float32 => {
                let slice = unsafe { std::slice::from_raw_parts_mut(ptr as *mut f32, axis_len) };
                slice.select_nth_unstable_by(kth, |a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            }
            DTypeKind::Int64 => {
                let slice = unsafe { std::slice::from_raw_parts_mut(ptr as *mut i64, axis_len) };
                slice.select_nth_unstable(kth);
            }
            DTypeKind::Int32 => {
                let slice = unsafe { std::slice::from_raw_parts_mut(ptr as *mut i32, axis_len) };
                slice.select_nth_unstable(kth);
            }
            DTypeKind::Int16 => {
                let slice = unsafe { std::slice::from_raw_parts_mut(ptr as *mut i16, axis_len) };
                slice.select_nth_unstable(kth);
            }
            DTypeKind::Uint64 => {
                let slice = unsafe { std::slice::from_raw_parts_mut(ptr as *mut u64, axis_len) };
                slice.select_nth_unstable(kth);
            }
            DTypeKind::Uint32 => {
                let slice = unsafe { std::slice::from_raw_parts_mut(ptr as *mut u32, axis_len) };
                slice.select_nth_unstable(kth);
            }
            DTypeKind::Uint16 => {
                let slice = unsafe { std::slice::from_raw_parts_mut(ptr as *mut u16, axis_len) };
                slice.select_nth_unstable(kth);
            }
            DTypeKind::Uint8 => {
                let slice = unsafe { std::slice::from_raw_parts_mut(ptr as *mut u8, axis_len) };
                slice.select_nth_unstable(kth);
            }
            _ => {
                // Fallback: use indices and DTypeOps comparison via partition_lane_typed
                partition_lane_typed(ptr, 0, axis_len, itemsize as isize, kth, &dtype, itemsize);
            }
        }
        return;
    }

    // Multi-dimensional case: partition each lane along axis
    let mut out_shape: Vec<usize> = shape[..axis].to_vec();
    out_shape.extend_from_slice(&shape[axis + 1..]);
    if out_shape.is_empty() {
        out_shape = vec![1];
    }
    let out_size: usize = out_shape.iter().product();
    let axis_stride = strides[axis];

    let buffer = Arc::get_mut(arr.buffer_mut()).expect("unique");
    let ptr = buffer.as_mut_ptr();

    let mut out_indices = vec![0usize; out_shape.len()];
    for _ in 0..out_size {
        // Calculate base offset for this lane
        let mut base_offset: isize = 0;
        let mut idx_pos = 0;
        for (d, _) in shape.iter().enumerate() {
            if d == axis {
                continue;
            }
            base_offset += (out_indices[idx_pos] as isize) * strides[d];
            idx_pos += 1;
        }

        // Partition this lane using typed operations
        partition_lane_typed(ptr, base_offset, axis_len, axis_stride, kth, &dtype, itemsize);

        increment_indices(&mut out_indices, &out_shape);
    }
}

/// Partition a single lane (1D slice) in-place using typed operations.
fn partition_lane_typed(
    ptr: *mut u8,
    base_offset: isize,
    len: usize,
    stride: isize,
    kth: usize,
    dtype: &DType,
    itemsize: usize,
) {
    // For strided access, we need to work with indices
    let mut indices: Vec<usize> = (0..len).collect();
    let ops = dtype.ops();

    // Partition indices based on values
    indices.select_nth_unstable_by(kth, |&a, &b| {
        let a_off = base_offset + (a as isize) * stride;
        let b_off = base_offset + (b as isize) * stride;
        unsafe { ops.compare_elements(ptr, a_off, ptr, b_off) }
    });

    // Now reorder data according to partitioned indices
    // Use a temporary buffer to avoid in-place permutation complexity
    let mut temp = vec![0u8; len * itemsize];
    for (new_pos, &old_pos) in indices.iter().enumerate() {
        let src_off = base_offset + (old_pos as isize) * stride;
        unsafe {
            std::ptr::copy_nonoverlapping(
                ptr.offset(src_off),
                temp.as_mut_ptr().add(new_pos * itemsize),
                itemsize,
            );
        }
    }
    // Copy back
    for i in 0..len {
        let dst_off = base_offset + (i as isize) * stride;
        unsafe {
            std::ptr::copy_nonoverlapping(
                temp.as_ptr().add(i * itemsize),
                ptr.offset(dst_off),
                itemsize,
            );
        }
    }
}

/// Compute argpartition: return indices that would partition the array.
fn argpartition_impl(arr: &RumpyArray, kth: usize, axis: usize) -> RumpyArray {
    let shape = arr.shape();
    let strides = arr.strides();
    let axis_len = shape[axis];
    let dtype = arr.dtype().clone();
    let ndim = arr.ndim();

    // Output shape is same as input
    let out_shape = shape.to_vec();
    let out_size: usize = out_shape.iter().product();

    // Create output array of indices (int64)
    let mut result = RumpyArray::zeros(out_shape.clone(), DType::int64());

    if out_size == 0 {
        return result;
    }

    let src_ptr = arr.data_ptr();
    let result_buffer = Arc::get_mut(result.buffer_mut()).expect("unique");
    let result_ptr = result_buffer.as_mut_ptr() as *mut i64;

    // For 1D contiguous arrays, use optimized typed comparison
    if ndim == 1 && arr.is_c_contiguous() {
        let mut indices: Vec<i64> = (0..axis_len as i64).collect();

        match dtype.kind() {
            DTypeKind::Float64 => {
                let slice = unsafe { std::slice::from_raw_parts(src_ptr as *const f64, axis_len) };
                indices.select_nth_unstable_by(kth, |&a, &b| {
                    let va = slice[a as usize];
                    let vb = slice[b as usize];
                    va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            DTypeKind::Float32 => {
                let slice = unsafe { std::slice::from_raw_parts(src_ptr as *const f32, axis_len) };
                indices.select_nth_unstable_by(kth, |&a, &b| {
                    let va = slice[a as usize];
                    let vb = slice[b as usize];
                    va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            DTypeKind::Int64 => {
                let slice = unsafe { std::slice::from_raw_parts(src_ptr as *const i64, axis_len) };
                indices.select_nth_unstable_by(kth, |&a, &b| {
                    slice[a as usize].cmp(&slice[b as usize])
                });
            }
            DTypeKind::Int32 => {
                let slice = unsafe { std::slice::from_raw_parts(src_ptr as *const i32, axis_len) };
                indices.select_nth_unstable_by(kth, |&a, &b| {
                    slice[a as usize].cmp(&slice[b as usize])
                });
            }
            DTypeKind::Int16 => {
                let slice = unsafe { std::slice::from_raw_parts(src_ptr as *const i16, axis_len) };
                indices.select_nth_unstable_by(kth, |&a, &b| {
                    slice[a as usize].cmp(&slice[b as usize])
                });
            }
            DTypeKind::Uint64 => {
                let slice = unsafe { std::slice::from_raw_parts(src_ptr as *const u64, axis_len) };
                indices.select_nth_unstable_by(kth, |&a, &b| {
                    slice[a as usize].cmp(&slice[b as usize])
                });
            }
            DTypeKind::Uint32 => {
                let slice = unsafe { std::slice::from_raw_parts(src_ptr as *const u32, axis_len) };
                indices.select_nth_unstable_by(kth, |&a, &b| {
                    slice[a as usize].cmp(&slice[b as usize])
                });
            }
            DTypeKind::Uint16 => {
                let slice = unsafe { std::slice::from_raw_parts(src_ptr as *const u16, axis_len) };
                indices.select_nth_unstable_by(kth, |&a, &b| {
                    slice[a as usize].cmp(&slice[b as usize])
                });
            }
            DTypeKind::Uint8 => {
                let slice = unsafe { std::slice::from_raw_parts(src_ptr as *const u8, axis_len) };
                indices.select_nth_unstable_by(kth, |&a, &b| {
                    slice[a as usize].cmp(&slice[b as usize])
                });
            }
            _ => {
                // Generic fallback using DTypeOps
                let ops = dtype.ops();
                indices.select_nth_unstable_by(kth, |&a, &b| {
                    let a_off = (a as isize) * strides[0];
                    let b_off = (b as isize) * strides[0];
                    unsafe { ops.compare_elements(src_ptr, a_off, src_ptr, b_off) }
                });
            }
        }

        // Copy indices to result
        unsafe {
            std::ptr::copy_nonoverlapping(indices.as_ptr(), result_ptr, axis_len);
        }
        return result;
    }

    // Multi-dimensional case: argpartition each lane along axis
    let axis_stride = strides[axis];
    let ops = dtype.ops();

    // Iterate over all positions except the axis dimension
    let mut lane_shape: Vec<usize> = shape[..axis].to_vec();
    lane_shape.extend_from_slice(&shape[axis + 1..]);
    if lane_shape.is_empty() {
        lane_shape = vec![1];
    }
    let lane_count: usize = lane_shape.iter().product();

    let result_strides = result.strides().to_vec();
    let result_axis_stride = result_strides[axis];

    let mut lane_indices = vec![0usize; lane_shape.len()];
    for _ in 0..lane_count {
        // Calculate base offset for this lane in source and result
        let mut src_base: isize = 0;
        let mut result_base: isize = 0;
        let mut idx_pos = 0;
        for (d, _) in shape.iter().enumerate() {
            if d == axis {
                continue;
            }
            src_base += (lane_indices[idx_pos] as isize) * strides[d];
            result_base += (lane_indices[idx_pos] as isize) * result_strides[d];
            idx_pos += 1;
        }

        // Create indices for this lane and argpartition
        let mut indices: Vec<i64> = (0..axis_len as i64).collect();
        indices.select_nth_unstable_by(kth, |&a, &b| {
            let a_off = src_base + (a as isize) * axis_stride;
            let b_off = src_base + (b as isize) * axis_stride;
            unsafe { ops.compare_elements(src_ptr, a_off, src_ptr, b_off) }
        });

        // Write indices to result
        for (i, &idx) in indices.iter().enumerate() {
            let result_off = result_base + (i as isize) * result_axis_stride;
            unsafe {
                *result_ptr.offset(result_off / 8) = idx;
            }
        }

        increment_indices(&mut lane_indices, &lane_shape);
    }

    result
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

/// Perform indirect sort using multiple keys.
/// Keys are sorted in reverse order: last key is primary, second-to-last is secondary, etc.
/// All keys must be 1-D arrays of the same length.
/// Returns indices that would sort the keys.
pub fn lexsort(keys: &[&RumpyArray]) -> Option<RumpyArray> {
    if keys.is_empty() {
        return None;
    }

    // All keys must be 1-D
    for key in keys {
        if key.ndim() != 1 {
            return None;
        }
    }

    // All keys must have the same length
    let n = keys[0].size();
    for key in &keys[1..] {
        if key.size() != n {
            return None;
        }
    }

    if n == 0 {
        return Some(RumpyArray::zeros(vec![0], DType::int64()));
    }

    // Collect indices and sort by key comparisons
    // Last key is primary (highest priority)
    let mut indices: Vec<usize> = (0..n).collect();

    // Store key metadata for comparison - use pointers, dtypes, strides
    struct KeyInfo {
        ptr: *const u8,
        dtype: DType,
        stride: isize,
        is_contiguous: bool,
    }

    let key_infos: Vec<KeyInfo> = keys
        .iter()
        .map(|key| KeyInfo {
            ptr: key.data_ptr(),
            dtype: key.dtype().clone(),
            stride: key.strides()[0],
            is_contiguous: key.is_c_contiguous(),
        })
        .collect();

    indices.sort_by(|&a, &b| {
        // Compare from last key (primary) to first key (least significant)
        for info in key_infos.iter().rev() {
            let ord = if info.is_contiguous {
                match info.dtype.kind() {
                    DTypeKind::Float64 => {
                        let slice = unsafe { std::slice::from_raw_parts(info.ptr as *const f64, n) };
                        slice[a].partial_cmp(&slice[b]).unwrap_or(std::cmp::Ordering::Equal)
                    }
                    DTypeKind::Float32 => {
                        let slice = unsafe { std::slice::from_raw_parts(info.ptr as *const f32, n) };
                        slice[a].partial_cmp(&slice[b]).unwrap_or(std::cmp::Ordering::Equal)
                    }
                    DTypeKind::Int64 => {
                        let slice = unsafe { std::slice::from_raw_parts(info.ptr as *const i64, n) };
                        slice[a].cmp(&slice[b])
                    }
                    DTypeKind::Int32 => {
                        let slice = unsafe { std::slice::from_raw_parts(info.ptr as *const i32, n) };
                        slice[a].cmp(&slice[b])
                    }
                    DTypeKind::Int16 => {
                        let slice = unsafe { std::slice::from_raw_parts(info.ptr as *const i16, n) };
                        slice[a].cmp(&slice[b])
                    }
                    DTypeKind::Uint64 => {
                        let slice = unsafe { std::slice::from_raw_parts(info.ptr as *const u64, n) };
                        slice[a].cmp(&slice[b])
                    }
                    DTypeKind::Uint32 => {
                        let slice = unsafe { std::slice::from_raw_parts(info.ptr as *const u32, n) };
                        slice[a].cmp(&slice[b])
                    }
                    DTypeKind::Uint16 => {
                        let slice = unsafe { std::slice::from_raw_parts(info.ptr as *const u16, n) };
                        slice[a].cmp(&slice[b])
                    }
                    DTypeKind::Uint8 => {
                        let slice = unsafe { std::slice::from_raw_parts(info.ptr as *const u8, n) };
                        slice[a].cmp(&slice[b])
                    }
                    _ => {
                        let ops = info.dtype.ops();
                        let itemsize = info.dtype.itemsize() as isize;
                        let a_off = (a as isize) * itemsize;
                        let b_off = (b as isize) * itemsize;
                        unsafe { ops.compare_elements(info.ptr, a_off, info.ptr, b_off) }
                    }
                }
            } else {
                // Non-contiguous: use DTypeOps with actual strides
                let ops = info.dtype.ops();
                let a_off = (a as isize) * info.stride;
                let b_off = (b as isize) * info.stride;
                unsafe { ops.compare_elements(info.ptr, a_off, info.ptr, b_off) }
            };

            if ord != std::cmp::Ordering::Equal {
                return ord;
            }
        }
        std::cmp::Ordering::Equal
    });

    // Create result array
    let mut result = RumpyArray::zeros(vec![n], DType::int64());
    let result_buffer = Arc::get_mut(result.buffer_mut()).expect("unique");
    let result_ptr = result_buffer.as_mut_ptr() as *mut i64;
    for (i, &idx) in indices.iter().enumerate() {
        unsafe { *result_ptr.add(i) = idx as i64; }
    }

    Some(result)
}
