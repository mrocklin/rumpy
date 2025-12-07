//! Reduction operations on RumpyArray (sum, mean, var, std, etc.)
//! Includes both standard and NaN-aware variants.

use crate::array::{increment_indices, DType, RumpyArray};
use crate::array::dtype::{ReduceOp, UnaryOp};
use crate::ops::{map_unary_op, reduce_all_f64, reduce_axis_op, BinaryOp};
use crate::ops::ufunc::variance_f64_contiguous;
use std::sync::Arc;

impl RumpyArray {
    // ========================================================================
    // Basic reductions
    // ========================================================================

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
}
