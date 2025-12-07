//! Statistical operations: median, histogram, cov, corrcoef, average, ptp.
//!
//! These are higher-level operations that build on the core ufunc machinery.

use crate::array::{increment_indices, DType, RumpyArray};
use crate::ops::{BinaryOp, matmul};
use std::sync::Arc;

// ============================================================================
// Module-level functions
// ============================================================================

/// Compute histogram of 1D array.
/// Returns (counts, bin_edges) tuple.
pub fn histogram(arr: &RumpyArray, bins: usize, range: Option<(f64, f64)>) -> (RumpyArray, RumpyArray) {
    let values = arr.to_vec();

    let (min_val, max_val) = range.unwrap_or_else(|| {
        if values.is_empty() {
            (0.0, 1.0)
        } else {
            let min_v = values.iter().cloned().fold(f64::INFINITY, f64::min);
            let max_v = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            (min_v, max_v)
        }
    });

    // Create bin edges
    let mut bin_edges = Vec::with_capacity(bins + 1);
    let bin_width = (max_val - min_val) / bins as f64;
    for i in 0..=bins {
        bin_edges.push(min_val + i as f64 * bin_width);
    }

    // Count values in each bin
    let mut counts = vec![0i64; bins];
    for &val in &values {
        if val >= min_val && val <= max_val {
            let mut bin_idx = ((val - min_val) / bin_width) as usize;
            // Handle edge case: value == max_val goes in last bin
            if bin_idx >= bins {
                bin_idx = bins - 1;
            }
            counts[bin_idx] += 1;
        }
    }

    let counts_data: Vec<f64> = counts.iter().map(|&c| c as f64).collect();
    let counts_arr = RumpyArray::from_vec(counts_data, DType::int64());
    let edges_arr = RumpyArray::from_vec(bin_edges, DType::float64());

    (counts_arr, edges_arr)
}

/// Compute covariance matrix.
/// x should be 2D with each row representing a variable and each column an observation.
/// Returns a 2D array where element [i,j] is cov(row_i, row_j).
pub fn cov(arr: &RumpyArray, ddof: usize) -> RumpyArray {
    // For 1D input, compute variance
    if arr.ndim() == 1 {
        let n = arr.size();
        if n <= ddof {
            return RumpyArray::from_vec(vec![f64::NAN], DType::float64());
        }
        let mean = arr.mean();
        // Use to_vec for fast access, then compute variance
        let values = arr.to_vec();
        let sum_sq: f64 = values.iter().map(|&v| (v - mean) * (v - mean)).sum();
        let var = sum_sq / (n - ddof) as f64;
        return RumpyArray::from_vec(vec![var], DType::float64());
    }

    // 2D case: each row is a variable
    let shape = arr.shape();
    let nvars = shape[0];  // number of variables (rows)
    let nobs = shape[1];   // number of observations (columns)

    if nobs <= ddof {
        // Return NaN matrix
        return RumpyArray::full(vec![nvars, nvars], f64::NAN, DType::float64());
    }

    // Compute means for each variable and center the data
    let means = arr.mean_axis(1);
    let means_broadcasted = means.reshape(vec![nvars, 1]).unwrap();
    let centered = arr.binary_op(&means_broadcasted, BinaryOp::Sub).expect("broadcast works");

    // Compute covariance: centered @ centered.T / (n - ddof)
    let centered_t = centered.transpose();
    let cov_unnorm = matmul::matmul(&centered, &centered_t).expect("matmul works");

    // Divide by (n - ddof)
    let divisor = RumpyArray::full(vec![1], (nobs - ddof) as f64, DType::float64());
    cov_unnorm.binary_op(&divisor, BinaryOp::Div).expect("broadcast works")
}

/// Compute Pearson correlation coefficient matrix.
/// Input should be 2D with each row representing a variable.
pub fn corrcoef(arr: &RumpyArray) -> RumpyArray {
    // For 1D input, return 1x1 matrix with 1.0 (perfect correlation with itself)
    if arr.ndim() == 1 {
        return RumpyArray::from_vec(vec![1.0], DType::float64()).reshape(vec![1, 1]).unwrap();
    }

    // Compute covariance matrix first
    let cov_matrix = cov(arr, 1);
    let shape = cov_matrix.shape();
    let nvars = shape[0];

    // Extract diagonal (standard deviations)
    let diag = cov_matrix.diagonal();

    // Compute correlation: corr[i,j] = cov[i,j] / (std[i] * std[j])
    let mut result = RumpyArray::zeros(vec![nvars, nvars], DType::float64());
    let buffer = result.buffer_mut();
    let result_buffer = Arc::get_mut(buffer).expect("buffer must be unique");
    let result_ptr = result_buffer.as_mut_ptr() as *mut f64;

    for i in 0..nvars {
        let std_i = diag.get_element(&[i]).sqrt();
        for j in 0..nvars {
            let std_j = diag.get_element(&[j]).sqrt();
            let cov_ij = cov_matrix.get_element(&[i, j]);
            let corr = if std_i == 0.0 || std_j == 0.0 {
                if i == j { 1.0 } else { f64::NAN }
            } else {
                cov_ij / (std_i * std_j)
            };
            unsafe { *result_ptr.add(i * nvars + j) = corr; }
        }
    }

    result
}

// ============================================================================
// RumpyArray method implementations
// ============================================================================

impl RumpyArray {
    /// Median of all elements (flattened).
    /// Uses select_nth_unstable for O(n) instead of O(n log n) sort.
    pub fn median(&self) -> f64 {
        let size = self.size();
        if size == 0 {
            return f64::NAN;
        }
        let mut values = self.to_vec();
        let mid = size / 2;

        // select_nth_unstable partitions so values[mid] is the correct element
        values.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        if size % 2 == 0 {
            // For even length, need max of left partition (which is unsorted)
            let left_max = values[..mid].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            (left_max + values[mid]) / 2.0
        } else {
            values[mid]
        }
    }

    /// Median along axis.
    pub fn median_axis(&self, axis: usize) -> RumpyArray {
        let shape = self.shape();
        let ndim = self.ndim();
        let axis_len = shape[axis];

        // Output shape: remove the axis dimension
        let mut out_shape: Vec<usize> = shape[..axis].to_vec();
        out_shape.extend_from_slice(&shape[axis + 1..]);
        if out_shape.is_empty() {
            out_shape = vec![1];
        }

        let out_size: usize = out_shape.iter().product();
        let mut result = RumpyArray::zeros(out_shape.clone(), DType::float64());

        if out_size == 0 || axis_len == 0 {
            return result;
        }

        let buffer = result.buffer_mut();
        let result_buffer = Arc::get_mut(buffer).expect("buffer must be unique");
        let result_ptr = result_buffer.as_mut_ptr() as *mut f64;

        let mut outer_indices = vec![0usize; out_shape.len()];
        for out_i in 0..out_size {
            // Collect values along axis
            let mut values = Vec::with_capacity(axis_len);
            for k in 0..axis_len {
                // Build input indices: insert k at position `axis`
                let mut in_indices = Vec::with_capacity(ndim);
                in_indices.extend_from_slice(&outer_indices[..axis]);
                in_indices.push(k);
                in_indices.extend_from_slice(&outer_indices[axis..]);
                values.push(self.get_element(&in_indices));
            }

            // Sort and compute median
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let median = if axis_len.is_multiple_of(2) {
                (values[axis_len / 2 - 1] + values[axis_len / 2]) / 2.0
            } else {
                values[axis_len / 2]
            };

            unsafe { *result_ptr.add(out_i) = median; }
            increment_indices(&mut outer_indices, &out_shape);
        }
        result
    }

    /// Peak-to-peak (max - min) of all elements.
    pub fn ptp(&self) -> f64 {
        if self.size() == 0 {
            return f64::NAN;
        }
        self.max() - self.min()
    }

    /// Peak-to-peak along axis.
    pub fn ptp_axis(&self, axis: usize) -> RumpyArray {
        let max_arr = self.max_axis(axis);
        let min_arr = self.min_axis(axis);
        max_arr.binary_op(&min_arr, BinaryOp::Sub).expect("shapes match")
    }

    /// Weighted average of all elements.
    /// If weights is None, computes simple mean.
    pub fn average(&self, weights: Option<&RumpyArray>) -> f64 {
        match weights {
            None => self.mean(),
            Some(w) => {
                // weighted_sum / sum_of_weights
                let product = self.binary_op(w, BinaryOp::Mul).expect("broadcast works");
                let weighted_sum = product.sum();
                let total_weight = w.sum();
                if total_weight == 0.0 {
                    f64::NAN
                } else {
                    weighted_sum / total_weight
                }
            }
        }
    }

    /// Weighted average along axis.
    pub fn average_axis(&self, axis: usize, weights: Option<&RumpyArray>) -> RumpyArray {
        match weights {
            None => self.mean_axis(axis),
            Some(w) => {
                // Broadcast weights and compute weighted mean
                let product = self.binary_op(w, BinaryOp::Mul).expect("broadcast works");
                let weighted_sum = product.sum_axis(axis);
                let total_weight = w.sum_axis(axis);
                weighted_sum.binary_op(&total_weight, BinaryOp::Div).expect("shapes match")
            }
        }
    }
}
