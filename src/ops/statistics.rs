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

/// Compute bin edges for histogram without computing the histogram itself.
/// This is useful when you want to use consistent bins across multiple histograms.
pub fn histogram_bin_edges(arr: &RumpyArray, bins: usize, range: Option<(f64, f64)>) -> RumpyArray {
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

    RumpyArray::from_vec(bin_edges, DType::float64())
}

/// Compute 2D histogram of two data samples.
/// Returns (histogram, x_edges, y_edges).
pub fn histogram2d(
    x: &RumpyArray,
    y: &RumpyArray,
    bins_x: usize,
    bins_y: usize,
    range: Option<[[f64; 2]; 2]>,
    density: bool,
) -> Result<(RumpyArray, RumpyArray, RumpyArray), &'static str> {
    let x_values = x.to_vec();
    let y_values = y.to_vec();

    if x_values.len() != y_values.len() {
        return Err("x and y must have the same length");
    }

    // Determine ranges
    let (x_min, x_max, y_min, y_max) = match range {
        Some(r) => (r[0][0], r[0][1], r[1][0], r[1][1]),
        None => {
            if x_values.is_empty() {
                (0.0, 1.0, 0.0, 1.0)
            } else {
                let x_min = x_values.iter().cloned().fold(f64::INFINITY, f64::min);
                let x_max = x_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let y_min = y_values.iter().cloned().fold(f64::INFINITY, f64::min);
                let y_max = y_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                (x_min, x_max, y_min, y_max)
            }
        }
    };

    // Create bin edges
    let x_bin_width = (x_max - x_min) / bins_x as f64;
    let y_bin_width = (y_max - y_min) / bins_y as f64;

    let mut x_edges = Vec::with_capacity(bins_x + 1);
    for i in 0..=bins_x {
        x_edges.push(x_min + i as f64 * x_bin_width);
    }

    let mut y_edges = Vec::with_capacity(bins_y + 1);
    for i in 0..=bins_y {
        y_edges.push(y_min + i as f64 * y_bin_width);
    }

    // Count values in each 2D bin
    let mut counts = vec![0.0f64; bins_x * bins_y];

    for i in 0..x_values.len() {
        let xv = x_values[i];
        let yv = y_values[i];

        if xv >= x_min && xv <= x_max && yv >= y_min && yv <= y_max {
            let mut x_bin = ((xv - x_min) / x_bin_width) as usize;
            let mut y_bin = ((yv - y_min) / y_bin_width) as usize;

            // Handle edge case: value == max goes in last bin
            if x_bin >= bins_x {
                x_bin = bins_x - 1;
            }
            if y_bin >= bins_y {
                y_bin = bins_y - 1;
            }

            counts[x_bin * bins_y + y_bin] += 1.0;
        }
    }

    // Apply density normalization if requested
    if density && !x_values.is_empty() {
        let bin_area = x_bin_width * y_bin_width;
        let total: f64 = counts.iter().sum();
        if total > 0.0 && bin_area > 0.0 {
            let norm = total * bin_area;
            for c in &mut counts {
                *c /= norm;
            }
        }
    }

    let h_arr = RumpyArray::from_vec(counts, DType::float64())
        .reshape(vec![bins_x, bins_y])
        .unwrap();
    let x_edges_arr = RumpyArray::from_vec(x_edges, DType::float64());
    let y_edges_arr = RumpyArray::from_vec(y_edges, DType::float64());

    Ok((h_arr, x_edges_arr, y_edges_arr))
}

/// Compute N-dimensional histogram.
/// sample is a 2D array with shape (N, D) where N is the number of samples and D is the number of dimensions.
/// Returns (histogram, list of edges per dimension).
pub fn histogramdd(
    sample: &RumpyArray,
    bins_per_dim: &[usize],
    range: Option<&[[f64; 2]]>,
    density: bool,
) -> Result<(RumpyArray, Vec<RumpyArray>), &'static str> {
    let shape = sample.shape();

    // Handle empty sample
    if shape[0] == 0 {
        // For empty sample, still create properly shaped output
        let ndim = if shape.len() == 2 { shape[1] } else { 1 };
        let bins: Vec<usize> = if bins_per_dim.len() == 1 {
            vec![bins_per_dim[0]; ndim]
        } else if bins_per_dim.len() == ndim {
            bins_per_dim.to_vec()
        } else {
            return Err("bins must have one element or match sample dimensions");
        };

        let h_arr = RumpyArray::zeros(bins.clone(), DType::float64());
        let edges: Vec<RumpyArray> = bins.iter().enumerate().map(|(d, &b)| {
            let (lo, hi) = if let Some(r) = range {
                (r[d][0], r[d][1])
            } else {
                (0.0, 1.0)
            };
            let width = (hi - lo) / b as f64;
            let edge_vals: Vec<f64> = (0..=b).map(|i| lo + i as f64 * width).collect();
            RumpyArray::from_vec(edge_vals, DType::float64())
        }).collect();

        return Ok((h_arr, edges));
    }

    // Determine number of dimensions from sample
    let ndim = if shape.len() == 2 { shape[1] } else { 1 };
    let nsamples = shape[0];

    // Expand bins if only one provided
    let bins: Vec<usize> = if bins_per_dim.len() == 1 {
        vec![bins_per_dim[0]; ndim]
    } else if bins_per_dim.len() == ndim {
        bins_per_dim.to_vec()
    } else {
        return Err("bins must have one element or match sample dimensions");
    };

    // Get data for each dimension
    let dim_data: Vec<Vec<f64>> = if ndim == 1 && shape.len() == 1 {
        // 1D case: single column
        vec![sample.to_vec()]
    } else {
        // Multi-D case: each column is a dimension
        (0..ndim)
            .map(|d| {
                (0..nsamples)
                    .map(|i| sample.get_element(&[i, d]))
                    .collect()
            })
            .collect()
    };

    // Compute ranges for each dimension
    let ranges: Vec<(f64, f64)> = (0..ndim)
        .map(|d| {
            if let Some(r) = range {
                (r[d][0], r[d][1])
            } else {
                let min_v = dim_data[d].iter().cloned().fold(f64::INFINITY, f64::min);
                let max_v = dim_data[d].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                (min_v, max_v)
            }
        })
        .collect();

    // Compute bin widths
    let bin_widths: Vec<f64> = (0..ndim)
        .map(|d| (ranges[d].1 - ranges[d].0) / bins[d] as f64)
        .collect();

    // Create edges for each dimension
    let edges: Vec<RumpyArray> = (0..ndim)
        .map(|d| {
            let edge_vals: Vec<f64> = (0..=bins[d])
                .map(|i| ranges[d].0 + i as f64 * bin_widths[d])
                .collect();
            RumpyArray::from_vec(edge_vals, DType::float64())
        })
        .collect();

    // Compute total number of bins and create histogram array
    let total_bins: usize = bins.iter().product();
    let mut counts = vec![0.0f64; total_bins];

    // Count samples in bins
    'sample: for i in 0..nsamples {
        // Find bin indices for this sample
        let mut bin_indices = Vec::with_capacity(ndim);
        for d in 0..ndim {
            let val = dim_data[d][i];
            let (lo, hi) = ranges[d];
            if val < lo || val > hi {
                continue 'sample;
            }
            let mut bin = ((val - lo) / bin_widths[d]) as usize;
            if bin >= bins[d] {
                bin = bins[d] - 1;
            }
            bin_indices.push(bin);
        }

        // Convert N-D bin indices to flat index (row-major order)
        let mut flat_idx = 0;
        let mut stride = 1;
        for d in (0..ndim).rev() {
            flat_idx += bin_indices[d] * stride;
            stride *= bins[d];
        }

        counts[flat_idx] += 1.0;
    }

    // Apply density normalization if requested
    if density && nsamples > 0 {
        let bin_volume: f64 = bin_widths.iter().product();
        let total: f64 = counts.iter().sum();
        if total > 0.0 && bin_volume > 0.0 {
            let norm = total * bin_volume;
            for c in &mut counts {
                *c /= norm;
            }
        }
    }

    let h_arr = RumpyArray::from_vec(counts, DType::float64())
        .reshape(bins)
        .unwrap();

    Ok((h_arr, edges))
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

        if size.is_multiple_of(2) {
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

    // ========================================================================
    // NaN-aware statistics
    // ========================================================================

    /// Median of all elements ignoring NaN values.
    pub fn nanmedian(&self) -> f64 {
        let size = self.size();
        if size == 0 {
            return f64::NAN;
        }
        // Collect non-NaN values
        let mut values: Vec<f64> = self.to_vec().into_iter().filter(|v| !v.is_nan()).collect();
        if values.is_empty() {
            return f64::NAN;
        }
        let n = values.len();
        let mid = n / 2;

        // Use select_nth_unstable for O(n) median
        values.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        if n % 2 == 0 {
            // For even length, need max of left partition
            let left_max = values[..mid].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            (left_max + values[mid]) / 2.0
        } else {
            values[mid]
        }
    }

    /// Median along axis ignoring NaN values.
    pub fn nanmedian_axis(&self, axis: usize) -> RumpyArray {
        let shape = self.shape();
        let axis_len = shape[axis];
        let axis_stride = self.strides()[axis];

        // Output shape: remove the axis dimension
        let mut out_shape: Vec<usize> = shape[..axis].to_vec();
        out_shape.extend_from_slice(&shape[axis + 1..]);
        if out_shape.is_empty() {
            out_shape = vec![1];
        }

        let out_size: usize = out_shape.iter().product();
        let mut result = RumpyArray::zeros(out_shape, DType::float64());

        if out_size == 0 || axis_len == 0 {
            return result;
        }

        let buffer = result.buffer_mut();
        let result_buffer = Arc::get_mut(buffer).expect("buffer must be unique");
        let result_ptr = result_buffer.as_mut_ptr() as *mut f64;

        let src_ptr = self.data_ptr();
        let dtype = self.dtype();
        let ops = dtype.ops();

        // Use axis_offsets for efficient strided access
        for (i, base_offset) in self.axis_offsets(axis).enumerate() {
            // Collect non-NaN values along axis using strided pointer
            let mut values = Vec::with_capacity(axis_len);
            let mut ptr = unsafe { src_ptr.offset(base_offset) };
            for _ in 0..axis_len {
                let v = unsafe { ops.read_f64(ptr, 0) }.unwrap_or(0.0);
                if !v.is_nan() {
                    values.push(v);
                }
                ptr = unsafe { ptr.offset(axis_stride) };
            }

            // Compute median of non-NaN values
            let median = if values.is_empty() {
                f64::NAN
            } else {
                values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let n = values.len();
                if n % 2 == 0 {
                    (values[n / 2 - 1] + values[n / 2]) / 2.0
                } else {
                    values[n / 2]
                }
            };

            unsafe { *result_ptr.add(i) = median; }
        }
        result
    }
}
