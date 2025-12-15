//! FFT operations using rustfft.

use crate::array::{DType, RumpyArray, increment_indices};
use rustfft::{FftPlanner, num_complex::Complex64};

/// Compute C-contiguous strides for a shape.
fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let ndim = shape.len();
    let mut strides = vec![1usize; ndim];
    for i in (0..ndim.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Convert array to complex buffer for FFT.
/// Uses fast path for contiguous float64/complex128 arrays.
fn to_complex_buffer(arr: &RumpyArray) -> Vec<Complex64> {
    let size = arr.size();
    let ptr = arr.data_ptr();
    let dtype = arr.dtype();

    // Fast path for contiguous float64
    if arr.is_c_contiguous() && dtype == DType::float64() {
        let f64_ptr = ptr as *const f64;
        let mut buffer = Vec::with_capacity(size);
        for i in 0..size {
            let re = unsafe { *f64_ptr.add(i) };
            buffer.push(Complex64::new(re, 0.0));
        }
        return buffer;
    }

    // Fast path for contiguous complex128
    if arr.is_c_contiguous() && dtype == DType::complex128() {
        let f64_ptr = ptr as *const f64;
        let mut buffer = Vec::with_capacity(size);
        for i in 0..size {
            let re = unsafe { *f64_ptr.add(2 * i) };
            let im = unsafe { *f64_ptr.add(2 * i + 1) };
            buffer.push(Complex64::new(re, im));
        }
        return buffer;
    }

    // Slow path for strided/other dtypes
    let ops = dtype.ops();
    let mut buffer = Vec::with_capacity(size);
    for offset in arr.iter_offsets() {
        let (re, im) = unsafe { ops.read_complex(ptr, offset).unwrap_or((0.0, 0.0)) };
        buffer.push(Complex64::new(re, im));
    }
    buffer
}

/// Convert complex buffer to RumpyArray with given shape.
fn from_complex_buffer(buffer: &[Complex64], shape: Vec<usize>) -> RumpyArray {
    let mut result = RumpyArray::zeros(shape, DType::complex128());
    let data = result.buffer_mut();
    let data = std::sync::Arc::get_mut(data).expect("buffer must be unique");
    let ptr = data.as_mut_ptr() as *mut f64;

    for (i, c) in buffer.iter().enumerate() {
        unsafe {
            *ptr.add(2 * i) = c.re;
            *ptr.add(2 * i + 1) = c.im;
        }
    }
    result
}

/// 1D FFT.
pub fn fft(arr: &RumpyArray) -> Option<RumpyArray> {
    if arr.ndim() != 1 {
        return None;
    }

    let n = arr.size();
    if n == 0 {
        return Some(RumpyArray::zeros(vec![0], DType::complex128()));
    }

    let mut buffer = to_complex_buffer(arr);
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);
    fft.process(&mut buffer);

    Some(from_complex_buffer(&buffer, vec![n]))
}

/// 1D inverse FFT.
pub fn ifft(arr: &RumpyArray) -> Option<RumpyArray> {
    if arr.ndim() != 1 {
        return None;
    }

    let n = arr.size();
    if n == 0 {
        return Some(RumpyArray::zeros(vec![0], DType::complex128()));
    }

    let mut buffer = to_complex_buffer(arr);
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_inverse(n);
    fft.process(&mut buffer);

    // Normalize by 1/n (numpy convention)
    let scale = 1.0 / n as f64;
    for c in buffer.iter_mut() {
        *c *= scale;
    }

    Some(from_complex_buffer(&buffer, vec![n]))
}

/// 2D FFT.
pub fn fft2(arr: &RumpyArray) -> Option<RumpyArray> {
    if arr.ndim() != 2 {
        return None;
    }
    fftn(arr, Some(&[0, 1]))
}

/// 2D inverse FFT.
pub fn ifft2(arr: &RumpyArray) -> Option<RumpyArray> {
    if arr.ndim() != 2 {
        return None;
    }
    ifftn(arr, Some(&[0, 1]))
}

/// Real FFT - returns only positive frequencies.
pub fn rfft(arr: &RumpyArray) -> Option<RumpyArray> {
    if arr.ndim() != 1 {
        return None;
    }

    let n = arr.size();
    if n == 0 {
        return Some(RumpyArray::zeros(vec![0], DType::complex128()));
    }

    let mut buffer = to_complex_buffer(arr);
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);
    fft.process(&mut buffer);

    let out_len = n / 2 + 1;
    Some(from_complex_buffer(&buffer[..out_len], vec![out_len]))
}

/// Inverse real FFT - returns real array.
pub fn irfft(arr: &RumpyArray) -> Option<RumpyArray> {
    if arr.ndim() != 1 {
        return None;
    }

    let n_freq = arr.size();
    if n_freq == 0 {
        return Some(RumpyArray::zeros(vec![0], DType::float64()));
    }

    let n = 2 * (n_freq - 1);
    let input = to_complex_buffer(arr);

    // Reconstruct full spectrum using Hermitian symmetry
    let mut buffer = vec![Complex64::new(0.0, 0.0); n];
    for (i, &c) in input.iter().enumerate() {
        buffer[i] = c;
    }
    for i in 1..n_freq - 1 {
        buffer[n - i] = input[i].conj();
    }

    let mut planner = FftPlanner::new();
    let ifft = planner.plan_fft_inverse(n);
    ifft.process(&mut buffer);

    // Normalize and extract real part
    let scale = 1.0 / n as f64;
    let mut result = RumpyArray::zeros(vec![n], DType::float64());
    let data = result.buffer_mut();
    let data = std::sync::Arc::get_mut(data).expect("buffer must be unique");
    let ptr = data.as_mut_ptr() as *mut f64;

    for (i, c) in buffer.iter().enumerate() {
        unsafe { *ptr.add(i) = c.re * scale; }
    }

    Some(result)
}

/// Shift array by given amounts along each axis.
fn shift_array(arr: &RumpyArray, shifts: &[usize]) -> RumpyArray {
    let shape = arr.shape();
    let ptr = arr.data_ptr();
    let dtype = arr.dtype();
    let ops = dtype.ops();

    let mut result = RumpyArray::zeros(shape.to_vec(), dtype.clone());
    let res_data = result.buffer_mut();
    let res_data = std::sync::Arc::get_mut(res_data).expect("buffer must be unique");
    let res_ptr = res_data.as_mut_ptr();

    let size = arr.size();
    let mut dst_indices = vec![0usize; shape.len()];
    for i in 0..size {
        // Compute source indices by shifting
        let src_indices: Vec<usize> = dst_indices.iter()
            .zip(shifts.iter())
            .zip(shape.iter())
            .map(|((&d, &shift), &dim)| (d + dim - shift) % dim)
            .collect();

        let src_offset = arr.byte_offset_for(&src_indices);
        unsafe { ops.copy_element(ptr, src_offset, res_ptr, i); }
        increment_indices(&mut dst_indices, shape);
    }
    result
}

/// Shift zero-frequency component to center.
pub fn fftshift(arr: &RumpyArray) -> Option<RumpyArray> {
    let ndim = arr.ndim();
    if ndim == 0 || ndim > 2 {
        return None;
    }
    let shifts: Vec<usize> = arr.shape().iter().map(|&n| n / 2).collect();
    Some(shift_array(arr, &shifts))
}

/// Inverse of fftshift.
pub fn ifftshift(arr: &RumpyArray) -> Option<RumpyArray> {
    let ndim = arr.ndim();
    if ndim == 0 || ndim > 2 {
        return None;
    }
    let shifts: Vec<usize> = arr.shape().iter().map(|&n| n.div_ceil(2)).collect();
    Some(shift_array(arr, &shifts))
}

/// Return FFT sample frequencies.
pub fn fftfreq(n: usize, d: f64) -> RumpyArray {
    let mut result = RumpyArray::zeros(vec![n], DType::float64());
    let data = result.buffer_mut();
    let data = std::sync::Arc::get_mut(data).expect("buffer must be unique");
    let ptr = data.as_mut_ptr() as *mut f64;

    let val = 1.0 / (n as f64 * d);
    let mid = n.div_ceil(2);

    for i in 0..mid {
        unsafe { *ptr.add(i) = i as f64 * val; }
    }
    for i in mid..n {
        unsafe { *ptr.add(i) = (i as isize - n as isize) as f64 * val; }
    }

    result
}

/// Return FFT sample frequencies for rfft.
pub fn rfftfreq(n: usize, d: f64) -> RumpyArray {
    let out_len = n / 2 + 1;
    let mut result = RumpyArray::zeros(vec![out_len], DType::float64());
    let data = result.buffer_mut();
    let data = std::sync::Arc::get_mut(data).expect("buffer must be unique");
    let ptr = data.as_mut_ptr() as *mut f64;

    let val = 1.0 / (n as f64 * d);
    for i in 0..out_len {
        unsafe { *ptr.add(i) = i as f64 * val; }
    }

    result
}

/// N-dimensional FFT.
/// Applies FFT along each axis in sequence.
pub fn fftn(arr: &RumpyArray, axes: Option<&[usize]>) -> Option<RumpyArray> {
    let shape = arr.shape();
    let ndim = arr.ndim();

    if ndim == 0 {
        return None;
    }

    let axes_to_use = validate_axes(axes, ndim)?;
    let mut buffer = to_complex_buffer(arr);
    let mut planner = FftPlanner::new();

    for &axis in &axes_to_use {
        let axis_len = shape[axis];
        if axis_len > 0 {
            let fft_plan = planner.plan_fft_forward(axis_len);
            fft_along_axis(&mut buffer, shape, axis, &fft_plan);
        }
    }

    Some(from_complex_buffer(&buffer, shape.to_vec()))
}

/// N-dimensional inverse FFT.
pub fn ifftn(arr: &RumpyArray, axes: Option<&[usize]>) -> Option<RumpyArray> {
    let shape = arr.shape();
    let ndim = arr.ndim();

    if ndim == 0 {
        return None;
    }

    let axes_to_use = validate_axes(axes, ndim)?;
    let mut buffer = to_complex_buffer(arr);
    let mut planner = FftPlanner::new();

    for &axis in &axes_to_use {
        let axis_len = shape[axis];
        if axis_len > 0 {
            let fft_plan = planner.plan_fft_inverse(axis_len);
            fft_along_axis(&mut buffer, shape, axis, &fft_plan);
        }
    }

    // Normalize by total size of transformed axes
    let total_size: usize = axes_to_use.iter().map(|&ax| shape[ax]).product();
    if total_size > 0 {
        let scale = 1.0 / total_size as f64;
        for c in buffer.iter_mut() {
            *c *= scale;
        }
    }

    Some(from_complex_buffer(&buffer, shape.to_vec()))
}

/// Helper to apply FFT along a specific axis.
fn fft_along_axis(
    buffer: &mut [Complex64],
    shape: &[usize],
    axis: usize,
    fft_plan: &std::sync::Arc<dyn rustfft::Fft<f64>>,
) {
    let axis_len = shape[axis];
    let strides = compute_strides(shape);

    // Number of 1D transforms = product of all dimensions except axis
    let n_transforms: usize = shape.iter().enumerate()
        .filter(|&(i, _)| i != axis)
        .map(|(_, &s)| s)
        .product();

    if n_transforms == 0 || axis_len == 0 {
        return;
    }

    let mut work_buffer = vec![Complex64::new(0.0, 0.0); axis_len];

    let other_shape: Vec<usize> = shape.iter().enumerate()
        .filter(|&(i, _)| i != axis)
        .map(|(_, &s)| s)
        .collect();

    let other_strides: Vec<usize> = strides.iter().enumerate()
        .filter(|&(i, _)| i != axis)
        .map(|(_, &s)| s)
        .collect();

    let axis_stride = strides[axis];

    for transform_idx in 0..n_transforms {
        let base_idx = index_from_transform_idx(transform_idx, &other_shape, &other_strides);

        // Extract, transform, write back
        for i in 0..axis_len {
            work_buffer[i] = buffer[base_idx + i * axis_stride];
        }
        fft_plan.process(&mut work_buffer);
        for i in 0..axis_len {
            buffer[base_idx + i * axis_stride] = work_buffer[i];
        }
    }
}

/// Compute linear index from transform index and shape/stride info.
fn index_from_transform_idx(transform_idx: usize, shape: &[usize], strides: &[usize]) -> usize {
    let mut idx = 0;
    let mut remaining = transform_idx;
    for (&dim, &stride) in shape.iter().zip(strides.iter()).rev() {
        if dim > 0 {
            idx += (remaining % dim) * stride;
            remaining /= dim;
        }
    }
    idx
}

/// Real N-dimensional FFT.
/// Returns complex array with shape where last axis is n//2 + 1.
///
/// Strategy (matches NumPy):
/// 1. Do rfft on the last axis first (output is truncated)
/// 2. Then do fft on remaining axes (in reverse order)
pub fn rfftn(arr: &RumpyArray, axes: Option<&[usize]>) -> Option<RumpyArray> {
    let shape = arr.shape();
    let ndim = arr.ndim();

    if ndim == 0 {
        return None;
    }

    let axes_to_use = validate_axes(axes, ndim)?;
    let last_axis = *axes_to_use.last().unwrap();
    let last_axis_len = shape[last_axis];
    let rfft_out_len = last_axis_len / 2 + 1;

    // Step 1: Do rfft (truncated FFT) on the last axis
    let buffer = to_complex_buffer(arr);
    let mut planner = FftPlanner::new();

    let mut current_shape = shape.to_vec();
    current_shape[last_axis] = rfft_out_len;
    let mut current_buffer = rfft_along_axis(
        &buffer, shape, &current_shape, last_axis, &mut planner
    );

    // Step 2: Apply FFT along remaining axes (in reverse order, like NumPy)
    for &axis in axes_to_use[..axes_to_use.len() - 1].iter().rev() {
        let axis_len = current_shape[axis];
        if axis_len > 0 {
            let fft_plan = planner.plan_fft_forward(axis_len);
            fft_along_axis(&mut current_buffer, &current_shape, axis, &fft_plan);
        }
    }

    Some(from_complex_buffer(&current_buffer, current_shape))
}

/// Validate and normalize axes parameter.
fn validate_axes(axes: Option<&[usize]>, ndim: usize) -> Option<Vec<usize>> {
    let axes_to_use: Vec<usize> = match axes {
        Some(ax) => ax.to_vec(),
        None => (0..ndim).collect(),
    };
    if axes_to_use.is_empty() || axes_to_use.iter().any(|&a| a >= ndim) {
        return None;
    }
    Some(axes_to_use)
}

/// Apply FFT along one axis and truncate output (for rfft).
fn rfft_along_axis(
    buffer: &[Complex64],
    in_shape: &[usize],
    out_shape: &[usize],
    axis: usize,
    planner: &mut FftPlanner<f64>,
) -> Vec<Complex64> {
    let in_strides = compute_strides(in_shape);
    let out_strides = compute_strides(out_shape);
    let axis_in_len = in_shape[axis];
    let axis_out_len = out_shape[axis];

    let n_other: usize = in_shape.iter().enumerate()
        .filter(|&(i, _)| i != axis)
        .map(|(_, &s)| s)
        .product();

    let other_shape: Vec<usize> = in_shape.iter().enumerate()
        .filter(|&(i, _)| i != axis)
        .map(|(_, &s)| s)
        .collect();
    let other_in_strides: Vec<usize> = in_strides.iter().enumerate()
        .filter(|&(i, _)| i != axis)
        .map(|(_, &s)| s)
        .collect();
    let other_out_strides: Vec<usize> = out_strides.iter().enumerate()
        .filter(|&(i, _)| i != axis)
        .map(|(_, &s)| s)
        .collect();

    let axis_in_stride = in_strides[axis];
    let axis_out_stride = out_strides[axis];

    let out_size: usize = out_shape.iter().product();
    let mut out_buffer = vec![Complex64::new(0.0, 0.0); out_size];

    let fft_plan = planner.plan_fft_forward(axis_in_len);
    let mut work_buffer = vec![Complex64::new(0.0, 0.0); axis_in_len];

    for other_idx in 0..n_other.max(1) {
        let in_base = index_from_transform_idx(other_idx, &other_shape, &other_in_strides);
        let out_base = index_from_transform_idx(other_idx, &other_shape, &other_out_strides);

        for i in 0..axis_in_len {
            work_buffer[i] = buffer[in_base + i * axis_in_stride];
        }
        fft_plan.process(&mut work_buffer);
        for i in 0..axis_out_len {
            out_buffer[out_base + i * axis_out_stride] = work_buffer[i];
        }
    }

    out_buffer
}

/// Inverse real N-dimensional FFT.
/// Takes complex input with truncated last axis, returns real output.
///
/// Strategy:
/// 1. First do inverse FFT on all axes except the last (real) axis
/// 2. Then for each slice along non-last axes, do irfft (Hermitian reconstruction + ifft)
pub fn irfftn(arr: &RumpyArray, shape_hint: Option<&[usize]>, axes: Option<&[usize]>) -> Option<RumpyArray> {
    let in_shape = arr.shape();
    let ndim = arr.ndim();

    if ndim == 0 {
        return None;
    }

    let axes_to_use = validate_axes(axes, ndim)?;
    let last_axis = *axes_to_use.last().unwrap();
    let n_freq = in_shape[last_axis];

    // Determine output size on last axis
    let out_last_len = match shape_hint {
        Some(s) if last_axis < s.len() => s[last_axis],
        _ => 2 * (n_freq - 1),  // Default: even length
    };

    // Convert to complex buffer
    let mut buffer = to_complex_buffer(arr);
    let mut planner = FftPlanner::new();

    // Step 1: Apply inverse FFT along all axes except the last one
    for &axis in &axes_to_use[..axes_to_use.len() - 1] {
        let axis_len = in_shape[axis];
        if axis_len > 0 {
            let fft_plan = planner.plan_fft_inverse(axis_len);
            fft_along_axis(&mut buffer, in_shape, axis, &fft_plan);
        }
    }

    // Step 2: For the last axis, expand using Hermitian symmetry and apply ifft
    let mut out_shape = in_shape.to_vec();
    out_shape[last_axis] = out_last_len;

    let in_strides = compute_strides(in_shape);
    let out_strides = compute_strides(&out_shape);

    let n_other: usize = in_shape.iter().enumerate()
        .filter(|&(i, _)| i != last_axis)
        .map(|(_, &s)| s)
        .product();

    let other_shape: Vec<usize> = in_shape.iter().enumerate()
        .filter(|&(i, _)| i != last_axis)
        .map(|(_, &s)| s)
        .collect();
    let other_in_strides: Vec<usize> = in_strides.iter().enumerate()
        .filter(|&(i, _)| i != last_axis)
        .map(|(_, &s)| s)
        .collect();
    let other_out_strides: Vec<usize> = out_strides.iter().enumerate()
        .filter(|&(i, _)| i != last_axis)
        .map(|(_, &s)| s)
        .collect();

    let last_in_stride = in_strides[last_axis];
    let last_out_stride = out_strides[last_axis];

    let out_size: usize = out_shape.iter().product();
    let mut out_buffer = vec![Complex64::new(0.0, 0.0); out_size];

    let ifft_plan = planner.plan_fft_inverse(out_last_len);
    let mut work_buffer = vec![Complex64::new(0.0, 0.0); out_last_len];

    for other_idx in 0..n_other.max(1) {
        let in_base = index_from_transform_idx(other_idx, &other_shape, &other_in_strides);
        let out_base = index_from_transform_idx(other_idx, &other_shape, &other_out_strides);

        // Fill work buffer: positive frequencies from input
        for i in 0..n_freq {
            work_buffer[i] = buffer[in_base + i * last_in_stride];
        }
        // Clear rest first
        for i in n_freq..out_last_len {
            work_buffer[i] = Complex64::new(0.0, 0.0);
        }
        // Fill negative frequencies using Hermitian symmetry
        for k in 1..(n_freq - 1) {
            if out_last_len > k {
                work_buffer[out_last_len - k] = buffer[in_base + k * last_in_stride].conj();
            }
        }

        // Apply inverse FFT
        ifft_plan.process(&mut work_buffer);

        // Copy to output
        for i in 0..out_last_len {
            out_buffer[out_base + i * last_out_stride] = work_buffer[i];
        }
    }

    // Normalize
    let total_size: usize = axes_to_use.iter()
        .map(|&ax| if ax == last_axis { out_last_len } else { in_shape[ax] })
        .product();
    let scale = 1.0 / total_size as f64;

    // Extract real part
    let mut result = RumpyArray::zeros(out_shape.clone(), DType::float64());
    let data = result.buffer_mut();
    let data = std::sync::Arc::get_mut(data).expect("buffer must be unique");
    let ptr = data.as_mut_ptr() as *mut f64;

    for (i, c) in out_buffer.iter().enumerate() {
        unsafe { *ptr.add(i) = c.re * scale; }
    }

    Some(result)
}

/// 2D real FFT.
pub fn rfft2(arr: &RumpyArray) -> Option<RumpyArray> {
    if arr.ndim() != 2 {
        return None;
    }
    rfftn(arr, Some(&[0, 1]))
}

/// Inverse 2D real FFT.
pub fn irfft2(arr: &RumpyArray, s: Option<&[usize]>) -> Option<RumpyArray> {
    if arr.ndim() != 2 {
        return None;
    }
    irfftn(arr, s, Some(&[0, 1]))
}

/// Hermitian FFT - FFT for signals with Hermitian symmetry.
/// Input is assumed to be Hermitian symmetric (only positive frequencies stored).
/// Output is real.
/// hfft(x, n) = n * irfft(conj(zero_pad(x, n//2+1)))
pub fn hfft(arr: &RumpyArray, n: Option<usize>) -> Option<RumpyArray> {
    if arr.ndim() != 1 {
        return None;
    }

    let n_freq = arr.size();
    if n_freq == 0 {
        return Some(RumpyArray::zeros(vec![0], DType::float64()));
    }

    // Output length
    let out_len = n.unwrap_or(2 * (n_freq - 1));
    if out_len == 0 {
        return Some(RumpyArray::zeros(vec![0], DType::float64()));
    }

    let input = to_complex_buffer(arr);

    // hfft(x, n) = irfft(conj(padded_x), n) where padded_x has n//2+1 elements
    // The n//2+1 elements represent the positive frequencies for output length n
    let padded_len = out_len / 2 + 1;

    // Create padded input (zero-padded or truncated)
    let mut padded = vec![Complex64::new(0.0, 0.0); padded_len];
    for (i, &c) in input.iter().enumerate().take(padded_len) {
        padded[i] = c.conj();
    }

    // Reconstruct full spectrum
    let mut buffer = vec![Complex64::new(0.0, 0.0); out_len];

    // Copy positive frequencies
    for (i, &c) in padded.iter().enumerate() {
        buffer[i] = c;
    }

    // Fill negative frequencies using Hermitian symmetry
    // buffer[out_len - k] = conj(padded[k]) = input[k] for k = 1 to padded_len-2
    for k in 1..(padded_len.min(out_len) - 1) {
        if out_len > k && k < input.len() {
            buffer[out_len - k] = input[k];
        } else if out_len > k {
            // Zero padding case
            buffer[out_len - k] = Complex64::new(0.0, 0.0);
        }
    }

    let mut planner = FftPlanner::new();
    let ifft = planner.plan_fft_inverse(out_len);
    ifft.process(&mut buffer);

    // Extract real part
    let mut result = RumpyArray::zeros(vec![out_len], DType::float64());
    let data = result.buffer_mut();
    let data = std::sync::Arc::get_mut(data).expect("buffer must be unique");
    let ptr = data.as_mut_ptr() as *mut f64;

    for (i, c) in buffer.iter().enumerate() {
        unsafe { *ptr.add(i) = c.re; }
    }

    Some(result)
}

/// Inverse Hermitian FFT.
/// Input is real, output is Hermitian symmetric (only positive frequencies).
/// ihfft(x) = conj(rfft(x)) / n
pub fn ihfft(arr: &RumpyArray) -> Option<RumpyArray> {
    if arr.ndim() != 1 {
        return None;
    }

    let n = arr.size();
    if n == 0 {
        return Some(RumpyArray::zeros(vec![0], DType::complex128()));
    }

    // Compute FFT
    let mut buffer = to_complex_buffer(arr);
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);
    fft.process(&mut buffer);

    // Conjugate and normalize
    let scale = 1.0 / n as f64;
    for c in buffer.iter_mut() {
        *c = Complex64::new(c.re * scale, -c.im * scale);
    }

    // Return only positive frequencies
    let out_len = n / 2 + 1;
    Some(from_complex_buffer(&buffer[..out_len], vec![out_len]))
}
