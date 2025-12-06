//! FFT operations using rustfft.

use crate::array::{DType, RumpyArray, increment_indices};
use rustfft::{FftPlanner, num_complex::Complex64};

/// Convert array to complex buffer for FFT.
fn to_complex_buffer(arr: &RumpyArray) -> Vec<Complex64> {
    let size = arr.size();
    let mut buffer = Vec::with_capacity(size);
    let ptr = arr.data_ptr();
    let dtype = arr.dtype();
    let ops = dtype.ops();

    let mut indices = vec![0usize; arr.ndim()];
    for _ in 0..size {
        let offset = arr.byte_offset_for(&indices);
        let (re, im) = unsafe { ops.read_complex(ptr, offset).unwrap_or((0.0, 0.0)) };
        buffer.push(Complex64::new(re, im));
        increment_indices(&mut indices, arr.shape());
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

    let shape = arr.shape();
    let (rows, cols) = (shape[0], shape[1]);

    if rows == 0 || cols == 0 {
        return Some(RumpyArray::zeros(shape.to_vec(), DType::complex128()));
    }

    let mut buffer = to_complex_buffer(arr);
    let mut planner = FftPlanner::new();

    // FFT along rows
    let row_fft = planner.plan_fft_forward(cols);
    for row in 0..rows {
        let start = row * cols;
        row_fft.process(&mut buffer[start..start + cols]);
    }

    // FFT along columns
    let col_fft = planner.plan_fft_forward(rows);
    let mut col_buffer = vec![Complex64::new(0.0, 0.0); rows];
    for col in 0..cols {
        for row in 0..rows {
            col_buffer[row] = buffer[row * cols + col];
        }
        col_fft.process(&mut col_buffer);
        for row in 0..rows {
            buffer[row * cols + col] = col_buffer[row];
        }
    }

    Some(from_complex_buffer(&buffer, vec![rows, cols]))
}

/// 2D inverse FFT.
pub fn ifft2(arr: &RumpyArray) -> Option<RumpyArray> {
    if arr.ndim() != 2 {
        return None;
    }

    let shape = arr.shape();
    let (rows, cols) = (shape[0], shape[1]);

    if rows == 0 || cols == 0 {
        return Some(RumpyArray::zeros(shape.to_vec(), DType::complex128()));
    }

    let mut buffer = to_complex_buffer(arr);
    let mut planner = FftPlanner::new();

    // IFFT along rows
    let row_ifft = planner.plan_fft_inverse(cols);
    for row in 0..rows {
        let start = row * cols;
        row_ifft.process(&mut buffer[start..start + cols]);
    }

    // IFFT along columns
    let col_ifft = planner.plan_fft_inverse(rows);
    let mut col_buffer = vec![Complex64::new(0.0, 0.0); rows];
    for col in 0..cols {
        for row in 0..rows {
            col_buffer[row] = buffer[row * cols + col];
        }
        col_ifft.process(&mut col_buffer);
        for row in 0..rows {
            buffer[row * cols + col] = col_buffer[row];
        }
    }

    // Normalize
    let scale = 1.0 / (rows * cols) as f64;
    for c in buffer.iter_mut() {
        *c *= scale;
    }

    Some(from_complex_buffer(&buffer, vec![rows, cols]))
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
