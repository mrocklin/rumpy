//! Numerical operations: gradient, trapezoid, interp, correlate.
//!
//! Uses typed inner loops for SIMD performance via macro-generated dispatch.

use crate::array::dtype::DTypeKind;
use crate::array::{DType, RumpyArray};
use std::sync::Arc;

// ============================================================================
// Typed loop generation macros
// ============================================================================

/// Generate typed inner loops for all numeric types.
/// Uses `as f64` for conversion since From<T> doesn't exist for i64/u64.
macro_rules! impl_typed_loops {
    ($gradient_name:ident, $gradient_coords_name:ident, $trapezoid_name:ident, $T:ty) => {
        #[inline]
        unsafe fn $gradient_name(
            src: *const $T, out: *mut f64, n: usize, h: f64, h2: f64,
            src_stride: isize, out_stride: isize,
        ) {
            let src_step = src_stride / std::mem::size_of::<$T>() as isize;
            let out_step = out_stride / 8;

            *out = (*src.offset(src_step) as f64 - *src as f64) / h;

            for i in 1..n - 1 {
                let v_prev = *src.offset((i as isize - 1) * src_step) as f64;
                let v_next = *src.offset((i as isize + 1) * src_step) as f64;
                *out.offset(i as isize * out_step) = (v_next - v_prev) / h2;
            }

            let v_last = *src.offset((n as isize - 1) * src_step) as f64;
            let v_prev = *src.offset((n as isize - 2) * src_step) as f64;
            *out.offset((n as isize - 1) * out_step) = (v_last - v_prev) / h;
        }

        #[inline]
        unsafe fn $gradient_coords_name(
            src: *const $T, out: *mut f64, x: &[f64], n: usize,
            src_stride: isize, out_stride: isize,
        ) {
            let src_step = src_stride / std::mem::size_of::<$T>() as isize;
            let out_step = out_stride / 8;

            *out = (*src.offset(src_step) as f64 - *src as f64) / (x[1] - x[0]);

            for i in 1..n - 1 {
                let v_prev = *src.offset((i as isize - 1) * src_step) as f64;
                let v_next = *src.offset((i as isize + 1) * src_step) as f64;
                *out.offset(i as isize * out_step) = (v_next - v_prev) / (x[i + 1] - x[i - 1]);
            }

            let v_last = *src.offset((n as isize - 1) * src_step) as f64;
            let v_prev = *src.offset((n as isize - 2) * src_step) as f64;
            *out.offset((n as isize - 1) * out_step) = (v_last - v_prev) / (x[n - 1] - x[n - 2]);
        }

        #[inline]
        unsafe fn $trapezoid_name(
            src: *const $T, n: usize, stride: isize, dx: f64, x_vals: Option<&[f64]>,
        ) -> f64 {
            let step = stride / std::mem::size_of::<$T>() as isize;
            let mut sum = 0.0;

            if let Some(x) = x_vals {
                for j in 0..n - 1 {
                    let y0 = *src.offset(j as isize * step) as f64;
                    let y1 = *src.offset((j + 1) as isize * step) as f64;
                    sum += (y0 + y1) * (x[j + 1] - x[j]) * 0.5;
                }
            } else {
                for j in 0..n - 1 {
                    let y0 = *src.offset(j as isize * step) as f64;
                    let y1 = *src.offset((j + 1) as isize * step) as f64;
                    sum += (y0 + y1) * dx * 0.5;
                }
            }
            sum
        }
    };
}

impl_typed_loops!(gradient_loop_f64, gradient_coords_loop_f64, trapezoid_loop_f64, f64);
impl_typed_loops!(gradient_loop_f32, gradient_coords_loop_f32, trapezoid_loop_f32, f32);
impl_typed_loops!(gradient_loop_i64, gradient_coords_loop_i64, trapezoid_loop_i64, i64);
impl_typed_loops!(gradient_loop_i32, gradient_coords_loop_i32, trapezoid_loop_i32, i32);
impl_typed_loops!(gradient_loop_i16, gradient_coords_loop_i16, trapezoid_loop_i16, i16);
impl_typed_loops!(gradient_loop_u64, gradient_coords_loop_u64, trapezoid_loop_u64, u64);
impl_typed_loops!(gradient_loop_u32, gradient_coords_loop_u32, trapezoid_loop_u32, u32);
impl_typed_loops!(gradient_loop_u16, gradient_coords_loop_u16, trapezoid_loop_u16, u16);
impl_typed_loops!(gradient_loop_u8, gradient_coords_loop_u8, trapezoid_loop_u8, u8);

// ============================================================================
// Gradient
// ============================================================================

/// Compute numerical gradient of an array.
pub fn gradient(f: &RumpyArray, spacing: Option<f64>, axis: Option<usize>) -> Option<Vec<RumpyArray>> {
    if f.size() == 0 || f.shape().iter().any(|&d| d < 2) {
        return None;
    }

    let sp = spacing.unwrap_or(1.0);
    match axis {
        Some(ax) if ax < f.ndim() => Some(vec![gradient_axis(f, sp, ax)]),
        Some(_) => None,
        None => Some((0..f.ndim()).map(|ax| gradient_axis(f, sp, ax)).collect()),
    }
}

/// Compute gradient along axis with non-uniform spacing.
pub fn gradient_with_coords(f: &RumpyArray, coords: &RumpyArray, axis: usize) -> Option<RumpyArray> {
    if axis >= f.ndim() || coords.ndim() != 1 || coords.size() != f.shape()[axis] {
        return None;
    }

    let shape = f.shape();
    let n = shape[axis];
    let mut result = RumpyArray::zeros(shape.to_vec(), DType::float64());
    if n <= 1 {
        return Some(result);
    }

    let x = read_to_f64_vec(coords);
    let buffer = Arc::get_mut(result.buffer_mut()).expect("unique");
    let result_ptr = buffer.as_mut_ptr() as *mut f64;
    let (src_stride, out_stride) = (f.strides()[axis], result.strides()[axis]);

    macro_rules! run {
        ($loop_fn:ident, $src_ptr:expr) => {
            for base in f.axis_offsets(axis) {
                unsafe { $loop_fn($src_ptr.byte_offset(base), result_ptr.byte_offset(base), &x, n, src_stride, out_stride); }
            }
        };
    }

    match f.dtype().kind() {
        DTypeKind::Float64 => run!(gradient_coords_loop_f64, f.data_ptr() as *const f64),
        DTypeKind::Float32 => run!(gradient_coords_loop_f32, f.data_ptr() as *const f32),
        DTypeKind::Int64 => run!(gradient_coords_loop_i64, f.data_ptr() as *const i64),
        DTypeKind::Int32 => run!(gradient_coords_loop_i32, f.data_ptr() as *const i32),
        DTypeKind::Int16 => run!(gradient_coords_loop_i16, f.data_ptr() as *const i16),
        DTypeKind::Uint64 => run!(gradient_coords_loop_u64, f.data_ptr() as *const u64),
        DTypeKind::Uint32 => run!(gradient_coords_loop_u32, f.data_ptr() as *const u32),
        DTypeKind::Uint16 => run!(gradient_coords_loop_u16, f.data_ptr() as *const u16),
        DTypeKind::Uint8 => run!(gradient_coords_loop_u8, f.data_ptr() as *const u8),
        _ => return gradient_fallback(f, Some(&x), None, axis),
    }

    Some(result)
}

fn gradient_axis(f: &RumpyArray, spacing: f64, axis: usize) -> RumpyArray {
    let shape = f.shape();
    let n = shape[axis];
    let mut result = RumpyArray::zeros(shape.to_vec(), DType::float64());
    if n <= 1 {
        return result;
    }

    let buffer = Arc::get_mut(result.buffer_mut()).expect("unique");
    let result_ptr = buffer.as_mut_ptr() as *mut f64;
    let (src_stride, out_stride) = (f.strides()[axis], result.strides()[axis]);
    let (h, h2) = (spacing, 2.0 * spacing);

    macro_rules! run {
        ($loop_fn:ident, $src_ptr:expr) => {
            for base in f.axis_offsets(axis) {
                unsafe { $loop_fn($src_ptr.byte_offset(base), result_ptr.byte_offset(base), n, h, h2, src_stride, out_stride); }
            }
        };
    }

    match f.dtype().kind() {
        DTypeKind::Float64 => run!(gradient_loop_f64, f.data_ptr() as *const f64),
        DTypeKind::Float32 => run!(gradient_loop_f32, f.data_ptr() as *const f32),
        DTypeKind::Int64 => run!(gradient_loop_i64, f.data_ptr() as *const i64),
        DTypeKind::Int32 => run!(gradient_loop_i32, f.data_ptr() as *const i32),
        DTypeKind::Int16 => run!(gradient_loop_i16, f.data_ptr() as *const i16),
        DTypeKind::Uint64 => run!(gradient_loop_u64, f.data_ptr() as *const u64),
        DTypeKind::Uint32 => run!(gradient_loop_u32, f.data_ptr() as *const u32),
        DTypeKind::Uint16 => run!(gradient_loop_u16, f.data_ptr() as *const u16),
        DTypeKind::Uint8 => run!(gradient_loop_u8, f.data_ptr() as *const u8),
        _ => return gradient_fallback(f, None, Some(spacing), axis).unwrap(),
    }

    result
}

fn gradient_fallback(f: &RumpyArray, x: Option<&[f64]>, spacing: Option<f64>, axis: usize) -> Option<RumpyArray> {
    let shape = f.shape();
    let n = shape[axis];
    let mut result = RumpyArray::zeros(shape.to_vec(), DType::float64());

    let buffer = Arc::get_mut(result.buffer_mut()).expect("unique");
    let result_ptr = buffer.as_mut_ptr() as *mut f64;
    let axis_stride = f.strides()[axis];
    let out_stride = result.strides()[axis];
    let f_dtype = f.dtype();
    let f_ops = f_dtype.ops();

    for base in f.axis_offsets(axis) {
        let vals = read_axis_values(f.data_ptr(), base, axis_stride, n, &*f_ops);
        let mut grad = vec![0.0; n];

        if let Some(x) = x {
            grad[0] = (vals[1] - vals[0]) / (x[1] - x[0]);
            for i in 1..n - 1 {
                grad[i] = (vals[i + 1] - vals[i - 1]) / (x[i + 1] - x[i - 1]);
            }
            grad[n - 1] = (vals[n - 1] - vals[n - 2]) / (x[n - 1] - x[n - 2]);
        } else {
            let h = spacing.unwrap_or(1.0);
            let h2 = 2.0 * h;
            grad[0] = (vals[1] - vals[0]) / h;
            for i in 1..n - 1 {
                grad[i] = (vals[i + 1] - vals[i - 1]) / h2;
            }
            grad[n - 1] = (vals[n - 1] - vals[n - 2]) / h;
        }

        for (i, &val) in grad.iter().enumerate() {
            let idx = (base + i as isize * out_stride) / 8;
            unsafe { *result_ptr.offset(idx) = val; }
        }
    }

    Some(result)
}

// ============================================================================
// Trapezoid
// ============================================================================

/// Trapezoidal integration along an axis.
pub fn trapezoid(y: &RumpyArray, x: Option<&RumpyArray>, dx: f64, axis: isize) -> Option<RumpyArray> {
    let ndim = y.ndim();
    if ndim == 0 {
        return Some(RumpyArray::full(vec![1], 0.0, DType::float64()));
    }

    let axis = if axis < 0 { (ndim as isize + axis) as usize } else { axis as usize };
    if axis >= ndim {
        return None;
    }

    let shape = y.shape();
    let n = shape[axis];
    let out_shape: Vec<usize> = shape.iter().enumerate()
        .filter(|&(i, _)| i != axis).map(|(_, &d)| d).collect();
    let out_shape = if out_shape.is_empty() { vec![1] } else { out_shape };

    if n <= 1 {
        return Some(RumpyArray::zeros(out_shape, DType::float64()));
    }

    let mut result = RumpyArray::zeros(out_shape, DType::float64());
    let buffer = Arc::get_mut(result.buffer_mut()).expect("unique");
    let result_ptr = buffer.as_mut_ptr() as *mut f64;

    let x_vals: Option<Vec<f64>> = x.and_then(|xa| {
        if xa.ndim() != 1 || xa.size() != n { return None; }
        Some(read_to_f64_vec(xa))
    });

    let axis_stride = y.strides()[axis];

    macro_rules! run {
        ($loop_fn:ident, $src_ptr:expr) => {
            for (i, base) in y.axis_offsets(axis).enumerate() {
                unsafe { *result_ptr.add(i) = $loop_fn($src_ptr.byte_offset(base), n, axis_stride, dx, x_vals.as_deref()); }
            }
        };
    }

    match y.dtype().kind() {
        DTypeKind::Float64 => run!(trapezoid_loop_f64, y.data_ptr() as *const f64),
        DTypeKind::Float32 => run!(trapezoid_loop_f32, y.data_ptr() as *const f32),
        DTypeKind::Int64 => run!(trapezoid_loop_i64, y.data_ptr() as *const i64),
        DTypeKind::Int32 => run!(trapezoid_loop_i32, y.data_ptr() as *const i32),
        DTypeKind::Int16 => run!(trapezoid_loop_i16, y.data_ptr() as *const i16),
        DTypeKind::Uint64 => run!(trapezoid_loop_u64, y.data_ptr() as *const u64),
        DTypeKind::Uint32 => run!(trapezoid_loop_u32, y.data_ptr() as *const u32),
        DTypeKind::Uint16 => run!(trapezoid_loop_u16, y.data_ptr() as *const u16),
        DTypeKind::Uint8 => run!(trapezoid_loop_u8, y.data_ptr() as *const u8),
        _ => return trapezoid_fallback(y, x_vals.as_deref(), dx, axis),
    }

    Some(result)
}

fn trapezoid_fallback(y: &RumpyArray, x_vals: Option<&[f64]>, dx: f64, axis: usize) -> Option<RumpyArray> {
    let shape = y.shape();
    let n = shape[axis];
    let out_shape: Vec<usize> = shape.iter().enumerate()
        .filter(|&(i, _)| i != axis).map(|(_, &d)| d).collect();
    let out_shape = if out_shape.is_empty() { vec![1] } else { out_shape };

    let mut result = RumpyArray::zeros(out_shape, DType::float64());
    let buffer = Arc::get_mut(result.buffer_mut()).expect("unique");
    let result_ptr = buffer.as_mut_ptr() as *mut f64;

    let y_dtype = y.dtype();
    let y_ops = y_dtype.ops();

    for (i, base) in y.axis_offsets(axis).enumerate() {
        let vals = read_axis_values(y.data_ptr(), base, y.strides()[axis], n, &*y_ops);
        let integral = if let Some(x) = x_vals {
            (0..n - 1).map(|j| (vals[j] + vals[j + 1]) * (x[j + 1] - x[j]) * 0.5).sum()
        } else {
            (0..n - 1).map(|j| (vals[j] + vals[j + 1]) * dx * 0.5).sum()
        };
        unsafe { *result_ptr.add(i) = integral; }
    }

    Some(result)
}

// ============================================================================
// Interp
// ============================================================================

/// 1D linear interpolation.
pub fn interp(
    x: &RumpyArray, xp: &RumpyArray, fp: &RumpyArray,
    left: Option<f64>, right: Option<f64>,
) -> Option<RumpyArray> {
    if xp.ndim() != 1 || fp.ndim() != 1 || xp.size() != fp.size() {
        return None;
    }

    let n = xp.size();
    if n == 0 {
        return Some(RumpyArray::zeros(x.shape().to_vec(), DType::float64()));
    }

    let xp_vals = read_to_f64_vec(xp);
    let fp_vals = read_to_f64_vec(fp);
    let (left_val, right_val) = (left.unwrap_or(fp_vals[0]), right.unwrap_or(fp_vals[n - 1]));

    let mut result = RumpyArray::zeros(x.shape().to_vec(), DType::float64());
    if x.size() == 0 {
        return Some(result);
    }

    let buffer = Arc::get_mut(result.buffer_mut()).expect("unique");
    let result_ptr = buffer.as_mut_ptr() as *mut f64;

    macro_rules! run {
        ($T:ty) => {{
            let x_ptr = x.data_ptr() as *const $T;
            for (i, offset) in x.iter_offsets().enumerate() {
                let xi = unsafe { *x_ptr.offset(offset / std::mem::size_of::<$T>() as isize) as f64 };
                unsafe { *result_ptr.add(i) = interp_single(xi, &xp_vals, &fp_vals, left_val, right_val); }
            }
        }};
    }

    match x.dtype().kind() {
        DTypeKind::Float64 => run!(f64),
        DTypeKind::Float32 => run!(f32),
        DTypeKind::Int64 => run!(i64),
        DTypeKind::Int32 => run!(i32),
        _ => {
            let x_dtype = x.dtype();
            let x_ops = x_dtype.ops();
            let x_ptr = x.data_ptr();
            for (i, offset) in x.iter_offsets().enumerate() {
                let xi = unsafe { x_ops.read_f64(x_ptr, offset) }.unwrap_or(0.0);
                unsafe { *result_ptr.add(i) = interp_single(xi, &xp_vals, &fp_vals, left_val, right_val); }
            }
        }
    }

    Some(result)
}

#[inline]
fn interp_single(xi: f64, xp: &[f64], fp: &[f64], left: f64, right: f64) -> f64 {
    let n = xp.len();
    if xi <= xp[0] { return left; }
    if xi >= xp[n - 1] { return right; }

    let (mut lo, mut hi) = (0, n - 1);
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if xp[mid] <= xi { lo = mid; } else { hi = mid; }
    }

    let t = (xi - xp[lo]) / (xp[hi] - xp[lo]);
    fp[lo] + t * (fp[hi] - fp[lo])
}

// ============================================================================
// Correlate
// ============================================================================

/// Cross-correlation of two 1D arrays.
pub fn correlate(a: &RumpyArray, v: &RumpyArray, mode: &str) -> Option<RumpyArray> {
    if a.ndim() != 1 || v.ndim() != 1 {
        return None;
    }

    let (n, m) = (a.size(), v.size());
    if n == 0 || m == 0 {
        let out_len = match mode {
            "full" => (n + m).saturating_sub(1),
            "same" => n,
            "valid" => n.saturating_sub(m.saturating_sub(1)),
            _ => return None,
        };
        return Some(RumpyArray::zeros(vec![out_len.max(0)], DType::float64()));
    }

    let a_vals = read_to_f64_vec(a);
    let mut v_vals = read_to_f64_vec(v);
    v_vals.reverse();  // correlate(a, v) = convolve(a, reverse(v))

    let full_result = if n * m > 500_000 {
        convolve_fft(&a_vals, &v_vals)
    } else {
        convolve_direct(&a_vals, &v_vals)
    };

    let (out_len, start) = match mode {
        "full" => (n + m - 1, 0),
        "same" => (n, (m - 1) / 2),
        "valid" => (n.saturating_sub(m - 1).max(1), m - 1),
        _ => return None,
    };

    let mut result = RumpyArray::zeros(vec![out_len], DType::float64());
    let buffer = Arc::get_mut(result.buffer_mut()).expect("unique");
    let result_ptr = buffer.as_mut_ptr() as *mut f64;

    for i in 0..out_len {
        unsafe { *result_ptr.add(i) = full_result[start + i]; }
    }

    Some(result)
}

fn convolve_direct(a: &[f64], v: &[f64]) -> Vec<f64> {
    let (n, m, out_len) = (a.len(), v.len(), a.len() + v.len() - 1);
    let mut result = vec![0.0; out_len];

    for k in 0..out_len {
        result[k] = (k.saturating_sub(n - 1)..m.min(k + 1))
            .map(|j| a[k - j] * v[j]).sum();
    }
    result
}

fn convolve_fft(a: &[f64], v: &[f64]) -> Vec<f64> {
    use rustfft::{FftPlanner, num_complex::Complex64};

    let out_len = a.len() + v.len() - 1;
    let fft_len = out_len.next_power_of_two();

    let mut a_c: Vec<_> = a.iter().map(|&x| Complex64::new(x, 0.0)).collect();
    a_c.resize(fft_len, Complex64::new(0.0, 0.0));
    let mut v_c: Vec<_> = v.iter().map(|&x| Complex64::new(x, 0.0)).collect();
    v_c.resize(fft_len, Complex64::new(0.0, 0.0));

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(fft_len);
    fft.process(&mut a_c);
    fft.process(&mut v_c);

    for i in 0..fft_len { a_c[i] *= v_c[i]; }

    planner.plan_fft_inverse(fft_len).process(&mut a_c);
    a_c[..out_len].iter().map(|c| c.re / fft_len as f64).collect()
}

// ============================================================================
// Utilities
// ============================================================================

fn read_to_f64_vec(arr: &RumpyArray) -> Vec<f64> {
    let ptr = arr.data_ptr();
    let dtype = arr.dtype();

    macro_rules! read {
        ($T:ty) => {{
            let typed = ptr as *const $T;
            arr.iter_offsets().map(|off| unsafe { *typed.offset(off / std::mem::size_of::<$T>() as isize) as f64 }).collect()
        }};
    }

    match dtype.kind() {
        DTypeKind::Float64 => read!(f64),
        DTypeKind::Float32 => read!(f32),
        DTypeKind::Int64 => read!(i64),
        DTypeKind::Int32 => read!(i32),
        DTypeKind::Int16 => read!(i16),
        DTypeKind::Uint64 => read!(u64),
        DTypeKind::Uint32 => read!(u32),
        DTypeKind::Uint16 => read!(u16),
        DTypeKind::Uint8 => read!(u8),
        _ => {
            let ops = dtype.ops();
            arr.iter_offsets().map(|off| unsafe { ops.read_f64(ptr, off) }.unwrap_or(0.0)).collect()
        }
    }
}

fn read_axis_values(ptr: *const u8, base: isize, stride: isize, n: usize, ops: &dyn crate::array::dtype::DTypeOps) -> Vec<f64> {
    let mut vals = Vec::with_capacity(n);
    let mut p = unsafe { ptr.offset(base) };
    for _ in 0..n {
        vals.push(unsafe { ops.read_f64(p, 0) }.unwrap_or(0.0));
        p = unsafe { p.offset(stride) };
    }
    vals
}
