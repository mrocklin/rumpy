//! Type-aware dispatch layer for ufunc operations.
//!
//! This module bridges the kernel/loop architecture with the RumpyArray type.
//! Layout selection happens here ONCE, not embedded in every kernel.
//!
//! The dispatch functions handle:
//! 1. DType resolution (which kernel impl to use)
//! 2. Layout detection (contiguous vs strided)
//! 3. Loop selection (call the appropriate loop strategy)

use crate::array::{DType, RumpyArray};
use crate::array::dtype::DTypeKind;
use crate::ops::kernels::{BinaryKernel, UnaryKernel, ReduceKernel, CompareKernel};
use crate::ops::kernels::bitwise::{And, Or, Xor, LeftShift, RightShift, Not};
use crate::ops::kernels::comparison::{Gt, Lt, Ge, Le, Eq, Ne};
use crate::ops::kernels::arithmetic::{
    Add, Sub, Mul, Div, Sum, Prod, Max, Min,
    Pow, Mod, FloorDiv, Maximum, Minimum,
    Arctan2, Hypot, FMax, FMin, Copysign, Logaddexp, Logaddexp2, Nextafter,
};
use crate::ops::kernels::math::{
    Neg, Abs, Sqrt, Exp, Log, Log10, Log2, Sin, Cos, Tan, Floor, Ceil, Square,
    Sinh, Cosh, Tanh, Arcsin, Arccos, Arctan, Sign, Positive, Reciprocal,
    Exp2, Expm1, Log1p, Cbrt, Trunc, Rint, Arcsinh, Arccosh, Arctanh,
};
use crate::ops::loops;
use std::sync::Arc;
use num_complex::Complex;
use half::f16;

/// Dispatch a binary operation using the kernel/loop architecture.
///
/// Returns Some(result) if the operation is handled, None if fallback needed.
pub fn dispatch_binary_add(
    a: &RumpyArray,
    b: &RumpyArray,
    out_shape: &[usize],
) -> Option<RumpyArray> {
    dispatch_binary_kernel(a, b, out_shape, Add)
}

pub fn dispatch_binary_sub(
    a: &RumpyArray,
    b: &RumpyArray,
    out_shape: &[usize],
) -> Option<RumpyArray> {
    dispatch_binary_kernel(a, b, out_shape, Sub)
}

pub fn dispatch_binary_mul(
    a: &RumpyArray,
    b: &RumpyArray,
    out_shape: &[usize],
) -> Option<RumpyArray> {
    dispatch_binary_kernel(a, b, out_shape, Mul)
}

pub fn dispatch_binary_div(
    a: &RumpyArray,
    b: &RumpyArray,
    out_shape: &[usize],
) -> Option<RumpyArray> {
    dispatch_binary_kernel(a, b, out_shape, Div)
}

pub fn dispatch_binary_pow(a: &RumpyArray, b: &RumpyArray, out_shape: &[usize]) -> Option<RumpyArray> {
    dispatch_binary_kernel_float(a, b, out_shape, Pow)
}

pub fn dispatch_binary_mod(a: &RumpyArray, b: &RumpyArray, out_shape: &[usize]) -> Option<RumpyArray> {
    dispatch_binary_kernel(a, b, out_shape, Mod)
}

pub fn dispatch_binary_floor_div(a: &RumpyArray, b: &RumpyArray, out_shape: &[usize]) -> Option<RumpyArray> {
    dispatch_binary_kernel(a, b, out_shape, FloorDiv)
}

pub fn dispatch_binary_maximum(a: &RumpyArray, b: &RumpyArray, out_shape: &[usize]) -> Option<RumpyArray> {
    dispatch_binary_kernel(a, b, out_shape, Maximum)
}

pub fn dispatch_binary_minimum(a: &RumpyArray, b: &RumpyArray, out_shape: &[usize]) -> Option<RumpyArray> {
    dispatch_binary_kernel(a, b, out_shape, Minimum)
}

pub fn dispatch_binary_arctan2(a: &RumpyArray, b: &RumpyArray, out_shape: &[usize]) -> Option<RumpyArray> {
    dispatch_binary_kernel_float(a, b, out_shape, Arctan2)
}

pub fn dispatch_binary_hypot(a: &RumpyArray, b: &RumpyArray, out_shape: &[usize]) -> Option<RumpyArray> {
    dispatch_binary_kernel_float(a, b, out_shape, Hypot)
}

pub fn dispatch_binary_fmax(a: &RumpyArray, b: &RumpyArray, out_shape: &[usize]) -> Option<RumpyArray> {
    dispatch_binary_kernel_float(a, b, out_shape, FMax)
}

pub fn dispatch_binary_fmin(a: &RumpyArray, b: &RumpyArray, out_shape: &[usize]) -> Option<RumpyArray> {
    dispatch_binary_kernel_float(a, b, out_shape, FMin)
}

pub fn dispatch_binary_copysign(a: &RumpyArray, b: &RumpyArray, out_shape: &[usize]) -> Option<RumpyArray> {
    dispatch_binary_kernel_float(a, b, out_shape, Copysign)
}

pub fn dispatch_binary_logaddexp(a: &RumpyArray, b: &RumpyArray, out_shape: &[usize]) -> Option<RumpyArray> {
    dispatch_binary_kernel_float(a, b, out_shape, Logaddexp)
}

pub fn dispatch_binary_logaddexp2(a: &RumpyArray, b: &RumpyArray, out_shape: &[usize]) -> Option<RumpyArray> {
    dispatch_binary_kernel_float(a, b, out_shape, Logaddexp2)
}

pub fn dispatch_binary_nextafter(a: &RumpyArray, b: &RumpyArray, out_shape: &[usize]) -> Option<RumpyArray> {
    dispatch_binary_kernel_float(a, b, out_shape, Nextafter)
}

/// Generic dispatch for binary kernels that support floats and complex.
fn dispatch_binary_kernel_float<K>(
    a: &RumpyArray,
    b: &RumpyArray,
    out_shape: &[usize],
    kernel: K,
) -> Option<RumpyArray>
where
    K: BinaryKernel<f64> + BinaryKernel<f32> + BinaryKernel<f16> + BinaryKernel<Complex<f64>> + BinaryKernel<Complex<f32>>,
{
    let a_kind = a.dtype().kind();
    let b_kind = b.dtype().kind();
    if a_kind != b_kind {
        return None;
    }
    match a_kind {
        DTypeKind::Float64 => dispatch_binary_typed::<f64, K>(a, b, out_shape, kernel, DType::float64()),
        DTypeKind::Float32 => dispatch_binary_typed::<f32, K>(a, b, out_shape, kernel, DType::float32()),
        DTypeKind::Float16 => dispatch_binary_typed::<f16, K>(a, b, out_shape, kernel, DType::float16()),
        DTypeKind::Complex128 => dispatch_binary_typed::<Complex<f64>, K>(a, b, out_shape, kernel, DType::complex128()),
        DTypeKind::Complex64 => dispatch_binary_typed::<Complex<f32>, K>(a, b, out_shape, kernel, DType::complex64()),
        _ => None,
    }
}

/// Generic dispatch for binary kernels that support common numeric types.
fn dispatch_binary_kernel<K>(
    a: &RumpyArray,
    b: &RumpyArray,
    out_shape: &[usize],
    kernel: K,
) -> Option<RumpyArray>
where
    K: BinaryKernel<f64>
        + BinaryKernel<f32>
        + BinaryKernel<f16>
        + BinaryKernel<i64>
        + BinaryKernel<i32>
        + BinaryKernel<i16>
        + BinaryKernel<u64>
        + BinaryKernel<u32>
        + BinaryKernel<u16>
        + BinaryKernel<u8>
        + BinaryKernel<Complex<f64>>
        + BinaryKernel<Complex<f32>>,
{
    let a_kind = a.dtype().kind();
    let b_kind = b.dtype().kind();

    // Only handle same-type operations for now
    if a_kind != b_kind {
        return None;
    }

    match a_kind {
        DTypeKind::Float64 => dispatch_binary_typed::<f64, K>(a, b, out_shape, kernel, DType::float64()),
        DTypeKind::Float32 => dispatch_binary_typed::<f32, K>(a, b, out_shape, kernel, DType::float32()),
        DTypeKind::Float16 => dispatch_binary_typed::<f16, K>(a, b, out_shape, kernel, DType::float16()),
        DTypeKind::Int64 => dispatch_binary_typed::<i64, K>(a, b, out_shape, kernel, DType::int64()),
        DTypeKind::Int32 => dispatch_binary_typed::<i32, K>(a, b, out_shape, kernel, DType::int32()),
        DTypeKind::Int16 => dispatch_binary_typed::<i16, K>(a, b, out_shape, kernel, DType::int16()),
        DTypeKind::Uint64 => dispatch_binary_typed::<u64, K>(a, b, out_shape, kernel, DType::uint64()),
        DTypeKind::Uint32 => dispatch_binary_typed::<u32, K>(a, b, out_shape, kernel, DType::uint32()),
        DTypeKind::Uint16 => dispatch_binary_typed::<u16, K>(a, b, out_shape, kernel, DType::uint16()),
        DTypeKind::Uint8 => dispatch_binary_typed::<u8, K>(a, b, out_shape, kernel, DType::uint8()),
        DTypeKind::Complex128 => dispatch_binary_typed::<Complex<f64>, K>(a, b, out_shape, kernel, DType::complex128()),
        DTypeKind::Complex64 => dispatch_binary_typed::<Complex<f32>, K>(a, b, out_shape, kernel, DType::complex64()),
        _ => None, // datetime, etc. fall back to trait dispatch
    }
}

/// Type-specific binary dispatch with layout detection.
fn dispatch_binary_typed<T: Copy, K: BinaryKernel<T>>(
    a: &RumpyArray,
    b: &RumpyArray,
    out_shape: &[usize],
    kernel: K,
    dtype: DType,
) -> Option<RumpyArray> {
    let size: usize = out_shape.iter().product();
    if size == 0 {
        return Some(RumpyArray::zeros(out_shape.to_vec(), dtype));
    }

    let mut result = RumpyArray::zeros(out_shape.to_vec(), dtype);
    let buffer = result.buffer_mut();
    let result_buffer = Arc::get_mut(buffer).expect("buffer must be unique");
    let result_ptr = result_buffer.as_mut_ptr() as *mut T;

    // Layout detection: both must be contiguous AND not broadcast
    let a_full_contig = a.is_c_contiguous() && a.shape() == out_shape;
    let b_full_contig = b.is_c_contiguous() && b.shape() == out_shape;

    if a_full_contig && b_full_contig {
        // Fast path: contiguous loop
        let a_slice = unsafe { std::slice::from_raw_parts(a.data_ptr() as *const T, size) };
        let b_slice = unsafe { std::slice::from_raw_parts(b.data_ptr() as *const T, size) };
        let out_slice = unsafe { std::slice::from_raw_parts_mut(result_ptr, size) };
        loops::map_binary(a_slice, b_slice, out_slice, kernel);
    } else {
        // Strided path
        let itemsize = std::mem::size_of::<T>() as isize;
        let ndim = out_shape.len();

        if ndim <= 1 {
            // 1D or scalar: simple strided loop
            let a_stride = if ndim == 0 { 0 } else { a.strides()[0] };
            let b_stride = if ndim == 0 { 0 } else { b.strides()[0] };
            unsafe {
                loops::map_binary_strided(
                    a.data_ptr() as *const T, a_stride,
                    b.data_ptr() as *const T, b_stride,
                    result_ptr, itemsize,
                    size, kernel,
                );
            }
        } else {
            // nD: iterate over outer dimensions, call strided loop per inner row
            let inner_size = out_shape[ndim - 1];
            let a_inner_stride = a.strides()[ndim - 1];
            let b_inner_stride = b.strides()[ndim - 1];
            let outer_shape = &out_shape[..ndim - 1];
            let outer_size: usize = outer_shape.iter().product();

            let a_strides = a.strides();
            let b_strides = b.strides();

            let mut outer_indices = vec![0usize; ndim - 1];
            for i in 0..outer_size {
                let a_offset: isize = outer_indices.iter().zip(a_strides).map(|(&idx, &s)| idx as isize * s).sum();
                let b_offset: isize = outer_indices.iter().zip(b_strides).map(|(&idx, &s)| idx as isize * s).sum();

                unsafe {
                    loops::map_binary_strided(
                        (a.data_ptr() as *const T).byte_offset(a_offset), a_inner_stride,
                        (b.data_ptr() as *const T).byte_offset(b_offset), b_inner_stride,
                        result_ptr.add(i * inner_size), itemsize,
                        inner_size, kernel,
                    );
                }
                crate::array::increment_indices(&mut outer_indices, outer_shape);
            }
        }
    }

    Some(result)
}

// ============================================================================
// Reduce dispatch (full-array)
// ============================================================================

/// Dispatch a reduce operation using the kernel/loop architecture.
/// Returns Some(result) as a scalar RumpyArray if handled, None if fallback needed.
pub fn dispatch_reduce_sum(arr: &RumpyArray) -> Option<RumpyArray> {
    dispatch_reduce_to_array(arr, Sum)
}

pub fn dispatch_reduce_prod(arr: &RumpyArray) -> Option<RumpyArray> {
    dispatch_reduce_to_array(arr, Prod)
}

pub fn dispatch_reduce_max(arr: &RumpyArray) -> Option<RumpyArray> {
    dispatch_reduce_to_array(arr, Max)
}

pub fn dispatch_reduce_min(arr: &RumpyArray) -> Option<RumpyArray> {
    dispatch_reduce_to_array(arr, Min)
}

/// Generic reduce dispatch returning a 0-d array.
fn dispatch_reduce_to_array<K>(arr: &RumpyArray, kernel: K) -> Option<RumpyArray>
where
    K: ReduceKernel<f64> + ReduceKernel<f32> + ReduceKernel<f16> + ReduceKernel<i64> + ReduceKernel<i32> + ReduceKernel<i16>
        + ReduceKernel<u64> + ReduceKernel<u32> + ReduceKernel<u16> + ReduceKernel<u8>
        + ReduceKernel<Complex<f64>> + ReduceKernel<Complex<f32>>,
{
    let kind = arr.dtype().kind();
    let size = arr.size();

    match kind {
        DTypeKind::Float64 => {
            let val = dispatch_reduce_typed::<f64, K>(arr, size, kernel)?;
            Some(RumpyArray::full(vec![1], val, DType::float64()))
        }
        DTypeKind::Float32 => {
            let val = dispatch_reduce_typed::<f32, K>(arr, size, kernel)?;
            Some(RumpyArray::full(vec![1], val as f64, DType::float32()))
        }
        DTypeKind::Float16 => {
            let val = dispatch_reduce_typed::<f16, K>(arr, size, kernel)?;
            Some(RumpyArray::full(vec![1], val.to_f64(), DType::float16()))
        }
        DTypeKind::Int64 => {
            let val = dispatch_reduce_typed::<i64, K>(arr, size, kernel)?;
            Some(RumpyArray::full(vec![1], val as f64, DType::int64()))
        }
        DTypeKind::Int32 => {
            let val = dispatch_reduce_typed::<i32, K>(arr, size, kernel)?;
            Some(RumpyArray::full(vec![1], val as f64, DType::int32()))
        }
        DTypeKind::Complex128 => {
            let val = dispatch_reduce_typed::<Complex<f64>, K>(arr, size, kernel)?;
            Some(RumpyArray::full_complex(vec![1], val.re, val.im, DType::complex128()))
        }
        DTypeKind::Complex64 => {
            let val = dispatch_reduce_typed::<Complex<f32>, K>(arr, size, kernel)?;
            Some(RumpyArray::full_complex(vec![1], val.re as f64, val.im as f64, DType::complex64()))
        }
        _ => None,
    }
}

/// Type-specific reduce returning native type.
fn dispatch_reduce_typed<T: Copy, K: ReduceKernel<T>>(
    arr: &RumpyArray,
    size: usize,
    kernel: K,
) -> Option<T> {
    if size == 0 {
        return Some(K::init());
    }

    if arr.is_c_contiguous() {
        let slice = unsafe { std::slice::from_raw_parts(arr.data_ptr() as *const T, size) };
        Some(loops::reduce(slice, kernel))
    } else {
        // For non-contiguous, use strided reduce
        let ndim = arr.ndim();
        if ndim <= 1 {
            let stride = if ndim == 0 { 0 } else { arr.strides()[0] };
            Some(unsafe { loops::reduce_strided(arr.data_ptr() as *const T, stride, size, kernel) })
        } else {
            None // Fall back for complex strided patterns
        }
    }
}

// ============================================================================
// Reduce dispatch (axis)
// ============================================================================

/// Dispatch axis reduction for Sum using kernel/loop architecture.
pub fn dispatch_reduce_axis_sum(arr: &RumpyArray, axis: usize) -> Option<RumpyArray> {
    dispatch_reduce_axis_kernel(arr, axis, Sum)
}

/// Dispatch axis reduction for Prod using kernel/loop architecture.
pub fn dispatch_reduce_axis_prod(arr: &RumpyArray, axis: usize) -> Option<RumpyArray> {
    dispatch_reduce_axis_kernel(arr, axis, Prod)
}

/// Dispatch axis reduction for Max using kernel/loop architecture.
pub fn dispatch_reduce_axis_max(arr: &RumpyArray, axis: usize) -> Option<RumpyArray> {
    dispatch_reduce_axis_kernel(arr, axis, Max)
}

/// Dispatch axis reduction for Min using kernel/loop architecture.
pub fn dispatch_reduce_axis_min(arr: &RumpyArray, axis: usize) -> Option<RumpyArray> {
    dispatch_reduce_axis_kernel(arr, axis, Min)
}

/// Generic axis reduce dispatch.
fn dispatch_reduce_axis_kernel<K>(arr: &RumpyArray, axis: usize, kernel: K) -> Option<RumpyArray>
where
    K: ReduceKernel<f64> + ReduceKernel<f32> + ReduceKernel<f16> + ReduceKernel<i64> + ReduceKernel<i32> + ReduceKernel<i16>
        + ReduceKernel<u64> + ReduceKernel<u32> + ReduceKernel<u16> + ReduceKernel<u8>
        + ReduceKernel<Complex<f64>> + ReduceKernel<Complex<f32>>,
{
    let kind = arr.dtype().kind();
    match kind {
        DTypeKind::Float64 => dispatch_reduce_axis_typed::<f64, K>(arr, axis, kernel, DType::float64()),
        DTypeKind::Float32 => dispatch_reduce_axis_typed::<f32, K>(arr, axis, kernel, DType::float32()),
        DTypeKind::Float16 => dispatch_reduce_axis_typed::<f16, K>(arr, axis, kernel, DType::float16()),
        DTypeKind::Int64 => dispatch_reduce_axis_typed::<i64, K>(arr, axis, kernel, DType::int64()),
        DTypeKind::Int32 => dispatch_reduce_axis_typed::<i32, K>(arr, axis, kernel, DType::int32()),
        DTypeKind::Int16 => dispatch_reduce_axis_typed::<i16, K>(arr, axis, kernel, DType::int16()),
        DTypeKind::Uint64 => dispatch_reduce_axis_typed::<u64, K>(arr, axis, kernel, DType::uint64()),
        DTypeKind::Uint32 => dispatch_reduce_axis_typed::<u32, K>(arr, axis, kernel, DType::uint32()),
        DTypeKind::Uint16 => dispatch_reduce_axis_typed::<u16, K>(arr, axis, kernel, DType::uint16()),
        DTypeKind::Uint8 => dispatch_reduce_axis_typed::<u8, K>(arr, axis, kernel, DType::uint8()),
        DTypeKind::Complex128 => dispatch_reduce_axis_typed::<Complex<f64>, K>(arr, axis, kernel, DType::complex128()),
        DTypeKind::Complex64 => dispatch_reduce_axis_typed::<Complex<f32>, K>(arr, axis, kernel, DType::complex64()),
        _ => None, // Bool, DateTime fall back to registry/trait
    }
}

/// Type-specific axis reduce dispatch.
///
/// For each output position, reduces along the specified axis using the kernel.
fn dispatch_reduce_axis_typed<T: Copy, K: ReduceKernel<T>>(
    arr: &RumpyArray,
    axis: usize,
    kernel: K,
    dtype: DType,
) -> Option<RumpyArray> {
    // Output shape: remove the reduction axis
    let mut out_shape: Vec<usize> = arr.shape().to_vec();
    let axis_len = out_shape.remove(axis);

    if out_shape.is_empty() {
        out_shape = vec![1]; // Scalar result wrapped in 1D array
    }

    let axis_stride = arr.strides()[axis];

    let mut result = RumpyArray::zeros(out_shape.clone(), dtype);
    let out_size = result.size();

    if out_size == 0 || axis_len == 0 {
        // Initialize with identity values for empty axis
        let buffer = result.buffer_mut();
        let result_buffer = Arc::get_mut(buffer).expect("buffer must be unique");
        let result_ptr = result_buffer.as_mut_ptr() as *mut T;
        for i in 0..out_size {
            unsafe { *result_ptr.add(i) = K::init(); }
        }
        return Some(result);
    }

    let buffer = result.buffer_mut();
    let result_buffer = Arc::get_mut(buffer).expect("buffer must be unique");
    let result_ptr = result_buffer.as_mut_ptr() as *mut T;

    // Check if reduction axis is contiguous (last axis for C-order)
    let itemsize = std::mem::size_of::<T>() as isize;
    if axis_stride == itemsize {
        // Contiguous reduction axis: use per-output-position reduce (cache-friendly)
        for (i, base_offset) in arr.axis_offsets(axis).enumerate() {
            let src_start = unsafe { (arr.data_ptr() as *const T).byte_offset(base_offset) };
            let slice = unsafe { std::slice::from_raw_parts(src_start, axis_len) };
            unsafe { *result_ptr.add(i) = loops::reduce(slice, kernel); }
        }
    } else if arr.is_c_contiguous() {
        // C-contiguous array: use row-major iteration for better cache behavior
        // Key insight: iterate through source memory sequentially, accumulate to output
        //
        // For shape [d0, d1, ..., d_axis, ..., d_{n-1}] reducing axis k:
        // - outer_shape = [d0, ..., d_{k-1}]       (axes before reduction)
        // - axis_len = d_k                         (reduction axis)
        // - inner_size = d_{k+1} * ... * d_{n-1}   (axes after reduction = contiguous block)
        //
        // Memory layout: outer_idx * (axis_len * inner_size) + axis_idx * inner_size + inner_idx
        // Output layout: outer_idx * inner_size + inner_idx

        let shape = arr.shape();
        let outer_shape = &shape[..axis];
        let inner_shape = &shape[axis + 1..];
        let outer_size: usize = outer_shape.iter().product::<usize>().max(1);
        let inner_size: usize = inner_shape.iter().product::<usize>().max(1);
        let src_ptr = arr.data_ptr() as *const T;

        // Initialize result with identity
        for i in 0..out_size {
            unsafe { *result_ptr.add(i) = K::init(); }
        }

        // Iterate in row-major order (sequential memory access through source)
        let mut src_idx = 0usize;
        for outer_idx in 0..outer_size {
            let out_base = outer_idx * inner_size;
            for _ in 0..axis_len {
                for inner_idx in 0..inner_size {
                    unsafe {
                        let v = *src_ptr.add(src_idx);
                        let acc = result_ptr.add(out_base + inner_idx);
                        *acc = K::combine(*acc, v);
                    }
                    src_idx += 1;
                }
            }
        }
    } else {
        // General strided case: use per-output-position strided reduce
        for (i, base_offset) in arr.axis_offsets(axis).enumerate() {
            let src_start = unsafe { (arr.data_ptr() as *const T).byte_offset(base_offset) };
            unsafe {
                *result_ptr.add(i) = loops::reduce_strided(src_start, axis_stride, axis_len, kernel);
            }
        }
    }

    Some(result)
}

// ============================================================================
// Unary dispatch
// ============================================================================

pub fn dispatch_unary_neg(arr: &RumpyArray) -> Option<RumpyArray> {
    dispatch_unary_kernel_int(arr, Neg)
}

pub fn dispatch_unary_abs(arr: &RumpyArray) -> Option<RumpyArray> {
    dispatch_unary_kernel_int(arr, Abs)
}

pub fn dispatch_unary_square(arr: &RumpyArray) -> Option<RumpyArray> {
    dispatch_unary_kernel_int(arr, Square)
}

pub fn dispatch_unary_sqrt(arr: &RumpyArray) -> Option<RumpyArray> {
    dispatch_unary_kernel_float(arr, Sqrt)
}

pub fn dispatch_unary_exp(arr: &RumpyArray) -> Option<RumpyArray> {
    dispatch_unary_kernel_float(arr, Exp)
}

pub fn dispatch_unary_log(arr: &RumpyArray) -> Option<RumpyArray> {
    dispatch_unary_kernel_float(arr, Log)
}

pub fn dispatch_unary_log10(arr: &RumpyArray) -> Option<RumpyArray> {
    dispatch_unary_kernel_float(arr, Log10)
}

pub fn dispatch_unary_log2(arr: &RumpyArray) -> Option<RumpyArray> {
    dispatch_unary_kernel_float(arr, Log2)
}

pub fn dispatch_unary_sin(arr: &RumpyArray) -> Option<RumpyArray> {
    dispatch_unary_kernel_float(arr, Sin)
}

pub fn dispatch_unary_cos(arr: &RumpyArray) -> Option<RumpyArray> {
    dispatch_unary_kernel_float(arr, Cos)
}

pub fn dispatch_unary_tan(arr: &RumpyArray) -> Option<RumpyArray> {
    dispatch_unary_kernel_float(arr, Tan)
}

pub fn dispatch_unary_floor(arr: &RumpyArray) -> Option<RumpyArray> {
    dispatch_unary_kernel_float(arr, Floor)
}

pub fn dispatch_unary_ceil(arr: &RumpyArray) -> Option<RumpyArray> {
    dispatch_unary_kernel_float(arr, Ceil)
}

pub fn dispatch_unary_sinh(arr: &RumpyArray) -> Option<RumpyArray> {
    dispatch_unary_kernel_float(arr, Sinh)
}

pub fn dispatch_unary_cosh(arr: &RumpyArray) -> Option<RumpyArray> {
    dispatch_unary_kernel_float(arr, Cosh)
}

pub fn dispatch_unary_tanh(arr: &RumpyArray) -> Option<RumpyArray> {
    dispatch_unary_kernel_float(arr, Tanh)
}

pub fn dispatch_unary_arcsin(arr: &RumpyArray) -> Option<RumpyArray> {
    dispatch_unary_kernel_float(arr, Arcsin)
}

pub fn dispatch_unary_arccos(arr: &RumpyArray) -> Option<RumpyArray> {
    dispatch_unary_kernel_float(arr, Arccos)
}

pub fn dispatch_unary_arctan(arr: &RumpyArray) -> Option<RumpyArray> {
    dispatch_unary_kernel_float(arr, Arctan)
}

pub fn dispatch_unary_sign(arr: &RumpyArray) -> Option<RumpyArray> {
    dispatch_unary_kernel_int(arr, Sign)
}

pub fn dispatch_unary_positive(arr: &RumpyArray) -> Option<RumpyArray> {
    dispatch_unary_kernel_int(arr, Positive)
}

pub fn dispatch_unary_reciprocal(arr: &RumpyArray) -> Option<RumpyArray> {
    dispatch_unary_kernel_float(arr, Reciprocal)
}

pub fn dispatch_unary_exp2(arr: &RumpyArray) -> Option<RumpyArray> {
    dispatch_unary_kernel_float(arr, Exp2)
}

pub fn dispatch_unary_expm1(arr: &RumpyArray) -> Option<RumpyArray> {
    dispatch_unary_kernel_float(arr, Expm1)
}

pub fn dispatch_unary_log1p(arr: &RumpyArray) -> Option<RumpyArray> {
    dispatch_unary_kernel_float(arr, Log1p)
}

pub fn dispatch_unary_cbrt(arr: &RumpyArray) -> Option<RumpyArray> {
    dispatch_unary_kernel_float(arr, Cbrt)
}

pub fn dispatch_unary_trunc(arr: &RumpyArray) -> Option<RumpyArray> {
    dispatch_unary_kernel_float(arr, Trunc)
}

pub fn dispatch_unary_rint(arr: &RumpyArray) -> Option<RumpyArray> {
    dispatch_unary_kernel_float(arr, Rint)
}

pub fn dispatch_unary_arcsinh(arr: &RumpyArray) -> Option<RumpyArray> {
    dispatch_unary_kernel_float(arr, Arcsinh)
}

pub fn dispatch_unary_arccosh(arr: &RumpyArray) -> Option<RumpyArray> {
    dispatch_unary_kernel_float(arr, Arccosh)
}

pub fn dispatch_unary_arctanh(arr: &RumpyArray) -> Option<RumpyArray> {
    dispatch_unary_kernel_float(arr, Arctanh)
}

/// Dispatch for kernels that support floats and complex.
fn dispatch_unary_kernel_float<K>(arr: &RumpyArray, kernel: K) -> Option<RumpyArray>
where
    K: UnaryKernel<f64> + UnaryKernel<f32> + UnaryKernel<f16> + UnaryKernel<Complex<f64>> + UnaryKernel<Complex<f32>>,
{
    match arr.dtype().kind() {
        DTypeKind::Float64 => dispatch_unary_typed::<f64, K>(arr, kernel, DType::float64()),
        DTypeKind::Float32 => dispatch_unary_typed::<f32, K>(arr, kernel, DType::float32()),
        DTypeKind::Float16 => dispatch_unary_typed::<f16, K>(arr, kernel, DType::float16()),
        DTypeKind::Complex128 => dispatch_unary_typed::<Complex<f64>, K>(arr, kernel, DType::complex128()),
        DTypeKind::Complex64 => dispatch_unary_typed::<Complex<f32>, K>(arr, kernel, DType::complex64()),
        _ => None,
    }
}

/// Dispatch for kernels that support floats, integers, and complex.
fn dispatch_unary_kernel_int<K>(arr: &RumpyArray, kernel: K) -> Option<RumpyArray>
where
    K: UnaryKernel<f64>
        + UnaryKernel<f32>
        + UnaryKernel<f16>
        + UnaryKernel<i64>
        + UnaryKernel<i32>
        + UnaryKernel<i16>
        + UnaryKernel<u64>
        + UnaryKernel<u32>
        + UnaryKernel<u16>
        + UnaryKernel<u8>
        + UnaryKernel<Complex<f64>>
        + UnaryKernel<Complex<f32>>,
{
    match arr.dtype().kind() {
        DTypeKind::Float64 => dispatch_unary_typed::<f64, K>(arr, kernel, DType::float64()),
        DTypeKind::Float32 => dispatch_unary_typed::<f32, K>(arr, kernel, DType::float32()),
        DTypeKind::Float16 => dispatch_unary_typed::<f16, K>(arr, kernel, DType::float16()),
        DTypeKind::Int64 => dispatch_unary_typed::<i64, K>(arr, kernel, DType::int64()),
        DTypeKind::Int32 => dispatch_unary_typed::<i32, K>(arr, kernel, DType::int32()),
        DTypeKind::Int16 => dispatch_unary_typed::<i16, K>(arr, kernel, DType::int16()),
        DTypeKind::Uint64 => dispatch_unary_typed::<u64, K>(arr, kernel, DType::uint64()),
        DTypeKind::Uint32 => dispatch_unary_typed::<u32, K>(arr, kernel, DType::uint32()),
        DTypeKind::Uint16 => dispatch_unary_typed::<u16, K>(arr, kernel, DType::uint16()),
        DTypeKind::Uint8 => dispatch_unary_typed::<u8, K>(arr, kernel, DType::uint8()),
        DTypeKind::Complex128 => dispatch_unary_typed::<Complex<f64>, K>(arr, kernel, DType::complex128()),
        DTypeKind::Complex64 => dispatch_unary_typed::<Complex<f32>, K>(arr, kernel, DType::complex64()),
        _ => None,
    }
}

/// Type-specific unary dispatch with layout detection.
fn dispatch_unary_typed<T: Copy, K: UnaryKernel<T>>(
    arr: &RumpyArray,
    kernel: K,
    dtype: DType,
) -> Option<RumpyArray> {
    let size = arr.size();
    if size == 0 {
        return Some(RumpyArray::zeros(arr.shape().to_vec(), dtype));
    }

    let mut result = RumpyArray::zeros(arr.shape().to_vec(), dtype);
    let buffer = result.buffer_mut();
    let result_buffer = Arc::get_mut(buffer).expect("buffer must be unique");
    let result_ptr = result_buffer.as_mut_ptr() as *mut T;

    if arr.is_c_contiguous() {
        let src_slice = unsafe { std::slice::from_raw_parts(arr.data_ptr() as *const T, size) };
        let out_slice = unsafe { std::slice::from_raw_parts_mut(result_ptr, size) };
        loops::map_unary(src_slice, out_slice, kernel);
    } else {
        let itemsize = std::mem::size_of::<T>() as isize;
        let ndim = arr.ndim();

        if ndim <= 1 {
            let stride = if ndim == 0 { 0 } else { arr.strides()[0] };
            unsafe {
                loops::map_unary_strided(
                    arr.data_ptr() as *const T, stride,
                    result_ptr, itemsize,
                    size, kernel,
                );
            }
        } else {
            // nD: iterate over outer dimensions
            let inner_size = arr.shape()[ndim - 1];
            let inner_stride = arr.strides()[ndim - 1];
            let outer_shape = &arr.shape()[..ndim - 1];
            let outer_size: usize = outer_shape.iter().product();
            let src_strides = arr.strides();

            let mut outer_indices = vec![0usize; ndim - 1];
            for i in 0..outer_size {
                let src_offset: isize = outer_indices.iter().zip(src_strides).map(|(&idx, &s)| idx as isize * s).sum();
                unsafe {
                    loops::map_unary_strided(
                        (arr.data_ptr() as *const T).byte_offset(src_offset), inner_stride,
                        result_ptr.add(i * inner_size), itemsize,
                        inner_size, kernel,
                    );
                }
                crate::array::increment_indices(&mut outer_indices, outer_shape);
            }
        }
    }

    Some(result)
}

// ============================================================================
// Comparison dispatch
// ============================================================================

/// Dispatch a greater-than comparison using the kernel/loop architecture.
pub fn dispatch_compare_gt(
    a: &RumpyArray,
    b: &RumpyArray,
    out_shape: &[usize],
) -> Option<RumpyArray> {
    dispatch_compare_kernel(a, b, out_shape, Gt)
}

pub fn dispatch_compare_lt(
    a: &RumpyArray,
    b: &RumpyArray,
    out_shape: &[usize],
) -> Option<RumpyArray> {
    dispatch_compare_kernel(a, b, out_shape, Lt)
}

pub fn dispatch_compare_ge(
    a: &RumpyArray,
    b: &RumpyArray,
    out_shape: &[usize],
) -> Option<RumpyArray> {
    dispatch_compare_kernel(a, b, out_shape, Ge)
}

pub fn dispatch_compare_le(
    a: &RumpyArray,
    b: &RumpyArray,
    out_shape: &[usize],
) -> Option<RumpyArray> {
    dispatch_compare_kernel(a, b, out_shape, Le)
}

pub fn dispatch_compare_eq(
    a: &RumpyArray,
    b: &RumpyArray,
    out_shape: &[usize],
) -> Option<RumpyArray> {
    dispatch_compare_kernel(a, b, out_shape, Eq)
}

pub fn dispatch_compare_ne(
    a: &RumpyArray,
    b: &RumpyArray,
    out_shape: &[usize],
) -> Option<RumpyArray> {
    dispatch_compare_kernel(a, b, out_shape, Ne)
}

/// Generic compare dispatch for all dtypes.
fn dispatch_compare_kernel<K>(
    a: &RumpyArray,
    b: &RumpyArray,
    out_shape: &[usize],
    kernel: K,
) -> Option<RumpyArray>
where
    K: CompareKernel<f64>
        + CompareKernel<f32>
        + CompareKernel<f16>
        + CompareKernel<i64>
        + CompareKernel<i32>
        + CompareKernel<i16>
        + CompareKernel<u64>
        + CompareKernel<u32>
        + CompareKernel<u16>
        + CompareKernel<u8>
        + CompareKernel<Complex<f64>>
        + CompareKernel<Complex<f32>>,
{
    let a_kind = a.dtype().kind();
    let b_kind = b.dtype().kind();

    // Only handle same-type operations for now
    if a_kind != b_kind {
        return None;
    }

    match a_kind {
        DTypeKind::Float64 => dispatch_compare_typed::<f64, K>(a, b, out_shape, kernel),
        DTypeKind::Float32 => dispatch_compare_typed::<f32, K>(a, b, out_shape, kernel),
        DTypeKind::Float16 => dispatch_compare_typed::<f16, K>(a, b, out_shape, kernel),
        DTypeKind::Int64 => dispatch_compare_typed::<i64, K>(a, b, out_shape, kernel),
        DTypeKind::Int32 => dispatch_compare_typed::<i32, K>(a, b, out_shape, kernel),
        DTypeKind::Int16 => dispatch_compare_typed::<i16, K>(a, b, out_shape, kernel),
        DTypeKind::Uint64 => dispatch_compare_typed::<u64, K>(a, b, out_shape, kernel),
        DTypeKind::Uint32 => dispatch_compare_typed::<u32, K>(a, b, out_shape, kernel),
        DTypeKind::Uint16 => dispatch_compare_typed::<u16, K>(a, b, out_shape, kernel),
        DTypeKind::Uint8 => dispatch_compare_typed::<u8, K>(a, b, out_shape, kernel),
        DTypeKind::Complex128 => dispatch_compare_typed::<Complex<f64>, K>(a, b, out_shape, kernel),
        DTypeKind::Complex64 => dispatch_compare_typed::<Complex<f32>, K>(a, b, out_shape, kernel),
        _ => None, // datetime, bool fall back to trait dispatch
    }
}

/// Type-specific compare dispatch with layout detection.
fn dispatch_compare_typed<T: Copy, K: CompareKernel<T>>(
    a: &RumpyArray,
    b: &RumpyArray,
    out_shape: &[usize],
    kernel: K,
) -> Option<RumpyArray> {
    let size: usize = out_shape.iter().product();
    if size == 0 {
        return Some(RumpyArray::zeros(out_shape.to_vec(), DType::bool()));
    }

    let mut result = RumpyArray::zeros(out_shape.to_vec(), DType::bool());
    let buffer = result.buffer_mut();
    let result_buffer = Arc::get_mut(buffer).expect("buffer must be unique");
    let result_ptr = result_buffer.as_mut_ptr();

    // Layout detection: both must be contiguous AND not broadcast
    let a_full_contig = a.is_c_contiguous() && a.shape() == out_shape;
    let b_full_contig = b.is_c_contiguous() && b.shape() == out_shape;

    if a_full_contig && b_full_contig {
        // Fast path: contiguous loop
        let a_slice = unsafe { std::slice::from_raw_parts(a.data_ptr() as *const T, size) };
        let b_slice = unsafe { std::slice::from_raw_parts(b.data_ptr() as *const T, size) };
        let out_slice = unsafe { std::slice::from_raw_parts_mut(result_ptr, size) };
        loops::map_compare(a_slice, b_slice, out_slice, kernel);
    } else {
        // Strided path
        let ndim = out_shape.len();

        if ndim <= 1 {
            // 1D or scalar: simple strided loop
            let a_stride = if ndim == 0 { 0 } else { a.strides()[0] };
            let b_stride = if ndim == 0 { 0 } else { b.strides()[0] };
            unsafe {
                loops::map_compare_strided(
                    a.data_ptr() as *const T, a_stride,
                    b.data_ptr() as *const T, b_stride,
                    result_ptr,
                    size, kernel,
                );
            }
        } else {
            // nD: iterate over outer dimensions, call strided loop per inner row
            let inner_size = out_shape[ndim - 1];
            let a_inner_stride = a.strides()[ndim - 1];
            let b_inner_stride = b.strides()[ndim - 1];
            let outer_shape = &out_shape[..ndim - 1];
            let outer_size: usize = outer_shape.iter().product();

            let a_strides = a.strides();
            let b_strides = b.strides();

            let mut outer_indices = vec![0usize; ndim - 1];
            for i in 0..outer_size {
                let a_offset: isize = outer_indices.iter().zip(a_strides).map(|(&idx, &s)| idx as isize * s).sum();
                let b_offset: isize = outer_indices.iter().zip(b_strides).map(|(&idx, &s)| idx as isize * s).sum();

                unsafe {
                    loops::map_compare_strided(
                        (a.data_ptr() as *const T).byte_offset(a_offset), a_inner_stride,
                        (b.data_ptr() as *const T).byte_offset(b_offset), b_inner_stride,
                        result_ptr.add(i * inner_size),
                        inner_size, kernel,
                    );
                }
                crate::array::increment_indices(&mut outer_indices, outer_shape);
            }
        }
    }

    Some(result)
}

// ============================================================================
// Bitwise dispatch
// ============================================================================

/// Dispatch bitwise AND using the kernel/loop architecture.
pub fn dispatch_bitwise_and(
    a: &RumpyArray,
    b: &RumpyArray,
    out_shape: &[usize],
) -> Option<RumpyArray> {
    dispatch_bitwise_binary_kernel(a, b, out_shape, And)
}

/// Dispatch bitwise OR using the kernel/loop architecture.
pub fn dispatch_bitwise_or(
    a: &RumpyArray,
    b: &RumpyArray,
    out_shape: &[usize],
) -> Option<RumpyArray> {
    dispatch_bitwise_binary_kernel(a, b, out_shape, Or)
}

/// Dispatch bitwise XOR using the kernel/loop architecture.
pub fn dispatch_bitwise_xor(
    a: &RumpyArray,
    b: &RumpyArray,
    out_shape: &[usize],
) -> Option<RumpyArray> {
    dispatch_bitwise_binary_kernel(a, b, out_shape, Xor)
}

/// Dispatch left shift using the kernel/loop architecture.
pub fn dispatch_left_shift(
    a: &RumpyArray,
    b: &RumpyArray,
    out_shape: &[usize],
) -> Option<RumpyArray> {
    dispatch_bitwise_shift_kernel(a, b, out_shape, LeftShift)
}

/// Dispatch right shift using the kernel/loop architecture.
pub fn dispatch_right_shift(
    a: &RumpyArray,
    b: &RumpyArray,
    out_shape: &[usize],
) -> Option<RumpyArray> {
    dispatch_bitwise_shift_kernel(a, b, out_shape, RightShift)
}

/// Dispatch bitwise NOT using the kernel/loop architecture.
pub fn dispatch_bitwise_not(arr: &RumpyArray) -> Option<RumpyArray> {
    dispatch_bitwise_not_kernel(arr, Not)
}

/// Generic dispatch for binary bitwise kernels (And, Or, Xor).
/// Supports integer types and bool.
fn dispatch_bitwise_binary_kernel<K>(
    a: &RumpyArray,
    b: &RumpyArray,
    out_shape: &[usize],
    kernel: K,
) -> Option<RumpyArray>
where
    K: BinaryKernel<i64>
        + BinaryKernel<i32>
        + BinaryKernel<i16>
        + BinaryKernel<u64>
        + BinaryKernel<u32>
        + BinaryKernel<u16>
        + BinaryKernel<u8>
        + BinaryKernel<bool>,
{
    let a_kind = a.dtype().kind();
    let b_kind = b.dtype().kind();

    // Only handle same-type operations
    if a_kind != b_kind {
        return None;
    }

    match a_kind {
        DTypeKind::Int64 => dispatch_binary_typed::<i64, K>(a, b, out_shape, kernel, DType::int64()),
        DTypeKind::Int32 => dispatch_binary_typed::<i32, K>(a, b, out_shape, kernel, DType::int32()),
        DTypeKind::Int16 => dispatch_binary_typed::<i16, K>(a, b, out_shape, kernel, DType::int16()),
        DTypeKind::Uint64 => dispatch_binary_typed::<u64, K>(a, b, out_shape, kernel, DType::uint64()),
        DTypeKind::Uint32 => dispatch_binary_typed::<u32, K>(a, b, out_shape, kernel, DType::uint32()),
        DTypeKind::Uint16 => dispatch_binary_typed::<u16, K>(a, b, out_shape, kernel, DType::uint16()),
        DTypeKind::Uint8 => dispatch_binary_typed::<u8, K>(a, b, out_shape, kernel, DType::uint8()),
        DTypeKind::Bool => dispatch_bitwise_bool_typed(a, b, out_shape, kernel),
        _ => None,
    }
}

/// Generic dispatch for shift kernels (LeftShift, RightShift).
/// Only supports integer types (not bool).
fn dispatch_bitwise_shift_kernel<K>(
    a: &RumpyArray,
    b: &RumpyArray,
    out_shape: &[usize],
    kernel: K,
) -> Option<RumpyArray>
where
    K: BinaryKernel<i64>
        + BinaryKernel<i32>
        + BinaryKernel<i16>
        + BinaryKernel<u64>
        + BinaryKernel<u32>
        + BinaryKernel<u16>
        + BinaryKernel<u8>,
{
    let a_kind = a.dtype().kind();
    let b_kind = b.dtype().kind();

    // Only handle same-type operations
    if a_kind != b_kind {
        return None;
    }

    match a_kind {
        DTypeKind::Int64 => dispatch_binary_typed::<i64, K>(a, b, out_shape, kernel, DType::int64()),
        DTypeKind::Int32 => dispatch_binary_typed::<i32, K>(a, b, out_shape, kernel, DType::int32()),
        DTypeKind::Int16 => dispatch_binary_typed::<i16, K>(a, b, out_shape, kernel, DType::int16()),
        DTypeKind::Uint64 => dispatch_binary_typed::<u64, K>(a, b, out_shape, kernel, DType::uint64()),
        DTypeKind::Uint32 => dispatch_binary_typed::<u32, K>(a, b, out_shape, kernel, DType::uint32()),
        DTypeKind::Uint16 => dispatch_binary_typed::<u16, K>(a, b, out_shape, kernel, DType::uint16()),
        DTypeKind::Uint8 => dispatch_binary_typed::<u8, K>(a, b, out_shape, kernel, DType::uint8()),
        _ => None, // No shifts for bool or float
    }
}

/// Bool-specific binary bitwise dispatch.
/// Bool is stored as u8 but uses logical operators.
fn dispatch_bitwise_bool_typed<K: BinaryKernel<bool>>(
    a: &RumpyArray,
    b: &RumpyArray,
    out_shape: &[usize],
    _kernel: K,
) -> Option<RumpyArray> {
    let size: usize = out_shape.iter().product();
    if size == 0 {
        return Some(RumpyArray::zeros(out_shape.to_vec(), DType::bool()));
    }

    let mut result = RumpyArray::zeros(out_shape.to_vec(), DType::bool());
    let buffer = result.buffer_mut();
    let result_buffer = Arc::get_mut(buffer).expect("buffer must be unique");
    let result_ptr = result_buffer.as_mut_ptr();

    // Layout detection
    let a_full_contig = a.is_c_contiguous() && a.shape() == out_shape;
    let b_full_contig = b.is_c_contiguous() && b.shape() == out_shape;

    if a_full_contig && b_full_contig {
        // Fast path: contiguous
        let a_ptr = a.data_ptr();
        let b_ptr = b.data_ptr();
        for i in 0..size {
            let av = unsafe { *a_ptr.add(i) != 0 };
            let bv = unsafe { *b_ptr.add(i) != 0 };
            unsafe { *result_ptr.add(i) = K::apply(av, bv) as u8; }
        }
    } else {
        // Strided path
        let ndim = out_shape.len();
        let a_strides = a.strides();
        let b_strides = b.strides();
        let a_ptr = a.data_ptr();
        let b_ptr = b.data_ptr();

        let mut indices = vec![0usize; ndim];
        for i in 0..size {
            let a_offset: isize = indices.iter().zip(a_strides).map(|(&idx, &s)| idx as isize * s).sum();
            let b_offset: isize = indices.iter().zip(b_strides).map(|(&idx, &s)| idx as isize * s).sum();
            let av = unsafe { *a_ptr.offset(a_offset) != 0 };
            let bv = unsafe { *b_ptr.offset(b_offset) != 0 };
            unsafe { *result_ptr.add(i) = K::apply(av, bv) as u8; }
            crate::array::increment_indices(&mut indices, out_shape);
        }
    }

    Some(result)
}

/// Generic dispatch for bitwise NOT kernel.
fn dispatch_bitwise_not_kernel<K>(arr: &RumpyArray, kernel: K) -> Option<RumpyArray>
where
    K: UnaryKernel<i64>
        + UnaryKernel<i32>
        + UnaryKernel<i16>
        + UnaryKernel<u64>
        + UnaryKernel<u32>
        + UnaryKernel<u16>
        + UnaryKernel<u8>
        + UnaryKernel<bool>,
{
    match arr.dtype().kind() {
        DTypeKind::Int64 => dispatch_unary_typed::<i64, K>(arr, kernel, DType::int64()),
        DTypeKind::Int32 => dispatch_unary_typed::<i32, K>(arr, kernel, DType::int32()),
        DTypeKind::Int16 => dispatch_unary_typed::<i16, K>(arr, kernel, DType::int16()),
        DTypeKind::Uint64 => dispatch_unary_typed::<u64, K>(arr, kernel, DType::uint64()),
        DTypeKind::Uint32 => dispatch_unary_typed::<u32, K>(arr, kernel, DType::uint32()),
        DTypeKind::Uint16 => dispatch_unary_typed::<u16, K>(arr, kernel, DType::uint16()),
        DTypeKind::Uint8 => dispatch_unary_typed::<u8, K>(arr, kernel, DType::uint8()),
        DTypeKind::Bool => dispatch_bitwise_not_bool_typed(arr, kernel),
        _ => None,
    }
}

/// Bool-specific NOT dispatch.
fn dispatch_bitwise_not_bool_typed<K: UnaryKernel<bool>>(
    arr: &RumpyArray,
    _kernel: K,
) -> Option<RumpyArray> {
    let size = arr.size();
    if size == 0 {
        return Some(RumpyArray::zeros(arr.shape().to_vec(), DType::bool()));
    }

    let mut result = RumpyArray::zeros(arr.shape().to_vec(), DType::bool());
    let buffer = result.buffer_mut();
    let result_buffer = Arc::get_mut(buffer).expect("buffer must be unique");
    let result_ptr = result_buffer.as_mut_ptr();

    if arr.is_c_contiguous() {
        let src_ptr = arr.data_ptr();
        for i in 0..size {
            let v = unsafe { *src_ptr.add(i) != 0 };
            unsafe { *result_ptr.add(i) = K::apply(v) as u8; }
        }
    } else {
        let src_ptr = arr.data_ptr();
        let strides = arr.strides();
        let shape = arr.shape();
        let ndim = shape.len();
        let mut indices = vec![0usize; ndim];

        for i in 0..size {
            let offset: isize = indices.iter().zip(strides).map(|(&idx, &s)| idx as isize * s).sum();
            let v = unsafe { *src_ptr.offset(offset) != 0 };
            unsafe { *result_ptr.add(i) = K::apply(v) as u8; }
            crate::array::increment_indices(&mut indices, shape);
        }
    }

    Some(result)
}
