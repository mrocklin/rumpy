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
    NanSum, NanProd, NanMax, NanMin, SumOfSquares,
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
        + BinaryKernel<i8>
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
        DTypeKind::Int8 => dispatch_binary_typed::<i8, K>(a, b, out_shape, kernel, DType::int8()),
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

/// Sum of squares reduction: sum(x*x). Single pass, no intermediate allocation.
pub fn dispatch_reduce_sum_of_squares(arr: &RumpyArray) -> Option<f64> {
    let kind = arr.dtype().kind();
    let size = arr.size();

    match kind {
        DTypeKind::Float64 => dispatch_reduce_typed::<f64, SumOfSquares>(arr, size, SumOfSquares),
        DTypeKind::Float32 => dispatch_reduce_typed::<f32, SumOfSquares>(arr, size, SumOfSquares).map(|v| v as f64),
        DTypeKind::Float16 => dispatch_reduce_typed::<f16, SumOfSquares>(arr, size, SumOfSquares).map(|v| v.to_f64()),
        DTypeKind::Int64 => dispatch_reduce_typed::<i64, SumOfSquares>(arr, size, SumOfSquares).map(|v| v as f64),
        DTypeKind::Int32 => dispatch_reduce_typed::<i32, SumOfSquares>(arr, size, SumOfSquares).map(|v| v as f64),
        DTypeKind::Int16 => dispatch_reduce_typed::<i16, SumOfSquares>(arr, size, SumOfSquares).map(|v| v as f64),
        DTypeKind::Int8 => dispatch_reduce_typed::<i8, SumOfSquares>(arr, size, SumOfSquares).map(|v| v as f64),
        DTypeKind::Uint64 => dispatch_reduce_typed::<u64, SumOfSquares>(arr, size, SumOfSquares).map(|v| v as f64),
        DTypeKind::Uint32 => dispatch_reduce_typed::<u32, SumOfSquares>(arr, size, SumOfSquares).map(|v| v as f64),
        DTypeKind::Uint16 => dispatch_reduce_typed::<u16, SumOfSquares>(arr, size, SumOfSquares).map(|v| v as f64),
        DTypeKind::Uint8 => dispatch_reduce_typed::<u8, SumOfSquares>(arr, size, SumOfSquares).map(|v| v as f64),
        // Complex: return magnitude squared sum (re^2 + im^2)
        DTypeKind::Complex128 => {
            dispatch_reduce_typed::<Complex<f64>, SumOfSquares>(arr, size, SumOfSquares)
                .map(|c| c.re + c.im)
        }
        DTypeKind::Complex64 => {
            dispatch_reduce_typed::<Complex<f32>, SumOfSquares>(arr, size, SumOfSquares)
                .map(|c| (c.re + c.im) as f64)
        }
        _ => None,
    }
}

/// Generic reduce dispatch returning a 0-d array.
fn dispatch_reduce_to_array<K>(arr: &RumpyArray, kernel: K) -> Option<RumpyArray>
where
    K: ReduceKernel<f64> + ReduceKernel<f32> + ReduceKernel<f16> + ReduceKernel<i64> + ReduceKernel<i32> + ReduceKernel<i16> + ReduceKernel<i8>
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
        DTypeKind::Int16 => {
            let val = dispatch_reduce_typed::<i16, K>(arr, size, kernel)?;
            Some(RumpyArray::full(vec![1], val as f64, DType::int16()))
        }
        DTypeKind::Int8 => {
            let val = dispatch_reduce_typed::<i8, K>(arr, size, kernel)?;
            Some(RumpyArray::full(vec![1], val as f64, DType::int8()))
        }
        DTypeKind::Uint64 => {
            let val = dispatch_reduce_typed::<u64, K>(arr, size, kernel)?;
            Some(RumpyArray::full(vec![1], val as f64, DType::uint64()))
        }
        DTypeKind::Uint32 => {
            let val = dispatch_reduce_typed::<u32, K>(arr, size, kernel)?;
            Some(RumpyArray::full(vec![1], val as f64, DType::uint32()))
        }
        DTypeKind::Uint16 => {
            let val = dispatch_reduce_typed::<u16, K>(arr, size, kernel)?;
            Some(RumpyArray::full(vec![1], val as f64, DType::uint16()))
        }
        DTypeKind::Uint8 => {
            let val = dispatch_reduce_typed::<u8, K>(arr, size, kernel)?;
            Some(RumpyArray::full(vec![1], val as f64, DType::uint8()))
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
    K: ReduceKernel<f64> + ReduceKernel<f32> + ReduceKernel<f16> + ReduceKernel<i64> + ReduceKernel<i32> + ReduceKernel<i16> + ReduceKernel<i8>
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
        DTypeKind::Int8 => dispatch_reduce_axis_typed::<i8, K>(arr, axis, kernel, DType::int8()),
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
// NaN-aware reduce dispatch (floats only)
// ============================================================================

/// Dispatch NaN-aware sum (full array).
pub fn dispatch_nan_reduce_sum(arr: &RumpyArray) -> Option<f64> {
    dispatch_nan_reduce_float(arr, NanSum)
}

/// Dispatch NaN-aware product (full array).
pub fn dispatch_nan_reduce_prod(arr: &RumpyArray) -> Option<f64> {
    dispatch_nan_reduce_float(arr, NanProd)
}

/// Dispatch NaN-aware max (full array).
pub fn dispatch_nan_reduce_max(arr: &RumpyArray) -> Option<f64> {
    dispatch_nan_reduce_float(arr, NanMax)
}

/// Dispatch NaN-aware min (full array).
pub fn dispatch_nan_reduce_min(arr: &RumpyArray) -> Option<f64> {
    dispatch_nan_reduce_float(arr, NanMin)
}

/// Generic NaN-aware reduce dispatch for float types.
fn dispatch_nan_reduce_float<K>(arr: &RumpyArray, kernel: K) -> Option<f64>
where
    K: ReduceKernel<f64> + ReduceKernel<f32>,
{
    let kind = arr.dtype().kind();
    let size = arr.size();

    match kind {
        DTypeKind::Float64 => {
            let val = dispatch_reduce_typed::<f64, K>(arr, size, kernel)?;
            Some(val)
        }
        DTypeKind::Float32 => {
            let val = dispatch_reduce_typed::<f32, K>(arr, size, kernel)?;
            Some(val as f64)
        }
        _ => None, // Only floats support NaN-aware reductions
    }
}

/// Dispatch NaN-aware sum along axis.
pub fn dispatch_nan_reduce_axis_sum(arr: &RumpyArray, axis: usize) -> Option<RumpyArray> {
    dispatch_nan_reduce_axis_float(arr, axis, NanSum)
}

/// Dispatch NaN-aware product along axis.
pub fn dispatch_nan_reduce_axis_prod(arr: &RumpyArray, axis: usize) -> Option<RumpyArray> {
    dispatch_nan_reduce_axis_float(arr, axis, NanProd)
}

/// Dispatch NaN-aware max along axis.
pub fn dispatch_nan_reduce_axis_max(arr: &RumpyArray, axis: usize) -> Option<RumpyArray> {
    dispatch_nan_reduce_axis_float(arr, axis, NanMax)
}

/// Dispatch NaN-aware min along axis.
pub fn dispatch_nan_reduce_axis_min(arr: &RumpyArray, axis: usize) -> Option<RumpyArray> {
    dispatch_nan_reduce_axis_float(arr, axis, NanMin)
}

/// Generic NaN-aware axis reduce dispatch for float types.
fn dispatch_nan_reduce_axis_float<K>(arr: &RumpyArray, axis: usize, kernel: K) -> Option<RumpyArray>
where
    K: ReduceKernel<f64> + ReduceKernel<f32>,
{
    let kind = arr.dtype().kind();
    match kind {
        DTypeKind::Float64 => dispatch_reduce_axis_typed::<f64, K>(arr, axis, kernel, DType::float64()),
        DTypeKind::Float32 => dispatch_reduce_axis_typed::<f32, K>(arr, axis, kernel, DType::float32()),
        _ => None, // Only floats support NaN-aware reductions
    }
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
        + UnaryKernel<i8>
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
        DTypeKind::Int8 => dispatch_unary_typed::<i8, K>(arr, kernel, DType::int8()),
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
        + CompareKernel<i8>
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
        DTypeKind::Int8 => dispatch_compare_typed::<i8, K>(a, b, out_shape, kernel),
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
        + BinaryKernel<i8>
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
        DTypeKind::Int8 => dispatch_binary_typed::<i8, K>(a, b, out_shape, kernel, DType::int8()),
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
        + BinaryKernel<i8>
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
        DTypeKind::Int8 => dispatch_binary_typed::<i8, K>(a, b, out_shape, kernel, DType::int8()),
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
        + UnaryKernel<i8>
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
        DTypeKind::Int8 => dispatch_unary_typed::<i8, K>(arr, kernel, DType::int8()),
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

// ============================================================================
// Parameterized operations (clip, round)
// ============================================================================

/// Clip array values to [min, max] range with dtype-aware dispatch.
pub fn dispatch_clip(arr: &RumpyArray, a_min: Option<f64>, a_max: Option<f64>) -> Option<RumpyArray> {
    match arr.dtype().kind() {
        DTypeKind::Float64 => dispatch_clip_typed::<f64>(arr, a_min, a_max, DType::float64()),
        DTypeKind::Float32 => dispatch_clip_typed::<f32>(arr, a_min.map(|v| v as f32), a_max.map(|v| v as f32), DType::float32()),
        DTypeKind::Float16 => dispatch_clip_typed::<f16>(arr, a_min.map(f16::from_f64), a_max.map(f16::from_f64), DType::float16()),
        DTypeKind::Int64 => dispatch_clip_typed::<i64>(arr, a_min.map(|v| v as i64), a_max.map(|v| v as i64), DType::int64()),
        DTypeKind::Int32 => dispatch_clip_typed::<i32>(arr, a_min.map(|v| v as i32), a_max.map(|v| v as i32), DType::int32()),
        DTypeKind::Int16 => dispatch_clip_typed::<i16>(arr, a_min.map(|v| v as i16), a_max.map(|v| v as i16), DType::int16()),
        DTypeKind::Uint64 => dispatch_clip_typed::<u64>(arr, a_min.map(|v| v as u64), a_max.map(|v| v as u64), DType::uint64()),
        DTypeKind::Uint32 => dispatch_clip_typed::<u32>(arr, a_min.map(|v| v as u32), a_max.map(|v| v as u32), DType::uint32()),
        DTypeKind::Uint16 => dispatch_clip_typed::<u16>(arr, a_min.map(|v| v as u16), a_max.map(|v| v as u16), DType::uint16()),
        DTypeKind::Uint8 => dispatch_clip_typed::<u8>(arr, a_min.map(|v| v as u8), a_max.map(|v| v as u8), DType::uint8()),
        _ => None,  // Complex, Bool, DateTime fall back to old path
    }
}

fn dispatch_clip_typed<T>(arr: &RumpyArray, a_min: Option<T>, a_max: Option<T>, dtype: DType) -> Option<RumpyArray>
where
    T: Copy + PartialOrd,
{
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
        for i in 0..size {
            let mut v = src_slice[i];
            if let Some(min) = a_min {
                if v < min { v = min; }
            }
            if let Some(max) = a_max {
                if v > max { v = max; }
            }
            out_slice[i] = v;
        }
    } else {
        let itemsize = std::mem::size_of::<T>() as isize;
        for (i, offset) in arr.iter_offsets().enumerate() {
            let mut v = unsafe { *(arr.data_ptr() as *const T).byte_offset(offset) };
            if let Some(min) = a_min {
                if v < min { v = min; }
            }
            if let Some(max) = a_max {
                if v > max { v = max; }
            }
            unsafe { *result_ptr.byte_offset(i as isize * itemsize) = v; }
        }
    }

    Some(result)
}

/// Round array values to given decimal places with dtype-aware dispatch.
pub fn dispatch_round(arr: &RumpyArray, decimals: i32) -> Option<RumpyArray> {
    match arr.dtype().kind() {
        DTypeKind::Float64 => dispatch_round_f64(arr, decimals),
        DTypeKind::Float32 => dispatch_round_f32(arr, decimals),
        // For integers, round is a no-op if decimals >= 0
        DTypeKind::Int64 if decimals >= 0 => Some(arr.clone()),
        DTypeKind::Int32 if decimals >= 0 => Some(arr.clone()),
        DTypeKind::Int16 if decimals >= 0 => Some(arr.clone()),
        DTypeKind::Uint64 if decimals >= 0 => Some(arr.clone()),
        DTypeKind::Uint32 if decimals >= 0 => Some(arr.clone()),
        DTypeKind::Uint16 if decimals >= 0 => Some(arr.clone()),
        DTypeKind::Uint8 if decimals >= 0 => Some(arr.clone()),
        // Negative decimals on integers: round to tens, hundreds, etc.
        DTypeKind::Int64 => dispatch_round_i64(arr, decimals),
        DTypeKind::Int32 => dispatch_round_i32(arr, decimals),
        DTypeKind::Int16 => dispatch_round_i16(arr, decimals),
        DTypeKind::Uint64 => dispatch_round_u64(arr, decimals),
        DTypeKind::Uint32 => dispatch_round_u32(arr, decimals),
        DTypeKind::Uint16 => dispatch_round_u16(arr, decimals),
        DTypeKind::Uint8 => dispatch_round_u8(arr, decimals),
        _ => None,
    }
}

/// Dispatch float rounding
macro_rules! impl_round_float {
    ($fn_name:ident, $T:ty, $dtype:expr) => {
        fn $fn_name(arr: &RumpyArray, decimals: i32) -> Option<RumpyArray> {
            let size = arr.size();
            if size == 0 {
                return Some(RumpyArray::zeros(arr.shape().to_vec(), $dtype));
            }

            let scale = (10.0 as $T).powi(decimals);
            let mut result = RumpyArray::zeros(arr.shape().to_vec(), $dtype);
            let buffer = result.buffer_mut();
            let result_buffer = Arc::get_mut(buffer).expect("buffer must be unique");
            let result_ptr = result_buffer.as_mut_ptr() as *mut $T;

            if arr.is_c_contiguous() {
                let src_slice = unsafe { std::slice::from_raw_parts(arr.data_ptr() as *const $T, size) };
                let out_slice = unsafe { std::slice::from_raw_parts_mut(result_ptr, size) };
                for i in 0..size {
                    out_slice[i] = (src_slice[i] * scale).round() / scale;
                }
            } else {
                let itemsize = std::mem::size_of::<$T>() as isize;
                for (i, offset) in arr.iter_offsets().enumerate() {
                    let v = unsafe { *(arr.data_ptr() as *const $T).byte_offset(offset) };
                    unsafe { *result_ptr.byte_offset(i as isize * itemsize) = (v * scale).round() / scale; }
                }
            }

            Some(result)
        }
    };
}

impl_round_float!(dispatch_round_f64, f64, DType::float64());
impl_round_float!(dispatch_round_f32, f32, DType::float32());

/// Dispatch integer rounding (negative decimals only - rounds to tens, hundreds, etc.)
macro_rules! impl_round_int {
    ($fn_name:ident, $T:ty, $dtype:expr) => {
        fn $fn_name(arr: &RumpyArray, decimals: i32) -> Option<RumpyArray> {
            let size = arr.size();
            if size == 0 {
                return Some(RumpyArray::zeros(arr.shape().to_vec(), $dtype));
            }

            let scale = 10i64.pow((-decimals) as u32) as $T;
            let mut result = RumpyArray::zeros(arr.shape().to_vec(), $dtype);
            let buffer = result.buffer_mut();
            let result_buffer = Arc::get_mut(buffer).expect("buffer must be unique");
            let result_ptr = result_buffer.as_mut_ptr() as *mut $T;

            if arr.is_c_contiguous() {
                let src_slice = unsafe { std::slice::from_raw_parts(arr.data_ptr() as *const $T, size) };
                let out_slice = unsafe { std::slice::from_raw_parts_mut(result_ptr, size) };
                for i in 0..size {
                    out_slice[i] = (src_slice[i] / scale) * scale;
                }
            } else {
                let itemsize = std::mem::size_of::<$T>() as isize;
                for (i, offset) in arr.iter_offsets().enumerate() {
                    let v = unsafe { *(arr.data_ptr() as *const $T).byte_offset(offset) };
                    unsafe { *result_ptr.byte_offset(i as isize * itemsize) = (v / scale) * scale; }
                }
            }

            Some(result)
        }
    };
}

impl_round_int!(dispatch_round_i64, i64, DType::int64());
impl_round_int!(dispatch_round_i32, i32, DType::int32());
impl_round_int!(dispatch_round_i16, i16, DType::int16());
impl_round_int!(dispatch_round_u64, u64, DType::uint64());
impl_round_int!(dispatch_round_u32, u32, DType::uint32());
impl_round_int!(dispatch_round_u16, u16, DType::uint16());
impl_round_int!(dispatch_round_u8, u8, DType::uint8());

/// Dispatch where selection with dtype-aware typed path.
/// Returns None if result dtype doesn't match x/y dtypes (requires mixed-type fallback).
pub fn dispatch_where(
    cond: &RumpyArray,
    x: &RumpyArray,
    y: &RumpyArray,
    result_dtype: &DType,
    out_shape: &[usize],
) -> Option<RumpyArray> {
    // Only dispatch when x, y, and result all have the same dtype
    if x.dtype() != y.dtype() || x.dtype() != *result_dtype {
        return None;
    }

    match result_dtype.kind() {
        DTypeKind::Float64 => dispatch_where_typed::<f64>(cond, x, y, out_shape, DType::float64()),
        DTypeKind::Float32 => dispatch_where_typed::<f32>(cond, x, y, out_shape, DType::float32()),
        DTypeKind::Float16 => dispatch_where_typed::<f16>(cond, x, y, out_shape, DType::float16()),
        DTypeKind::Int64 => dispatch_where_typed::<i64>(cond, x, y, out_shape, DType::int64()),
        DTypeKind::Int32 => dispatch_where_typed::<i32>(cond, x, y, out_shape, DType::int32()),
        DTypeKind::Int16 => dispatch_where_typed::<i16>(cond, x, y, out_shape, DType::int16()),
        DTypeKind::Uint64 => dispatch_where_typed::<u64>(cond, x, y, out_shape, DType::uint64()),
        DTypeKind::Uint32 => dispatch_where_typed::<u32>(cond, x, y, out_shape, DType::uint32()),
        DTypeKind::Uint16 => dispatch_where_typed::<u16>(cond, x, y, out_shape, DType::uint16()),
        DTypeKind::Uint8 => dispatch_where_typed::<u8>(cond, x, y, out_shape, DType::uint8()),
        DTypeKind::Bool => dispatch_where_typed::<u8>(cond, x, y, out_shape, DType::bool()),
        DTypeKind::Complex128 => dispatch_where_typed::<Complex<f64>>(cond, x, y, out_shape, DType::complex128()),
        DTypeKind::Complex64 => dispatch_where_typed::<Complex<f32>>(cond, x, y, out_shape, DType::complex64()),
        _ => None,
    }
}

fn dispatch_where_typed<T: Copy>(
    cond: &RumpyArray,
    x: &RumpyArray,
    y: &RumpyArray,
    out_shape: &[usize],
    dtype: DType,
) -> Option<RumpyArray> {
    let size: usize = out_shape.iter().product();
    if size == 0 {
        return Some(RumpyArray::zeros(out_shape.to_vec(), dtype));
    }

    // Broadcast arrays
    let cond_bc = cond.broadcast_to(out_shape)?;
    let x_bc = x.broadcast_to(out_shape)?;
    let y_bc = y.broadcast_to(out_shape)?;

    let mut result = RumpyArray::zeros(out_shape.to_vec(), dtype);
    let buffer = result.buffer_mut();
    let result_buffer = Arc::get_mut(buffer).expect("buffer must be unique");
    let result_ptr = result_buffer.as_mut_ptr() as *mut T;

    let itemsize = std::mem::size_of::<T>() as isize;

    // Check for contiguous fast path: all arrays contiguous and same itemsize
    let all_contiguous = cond_bc.is_c_contiguous() && x_bc.is_c_contiguous() && y_bc.is_c_contiguous();

    if all_contiguous {
        let cond_ptr = cond_bc.data_ptr();
        let x_ptr = x_bc.data_ptr() as *const T;
        let y_ptr = y_bc.data_ptr() as *const T;
        let cond_itemsize = cond_bc.dtype().itemsize();

        for i in 0..size {
            let cond_val = unsafe {
                match cond_itemsize {
                    1 => *cond_ptr.add(i) != 0,
                    _ => {
                        // For other dtypes, check if any byte is non-zero
                        let ptr = cond_ptr.add(i * cond_itemsize);
                        (0..cond_itemsize).any(|j| *ptr.add(j) != 0)
                    }
                }
            };
            let val = if cond_val {
                unsafe { *x_ptr.add(i) }
            } else {
                unsafe { *y_ptr.add(i) }
            };
            unsafe { *result_ptr.add(i) = val; }
        }
    } else {
        // Strided path using offset iterators
        for (i, ((cond_off, x_off), y_off)) in cond_bc.iter_offsets()
            .zip(x_bc.iter_offsets())
            .zip(y_bc.iter_offsets())
            .enumerate()
        {
            let cond_val = unsafe { *cond_bc.data_ptr().byte_offset(cond_off) != 0 };
            let val = if cond_val {
                unsafe { *(x_bc.data_ptr() as *const T).byte_offset(x_off) }
            } else {
                unsafe { *(y_bc.data_ptr() as *const T).byte_offset(y_off) }
            };
            unsafe { *result_ptr.byte_offset(i as isize * itemsize) = val; }
        }
    }

    Some(result)
}

// ============================================================================
// Logical axis operations (all_axis, any_axis)
// ============================================================================

/// Dispatch all_axis with dtype-aware typed path.
pub fn dispatch_all_axis(arr: &RumpyArray, axis: usize) -> Option<RumpyArray> {
    match arr.dtype().kind() {
        DTypeKind::Float64 => dispatch_all_axis_typed::<f64>(arr, axis),
        DTypeKind::Float32 => dispatch_all_axis_typed::<f32>(arr, axis),
        DTypeKind::Float16 => dispatch_all_axis_typed::<f16>(arr, axis),
        DTypeKind::Int64 => dispatch_all_axis_typed::<i64>(arr, axis),
        DTypeKind::Int32 => dispatch_all_axis_typed::<i32>(arr, axis),
        DTypeKind::Int16 => dispatch_all_axis_typed::<i16>(arr, axis),
        DTypeKind::Uint64 => dispatch_all_axis_typed::<u64>(arr, axis),
        DTypeKind::Uint32 => dispatch_all_axis_typed::<u32>(arr, axis),
        DTypeKind::Uint16 => dispatch_all_axis_typed::<u16>(arr, axis),
        DTypeKind::Uint8 => dispatch_all_axis_typed::<u8>(arr, axis),
        DTypeKind::Bool => dispatch_all_axis_typed::<u8>(arr, axis), // bool stored as u8
        DTypeKind::Complex128 => dispatch_all_axis_complex::<f64>(arr, axis),
        DTypeKind::Complex64 => dispatch_all_axis_complex::<f32>(arr, axis),
        _ => None,
    }
}

/// Dispatch any_axis with dtype-aware typed path.
pub fn dispatch_any_axis(arr: &RumpyArray, axis: usize) -> Option<RumpyArray> {
    match arr.dtype().kind() {
        DTypeKind::Float64 => dispatch_any_axis_typed::<f64>(arr, axis),
        DTypeKind::Float32 => dispatch_any_axis_typed::<f32>(arr, axis),
        DTypeKind::Float16 => dispatch_any_axis_typed::<f16>(arr, axis),
        DTypeKind::Int64 => dispatch_any_axis_typed::<i64>(arr, axis),
        DTypeKind::Int32 => dispatch_any_axis_typed::<i32>(arr, axis),
        DTypeKind::Int16 => dispatch_any_axis_typed::<i16>(arr, axis),
        DTypeKind::Uint64 => dispatch_any_axis_typed::<u64>(arr, axis),
        DTypeKind::Uint32 => dispatch_any_axis_typed::<u32>(arr, axis),
        DTypeKind::Uint16 => dispatch_any_axis_typed::<u16>(arr, axis),
        DTypeKind::Uint8 => dispatch_any_axis_typed::<u8>(arr, axis),
        DTypeKind::Bool => dispatch_any_axis_typed::<u8>(arr, axis),
        DTypeKind::Complex128 => dispatch_any_axis_complex::<f64>(arr, axis),
        DTypeKind::Complex64 => dispatch_any_axis_complex::<f32>(arr, axis),
        _ => None,
    }
}

/// Helper for all_axis: returns false on first zero, true otherwise.
fn dispatch_all_axis_typed<T: Copy + Default + PartialEq>(arr: &RumpyArray, axis: usize) -> Option<RumpyArray> {
    dispatch_logical_axis_typed::<T>(arr, axis, true, |v, zero| v == zero)
}

/// Helper for any_axis: returns true on first non-zero, false otherwise.
fn dispatch_any_axis_typed<T: Copy + Default + PartialEq>(arr: &RumpyArray, axis: usize) -> Option<RumpyArray> {
    dispatch_logical_axis_typed::<T>(arr, axis, false, |v, zero| v != zero)
}

/// Generic logical axis reduction for numeric types.
/// `empty_val`: result for empty axis (true for all, false for any)
/// `stop_pred`: return true to short-circuit (found zero for all, found nonzero for any)
fn dispatch_logical_axis_typed<T: Copy + Default + PartialEq>(
    arr: &RumpyArray,
    axis: usize,
    empty_val: bool,
    stop_pred: fn(T, T) -> bool,
) -> Option<RumpyArray> {
    let mut out_shape: Vec<usize> = arr.shape().to_vec();
    let axis_len = out_shape.remove(axis);
    if out_shape.is_empty() {
        out_shape = vec![1];
    }

    let mut result = RumpyArray::zeros(out_shape.clone(), DType::bool());
    let out_size = result.size();
    if out_size == 0 {
        return Some(result);
    }
    if axis_len == 0 {
        // Empty axis: fill with empty_val
        if empty_val {
            let buffer = result.buffer_mut();
            let result_buffer = Arc::get_mut(buffer).expect("buffer must be unique");
            let result_ptr = result_buffer.as_mut_ptr();
            for i in 0..out_size {
                unsafe { *result_ptr.add(i) = 1; }
            }
        }
        return Some(result);
    }

    let buffer = result.buffer_mut();
    let result_buffer = Arc::get_mut(buffer).expect("buffer must be unique");
    let result_ptr = result_buffer.as_mut_ptr();

    let axis_stride = arr.strides()[axis];
    let zero = T::default();

    for (i, base_offset) in arr.axis_offsets(axis).enumerate() {
        let src_start = unsafe { (arr.data_ptr() as *const T).byte_offset(base_offset) };
        let mut val = empty_val;
        for j in 0..axis_len {
            let v = unsafe { *src_start.byte_offset(axis_stride * j as isize) };
            if stop_pred(v, zero) {
                val = !empty_val;
                break;
            }
        }
        unsafe { *result_ptr.add(i) = val as u8; }
    }

    Some(result)
}

/// Complex all_axis: truthy if either real or imag is non-zero.
fn dispatch_all_axis_complex<T: Copy + Default + PartialEq>(arr: &RumpyArray, axis: usize) -> Option<RumpyArray> {
    dispatch_logical_axis_complex::<T>(arr, axis, true)
}

/// Complex any_axis.
fn dispatch_any_axis_complex<T: Copy + Default + PartialEq>(arr: &RumpyArray, axis: usize) -> Option<RumpyArray> {
    dispatch_logical_axis_complex::<T>(arr, axis, false)
}

/// Generic logical axis reduction for complex types.
fn dispatch_logical_axis_complex<T: Copy + Default + PartialEq>(
    arr: &RumpyArray,
    axis: usize,
    is_all: bool, // true for all(), false for any()
) -> Option<RumpyArray> {
    let mut out_shape: Vec<usize> = arr.shape().to_vec();
    let axis_len = out_shape.remove(axis);
    if out_shape.is_empty() {
        out_shape = vec![1];
    }

    let mut result = RumpyArray::zeros(out_shape.clone(), DType::bool());
    let out_size = result.size();
    if out_size == 0 {
        return Some(result);
    }
    if axis_len == 0 {
        if is_all {
            let buffer = result.buffer_mut();
            let result_buffer = Arc::get_mut(buffer).expect("buffer must be unique");
            let result_ptr = result_buffer.as_mut_ptr();
            for i in 0..out_size {
                unsafe { *result_ptr.add(i) = 1; }
            }
        }
        return Some(result);
    }

    let buffer = result.buffer_mut();
    let result_buffer = Arc::get_mut(buffer).expect("buffer must be unique");
    let result_ptr = result_buffer.as_mut_ptr();

    let axis_stride = arr.strides()[axis];
    let zero = T::default();

    for (i, base_offset) in arr.axis_offsets(axis).enumerate() {
        let src_start = unsafe { (arr.data_ptr() as *const T).byte_offset(base_offset) };
        let mut val = is_all;
        for j in 0..axis_len {
            let ptr = unsafe { src_start.byte_offset(axis_stride * j as isize) };
            let re = unsafe { *ptr };
            let im = unsafe { *ptr.add(1) };
            let is_zero = re == zero && im == zero;
            // For all: stop on zero (found false), for any: stop on non-zero (found true)
            if is_all == is_zero {
                val = !is_all;
                break;
            }
        }
        unsafe { *result_ptr.add(i) = val as u8; }
    }

    Some(result)
}

// ============================================================================
// Cumulative operations (cumsum, cumprod)
// ============================================================================

use crate::ops::kernels::arithmetic::{Sum as SumK, Prod as ProdK};

/// Dispatch cumulative sum with dtype-aware typed path.
/// axis = None means flatten and cumsum over all elements.
pub fn dispatch_cumsum(arr: &RumpyArray, axis: Option<usize>) -> Option<RumpyArray> {
    dispatch_cumulative_kernel(arr, axis, SumK)
}

/// Dispatch cumulative product with dtype-aware typed path.
pub fn dispatch_cumprod(arr: &RumpyArray, axis: Option<usize>) -> Option<RumpyArray> {
    dispatch_cumulative_kernel(arr, axis, ProdK)
}

/// Generic cumulative dispatch for all numeric dtypes.
fn dispatch_cumulative_kernel<K>(arr: &RumpyArray, axis: Option<usize>, kernel: K) -> Option<RumpyArray>
where
    K: ReduceKernel<f64>
        + ReduceKernel<f32>
        + ReduceKernel<f16>
        + ReduceKernel<i64>
        + ReduceKernel<i32>
        + ReduceKernel<i16>
        + ReduceKernel<u64>
        + ReduceKernel<u32>
        + ReduceKernel<u16>
        + ReduceKernel<u8>
        + ReduceKernel<Complex<f64>>
        + ReduceKernel<Complex<f32>>,
{
    match arr.dtype().kind() {
        DTypeKind::Float64 => dispatch_cumulative_typed::<f64, K>(arr, axis, kernel, DType::float64()),
        DTypeKind::Float32 => dispatch_cumulative_typed::<f32, K>(arr, axis, kernel, DType::float32()),
        DTypeKind::Float16 => dispatch_cumulative_typed::<f16, K>(arr, axis, kernel, DType::float16()),
        DTypeKind::Int64 => dispatch_cumulative_typed::<i64, K>(arr, axis, kernel, DType::int64()),
        DTypeKind::Int32 => dispatch_cumulative_typed::<i32, K>(arr, axis, kernel, DType::int32()),
        DTypeKind::Int16 => dispatch_cumulative_typed::<i16, K>(arr, axis, kernel, DType::int16()),
        DTypeKind::Uint64 => dispatch_cumulative_typed::<u64, K>(arr, axis, kernel, DType::uint64()),
        DTypeKind::Uint32 => dispatch_cumulative_typed::<u32, K>(arr, axis, kernel, DType::uint32()),
        DTypeKind::Uint16 => dispatch_cumulative_typed::<u16, K>(arr, axis, kernel, DType::uint16()),
        DTypeKind::Uint8 => dispatch_cumulative_typed::<u8, K>(arr, axis, kernel, DType::uint8()),
        DTypeKind::Complex128 => dispatch_cumulative_typed::<Complex<f64>, K>(arr, axis, kernel, DType::complex128()),
        DTypeKind::Complex64 => dispatch_cumulative_typed::<Complex<f32>, K>(arr, axis, kernel, DType::complex64()),
        _ => None, // Bool, DateTime fall back to old path
    }
}

/// Type-specific cumulative dispatch.
fn dispatch_cumulative_typed<T: Copy, K: ReduceKernel<T>>(
    arr: &RumpyArray,
    axis: Option<usize>,
    kernel: K,
    dtype: DType,
) -> Option<RumpyArray> {
    match axis {
        None => dispatch_cumulative_flat::<T, K>(arr, kernel, dtype),
        Some(ax) => dispatch_cumulative_axis::<T, K>(arr, ax, kernel, dtype),
    }
}

/// Cumulative over flattened array.
fn dispatch_cumulative_flat<T: Copy, K: ReduceKernel<T>>(
    arr: &RumpyArray,
    kernel: K,
    dtype: DType,
) -> Option<RumpyArray> {
    let size = arr.size();
    if size == 0 {
        return Some(RumpyArray::zeros(vec![0], dtype));
    }

    let mut result = RumpyArray::zeros(vec![size], dtype);
    let buffer = result.buffer_mut();
    let result_buffer = Arc::get_mut(buffer).expect("buffer must be unique");
    let result_ptr = result_buffer.as_mut_ptr() as *mut T;

    let itemsize = std::mem::size_of::<T>() as isize;

    if arr.is_c_contiguous() {
        let src_slice = unsafe { std::slice::from_raw_parts(arr.data_ptr() as *const T, size) };
        let out_slice = unsafe { std::slice::from_raw_parts_mut(result_ptr, size) };
        loops::cumulative(src_slice, out_slice, kernel);
    } else {
        // Strided: accumulate while iterating offsets
        let mut acc = K::init();
        for (i, offset) in arr.iter_offsets().enumerate() {
            let v = unsafe { *(arr.data_ptr() as *const T).byte_offset(offset) };
            acc = K::combine(acc, v);
            unsafe { *result_ptr.byte_offset(i as isize * itemsize) = acc; }
        }
    }

    Some(result)
}

/// Cumulative along a specific axis.
fn dispatch_cumulative_axis<T: Copy, K: ReduceKernel<T>>(
    arr: &RumpyArray,
    axis: usize,
    kernel: K,
    dtype: DType,
) -> Option<RumpyArray> {
    let shape = arr.shape().to_vec();
    let size = arr.size();
    if size == 0 {
        return Some(RumpyArray::zeros(shape, dtype));
    }

    let mut result = RumpyArray::zeros(shape.clone(), dtype);
    let buffer = result.buffer_mut();
    let result_buffer = Arc::get_mut(buffer).expect("buffer must be unique");
    let result_ptr = result_buffer.as_mut_ptr() as *mut T;

    let axis_len = shape[axis];
    let axis_stride = arr.strides()[axis];
    let itemsize = std::mem::size_of::<T>() as isize;

    // Check if axis is contiguous (common case: cumsum along last axis of C-contiguous array)
    if axis_stride == itemsize && arr.is_c_contiguous() {
        // Contiguous axis: process each lane directly
        let outer_size = size / axis_len;
        let src_ptr = arr.data_ptr() as *const T;

        for lane in 0..outer_size {
            let lane_base = lane * axis_len;
            let src_slice = unsafe { std::slice::from_raw_parts(src_ptr.add(lane_base), axis_len) };
            let out_slice = unsafe { std::slice::from_raw_parts_mut(result_ptr.add(lane_base), axis_len) };
            loops::cumulative(src_slice, out_slice, kernel);
        }
    } else {
        // General strided case: iterate over positions orthogonal to axis
        // For each starting position, walk along axis with strides
        let ndim = shape.len();
        let strides = arr.strides();

        // Compute outer shape (shape without axis dimension)
        let mut outer_shape: Vec<usize> = shape[..axis].to_vec();
        outer_shape.extend_from_slice(&shape[axis + 1..]);
        if outer_shape.is_empty() {
            outer_shape = vec![1];
        }
        let outer_size: usize = outer_shape.iter().product();

        // For each position in outer dimensions
        let mut outer_indices = vec![0usize; outer_shape.len()];
        for _ in 0..outer_size {
            // Compute base offset for this outer position
            let mut base_offset: isize = 0;
            let mut outer_idx = 0;
            for d in 0..ndim {
                if d != axis {
                    base_offset += outer_indices[outer_idx] as isize * strides[d];
                    outer_idx += 1;
                }
            }

            // Walk along axis, accumulating
            let mut acc = K::init();
            for j in 0..axis_len {
                let src_offset = base_offset + j as isize * axis_stride;
                let v = unsafe { *(arr.data_ptr() as *const T).byte_offset(src_offset) };
                acc = K::combine(acc, v);

                // Write to result at same logical position
                // Result is C-contiguous, so compute flat index
                let mut flat_idx = 0usize;
                let mut stride = 1usize;
                for d in (0..ndim).rev() {
                    let idx = if d == axis { j } else if d < axis { outer_indices[d] } else { outer_indices[d - 1] };
                    flat_idx += idx * stride;
                    stride *= shape[d];
                }
                unsafe { *result_ptr.add(flat_idx) = acc; }
            }

            crate::array::increment_indices(&mut outer_indices, &outer_shape);
        }
    }

    Some(result)
}
