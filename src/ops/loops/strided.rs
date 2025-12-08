//! Strided memory traversal for non-contiguous arrays.
//!
//! These functions handle arbitrary strides using pointer arithmetic.
//! SIMD is not applied here - strided access patterns don't benefit from it.

use crate::ops::kernels::{BinaryKernel, UnaryKernel, ReduceKernel, CompareKernel};

/// Map binary operation over strided arrays.
///
/// # Safety
/// Caller must ensure pointers and strides are valid for n elements.
#[inline]
pub unsafe fn map_binary_strided<T: Copy, K: BinaryKernel<T>>(
    a_ptr: *const T,
    a_stride: isize,
    b_ptr: *const T,
    b_stride: isize,
    out_ptr: *mut T,
    out_stride: isize,
    n: usize,
    _kernel: K,
) {
    for i in 0..n {
        let offset = i as isize;
        let a = *a_ptr.byte_offset(a_stride * offset);
        let b = *b_ptr.byte_offset(b_stride * offset);
        *out_ptr.byte_offset(out_stride * offset) = K::apply(a, b);
    }
}

/// Map unary operation over strided array.
///
/// # Safety
/// Caller must ensure pointers and strides are valid for n elements.
#[inline]
pub unsafe fn map_unary_strided<T: Copy, K: UnaryKernel<T>>(
    src_ptr: *const T,
    src_stride: isize,
    out_ptr: *mut T,
    out_stride: isize,
    n: usize,
    _kernel: K,
) {
    for i in 0..n {
        let offset = i as isize;
        let v = *src_ptr.byte_offset(src_stride * offset);
        *out_ptr.byte_offset(out_stride * offset) = K::apply(v);
    }
}

/// Reduce strided array to single value.
///
/// # Safety
/// Caller must ensure pointer and stride are valid for n elements.
#[inline]
pub unsafe fn reduce_strided<T: Copy, K: ReduceKernel<T>>(
    ptr: *const T,
    stride: isize,
    n: usize,
    _kernel: K,
) -> T {
    let mut acc = K::init();
    for i in 0..n {
        let v = *ptr.byte_offset(stride * i as isize);
        acc = K::combine(acc, v);
    }
    acc
}

/// Map comparison operation over strided arrays, writing to bool (u8) output.
///
/// # Safety
/// Caller must ensure pointers and strides are valid for n elements.
#[inline]
pub unsafe fn map_compare_strided<T: Copy, K: CompareKernel<T>>(
    a_ptr: *const T,
    a_stride: isize,
    b_ptr: *const T,
    b_stride: isize,
    out_ptr: *mut u8,
    n: usize,
    _kernel: K,
) {
    for i in 0..n {
        let offset = i as isize;
        let a = *a_ptr.byte_offset(a_stride * offset);
        let b = *b_ptr.byte_offset(b_stride * offset);
        *out_ptr.add(i) = K::apply(a, b) as u8;
    }
}
