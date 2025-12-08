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
/// Uses 8-accumulator pattern matching NumPy's pairwise summation approach.
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
    if n == 0 {
        return K::init();
    }

    // Small arrays: simple loop
    if n < 8 {
        let mut acc = K::init();
        for i in 0..n {
            let v = *ptr.byte_offset(stride * i as isize);
            acc = K::combine(acc, v);
        }
        return acc;
    }

    // 8-accumulator pattern for ILP (matches NumPy's pairwise sum)
    // Use combine() for init to support NaN-skipping kernels
    let mut r = [
        K::combine(K::init(), *ptr),
        K::combine(K::init(), *ptr.byte_offset(stride)),
        K::combine(K::init(), *ptr.byte_offset(stride * 2)),
        K::combine(K::init(), *ptr.byte_offset(stride * 3)),
        K::combine(K::init(), *ptr.byte_offset(stride * 4)),
        K::combine(K::init(), *ptr.byte_offset(stride * 5)),
        K::combine(K::init(), *ptr.byte_offset(stride * 6)),
        K::combine(K::init(), *ptr.byte_offset(stride * 7)),
    ];

    // Process in chunks of 8
    let mut i = 8isize;
    let end = (n - (n % 8)) as isize;
    while i < end {
        r[0] = K::combine(r[0], *ptr.byte_offset(stride * i));
        r[1] = K::combine(r[1], *ptr.byte_offset(stride * (i + 1)));
        r[2] = K::combine(r[2], *ptr.byte_offset(stride * (i + 2)));
        r[3] = K::combine(r[3], *ptr.byte_offset(stride * (i + 3)));
        r[4] = K::combine(r[4], *ptr.byte_offset(stride * (i + 4)));
        r[5] = K::combine(r[5], *ptr.byte_offset(stride * (i + 5)));
        r[6] = K::combine(r[6], *ptr.byte_offset(stride * (i + 6)));
        r[7] = K::combine(r[7], *ptr.byte_offset(stride * (i + 7)));
        i += 8;
    }

    // Combine 8 accumulators in tree pattern (matches NumPy)
    let mut acc = K::combine(
        K::combine(K::combine(r[0], r[1]), K::combine(r[2], r[3])),
        K::combine(K::combine(r[4], r[5]), K::combine(r[6], r[7])),
    );

    // Handle remainder
    while i < n as isize {
        acc = K::combine(acc, *ptr.byte_offset(stride * i));
        i += 1;
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

/// Cumulative operation over strided array.
///
/// Writes all intermediate accumulator values to output.
///
/// # Safety
/// Caller must ensure pointers and strides are valid for n elements.
#[inline]
pub unsafe fn cumulative_strided<T: Copy, K: ReduceKernel<T>>(
    src_ptr: *const T,
    src_stride: isize,
    out_ptr: *mut T,
    out_stride: isize,
    n: usize,
    _kernel: K,
) {
    if n == 0 {
        return;
    }
    let mut acc = K::init();
    for i in 0..n {
        let offset = i as isize;
        acc = K::combine(acc, *src_ptr.byte_offset(src_stride * offset));
        *out_ptr.byte_offset(out_stride * offset) = acc;
    }
}
