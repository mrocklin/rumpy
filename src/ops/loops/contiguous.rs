//! Contiguous memory traversal with SIMD-friendly loops.
//!
//! All contiguous operations go through these functions.
//! SIMD optimizations (when added) live here, not in individual kernels.

use crate::ops::kernels::{BinaryKernel, UnaryKernel, ReduceKernel, CompareKernel};

/// Map binary operation over contiguous slices.
///
/// LLVM will auto-vectorize the simple loop. Explicit SIMD can be added here
/// without touching any kernel code.
#[inline]
pub fn map_binary<T: Copy, K: BinaryKernel<T>>(
    a: &[T],
    b: &[T],
    out: &mut [T],
    _kernel: K,
) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), out.len());

    for i in 0..a.len() {
        out[i] = K::apply(a[i], b[i]);
    }
}

/// Map unary operation over contiguous slices.
#[inline]
pub fn map_unary<T: Copy, K: UnaryKernel<T>>(
    src: &[T],
    out: &mut [T],
    _kernel: K,
) {
    debug_assert_eq!(src.len(), out.len());

    for i in 0..src.len() {
        out[i] = K::apply(src[i]);
    }
}

/// Map comparison operation over contiguous slices, writing to bool output.
#[inline]
pub fn map_compare<T: Copy, K: CompareKernel<T>>(
    a: &[T],
    b: &[T],
    out: &mut [u8],
    _kernel: K,
) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), out.len());

    for i in 0..a.len() {
        out[i] = K::apply(a[i], b[i]) as u8;
    }
}

/// Reduce contiguous slice to single value.
///
/// Uses 4-accumulator pattern for instruction-level parallelism.
/// This breaks associativity for floats but matches NumPy behavior.
#[inline]
pub fn reduce<T: Copy, K: ReduceKernel<T>>(data: &[T], _kernel: K) -> T {
    if data.is_empty() {
        return K::init();
    }

    // 4-accumulator pattern for ILP (instruction-level parallelism)
    let (mut s0, mut s1, mut s2, mut s3) = (K::init(), K::init(), K::init(), K::init());
    let chunks = data.chunks_exact(4);
    let remainder = chunks.remainder();

    for chunk in chunks {
        s0 = K::combine(s0, chunk[0]);
        s1 = K::combine(s1, chunk[1]);
        s2 = K::combine(s2, chunk[2]);
        s3 = K::combine(s3, chunk[3]);
    }

    // Combine the 4 accumulators
    let mut acc = K::combine(K::combine(s0, s1), K::combine(s2, s3));

    // Handle remainder
    for &v in remainder {
        acc = K::combine(acc, v);
    }

    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::kernels::arithmetic::{Add, Sum};

    #[test]
    fn test_map_binary_add_f64() {
        let a = vec![1.0f64, 2.0, 3.0, 4.0];
        let b = vec![10.0f64, 20.0, 30.0, 40.0];
        let mut out = vec![0.0f64; 4];
        map_binary(&a, &b, &mut out, Add);
        assert_eq!(out, vec![11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn test_reduce_sum_f64() {
        let data = vec![1.0f64, 2.0, 3.0, 4.0, 5.0];
        let result = reduce(&data, Sum);
        assert_eq!(result, 15.0);
    }

    #[test]
    fn test_reduce_empty() {
        let data: Vec<f64> = vec![];
        let result = reduce(&data, Sum);
        assert_eq!(result, 0.0);
    }
}
