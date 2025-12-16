//! Contiguous memory traversal with SIMD-friendly loops.
//!
//! All contiguous operations go through these functions.
//! SIMD optimizations (when added) live here, not in individual kernels.

use crate::ops::kernels::{BinaryKernel, UnaryKernel, ReduceKernel, CompareKernel, PredicateKernel};

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

/// Map unary predicate operation over contiguous slices, writing to bool output.
/// Used for isnan, isinf, isfinite, signbit, isneginf, isposinf.
#[inline]
pub fn map_predicate<T: Copy, K: PredicateKernel<T>>(
    src: &[T],
    out: &mut [u8],
    _kernel: K,
) {
    debug_assert_eq!(src.len(), out.len());

    for i in 0..src.len() {
        out[i] = K::apply(src[i]) as u8;
    }
}

/// Reduce contiguous slice to single value.
///
/// Uses 8-accumulator pattern matching NumPy's pairwise summation approach.
/// This provides instruction-level parallelism and allows AVX vectorization.
#[inline]
pub fn reduce<T: Copy, K: ReduceKernel<T>>(data: &[T], _kernel: K) -> T {
    let n = data.len();
    if n == 0 {
        return K::init();
    }

    // Small arrays: simple loop
    if n < 8 {
        let mut acc = K::init();
        for &v in data {
            acc = K::combine(acc, v);
        }
        return acc;
    }

    // 8-accumulator pattern for ILP (matches NumPy's pairwise sum)
    // Use combine() for init to support NaN-skipping kernels
    let mut r = [
        K::combine(K::init(), data[0]),
        K::combine(K::init(), data[1]),
        K::combine(K::init(), data[2]),
        K::combine(K::init(), data[3]),
        K::combine(K::init(), data[4]),
        K::combine(K::init(), data[5]),
        K::combine(K::init(), data[6]),
        K::combine(K::init(), data[7]),
    ];

    // Process in chunks of 8
    let mut i = 8;
    let end = n - (n % 8);
    while i < end {
        r[0] = K::combine(r[0], data[i]);
        r[1] = K::combine(r[1], data[i + 1]);
        r[2] = K::combine(r[2], data[i + 2]);
        r[3] = K::combine(r[3], data[i + 3]);
        r[4] = K::combine(r[4], data[i + 4]);
        r[5] = K::combine(r[5], data[i + 5]);
        r[6] = K::combine(r[6], data[i + 6]);
        r[7] = K::combine(r[7], data[i + 7]);
        i += 8;
    }

    // Combine 8 accumulators in tree pattern (matches NumPy)
    let mut acc = K::combine(
        K::combine(K::combine(r[0], r[1]), K::combine(r[2], r[3])),
        K::combine(K::combine(r[4], r[5]), K::combine(r[6], r[7])),
    );

    // Handle remainder
    while i < n {
        acc = K::combine(acc, data[i]);
        i += 1;
    }

    acc
}

/// Cumulative operation over contiguous slice.
///
/// Writes all intermediate accumulator values to output.
/// out[i] = K::combine(out[i-1], data[i]) for i > 0
/// out[0] = K::combine(K::init(), data[0])
#[inline]
pub fn cumulative<T: Copy, K: ReduceKernel<T>>(data: &[T], out: &mut [T], _kernel: K) {
    debug_assert_eq!(data.len(), out.len());
    if data.is_empty() {
        return;
    }
    let mut acc = K::init();
    for i in 0..data.len() {
        acc = K::combine(acc, data[i]);
        out[i] = acc;
    }
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
