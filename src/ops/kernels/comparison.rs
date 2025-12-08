//! Comparison kernels: Gt, Lt, Ge, Le, Eq, Ne.

use super::CompareKernel;
use num_complex::Complex;

// ============================================================================
// Kernel structs (zero-sized types)
// ============================================================================

#[derive(Clone, Copy)]
pub struct Gt;

#[derive(Clone, Copy)]
pub struct Lt;

#[derive(Clone, Copy)]
pub struct Ge;

#[derive(Clone, Copy)]
pub struct Le;

#[derive(Clone, Copy)]
pub struct Eq;

#[derive(Clone, Copy)]
pub struct Ne;

// ============================================================================
// Implementations for PartialOrd + PartialEq types (floats and integers)
// ============================================================================

macro_rules! impl_ord_comparison {
    ($T:ty) => {
        impl CompareKernel<$T> for Gt {
            #[inline(always)]
            fn apply(a: $T, b: $T) -> bool { a > b }
        }
        impl CompareKernel<$T> for Lt {
            #[inline(always)]
            fn apply(a: $T, b: $T) -> bool { a < b }
        }
        impl CompareKernel<$T> for Ge {
            #[inline(always)]
            fn apply(a: $T, b: $T) -> bool { a >= b }
        }
        impl CompareKernel<$T> for Le {
            #[inline(always)]
            fn apply(a: $T, b: $T) -> bool { a <= b }
        }
        impl CompareKernel<$T> for Eq {
            #[inline(always)]
            fn apply(a: $T, b: $T) -> bool { a == b }
        }
        impl CompareKernel<$T> for Ne {
            #[inline(always)]
            fn apply(a: $T, b: $T) -> bool { a != b }
        }
    };
}

impl_ord_comparison!(f64);
impl_ord_comparison!(f32);
impl_ord_comparison!(i64);
impl_ord_comparison!(i32);
impl_ord_comparison!(i16);
impl_ord_comparison!(u64);
impl_ord_comparison!(u32);
impl_ord_comparison!(u16);
impl_ord_comparison!(u8);

// ============================================================================
// Complex implementations
// For complex numbers, Eq/Ne compare exactly, Gt/Lt/Ge/Le compare by magnitude
// (matching NumPy behavior where ordering is not well-defined for complex)
// ============================================================================

macro_rules! impl_complex_comparison {
    ($F:ty) => {
        impl CompareKernel<Complex<$F>> for Gt {
            #[inline(always)]
            fn apply(a: Complex<$F>, b: Complex<$F>) -> bool { a.norm_sqr() > b.norm_sqr() }
        }
        impl CompareKernel<Complex<$F>> for Lt {
            #[inline(always)]
            fn apply(a: Complex<$F>, b: Complex<$F>) -> bool { a.norm_sqr() < b.norm_sqr() }
        }
        impl CompareKernel<Complex<$F>> for Ge {
            #[inline(always)]
            fn apply(a: Complex<$F>, b: Complex<$F>) -> bool { a.norm_sqr() >= b.norm_sqr() }
        }
        impl CompareKernel<Complex<$F>> for Le {
            #[inline(always)]
            fn apply(a: Complex<$F>, b: Complex<$F>) -> bool { a.norm_sqr() <= b.norm_sqr() }
        }
        impl CompareKernel<Complex<$F>> for Eq {
            #[inline(always)]
            fn apply(a: Complex<$F>, b: Complex<$F>) -> bool { a == b }
        }
        impl CompareKernel<Complex<$F>> for Ne {
            #[inline(always)]
            fn apply(a: Complex<$F>, b: Complex<$F>) -> bool { a != b }
        }
    };
}

impl_complex_comparison!(f64);
impl_complex_comparison!(f32);
