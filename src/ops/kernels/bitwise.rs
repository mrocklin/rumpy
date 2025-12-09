//! Bitwise operation kernels.
//!
//! Implements And, Or, Xor, LeftShift, RightShift for integer types,
//! and Not as a unary kernel.

use super::BinaryKernel;
use super::UnaryKernel;

// ============================================================================
// Binary bitwise kernels
// ============================================================================

#[derive(Clone, Copy)]
pub struct And;

#[derive(Clone, Copy)]
pub struct Or;

#[derive(Clone, Copy)]
pub struct Xor;

#[derive(Clone, Copy)]
pub struct LeftShift;

#[derive(Clone, Copy)]
pub struct RightShift;

// ============================================================================
// Unary bitwise kernel
// ============================================================================

#[derive(Clone, Copy)]
pub struct Not;

// ============================================================================
// Macro for integer implementations
// ============================================================================

macro_rules! impl_bitwise_int {
    ($T:ty) => {
        impl BinaryKernel<$T> for And {
            #[inline(always)]
            fn apply(a: $T, b: $T) -> $T {
                a & b
            }
        }

        impl BinaryKernel<$T> for Or {
            #[inline(always)]
            fn apply(a: $T, b: $T) -> $T {
                a | b
            }
        }

        impl BinaryKernel<$T> for Xor {
            #[inline(always)]
            fn apply(a: $T, b: $T) -> $T {
                a ^ b
            }
        }

        impl BinaryKernel<$T> for LeftShift {
            #[inline(always)]
            fn apply(a: $T, b: $T) -> $T {
                a.wrapping_shl(b as u32)
            }
        }

        impl BinaryKernel<$T> for RightShift {
            #[inline(always)]
            fn apply(a: $T, b: $T) -> $T {
                a.wrapping_shr(b as u32)
            }
        }

        impl UnaryKernel<$T> for Not {
            #[inline(always)]
            fn apply(v: $T) -> $T {
                !v
            }
        }
    };
}

impl_bitwise_int!(i64);
impl_bitwise_int!(i32);
impl_bitwise_int!(i16);
impl_bitwise_int!(i8);
impl_bitwise_int!(u64);
impl_bitwise_int!(u32);
impl_bitwise_int!(u16);
impl_bitwise_int!(u8);

// ============================================================================
// Bool implementations (logical operators)
// ============================================================================

impl BinaryKernel<bool> for And {
    #[inline(always)]
    fn apply(a: bool, b: bool) -> bool {
        a && b
    }
}

impl BinaryKernel<bool> for Or {
    #[inline(always)]
    fn apply(a: bool, b: bool) -> bool {
        a || b
    }
}

impl BinaryKernel<bool> for Xor {
    #[inline(always)]
    fn apply(a: bool, b: bool) -> bool {
        a != b
    }
}

impl UnaryKernel<bool> for Not {
    #[inline(always)]
    fn apply(v: bool) -> bool {
        !v
    }
}
