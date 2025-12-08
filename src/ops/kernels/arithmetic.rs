//! Arithmetic kernels: Add, Sub, Mul, Div, etc.

use super::{BinaryKernel, ReduceKernel};

// ============================================================================
// Kernel structs (zero-sized types)
// ============================================================================

#[derive(Clone, Copy)]
pub struct Add;

#[derive(Clone, Copy)]
pub struct Sub;

#[derive(Clone, Copy)]
pub struct Mul;

#[derive(Clone, Copy)]
pub struct Div;

#[derive(Clone, Copy)]
pub struct Sum;

#[derive(Clone, Copy)]
pub struct Prod;

#[derive(Clone, Copy)]
pub struct Max;

#[derive(Clone, Copy)]
pub struct Min;

#[derive(Clone, Copy)]
pub struct Pow;

#[derive(Clone, Copy)]
pub struct Mod;

#[derive(Clone, Copy)]
pub struct FloorDiv;

#[derive(Clone, Copy)]
pub struct Maximum;

#[derive(Clone, Copy)]
pub struct Minimum;

#[derive(Clone, Copy)]
pub struct Arctan2;

#[derive(Clone, Copy)]
pub struct Hypot;

#[derive(Clone, Copy)]
pub struct FMax;

#[derive(Clone, Copy)]
pub struct FMin;

#[derive(Clone, Copy)]
pub struct Copysign;

#[derive(Clone, Copy)]
pub struct Logaddexp;

#[derive(Clone, Copy)]
pub struct Logaddexp2;

#[derive(Clone, Copy)]
pub struct Nextafter;

// NaN-aware reduction kernels (skip NaN values)
#[derive(Clone, Copy)]
pub struct NanSum;

#[derive(Clone, Copy)]
pub struct NanProd;

#[derive(Clone, Copy)]
pub struct NanMax;

#[derive(Clone, Copy)]
pub struct NanMin;

// ============================================================================
// Float implementations
// ============================================================================

macro_rules! impl_float_arithmetic {
    ($T:ty) => {
        impl BinaryKernel<$T> for Add {
            #[inline(always)]
            fn apply(a: $T, b: $T) -> $T { a + b }
        }
        impl BinaryKernel<$T> for Sub {
            #[inline(always)]
            fn apply(a: $T, b: $T) -> $T { a - b }
        }
        impl BinaryKernel<$T> for Mul {
            #[inline(always)]
            fn apply(a: $T, b: $T) -> $T { a * b }
        }
        impl BinaryKernel<$T> for Div {
            #[inline(always)]
            fn apply(a: $T, b: $T) -> $T { a / b }
        }
        impl ReduceKernel<$T> for Sum {
            #[inline(always)]
            fn init() -> $T { 0.0 }
            #[inline(always)]
            fn combine(acc: $T, v: $T) -> $T { acc + v }
        }
        impl ReduceKernel<$T> for Prod {
            #[inline(always)]
            fn init() -> $T { 1.0 }
            #[inline(always)]
            fn combine(acc: $T, v: $T) -> $T { acc * v }
        }
        impl ReduceKernel<$T> for Max {
            #[inline(always)]
            fn init() -> $T { <$T>::NEG_INFINITY }
            #[inline(always)]
            fn combine(acc: $T, v: $T) -> $T { if v > acc { v } else { acc } }
        }
        impl ReduceKernel<$T> for Min {
            #[inline(always)]
            fn init() -> $T { <$T>::INFINITY }
            #[inline(always)]
            fn combine(acc: $T, v: $T) -> $T { if v < acc { v } else { acc } }
        }
        impl BinaryKernel<$T> for Pow {
            #[inline(always)]
            fn apply(a: $T, b: $T) -> $T { a.powf(b) }
        }
        impl BinaryKernel<$T> for Mod {
            #[inline(always)]
            fn apply(a: $T, b: $T) -> $T { a % b }
        }
        impl BinaryKernel<$T> for FloorDiv {
            #[inline(always)]
            fn apply(a: $T, b: $T) -> $T { (a / b).floor() }
        }
        impl BinaryKernel<$T> for Maximum {
            #[inline(always)]
            fn apply(a: $T, b: $T) -> $T {
                // Propagate NaN like NumPy
                if a.is_nan() || b.is_nan() { <$T>::NAN } else { a.max(b) }
            }
        }
        impl BinaryKernel<$T> for Minimum {
            #[inline(always)]
            fn apply(a: $T, b: $T) -> $T {
                // Propagate NaN like NumPy
                if a.is_nan() || b.is_nan() { <$T>::NAN } else { a.min(b) }
            }
        }
        impl BinaryKernel<$T> for Arctan2 {
            #[inline(always)]
            fn apply(a: $T, b: $T) -> $T { a.atan2(b) }
        }
        impl BinaryKernel<$T> for Hypot {
            #[inline(always)]
            fn apply(a: $T, b: $T) -> $T { a.hypot(b) }
        }
        impl BinaryKernel<$T> for FMax {
            #[inline(always)]
            fn apply(a: $T, b: $T) -> $T {
                if b.is_nan() { a } else if a.is_nan() { b } else { a.max(b) }
            }
        }
        impl BinaryKernel<$T> for FMin {
            #[inline(always)]
            fn apply(a: $T, b: $T) -> $T {
                if b.is_nan() { a } else if a.is_nan() { b } else { a.min(b) }
            }
        }
        impl BinaryKernel<$T> for Copysign {
            #[inline(always)]
            fn apply(a: $T, b: $T) -> $T { a.copysign(b) }
        }
        impl BinaryKernel<$T> for Logaddexp {
            #[inline(always)]
            fn apply(a: $T, b: $T) -> $T {
                let max = a.max(b);
                if max.is_infinite() { max } else { max + ((-((a - b).abs())).exp()).ln_1p() }
            }
        }
        impl BinaryKernel<$T> for Logaddexp2 {
            #[inline(always)]
            fn apply(a: $T, b: $T) -> $T {
                let max = a.max(b);
                if max.is_infinite() { max } else { max + (2.0 as $T).powf(-(a - b).abs()).ln_1p() / (2.0 as $T).ln() }
            }
        }
        impl BinaryKernel<$T> for Nextafter {
            #[inline(always)]
            fn apply(a: $T, b: $T) -> $T {
                if a == b { b }
                else if a.is_nan() || b.is_nan() { <$T>::NAN }
                else if a < b { <$T>::from_bits(a.to_bits().wrapping_add(1)) }
                else { <$T>::from_bits(a.to_bits().wrapping_sub(1)) }
            }
        }
    };
}

impl_float_arithmetic!(f64);
impl_float_arithmetic!(f32);

// ============================================================================
// NaN-aware reduction kernels (skip NaN values)
// ============================================================================

macro_rules! impl_nan_reduce {
    ($T:ty) => {
        impl ReduceKernel<$T> for NanSum {
            #[inline(always)]
            fn init() -> $T { 0.0 }
            #[inline(always)]
            fn combine(acc: $T, v: $T) -> $T {
                if v.is_nan() { acc } else { acc + v }
            }
        }
        impl ReduceKernel<$T> for NanProd {
            #[inline(always)]
            fn init() -> $T { 1.0 }
            #[inline(always)]
            fn combine(acc: $T, v: $T) -> $T {
                if v.is_nan() { acc } else { acc * v }
            }
        }
        impl ReduceKernel<$T> for NanMax {
            #[inline(always)]
            fn init() -> $T { <$T>::NEG_INFINITY }
            #[inline(always)]
            fn combine(acc: $T, v: $T) -> $T {
                if v.is_nan() { acc } else if v > acc { v } else { acc }
            }
        }
        impl ReduceKernel<$T> for NanMin {
            #[inline(always)]
            fn init() -> $T { <$T>::INFINITY }
            #[inline(always)]
            fn combine(acc: $T, v: $T) -> $T {
                if v.is_nan() { acc } else if v < acc { v } else { acc }
            }
        }
    };
}

impl_nan_reduce!(f64);
impl_nan_reduce!(f32);

// ============================================================================
// Float16 implementations (convert to f32, compute, convert back)
// ============================================================================

use half::f16;

impl BinaryKernel<f16> for Add {
    #[inline(always)]
    fn apply(a: f16, b: f16) -> f16 { f16::from_f32(a.to_f32() + b.to_f32()) }
}
impl BinaryKernel<f16> for Sub {
    #[inline(always)]
    fn apply(a: f16, b: f16) -> f16 { f16::from_f32(a.to_f32() - b.to_f32()) }
}
impl BinaryKernel<f16> for Mul {
    #[inline(always)]
    fn apply(a: f16, b: f16) -> f16 { f16::from_f32(a.to_f32() * b.to_f32()) }
}
impl BinaryKernel<f16> for Div {
    #[inline(always)]
    fn apply(a: f16, b: f16) -> f16 { f16::from_f32(a.to_f32() / b.to_f32()) }
}
impl ReduceKernel<f16> for Sum {
    #[inline(always)]
    fn init() -> f16 { f16::ZERO }
    #[inline(always)]
    fn combine(acc: f16, v: f16) -> f16 { f16::from_f32(acc.to_f32() + v.to_f32()) }
}
impl ReduceKernel<f16> for Prod {
    #[inline(always)]
    fn init() -> f16 { f16::ONE }
    #[inline(always)]
    fn combine(acc: f16, v: f16) -> f16 { f16::from_f32(acc.to_f32() * v.to_f32()) }
}
impl ReduceKernel<f16> for Max {
    #[inline(always)]
    fn init() -> f16 { f16::NEG_INFINITY }
    #[inline(always)]
    fn combine(acc: f16, v: f16) -> f16 { if v > acc { v } else { acc } }
}
impl ReduceKernel<f16> for Min {
    #[inline(always)]
    fn init() -> f16 { f16::INFINITY }
    #[inline(always)]
    fn combine(acc: f16, v: f16) -> f16 { if v < acc { v } else { acc } }
}
impl BinaryKernel<f16> for Pow {
    #[inline(always)]
    fn apply(a: f16, b: f16) -> f16 { f16::from_f32(a.to_f32().powf(b.to_f32())) }
}
impl BinaryKernel<f16> for Mod {
    #[inline(always)]
    fn apply(a: f16, b: f16) -> f16 { f16::from_f32(a.to_f32() % b.to_f32()) }
}
impl BinaryKernel<f16> for FloorDiv {
    #[inline(always)]
    fn apply(a: f16, b: f16) -> f16 { f16::from_f32((a.to_f32() / b.to_f32()).floor()) }
}
impl BinaryKernel<f16> for Maximum {
    #[inline(always)]
    fn apply(a: f16, b: f16) -> f16 {
        let af = a.to_f32();
        let bf = b.to_f32();
        if af.is_nan() || bf.is_nan() { f16::NAN } else { f16::from_f32(af.max(bf)) }
    }
}
impl BinaryKernel<f16> for Minimum {
    #[inline(always)]
    fn apply(a: f16, b: f16) -> f16 {
        let af = a.to_f32();
        let bf = b.to_f32();
        if af.is_nan() || bf.is_nan() { f16::NAN } else { f16::from_f32(af.min(bf)) }
    }
}
impl BinaryKernel<f16> for Arctan2 {
    #[inline(always)]
    fn apply(a: f16, b: f16) -> f16 { f16::from_f32(a.to_f32().atan2(b.to_f32())) }
}
impl BinaryKernel<f16> for Hypot {
    #[inline(always)]
    fn apply(a: f16, b: f16) -> f16 { f16::from_f32(a.to_f32().hypot(b.to_f32())) }
}
impl BinaryKernel<f16> for FMax {
    #[inline(always)]
    fn apply(a: f16, b: f16) -> f16 {
        let af = a.to_f32();
        let bf = b.to_f32();
        if bf.is_nan() { a } else if af.is_nan() { b } else { f16::from_f32(af.max(bf)) }
    }
}
impl BinaryKernel<f16> for FMin {
    #[inline(always)]
    fn apply(a: f16, b: f16) -> f16 {
        let af = a.to_f32();
        let bf = b.to_f32();
        if bf.is_nan() { a } else if af.is_nan() { b } else { f16::from_f32(af.min(bf)) }
    }
}
impl BinaryKernel<f16> for Copysign {
    #[inline(always)]
    fn apply(a: f16, b: f16) -> f16 { f16::from_f32(a.to_f32().copysign(b.to_f32())) }
}
impl BinaryKernel<f16> for Logaddexp {
    #[inline(always)]
    fn apply(a: f16, b: f16) -> f16 {
        let af = a.to_f32();
        let bf = b.to_f32();
        let max = af.max(bf);
        if max.is_infinite() { f16::from_f32(max) } else { f16::from_f32(max + ((-((af - bf).abs())).exp()).ln_1p()) }
    }
}
impl BinaryKernel<f16> for Logaddexp2 {
    #[inline(always)]
    fn apply(a: f16, b: f16) -> f16 {
        let af = a.to_f32();
        let bf = b.to_f32();
        let max = af.max(bf);
        if max.is_infinite() { f16::from_f32(max) } else { f16::from_f32(max + 2.0f32.powf(-(af - bf).abs()).ln_1p() / 2.0f32.ln()) }
    }
}
impl BinaryKernel<f16> for Nextafter {
    #[inline(always)]
    fn apply(a: f16, b: f16) -> f16 {
        if a == b { b }
        else if a.is_nan() || b.is_nan() { f16::NAN }
        else if a < b { f16::from_bits(a.to_bits().wrapping_add(1)) }
        else { f16::from_bits(a.to_bits().wrapping_sub(1)) }
    }
}

// ============================================================================
// Integer implementations (wrapping arithmetic)
// ============================================================================

macro_rules! impl_int_arithmetic {
    ($T:ty, $zero:expr, $one:expr) => {
        impl BinaryKernel<$T> for Add {
            #[inline(always)]
            fn apply(a: $T, b: $T) -> $T { a.wrapping_add(b) }
        }
        impl BinaryKernel<$T> for Sub {
            #[inline(always)]
            fn apply(a: $T, b: $T) -> $T { a.wrapping_sub(b) }
        }
        impl BinaryKernel<$T> for Mul {
            #[inline(always)]
            fn apply(a: $T, b: $T) -> $T { a.wrapping_mul(b) }
        }
        impl BinaryKernel<$T> for Div {
            #[inline(always)]
            fn apply(a: $T, b: $T) -> $T {
                if b == $zero { $zero } else { a.wrapping_div(b) }
            }
        }
        impl ReduceKernel<$T> for Sum {
            #[inline(always)]
            fn init() -> $T { $zero }
            #[inline(always)]
            fn combine(acc: $T, v: $T) -> $T { acc.wrapping_add(v) }
        }
        impl ReduceKernel<$T> for Prod {
            #[inline(always)]
            fn init() -> $T { $one }
            #[inline(always)]
            fn combine(acc: $T, v: $T) -> $T { acc.wrapping_mul(v) }
        }
        impl ReduceKernel<$T> for Max {
            #[inline(always)]
            fn init() -> $T { <$T>::MIN }
            #[inline(always)]
            fn combine(acc: $T, v: $T) -> $T { if v > acc { v } else { acc } }
        }
        impl ReduceKernel<$T> for Min {
            #[inline(always)]
            fn init() -> $T { <$T>::MAX }
            #[inline(always)]
            fn combine(acc: $T, v: $T) -> $T { if v < acc { v } else { acc } }
        }
        impl BinaryKernel<$T> for Mod {
            #[inline(always)]
            fn apply(a: $T, b: $T) -> $T { if b == $zero { $zero } else { a % b } }
        }
        impl BinaryKernel<$T> for FloorDiv {
            #[inline(always)]
            fn apply(a: $T, b: $T) -> $T { if b == $zero { $zero } else { a / b } }
        }
        impl BinaryKernel<$T> for Maximum {
            #[inline(always)]
            fn apply(a: $T, b: $T) -> $T { if a > b { a } else { b } }
        }
        impl BinaryKernel<$T> for Minimum {
            #[inline(always)]
            fn apply(a: $T, b: $T) -> $T { if a < b { a } else { b } }
        }
    };
}

impl_int_arithmetic!(i64, 0i64, 1i64);
impl_int_arithmetic!(i32, 0i32, 1i32);
impl_int_arithmetic!(i16, 0i16, 1i16);
impl_int_arithmetic!(u64, 0u64, 1u64);
impl_int_arithmetic!(u32, 0u32, 1u32);
impl_int_arithmetic!(u16, 0u16, 1u16);
impl_int_arithmetic!(u8, 0u8, 1u8);

// ============================================================================
// Complex implementations
// ============================================================================

use num_complex::Complex;

macro_rules! impl_complex_arithmetic {
    ($F:ty) => {
        impl BinaryKernel<Complex<$F>> for Add {
            #[inline(always)]
            fn apply(a: Complex<$F>, b: Complex<$F>) -> Complex<$F> { a + b }
        }
        impl BinaryKernel<Complex<$F>> for Sub {
            #[inline(always)]
            fn apply(a: Complex<$F>, b: Complex<$F>) -> Complex<$F> { a - b }
        }
        impl BinaryKernel<Complex<$F>> for Mul {
            #[inline(always)]
            fn apply(a: Complex<$F>, b: Complex<$F>) -> Complex<$F> { a * b }
        }
        impl BinaryKernel<Complex<$F>> for Div {
            #[inline(always)]
            fn apply(a: Complex<$F>, b: Complex<$F>) -> Complex<$F> { a / b }
        }
        impl ReduceKernel<Complex<$F>> for Sum {
            #[inline(always)]
            fn init() -> Complex<$F> { Complex::new(0.0 as $F, 0.0 as $F) }
            #[inline(always)]
            fn combine(acc: Complex<$F>, v: Complex<$F>) -> Complex<$F> { acc + v }
        }
        impl ReduceKernel<Complex<$F>> for Prod {
            #[inline(always)]
            fn init() -> Complex<$F> { Complex::new(1.0 as $F, 0.0 as $F) }
            #[inline(always)]
            fn combine(acc: Complex<$F>, v: Complex<$F>) -> Complex<$F> { acc * v }
        }
        impl BinaryKernel<Complex<$F>> for Pow {
            #[inline(always)]
            fn apply(a: Complex<$F>, b: Complex<$F>) -> Complex<$F> { a.powc(b) }
        }
        impl BinaryKernel<Complex<$F>> for Mod {
            #[inline(always)]
            fn apply(_a: Complex<$F>, _b: Complex<$F>) -> Complex<$F> {
                // Mod not defined for complex - return NaN
                Complex::new(<$F>::NAN, <$F>::NAN)
            }
        }
        impl BinaryKernel<Complex<$F>> for FloorDiv {
            #[inline(always)]
            fn apply(_a: Complex<$F>, _b: Complex<$F>) -> Complex<$F> {
                // FloorDiv not defined for complex - return NaN
                Complex::new(<$F>::NAN, <$F>::NAN)
            }
        }
        impl BinaryKernel<Complex<$F>> for Maximum {
            #[inline(always)]
            fn apply(a: Complex<$F>, b: Complex<$F>) -> Complex<$F> {
                // NaN propagation
                if a.re.is_nan() || a.im.is_nan() || b.re.is_nan() || b.im.is_nan() {
                    Complex::new(<$F>::NAN, <$F>::NAN)
                } else if a.re > b.re || (a.re == b.re && a.im >= b.im) { a } else { b }
            }
        }
        impl BinaryKernel<Complex<$F>> for Minimum {
            #[inline(always)]
            fn apply(a: Complex<$F>, b: Complex<$F>) -> Complex<$F> {
                // NaN propagation
                if a.re.is_nan() || a.im.is_nan() || b.re.is_nan() || b.im.is_nan() {
                    Complex::new(<$F>::NAN, <$F>::NAN)
                } else if a.re < b.re || (a.re == b.re && a.im <= b.im) { a } else { b }
            }
        }
        impl BinaryKernel<Complex<$F>> for Arctan2 {
            #[inline(always)]
            fn apply(a: Complex<$F>, b: Complex<$F>) -> Complex<$F> {
                // atan2(y, x) = atan(y/x) for complex
                (a / b).atan()
            }
        }
        impl BinaryKernel<Complex<$F>> for Hypot {
            #[inline(always)]
            fn apply(a: Complex<$F>, b: Complex<$F>) -> Complex<$F> {
                // hypot for complex: sqrt(|a|^2 + |b|^2), real result
                let mag = (a.norm_sqr() + b.norm_sqr()).sqrt();
                Complex::new(mag, 0.0 as $F)
            }
        }
        impl BinaryKernel<Complex<$F>> for FMax {
            #[inline(always)]
            fn apply(a: Complex<$F>, b: Complex<$F>) -> Complex<$F> {
                // Ignore NaN
                let a_nan = a.re.is_nan() || a.im.is_nan();
                let b_nan = b.re.is_nan() || b.im.is_nan();
                if a_nan && b_nan { Complex::new(<$F>::NAN, <$F>::NAN) }
                else if a_nan { b }
                else if b_nan { a }
                else if a.re > b.re || (a.re == b.re && a.im >= b.im) { a } else { b }
            }
        }
        impl BinaryKernel<Complex<$F>> for FMin {
            #[inline(always)]
            fn apply(a: Complex<$F>, b: Complex<$F>) -> Complex<$F> {
                // Ignore NaN
                let a_nan = a.re.is_nan() || a.im.is_nan();
                let b_nan = b.re.is_nan() || b.im.is_nan();
                if a_nan && b_nan { Complex::new(<$F>::NAN, <$F>::NAN) }
                else if a_nan { b }
                else if b_nan { a }
                else if a.re < b.re || (a.re == b.re && a.im <= b.im) { a } else { b }
            }
        }
        impl BinaryKernel<Complex<$F>> for Copysign {
            #[inline(always)]
            fn apply(a: Complex<$F>, b: Complex<$F>) -> Complex<$F> {
                Complex::new(a.re.copysign(b.re), a.im.copysign(b.im))
            }
        }
        impl BinaryKernel<Complex<$F>> for Logaddexp {
            #[inline(always)]
            fn apply(_a: Complex<$F>, _b: Complex<$F>) -> Complex<$F> {
                Complex::new(<$F>::NAN, <$F>::NAN) // Not defined for complex
            }
        }
        impl BinaryKernel<Complex<$F>> for Logaddexp2 {
            #[inline(always)]
            fn apply(_a: Complex<$F>, _b: Complex<$F>) -> Complex<$F> {
                Complex::new(<$F>::NAN, <$F>::NAN) // Not defined for complex
            }
        }
        impl BinaryKernel<Complex<$F>> for Nextafter {
            #[inline(always)]
            fn apply(_a: Complex<$F>, b: Complex<$F>) -> Complex<$F> { b } // Just return b
        }
        // Max/Min reductions for complex: compare by magnitude
        impl ReduceKernel<Complex<$F>> for Max {
            #[inline(always)]
            fn init() -> Complex<$F> { Complex::new(<$F>::NEG_INFINITY, 0.0 as $F) }
            #[inline(always)]
            fn combine(acc: Complex<$F>, v: Complex<$F>) -> Complex<$F> {
                if v.norm_sqr() > acc.norm_sqr() { v } else { acc }
            }
        }
        impl ReduceKernel<Complex<$F>> for Min {
            #[inline(always)]
            fn init() -> Complex<$F> { Complex::new(<$F>::INFINITY, 0.0 as $F) }
            #[inline(always)]
            fn combine(acc: Complex<$F>, v: Complex<$F>) -> Complex<$F> {
                if v.norm_sqr() < acc.norm_sqr() { v } else { acc }
            }
        }
    };
}

impl_complex_arithmetic!(f64);
impl_complex_arithmetic!(f32);
