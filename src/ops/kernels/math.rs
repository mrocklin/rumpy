//! Math kernels: Neg, Abs, Sqrt, Exp, Sin, etc.

use super::UnaryKernel;
use rustfft::num_traits::{Float, FloatConst};

// ============================================================================
// Kernel structs (zero-sized types)
// ============================================================================

#[derive(Clone, Copy)]
pub struct Neg;

#[derive(Clone, Copy)]
pub struct Abs;

#[derive(Clone, Copy)]
pub struct Sqrt;

#[derive(Clone, Copy)]
pub struct Exp;

#[derive(Clone, Copy)]
pub struct Log;

#[derive(Clone, Copy)]
pub struct Log10;

#[derive(Clone, Copy)]
pub struct Log2;

#[derive(Clone, Copy)]
pub struct Sin;

#[derive(Clone, Copy)]
pub struct Cos;

#[derive(Clone, Copy)]
pub struct Tan;

#[derive(Clone, Copy)]
pub struct Floor;

#[derive(Clone, Copy)]
pub struct Ceil;

#[derive(Clone, Copy)]
pub struct Square;

#[derive(Clone, Copy)]
pub struct Sinh;

#[derive(Clone, Copy)]
pub struct Cosh;

#[derive(Clone, Copy)]
pub struct Tanh;

#[derive(Clone, Copy)]
pub struct Arcsin;

#[derive(Clone, Copy)]
pub struct Arccos;

#[derive(Clone, Copy)]
pub struct Arctan;

#[derive(Clone, Copy)]
pub struct Sign;

#[derive(Clone, Copy)]
pub struct Isnan;

#[derive(Clone, Copy)]
pub struct Isinf;

#[derive(Clone, Copy)]
pub struct Isfinite;

#[derive(Clone, Copy)]
pub struct Positive;

#[derive(Clone, Copy)]
pub struct Reciprocal;

#[derive(Clone, Copy)]
pub struct Exp2;

#[derive(Clone, Copy)]
pub struct Expm1;

#[derive(Clone, Copy)]
pub struct Log1p;

#[derive(Clone, Copy)]
pub struct Cbrt;

#[derive(Clone, Copy)]
pub struct Trunc;

#[derive(Clone, Copy)]
pub struct Rint;

#[derive(Clone, Copy)]
pub struct Arcsinh;

#[derive(Clone, Copy)]
pub struct Arccosh;

#[derive(Clone, Copy)]
pub struct Arctanh;

#[derive(Clone, Copy)]
pub struct Signbit;

// ============================================================================
// Float implementations
// ============================================================================

macro_rules! impl_float_math {
    ($T:ty) => {
        impl UnaryKernel<$T> for Neg {
            #[inline(always)]
            fn apply(v: $T) -> $T { -v }
        }
        impl UnaryKernel<$T> for Abs {
            #[inline(always)]
            fn apply(v: $T) -> $T { v.abs() }
        }
        impl UnaryKernel<$T> for Sqrt {
            #[inline(always)]
            fn apply(v: $T) -> $T { v.sqrt() }
        }
        impl UnaryKernel<$T> for Exp {
            #[inline(always)]
            fn apply(v: $T) -> $T { v.exp() }
        }
        impl UnaryKernel<$T> for Log {
            #[inline(always)]
            fn apply(v: $T) -> $T { v.ln() }
        }
        impl UnaryKernel<$T> for Log10 {
            #[inline(always)]
            fn apply(v: $T) -> $T { v.log10() }
        }
        impl UnaryKernel<$T> for Log2 {
            #[inline(always)]
            fn apply(v: $T) -> $T { v.log2() }
        }
        impl UnaryKernel<$T> for Sin {
            #[inline(always)]
            fn apply(v: $T) -> $T { v.sin() }
        }
        impl UnaryKernel<$T> for Cos {
            #[inline(always)]
            fn apply(v: $T) -> $T { v.cos() }
        }
        impl UnaryKernel<$T> for Tan {
            #[inline(always)]
            fn apply(v: $T) -> $T { v.tan() }
        }
        impl UnaryKernel<$T> for Floor {
            #[inline(always)]
            fn apply(v: $T) -> $T { v.floor() }
        }
        impl UnaryKernel<$T> for Ceil {
            #[inline(always)]
            fn apply(v: $T) -> $T { v.ceil() }
        }
        impl UnaryKernel<$T> for Square {
            #[inline(always)]
            fn apply(v: $T) -> $T { v * v }
        }
        impl UnaryKernel<$T> for Sinh {
            #[inline(always)]
            fn apply(v: $T) -> $T { v.sinh() }
        }
        impl UnaryKernel<$T> for Cosh {
            #[inline(always)]
            fn apply(v: $T) -> $T { v.cosh() }
        }
        impl UnaryKernel<$T> for Tanh {
            #[inline(always)]
            fn apply(v: $T) -> $T { v.tanh() }
        }
        impl UnaryKernel<$T> for Arcsin {
            #[inline(always)]
            fn apply(v: $T) -> $T { v.asin() }
        }
        impl UnaryKernel<$T> for Arccos {
            #[inline(always)]
            fn apply(v: $T) -> $T { v.acos() }
        }
        impl UnaryKernel<$T> for Arctan {
            #[inline(always)]
            fn apply(v: $T) -> $T { v.atan() }
        }
        impl UnaryKernel<$T> for Sign {
            #[inline(always)]
            fn apply(v: $T) -> $T {
                if v.is_nan() { v } else if v > 0.0 { 1.0 } else if v < 0.0 { -1.0 } else { 0.0 }
            }
        }
        impl UnaryKernel<$T> for Positive {
            #[inline(always)]
            fn apply(v: $T) -> $T { v }
        }
        impl UnaryKernel<$T> for Reciprocal {
            #[inline(always)]
            fn apply(v: $T) -> $T { 1.0 / v }
        }
        impl UnaryKernel<$T> for Exp2 {
            #[inline(always)]
            fn apply(v: $T) -> $T { v.exp2() }
        }
        impl UnaryKernel<$T> for Expm1 {
            #[inline(always)]
            fn apply(v: $T) -> $T { v.exp_m1() }
        }
        impl UnaryKernel<$T> for Log1p {
            #[inline(always)]
            fn apply(v: $T) -> $T { v.ln_1p() }
        }
        impl UnaryKernel<$T> for Cbrt {
            #[inline(always)]
            fn apply(v: $T) -> $T { v.cbrt() }
        }
        impl UnaryKernel<$T> for Trunc {
            #[inline(always)]
            fn apply(v: $T) -> $T { v.trunc() }
        }
        impl UnaryKernel<$T> for Rint {
            #[inline(always)]
            fn apply(v: $T) -> $T { v.round() }
        }
        impl UnaryKernel<$T> for Arcsinh {
            #[inline(always)]
            fn apply(v: $T) -> $T { v.asinh() }
        }
        impl UnaryKernel<$T> for Arccosh {
            #[inline(always)]
            fn apply(v: $T) -> $T { v.acosh() }
        }
        impl UnaryKernel<$T> for Arctanh {
            #[inline(always)]
            fn apply(v: $T) -> $T { v.atanh() }
        }
    };
}

impl_float_math!(f64);
impl_float_math!(f32);

// ============================================================================
// Float16 implementations (convert to f32, compute, convert back)
// ============================================================================

use half::f16;

impl UnaryKernel<f16> for Neg {
    #[inline(always)]
    fn apply(v: f16) -> f16 { f16::from_f32(-v.to_f32()) }
}
impl UnaryKernel<f16> for Abs {
    #[inline(always)]
    fn apply(v: f16) -> f16 { f16::from_f32(v.to_f32().abs()) }
}
impl UnaryKernel<f16> for Sqrt {
    #[inline(always)]
    fn apply(v: f16) -> f16 { f16::from_f32(v.to_f32().sqrt()) }
}
impl UnaryKernel<f16> for Exp {
    #[inline(always)]
    fn apply(v: f16) -> f16 { f16::from_f32(v.to_f32().exp()) }
}
impl UnaryKernel<f16> for Log {
    #[inline(always)]
    fn apply(v: f16) -> f16 { f16::from_f32(v.to_f32().ln()) }
}
impl UnaryKernel<f16> for Log10 {
    #[inline(always)]
    fn apply(v: f16) -> f16 { f16::from_f32(v.to_f32().log10()) }
}
impl UnaryKernel<f16> for Log2 {
    #[inline(always)]
    fn apply(v: f16) -> f16 { f16::from_f32(v.to_f32().log2()) }
}
impl UnaryKernel<f16> for Sin {
    #[inline(always)]
    fn apply(v: f16) -> f16 { f16::from_f32(v.to_f32().sin()) }
}
impl UnaryKernel<f16> for Cos {
    #[inline(always)]
    fn apply(v: f16) -> f16 { f16::from_f32(v.to_f32().cos()) }
}
impl UnaryKernel<f16> for Tan {
    #[inline(always)]
    fn apply(v: f16) -> f16 { f16::from_f32(v.to_f32().tan()) }
}
impl UnaryKernel<f16> for Floor {
    #[inline(always)]
    fn apply(v: f16) -> f16 { f16::from_f32(v.to_f32().floor()) }
}
impl UnaryKernel<f16> for Ceil {
    #[inline(always)]
    fn apply(v: f16) -> f16 { f16::from_f32(v.to_f32().ceil()) }
}
impl UnaryKernel<f16> for Square {
    #[inline(always)]
    fn apply(v: f16) -> f16 { let vf = v.to_f32(); f16::from_f32(vf * vf) }
}
impl UnaryKernel<f16> for Sinh {
    #[inline(always)]
    fn apply(v: f16) -> f16 { f16::from_f32(v.to_f32().sinh()) }
}
impl UnaryKernel<f16> for Cosh {
    #[inline(always)]
    fn apply(v: f16) -> f16 { f16::from_f32(v.to_f32().cosh()) }
}
impl UnaryKernel<f16> for Tanh {
    #[inline(always)]
    fn apply(v: f16) -> f16 { f16::from_f32(v.to_f32().tanh()) }
}
impl UnaryKernel<f16> for Arcsin {
    #[inline(always)]
    fn apply(v: f16) -> f16 { f16::from_f32(v.to_f32().asin()) }
}
impl UnaryKernel<f16> for Arccos {
    #[inline(always)]
    fn apply(v: f16) -> f16 { f16::from_f32(v.to_f32().acos()) }
}
impl UnaryKernel<f16> for Arctan {
    #[inline(always)]
    fn apply(v: f16) -> f16 { f16::from_f32(v.to_f32().atan()) }
}
impl UnaryKernel<f16> for Sign {
    #[inline(always)]
    fn apply(v: f16) -> f16 {
        let vf = v.to_f32();
        if vf.is_nan() { v } else if vf > 0.0 { f16::ONE } else if vf < 0.0 { f16::NEG_ONE } else { f16::ZERO }
    }
}
impl UnaryKernel<f16> for Positive {
    #[inline(always)]
    fn apply(v: f16) -> f16 { v }
}
impl UnaryKernel<f16> for Reciprocal {
    #[inline(always)]
    fn apply(v: f16) -> f16 { f16::from_f32(1.0 / v.to_f32()) }
}
impl UnaryKernel<f16> for Exp2 {
    #[inline(always)]
    fn apply(v: f16) -> f16 { f16::from_f32(v.to_f32().exp2()) }
}
impl UnaryKernel<f16> for Expm1 {
    #[inline(always)]
    fn apply(v: f16) -> f16 { f16::from_f32(v.to_f32().exp_m1()) }
}
impl UnaryKernel<f16> for Log1p {
    #[inline(always)]
    fn apply(v: f16) -> f16 { f16::from_f32(v.to_f32().ln_1p()) }
}
impl UnaryKernel<f16> for Cbrt {
    #[inline(always)]
    fn apply(v: f16) -> f16 { f16::from_f32(v.to_f32().cbrt()) }
}
impl UnaryKernel<f16> for Trunc {
    #[inline(always)]
    fn apply(v: f16) -> f16 { f16::from_f32(v.to_f32().trunc()) }
}
impl UnaryKernel<f16> for Rint {
    #[inline(always)]
    fn apply(v: f16) -> f16 { f16::from_f32(v.to_f32().round()) }
}
impl UnaryKernel<f16> for Arcsinh {
    #[inline(always)]
    fn apply(v: f16) -> f16 { f16::from_f32(v.to_f32().asinh()) }
}
impl UnaryKernel<f16> for Arccosh {
    #[inline(always)]
    fn apply(v: f16) -> f16 { f16::from_f32(v.to_f32().acosh()) }
}
impl UnaryKernel<f16> for Arctanh {
    #[inline(always)]
    fn apply(v: f16) -> f16 { f16::from_f32(v.to_f32().atanh()) }
}

// ============================================================================
// Integer implementations
// ============================================================================

macro_rules! impl_signed_int_math {
    ($T:ty) => {
        impl UnaryKernel<$T> for Neg {
            #[inline(always)]
            fn apply(v: $T) -> $T { v.wrapping_neg() }
        }
        impl UnaryKernel<$T> for Abs {
            #[inline(always)]
            fn apply(v: $T) -> $T { v.abs() }
        }
        impl UnaryKernel<$T> for Square {
            #[inline(always)]
            fn apply(v: $T) -> $T { v.wrapping_mul(v) }
        }
        impl UnaryKernel<$T> for Sign {
            #[inline(always)]
            fn apply(v: $T) -> $T { v.signum() }
        }
        impl UnaryKernel<$T> for Positive {
            #[inline(always)]
            fn apply(v: $T) -> $T { v }
        }
    };
}

macro_rules! impl_unsigned_int_math {
    ($T:ty) => {
        impl UnaryKernel<$T> for Neg {
            #[inline(always)]
            fn apply(v: $T) -> $T { v.wrapping_neg() }
        }
        impl UnaryKernel<$T> for Abs {
            #[inline(always)]
            fn apply(v: $T) -> $T { v }  // unsigned already positive
        }
        impl UnaryKernel<$T> for Square {
            #[inline(always)]
            fn apply(v: $T) -> $T { v.wrapping_mul(v) }
        }
        impl UnaryKernel<$T> for Sign {
            #[inline(always)]
            fn apply(v: $T) -> $T { if v > 0 { 1 } else { 0 } }
        }
        impl UnaryKernel<$T> for Positive {
            #[inline(always)]
            fn apply(v: $T) -> $T { v }
        }
    };
}

impl_signed_int_math!(i64);
impl_signed_int_math!(i32);
impl_signed_int_math!(i16);
impl_signed_int_math!(i8);

impl_unsigned_int_math!(u64);
impl_unsigned_int_math!(u32);
impl_unsigned_int_math!(u16);
impl_unsigned_int_math!(u8);

// ============================================================================
// Complex implementations
// ============================================================================

use num_complex::Complex;

/// Complex square root that matches numpy's branch cut conventions.
/// The imaginary part of the result has the same sign as the input imaginary part.
/// This correctly handles signed zero: sqrt(-3-0j) != sqrt(-3+0j).
#[inline]
fn numpy_sqrt<F: Float + FloatConst>(z: Complex<F>) -> Complex<F> {
    let re = z.re;
    let im = z.im;
    if re == F::zero() && im == F::zero() {
        return Complex::new(F::zero(), im); // Preserve signed zero
    }
    let two = F::from(2.0).unwrap();
    let mag = (re * re + im * im).sqrt();
    let sqrt_r = ((mag + re) / two).sqrt();
    let sqrt_i = ((mag - re) / two).sqrt();
    // Sign of imaginary result matches sign of input imaginary (using is_sign_negative for -0.0)
    if im.is_sign_negative() {
        Complex::new(sqrt_r, -sqrt_i)
    } else {
        Complex::new(sqrt_r, sqrt_i)
    }
}

macro_rules! impl_complex_math {
    ($F:ty) => {
        impl UnaryKernel<Complex<$F>> for Neg {
            #[inline(always)]
            fn apply(v: Complex<$F>) -> Complex<$F> { -v }
        }
        impl UnaryKernel<Complex<$F>> for Abs {
            #[inline(always)]
            fn apply(v: Complex<$F>) -> Complex<$F> {
                // |z| is real, return as complex with zero imag
                Complex::new(v.norm(), 0.0 as $F)
            }
        }
        impl UnaryKernel<Complex<$F>> for Sqrt {
            #[inline(always)]
            fn apply(v: Complex<$F>) -> Complex<$F> { v.sqrt() }
        }
        impl UnaryKernel<Complex<$F>> for Exp {
            #[inline(always)]
            fn apply(v: Complex<$F>) -> Complex<$F> { v.exp() }
        }
        impl UnaryKernel<Complex<$F>> for Log {
            #[inline(always)]
            fn apply(v: Complex<$F>) -> Complex<$F> { v.ln() }
        }
        impl UnaryKernel<Complex<$F>> for Log10 {
            #[inline(always)]
            fn apply(v: Complex<$F>) -> Complex<$F> { v.log(10.0 as $F) }
        }
        impl UnaryKernel<Complex<$F>> for Log2 {
            #[inline(always)]
            fn apply(v: Complex<$F>) -> Complex<$F> { v.log(2.0 as $F) }
        }
        impl UnaryKernel<Complex<$F>> for Sin {
            #[inline(always)]
            fn apply(v: Complex<$F>) -> Complex<$F> { v.sin() }
        }
        impl UnaryKernel<Complex<$F>> for Cos {
            #[inline(always)]
            fn apply(v: Complex<$F>) -> Complex<$F> { v.cos() }
        }
        impl UnaryKernel<Complex<$F>> for Tan {
            #[inline(always)]
            fn apply(v: Complex<$F>) -> Complex<$F> { v.tan() }
        }
        impl UnaryKernel<Complex<$F>> for Floor {
            #[inline(always)]
            fn apply(v: Complex<$F>) -> Complex<$F> {
                Complex::new(v.re.floor(), v.im.floor())
            }
        }
        impl UnaryKernel<Complex<$F>> for Ceil {
            #[inline(always)]
            fn apply(v: Complex<$F>) -> Complex<$F> {
                Complex::new(v.re.ceil(), v.im.ceil())
            }
        }
        impl UnaryKernel<Complex<$F>> for Square {
            #[inline(always)]
            fn apply(v: Complex<$F>) -> Complex<$F> { v * v }
        }
        impl UnaryKernel<Complex<$F>> for Sinh {
            #[inline(always)]
            fn apply(v: Complex<$F>) -> Complex<$F> { v.sinh() }
        }
        impl UnaryKernel<Complex<$F>> for Cosh {
            #[inline(always)]
            fn apply(v: Complex<$F>) -> Complex<$F> { v.cosh() }
        }
        impl UnaryKernel<Complex<$F>> for Tanh {
            #[inline(always)]
            fn apply(v: Complex<$F>) -> Complex<$F> { v.tanh() }
        }
        impl UnaryKernel<Complex<$F>> for Arcsin {
            #[inline(always)]
            fn apply(v: Complex<$F>) -> Complex<$F> {
                // arcsin(z) = -i * log(iz + sqrt(1 - z^2))
                // Using numpy-compatible branch cuts
                //
                // Key issue: when computing 1 - z^2 for real z outside [-1,1],
                // the imaginary part should be -0.0 (negative zero) to get correct
                // sqrt branch. But complex arithmetic loses this sign.
                //
                // For real z with |z| > 1: 1-z^2 is negative real.
                // The sqrt branch cut gives sqrt(negative) = +i*sqrt(|negative|)
                // when computed from above the branch cut, but the formula requires
                // we approach from below for z > 1.
                //
                // Solution: Manually compute with proper signed zero handling.
                let one = 1.0 as $F;
                let zero = 0.0 as $F;

                // z^2 = (re^2 - im^2, 2*re*im)
                let z2_re = v.re * v.re - v.im * v.im;
                let z2_im = (2.0 as $F) * v.re * v.im;

                // 1 - z^2 = (1 - z2_re, -z2_im)
                // The key is: for z = x+0j with x > 0:
                //   z^2 = x^2 + 0j, z2_im = 0
                //   -z2_im should be -0.0 to preserve the sign
                let one_minus_z2_re = one - z2_re;
                let one_minus_z2_im = -z2_im;  // This preserves -0.0 when z2_im = +0.0

                let one_minus_z2 = Complex::new(one_minus_z2_re, one_minus_z2_im);
                let sqrt_val = numpy_sqrt(one_minus_z2);

                // iz = (-im, re)
                let iz = Complex::new(-v.im, v.re);

                let log_arg = iz + sqrt_val;
                let i = Complex::new(zero, one);
                -i * log_arg.ln()
            }
        }
        impl UnaryKernel<Complex<$F>> for Arccos {
            #[inline(always)]
            fn apply(v: Complex<$F>) -> Complex<$F> {
                // arccos(z) = pi/2 - arcsin(z)
                let pi_over_2 = <$F as FloatConst>::FRAC_PI_2();
                let arcsin_v = <Arcsin as UnaryKernel<Complex<$F>>>::apply(v);
                Complex::new(pi_over_2, 0.0 as $F) - arcsin_v
            }
        }
        impl UnaryKernel<Complex<$F>> for Arctan {
            #[inline(always)]
            fn apply(v: Complex<$F>) -> Complex<$F> { v.atan() }
        }
        impl UnaryKernel<Complex<$F>> for Sign {
            #[inline(always)]
            fn apply(v: Complex<$F>) -> Complex<$F> {
                let mag = v.norm();
                if mag == 0.0 { Complex::new(0.0 as $F, 0.0 as $F) } else { v / mag }
            }
        }
        impl UnaryKernel<Complex<$F>> for Positive {
            #[inline(always)]
            fn apply(v: Complex<$F>) -> Complex<$F> { v }
        }
        impl UnaryKernel<Complex<$F>> for Reciprocal {
            #[inline(always)]
            fn apply(v: Complex<$F>) -> Complex<$F> { v.inv() }
        }
        impl UnaryKernel<Complex<$F>> for Exp2 {
            #[inline(always)]
            fn apply(v: Complex<$F>) -> Complex<$F> {
                // 2^z = exp(z * ln(2))
                (v * (2.0 as $F).ln()).exp()
            }
        }
        impl UnaryKernel<Complex<$F>> for Expm1 {
            #[inline(always)]
            fn apply(v: Complex<$F>) -> Complex<$F> { v.exp() - 1.0 }
        }
        impl UnaryKernel<Complex<$F>> for Log1p {
            #[inline(always)]
            fn apply(v: Complex<$F>) -> Complex<$F> { (v + 1.0).ln() }
        }
        impl UnaryKernel<Complex<$F>> for Cbrt {
            #[inline(always)]
            fn apply(v: Complex<$F>) -> Complex<$F> {
                // z^(1/3) via powc
                v.powf(1.0 / 3.0)
            }
        }
        impl UnaryKernel<Complex<$F>> for Trunc {
            #[inline(always)]
            fn apply(v: Complex<$F>) -> Complex<$F> {
                Complex::new(v.re.trunc(), v.im.trunc())
            }
        }
        impl UnaryKernel<Complex<$F>> for Rint {
            #[inline(always)]
            fn apply(v: Complex<$F>) -> Complex<$F> {
                Complex::new(v.re.round(), v.im.round())
            }
        }
        impl UnaryKernel<Complex<$F>> for Arcsinh {
            #[inline(always)]
            fn apply(v: Complex<$F>) -> Complex<$F> {
                // arcsinh(z) = log(z + sqrt(z^2 + 1))
                let z2_plus_1 = v * v + Complex::new(1.0 as $F, 0.0 as $F);
                let sqrt_val = numpy_sqrt(z2_plus_1);
                (v + sqrt_val).ln()
            }
        }
        impl UnaryKernel<Complex<$F>> for Arccosh {
            #[inline(always)]
            fn apply(v: Complex<$F>) -> Complex<$F> {
                // arccosh(z) = log(z + sqrt(z^2 - 1))
                // For real x in [0, 1], result should be purely imaginary
                let z2_minus_1 = v * v - Complex::new(1.0 as $F, 0.0 as $F);
                let sqrt_val = numpy_sqrt(z2_minus_1);
                (v + sqrt_val).ln()
            }
        }
        impl UnaryKernel<Complex<$F>> for Arctanh {
            #[inline(always)]
            fn apply(v: Complex<$F>) -> Complex<$F> {
                // arctanh(z) = 0.5 * log((1+z)/(1-z))
                // For numpy compatibility, both z>1 and z<-1 give positive imaginary part.
                let one = 1.0 as $F;
                let half = 0.5 as $F;

                let num_re = one + v.re;
                let num_im = v.im;
                let den_re = one - v.re;
                let den_im = -v.im;

                // Complex division
                let den_mag2 = den_re * den_re + den_im * den_im;
                let ratio_re = (num_re * den_re + num_im * den_im) / den_mag2;
                let ratio_im = (num_im * den_re - num_re * den_im) / den_mag2;

                let ratio = Complex::new(ratio_re, ratio_im);

                // For negative real ratio, use +pi branch for numpy compatibility
                let log_val = if ratio_re < 0.0 as $F && ratio_im == 0.0 as $F {
                    Complex::new((-ratio_re).ln(), <$F as FloatConst>::PI())
                } else {
                    ratio.ln()
                };

                Complex::new(half * log_val.re, half * log_val.im)
            }
        }
    };
}

impl_complex_math!(f64);
impl_complex_math!(f32);

// ============================================================================
// Predicate kernels (return bool)
// ============================================================================

use super::PredicateKernel;

#[derive(Clone, Copy)]
pub struct Isneginf;

#[derive(Clone, Copy)]
pub struct Isposinf;

// Float predicate implementations
macro_rules! impl_float_predicates {
    ($T:ty) => {
        impl PredicateKernel<$T> for Isnan {
            #[inline(always)]
            fn apply(v: $T) -> bool { v.is_nan() }
        }
        impl PredicateKernel<$T> for Isinf {
            #[inline(always)]
            fn apply(v: $T) -> bool { v.is_infinite() }
        }
        impl PredicateKernel<$T> for Isfinite {
            #[inline(always)]
            fn apply(v: $T) -> bool { v.is_finite() }
        }
        impl PredicateKernel<$T> for Signbit {
            #[inline(always)]
            fn apply(v: $T) -> bool { v.is_sign_negative() }
        }
        impl PredicateKernel<$T> for Isneginf {
            #[inline(always)]
            fn apply(v: $T) -> bool { v == <$T>::NEG_INFINITY }
        }
        impl PredicateKernel<$T> for Isposinf {
            #[inline(always)]
            fn apply(v: $T) -> bool { v == <$T>::INFINITY }
        }
    };
}

impl_float_predicates!(f64);
impl_float_predicates!(f32);

// Float16 predicate implementations
impl PredicateKernel<f16> for Isnan {
    #[inline(always)]
    fn apply(v: f16) -> bool { v.is_nan() }
}
impl PredicateKernel<f16> for Isinf {
    #[inline(always)]
    fn apply(v: f16) -> bool { v.is_infinite() }
}
impl PredicateKernel<f16> for Isfinite {
    #[inline(always)]
    fn apply(v: f16) -> bool { v.is_finite() }
}
impl PredicateKernel<f16> for Signbit {
    #[inline(always)]
    fn apply(v: f16) -> bool { v.is_sign_negative() }
}
impl PredicateKernel<f16> for Isneginf {
    #[inline(always)]
    fn apply(v: f16) -> bool { v == f16::NEG_INFINITY }
}
impl PredicateKernel<f16> for Isposinf {
    #[inline(always)]
    fn apply(v: f16) -> bool { v == f16::INFINITY }
}

// Integer predicate implementations (integers can't be nan/inf)
macro_rules! impl_int_predicates {
    ($T:ty, $is_signed:expr) => {
        impl PredicateKernel<$T> for Isnan {
            #[inline(always)]
            fn apply(_v: $T) -> bool { false }
        }
        impl PredicateKernel<$T> for Isinf {
            #[inline(always)]
            fn apply(_v: $T) -> bool { false }
        }
        impl PredicateKernel<$T> for Isfinite {
            #[inline(always)]
            fn apply(_v: $T) -> bool { true }
        }
        impl PredicateKernel<$T> for Signbit {
            #[inline(always)]
            fn apply(v: $T) -> bool {
                if $is_signed { (v as i64) < 0 } else { false }
            }
        }
        impl PredicateKernel<$T> for Isneginf {
            #[inline(always)]
            fn apply(_v: $T) -> bool { false }
        }
        impl PredicateKernel<$T> for Isposinf {
            #[inline(always)]
            fn apply(_v: $T) -> bool { false }
        }
    };
}

impl_int_predicates!(i64, true);
impl_int_predicates!(i32, true);
impl_int_predicates!(i16, true);
impl_int_predicates!(i8, true);
impl_int_predicates!(u64, false);
impl_int_predicates!(u32, false);
impl_int_predicates!(u16, false);
impl_int_predicates!(u8, false);

// Complex predicate implementations
macro_rules! impl_complex_predicates {
    ($F:ty) => {
        impl PredicateKernel<Complex<$F>> for Isnan {
            #[inline(always)]
            fn apply(v: Complex<$F>) -> bool { v.re.is_nan() || v.im.is_nan() }
        }
        impl PredicateKernel<Complex<$F>> for Isinf {
            #[inline(always)]
            fn apply(v: Complex<$F>) -> bool { v.re.is_infinite() || v.im.is_infinite() }
        }
        impl PredicateKernel<Complex<$F>> for Isfinite {
            #[inline(always)]
            fn apply(v: Complex<$F>) -> bool { v.re.is_finite() && v.im.is_finite() }
        }
        impl PredicateKernel<Complex<$F>> for Signbit {
            #[inline(always)]
            fn apply(v: Complex<$F>) -> bool { v.re.is_sign_negative() }
        }
        impl PredicateKernel<Complex<$F>> for Isneginf {
            #[inline(always)]
            fn apply(v: Complex<$F>) -> bool { v.re == <$F>::NEG_INFINITY && v.im == 0.0 }
        }
        impl PredicateKernel<Complex<$F>> for Isposinf {
            #[inline(always)]
            fn apply(v: Complex<$F>) -> bool { v.re == <$F>::INFINITY && v.im == 0.0 }
        }
    };
}

impl_complex_predicates!(f64);
impl_complex_predicates!(f32);
