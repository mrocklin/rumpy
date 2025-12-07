//! Complex128 dtype implementation.

use super::{BinaryOp, DTypeKind, DTypeOps, ReduceOp, UnaryOp};
use std::cmp::Ordering;

/// Complex128 dtype operations (two f64: real, imag).
pub(super) struct Complex128Ops;

impl Complex128Ops {
    #[inline]
    unsafe fn read(ptr: *const u8, byte_offset: isize) -> (f64, f64) {
        let p = ptr.offset(byte_offset) as *const f64;
        (*p, *p.add(1))
    }

    #[inline]
    unsafe fn write(ptr: *mut u8, idx: usize, real: f64, imag: f64) {
        let p = (ptr as *mut f64).add(idx * 2);
        *p = real;
        *p.add(1) = imag;
    }

    /// Complex power: z^w = exp(w * ln(z))
    #[inline]
    fn pow(ar: f64, ai: f64, br: f64, bi: f64) -> (f64, f64) {
        // Handle special case: 0^w
        if ar == 0.0 && ai == 0.0 {
            return if br > 0.0 { (0.0, 0.0) } else { (f64::NAN, f64::NAN) };
        }
        // ln(a) = (ln(|a|), arg(a))
        let mag_a = (ar * ar + ai * ai).sqrt();
        let ln_r = mag_a.ln();
        let ln_i = ai.atan2(ar);
        // w * ln(a) - complex multiplication
        let prod_r = br * ln_r - bi * ln_i;
        let prod_i = br * ln_i + bi * ln_r;
        // exp(prod) = exp(prod_r) * (cos(prod_i) + i*sin(prod_i))
        let exp_r = prod_r.exp();
        (exp_r * prod_i.cos(), exp_r * prod_i.sin())
    }

    /// Complex arcsin: arcsin(z) = -i * log(iz + sqrt(1 - z^2))
    #[inline]
    fn arcsin(r: f64, i: f64) -> (f64, f64) {
        let iz = (-i, r);
        let one_minus_z2 = (1.0 - r * r + i * i, -2.0 * r * i);
        let mag = (one_minus_z2.0 * one_minus_z2.0 + one_minus_z2.1 * one_minus_z2.1).sqrt();
        let sqrt_r = ((mag + one_minus_z2.0) / 2.0).sqrt();
        let sqrt_i = one_minus_z2.1.signum() * ((mag - one_minus_z2.0) / 2.0).sqrt();
        let sum = (iz.0 + sqrt_r, iz.1 + sqrt_i);
        let log_mag = (sum.0 * sum.0 + sum.1 * sum.1).sqrt();
        let log_r = log_mag.ln();
        let log_i = sum.1.atan2(sum.0);
        (log_i, -log_r)
    }
}

impl DTypeOps for Complex128Ops {
    fn kind(&self) -> DTypeKind { DTypeKind::Complex128 }
    fn itemsize(&self) -> usize { 16 }
    fn typestr(&self) -> &'static str { "<c16" }
    fn format_char(&self) -> &'static str { "Zd" }
    fn name(&self) -> &'static str { "complex128" }
    fn promotion_priority(&self) -> u8 { 110 } // Higher than float64

    unsafe fn write_zero(&self, ptr: *mut u8, idx: usize) {
        Self::write(ptr, idx, 0.0, 0.0);
    }

    unsafe fn write_one(&self, ptr: *mut u8, idx: usize) {
        Self::write(ptr, idx, 1.0, 0.0);
    }

    unsafe fn copy_element(&self, src: *const u8, byte_offset: isize, dst: *mut u8, idx: usize) {
        let (r, i) = Self::read(src, byte_offset);
        Self::write(dst, idx, r, i);
    }

    unsafe fn unary_op(&self, op: UnaryOp, src: *const u8, byte_offset: isize, out: *mut u8, idx: usize) {
        let (r, i) = Self::read(src, byte_offset);
        let (out_r, out_i) = match op {
            UnaryOp::Neg => (-r, -i),
            UnaryOp::Abs => ((r * r + i * i).sqrt(), 0.0), // |z| is real
            UnaryOp::Sqrt => {
                // sqrt(a+bi) = sqrt((|z|+a)/2) + i*sign(b)*sqrt((|z|-a)/2)
                let mag = (r * r + i * i).sqrt();
                let real = ((mag + r) / 2.0).sqrt();
                let imag = i.signum() * ((mag - r) / 2.0).sqrt();
                (real, imag)
            }
            UnaryOp::Exp => {
                // exp(a+bi) = exp(a) * (cos(b) + i*sin(b))
                let exp_r = r.exp();
                (exp_r * i.cos(), exp_r * i.sin())
            }
            UnaryOp::Log => {
                // log(a+bi) = log(|z|) + i*arg(z)
                let mag = (r * r + i * i).sqrt();
                (mag.ln(), i.atan2(r))
            }
            UnaryOp::Sin => {
                // sin(a+bi) = sin(a)*cosh(b) + i*cos(a)*sinh(b)
                (r.sin() * i.cosh(), r.cos() * i.sinh())
            }
            UnaryOp::Cos => {
                // cos(a+bi) = cos(a)*cosh(b) - i*sin(a)*sinh(b)
                (r.cos() * i.cosh(), -r.sin() * i.sinh())
            }
            UnaryOp::Tan => {
                // tan(z) = sin(z) / cos(z)
                let sin_r = r.sin() * i.cosh();
                let sin_i = r.cos() * i.sinh();
                let cos_r = r.cos() * i.cosh();
                let cos_i = -r.sin() * i.sinh();
                // (sin_r + i*sin_i) / (cos_r + i*cos_i)
                let denom = cos_r * cos_r + cos_i * cos_i;
                ((sin_r * cos_r + sin_i * cos_i) / denom,
                 (sin_i * cos_r - sin_r * cos_i) / denom)
            }
            UnaryOp::Floor => (r.floor(), i.floor()),
            UnaryOp::Ceil => (r.ceil(), i.ceil()),
            UnaryOp::Arcsin => Self::arcsin(r, i),
            UnaryOp::Arccos => {
                // arccos(z) = pi/2 - arcsin(z)
                let asin = Self::arcsin(r, i);
                (std::f64::consts::FRAC_PI_2 - asin.0, -asin.1)
            }
            UnaryOp::Arctan => {
                // arctan(z) = (i/2) * log((1-iz)/(1+iz))
                let num = (1.0 + i, -r);  // 1 - iz
                let den = (1.0 - i, r);   // 1 + iz
                let den_mag2 = den.0 * den.0 + den.1 * den.1;
                let div_r = (num.0 * den.0 + num.1 * den.1) / den_mag2;
                let div_i = (num.1 * den.0 - num.0 * den.1) / den_mag2;
                let log_mag = (div_r * div_r + div_i * div_i).sqrt();
                let log_r = log_mag.ln();
                let log_i = div_i.atan2(div_r);
                (-log_i / 2.0, log_r / 2.0)
            }
            UnaryOp::Log10 => {
                // log10(z) = log(z) / log(10)
                let mag = (r * r + i * i).sqrt();
                let log_r = mag.ln();
                let log_i = i.atan2(r);
                let ln10 = 10.0_f64.ln();
                (log_r / ln10, log_i / ln10)
            }
            UnaryOp::Log2 => {
                // log2(z) = log(z) / log(2)
                let mag = (r * r + i * i).sqrt();
                let log_r = mag.ln();
                let log_i = i.atan2(r);
                let ln2 = 2.0_f64.ln();
                (log_r / ln2, log_i / ln2)
            }
            UnaryOp::Sinh => {
                // sinh(a+bi) = sinh(a)*cos(b) + i*cosh(a)*sin(b)
                (r.sinh() * i.cos(), r.cosh() * i.sin())
            }
            UnaryOp::Cosh => {
                // cosh(a+bi) = cosh(a)*cos(b) + i*sinh(a)*sin(b)
                (r.cosh() * i.cos(), r.sinh() * i.sin())
            }
            UnaryOp::Tanh => {
                // tanh(z) = sinh(z) / cosh(z)
                let sinh_r = r.sinh() * i.cos();
                let sinh_i = r.cosh() * i.sin();
                let cosh_r = r.cosh() * i.cos();
                let cosh_i = r.sinh() * i.sin();
                let denom = cosh_r * cosh_r + cosh_i * cosh_i;
                ((sinh_r * cosh_r + sinh_i * cosh_i) / denom,
                 (sinh_i * cosh_r - sinh_r * cosh_i) / denom)
            }
            UnaryOp::Sign => {
                let mag = (r * r + i * i).sqrt();
                if mag == 0.0 { (0.0, 0.0) } else { (r / mag, i / mag) }
            }
            UnaryOp::Isnan => (if r.is_nan() || i.is_nan() { 1.0 } else { 0.0 }, 0.0),
            UnaryOp::Isinf => (if r.is_infinite() || i.is_infinite() { 1.0 } else { 0.0 }, 0.0),
            UnaryOp::Isfinite => (if r.is_finite() && i.is_finite() { 1.0 } else { 0.0 }, 0.0),

            UnaryOp::Square => (r * r - i * i, 2.0 * r * i),
            UnaryOp::Positive => (r, i),
            UnaryOp::Reciprocal => {
                let denom = r * r + i * i;
                (r / denom, -i / denom)
            }
            UnaryOp::Exp2 => {
                // 2^(a+bi) = 2^a * (cos(b*ln2) + i*sin(b*ln2))
                let ln2 = 2.0_f64.ln();
                let exp_r = 2.0_f64.powf(r);
                (exp_r * (i * ln2).cos(), exp_r * (i * ln2).sin())
            }
            UnaryOp::Expm1 => {
                // expm1(z) = exp(z) - 1
                let exp_r = r.exp();
                (exp_r * i.cos() - 1.0, exp_r * i.sin())
            }
            UnaryOp::Log1p => {
                // log1p(z) = log(1+z)
                let nr = 1.0 + r;
                let mag = (nr * nr + i * i).sqrt();
                (mag.ln(), i.atan2(nr))
            }
            UnaryOp::Cbrt => {
                // cbrt(z) = |z|^(1/3) * exp(i*arg(z)/3)
                let mag = (r * r + i * i).sqrt().cbrt();
                let arg = i.atan2(r) / 3.0;
                (mag * arg.cos(), mag * arg.sin())
            }
            UnaryOp::Trunc => (r.trunc(), i.trunc()),
            UnaryOp::Rint => (r.round(), i.round()),
            UnaryOp::Arcsinh => {
                // arcsinh(z) = log(z + sqrt(z^2 + 1))
                let z2_plus_1 = (r * r - i * i + 1.0, 2.0 * r * i);
                let mag = (z2_plus_1.0 * z2_plus_1.0 + z2_plus_1.1 * z2_plus_1.1).sqrt();
                let sqrt_r = ((mag + z2_plus_1.0) / 2.0).sqrt();
                let sqrt_i = z2_plus_1.1.signum() * ((mag - z2_plus_1.0) / 2.0).sqrt();
                let sum = (r + sqrt_r, i + sqrt_i);
                let log_mag = (sum.0 * sum.0 + sum.1 * sum.1).sqrt();
                (log_mag.ln(), sum.1.atan2(sum.0))
            }
            UnaryOp::Arccosh => {
                // arccosh(z) = log(z + sqrt(z^2 - 1))
                let z2_minus_1 = (r * r - i * i - 1.0, 2.0 * r * i);
                let mag = (z2_minus_1.0 * z2_minus_1.0 + z2_minus_1.1 * z2_minus_1.1).sqrt();
                let sqrt_r = ((mag + z2_minus_1.0) / 2.0).sqrt();
                let sqrt_i = z2_minus_1.1.signum() * ((mag - z2_minus_1.0) / 2.0).sqrt();
                let sum = (r + sqrt_r, i + sqrt_i);
                let log_mag = (sum.0 * sum.0 + sum.1 * sum.1).sqrt();
                (log_mag.ln(), sum.1.atan2(sum.0))
            }
            UnaryOp::Arctanh => {
                // arctanh(z) = 0.5 * log((1+z)/(1-z))
                let num = (1.0 + r, i);
                let den = (1.0 - r, -i);
                let den_mag2 = den.0 * den.0 + den.1 * den.1;
                let div_r = (num.0 * den.0 + num.1 * den.1) / den_mag2;
                let div_i = (num.1 * den.0 - num.0 * den.1) / den_mag2;
                let log_mag = (div_r * div_r + div_i * div_i).sqrt();
                (0.5 * log_mag.ln(), 0.5 * div_i.atan2(div_r))
            }
            UnaryOp::Signbit => (if r.is_sign_negative() { 1.0 } else { 0.0 }, 0.0),
        };
        Self::write(out, idx, out_r, out_i);
    }

    unsafe fn binary_op(&self, op: BinaryOp, a: *const u8, a_offset: isize, b: *const u8, b_offset: isize, out: *mut u8, idx: usize) {
        let (ar, ai) = Self::read(a, a_offset);
        let (br, bi) = Self::read(b, b_offset);
        let (out_r, out_i) = match op {
            BinaryOp::Add => (ar + br, ai + bi),
            BinaryOp::Sub => (ar - br, ai - bi),
            BinaryOp::Mul => (ar * br - ai * bi, ar * bi + ai * br),
            BinaryOp::Div => {
                let denom = br * br + bi * bi;
                ((ar * br + ai * bi) / denom, (ai * br - ar * bi) / denom)
            }
            BinaryOp::Pow => Self::pow(ar, ai, br, bi),
            BinaryOp::Mod | BinaryOp::FloorDiv => (f64::NAN, f64::NAN),
            // Complex max/min: NumPy compares by real part first, then imaginary
            // NaN propagation: if any component is NaN, result is NaN
            BinaryOp::Maximum => if ar.is_nan() || ai.is_nan() || br.is_nan() || bi.is_nan() {
                (f64::NAN, f64::NAN)
            } else if ar > br || (ar == br && ai >= bi) { (ar, ai) } else { (br, bi) },
            BinaryOp::Minimum => if ar.is_nan() || ai.is_nan() || br.is_nan() || bi.is_nan() {
                (f64::NAN, f64::NAN)
            } else if ar < br || (ar == br && ai <= bi) { (ar, ai) } else { (br, bi) },
        };
        Self::write(out, idx, out_r, out_i);
    }

    unsafe fn reduce_init(&self, op: ReduceOp, out: *mut u8, idx: usize) {
        let (r, i) = match op {
            ReduceOp::Sum => (0.0, 0.0),
            ReduceOp::Prod => (1.0, 0.0),
            ReduceOp::Max | ReduceOp::Min => {
                // Max/Min not well-defined for complex; use -inf/inf magnitude placeholder
                (f64::NEG_INFINITY, 0.0)
            }
        };
        Self::write(out, idx, r, i);
    }

    unsafe fn reduce_acc(&self, op: ReduceOp, acc: *mut u8, idx: usize, val: *const u8, byte_offset: isize) {
        let (ar, ai) = Self::read(acc as *const u8, (idx * 16) as isize);
        let (vr, vi) = Self::read(val, byte_offset);
        let (out_r, out_i) = match op {
            ReduceOp::Sum => (ar + vr, ai + vi),
            ReduceOp::Prod => (ar * vr - ai * vi, ar * vi + ai * vr),
            ReduceOp::Max | ReduceOp::Min => {
                // Compare by magnitude
                let mag_a = ar * ar + ai * ai;
                let mag_v = vr * vr + vi * vi;
                if (op as u8 == ReduceOp::Max as u8 && mag_v > mag_a) ||
                   (op as u8 == ReduceOp::Min as u8 && mag_v < mag_a) {
                    (vr, vi)
                } else {
                    (ar, ai)
                }
            }
        };
        Self::write(acc, idx, out_r, out_i);
    }

    unsafe fn format_element(&self, ptr: *const u8, byte_offset: isize) -> String {
        let (r, i) = Self::read(ptr, byte_offset);
        // NumPy style: "(1+2j)" or "(1-2j)"
        if i >= 0.0 {
            format!("({:.8}+{:.8}j)", r, i).replace(".00000000", ".").replace("0j", "j")
        } else {
            format!("({:.8}{:.8}j)", r, i).replace(".00000000", ".").replace("0j", "j")
        }
    }

    unsafe fn compare_elements(&self, a: *const u8, a_offset: isize, b: *const u8, b_offset: isize) -> Ordering {
        // Compare by magnitude (not ideal but allows sorting)
        let (ar, ai) = Self::read(a, a_offset);
        let (br, bi) = Self::read(b, b_offset);
        let mag_a = ar * ar + ai * ai;
        let mag_b = br * br + bi * bi;
        mag_a.partial_cmp(&mag_b).unwrap_or(Ordering::Equal)
    }

    unsafe fn is_truthy(&self, ptr: *const u8, byte_offset: isize) -> bool {
        let (r, i) = Self::read(ptr, byte_offset);
        r != 0.0 || i != 0.0
    }

    unsafe fn write_f64(&self, ptr: *mut u8, idx: usize, val: f64) -> bool {
        Self::write(ptr, idx, val, 0.0);
        true
    }

    unsafe fn read_f64(&self, _ptr: *const u8, _byte_offset: isize) -> Option<f64> {
        // Complex can't be represented as single f64
        None
    }

    unsafe fn write_f64_at_byte_offset(&self, ptr: *mut u8, byte_offset: isize, val: f64) -> bool {
        // Write real part only, zero imaginary
        let complex_ptr = ptr.offset(byte_offset) as *mut [f64; 2];
        (*complex_ptr)[0] = val;
        (*complex_ptr)[1] = 0.0;
        true
    }

    unsafe fn write_complex(&self, ptr: *mut u8, idx: usize, real: f64, imag: f64) -> bool {
        Self::write(ptr, idx, real, imag);
        true
    }

    unsafe fn read_complex(&self, ptr: *const u8, byte_offset: isize) -> Option<(f64, f64)> {
        Some(Self::read(ptr, byte_offset))
    }
}
