//! Float16 dtype implementation.

use super::{BinaryOp, DTypeKind, DTypeOps, ReduceOp, UnaryOp};
use half::f16;
use std::cmp::Ordering;

/// Float16 dtype operations.
pub(super) struct Float16Ops;

impl Float16Ops {
    #[inline]
    unsafe fn read(ptr: *const u8, byte_offset: isize) -> f16 {
        *(ptr.offset(byte_offset) as *const f16)
    }

    #[inline]
    unsafe fn write(ptr: *mut u8, idx: usize, val: f16) {
        *(ptr as *mut f16).add(idx) = val;
    }
}

impl DTypeOps for Float16Ops {
    fn kind(&self) -> DTypeKind { DTypeKind::Float16 }
    fn itemsize(&self) -> usize { 2 }
    fn typestr(&self) -> &'static str { "<f2" }
    fn format_char(&self) -> &'static str { "e" }
    fn name(&self) -> &'static str { "float16" }
    fn promotion_priority(&self) -> u8 { 80 } // Between integers and float32

    unsafe fn write_zero(&self, ptr: *mut u8, idx: usize) {
        Self::write(ptr, idx, f16::ZERO);
    }

    unsafe fn write_one(&self, ptr: *mut u8, idx: usize) {
        Self::write(ptr, idx, f16::ONE);
    }

    unsafe fn copy_element(&self, src: *const u8, byte_offset: isize, dst: *mut u8, idx: usize) {
        Self::write(dst, idx, Self::read(src, byte_offset));
    }

    unsafe fn unary_op(&self, op: UnaryOp, src: *const u8, byte_offset: isize, out: *mut u8, idx: usize) {
        // Convert to f32 for math ops, then back to f16
        let v = Self::read(src, byte_offset).to_f32();
        let result = match op {
            UnaryOp::Neg => -v,
            UnaryOp::Abs => v.abs(),
            UnaryOp::Sqrt => v.sqrt(),
            UnaryOp::Exp => v.exp(),
            UnaryOp::Log => v.ln(),
            UnaryOp::Log10 => v.log10(),
            UnaryOp::Log2 => v.log2(),
            UnaryOp::Sin => v.sin(),
            UnaryOp::Cos => v.cos(),
            UnaryOp::Tan => v.tan(),
            UnaryOp::Sinh => v.sinh(),
            UnaryOp::Cosh => v.cosh(),
            UnaryOp::Tanh => v.tanh(),
            UnaryOp::Floor => v.floor(),
            UnaryOp::Ceil => v.ceil(),
            UnaryOp::Arcsin => v.asin(),
            UnaryOp::Arccos => v.acos(),
            UnaryOp::Arctan => v.atan(),
            UnaryOp::Sign => if v > 0.0 { 1.0 } else if v < 0.0 { -1.0 } else { 0.0 },
            UnaryOp::Isnan => if v.is_nan() { 1.0 } else { 0.0 },
            UnaryOp::Isinf => if v.is_infinite() { 1.0 } else { 0.0 },
            UnaryOp::Isfinite => if v.is_finite() { 1.0 } else { 0.0 },

            UnaryOp::Square => v * v,
            UnaryOp::Positive => v,
            UnaryOp::Reciprocal => 1.0 / v,
            UnaryOp::Exp2 => 2.0f32.powf(v),
            UnaryOp::Expm1 => v.exp_m1(),
            UnaryOp::Log1p => v.ln_1p(),
            UnaryOp::Cbrt => v.cbrt(),
            UnaryOp::Trunc => v.trunc(),
            UnaryOp::Rint => v.round(),
            UnaryOp::Arcsinh => v.asinh(),
            UnaryOp::Arccosh => v.acosh(),
            UnaryOp::Arctanh => v.atanh(),
            UnaryOp::Signbit => if v.is_sign_negative() { 1.0 } else { 0.0 },
            UnaryOp::Isneginf => if v == f32::NEG_INFINITY { 1.0 } else { 0.0 },
            UnaryOp::Isposinf => if v == f32::INFINITY { 1.0 } else { 0.0 },
            UnaryOp::Isreal => 1.0,    // real types are always real
            UnaryOp::Iscomplex => 0.0, // real types are never complex
        };
        Self::write(out, idx, f16::from_f32(result));
    }

    unsafe fn binary_op(&self, op: BinaryOp, a: *const u8, a_offset: isize, b: *const u8, b_offset: isize, out: *mut u8, idx: usize) {
        let av = Self::read(a, a_offset).to_f32();
        let bv = Self::read(b, b_offset).to_f32();
        let result = match op {
            BinaryOp::Add => av + bv,
            BinaryOp::Sub => av - bv,
            BinaryOp::Mul => av * bv,
            BinaryOp::Div => av / bv,
            BinaryOp::Pow => av.powf(bv),
            BinaryOp::Mod => av % bv,
            BinaryOp::FloorDiv => (av / bv).floor(),
            BinaryOp::Maximum => if av.is_nan() || bv.is_nan() { f32::NAN } else { av.max(bv) },
            BinaryOp::Minimum => if av.is_nan() || bv.is_nan() { f32::NAN } else { av.min(bv) },
            // Stream 2: Binary Math Operations
            BinaryOp::Arctan2 => av.atan2(bv),
            BinaryOp::Hypot => av.hypot(bv),
            BinaryOp::FMax => if bv.is_nan() { av } else if av.is_nan() { bv } else { av.max(bv) },
            BinaryOp::FMin => if bv.is_nan() { av } else if av.is_nan() { bv } else { av.min(bv) },
            BinaryOp::Copysign => av.copysign(bv),
            BinaryOp::Logaddexp => {
                let m = av.max(bv);
                if m.is_infinite() { m } else { m + (1.0_f32 + (-(av - bv).abs()).exp()).ln() }
            },
            BinaryOp::Logaddexp2 => {
                let m = av.max(bv);
                let ln2 = std::f32::consts::LN_2;
                if m.is_infinite() { m } else { m + ((1.0_f32 + (-(av - bv).abs() * ln2).exp()).ln() / ln2) }
            },
            BinaryOp::Nextafter => {
                // Use bit manipulation for nextafter
                if av.is_nan() || bv.is_nan() {
                    f32::NAN
                } else if av == bv {
                    bv
                } else if av < bv {
                    let bits = av.to_bits();
                    f32::from_bits(if av >= 0.0 { bits + 1 } else { bits - 1 })
                } else {
                    let bits = av.to_bits();
                    f32::from_bits(if av > 0.0 { bits - 1 } else { bits + 1 })
                }
            },
        };
        Self::write(out, idx, f16::from_f32(result));
    }

    unsafe fn reduce_init(&self, op: ReduceOp, out: *mut u8, idx: usize) {
        let v = match op {
            ReduceOp::Sum => f16::ZERO,
            ReduceOp::Prod => f16::ONE,
            ReduceOp::Max => f16::NEG_INFINITY,
            ReduceOp::Min => f16::INFINITY,
        };
        Self::write(out, idx, v);
    }

    unsafe fn reduce_acc(&self, op: ReduceOp, acc: *mut u8, idx: usize, val: *const u8, byte_offset: isize) {
        let a = Self::read(acc as *const u8, (idx * 2) as isize).to_f32();
        let v = Self::read(val, byte_offset).to_f32();
        let result = match op {
            ReduceOp::Sum => a + v,
            ReduceOp::Prod => a * v,
            ReduceOp::Max => a.max(v),
            ReduceOp::Min => a.min(v),
        };
        Self::write(acc, idx, f16::from_f32(result));
    }

    unsafe fn format_element(&self, ptr: *const u8, byte_offset: isize) -> String {
        let v = Self::read(ptr, byte_offset).to_f32();
        let s = format!("{:.4}", v);
        s.trim_end_matches('0').to_string()
    }

    unsafe fn compare_elements(&self, a: *const u8, a_offset: isize, b: *const u8, b_offset: isize) -> Ordering {
        let av = Self::read(a, a_offset);
        let bv = Self::read(b, b_offset);
        av.partial_cmp(&bv).unwrap_or(Ordering::Equal)
    }

    unsafe fn is_truthy(&self, ptr: *const u8, byte_offset: isize) -> bool {
        Self::read(ptr, byte_offset) != f16::ZERO
    }

    unsafe fn write_f64(&self, ptr: *mut u8, idx: usize, val: f64) -> bool {
        Self::write(ptr, idx, f16::from_f64(val));
        true
    }

    unsafe fn read_f64(&self, ptr: *const u8, byte_offset: isize) -> Option<f64> {
        Some(Self::read(ptr, byte_offset).to_f64())
    }

    unsafe fn read_i64(&self, ptr: *const u8, byte_offset: isize) -> Option<i64> {
        Some(Self::read(ptr, byte_offset).to_f64() as i64)
    }

    unsafe fn write_i64(&self, ptr: *mut u8, idx: usize, val: i64) -> bool {
        Self::write(ptr, idx, f16::from_f64(val as f64));
        true
    }

    unsafe fn write_f64_at_byte_offset(&self, ptr: *mut u8, byte_offset: isize, val: f64) -> bool {
        *(ptr.offset(byte_offset) as *mut f16) = f16::from_f64(val);
        true
    }

    unsafe fn write_complex(&self, ptr: *mut u8, idx: usize, real: f64, _imag: f64) -> bool {
        Self::write(ptr, idx, f16::from_f64(real));
        true
    }

    unsafe fn read_complex(&self, ptr: *const u8, byte_offset: isize) -> Option<(f64, f64)> {
        Some((Self::read(ptr, byte_offset).to_f64(), 0.0))
    }
}
