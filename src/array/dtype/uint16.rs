//! Uint16 dtype implementation.

use super::{BinaryOp, DTypeKind, DTypeOps, ReduceOp, UnaryOp};
use std::cmp::Ordering;

/// Uint16 dtype operations.
pub(super) struct Uint16Ops;

impl Uint16Ops {
    #[inline]
    unsafe fn read(ptr: *const u8, byte_offset: isize) -> u16 {
        *(ptr.offset(byte_offset) as *const u16)
    }

    #[inline]
    unsafe fn write(ptr: *mut u8, idx: usize, val: u16) {
        *(ptr as *mut u16).add(idx) = val;
    }
}

impl DTypeOps for Uint16Ops {
    fn kind(&self) -> DTypeKind { DTypeKind::Uint16 }
    fn itemsize(&self) -> usize { 2 }
    fn typestr(&self) -> &'static str { "<u2" }
    fn format_char(&self) -> &'static str { "H" }
    fn name(&self) -> &'static str { "uint16" }
    fn promotion_priority(&self) -> u8 { 50 }

    unsafe fn write_zero(&self, ptr: *mut u8, idx: usize) {
        Self::write(ptr, idx, 0);
    }

    unsafe fn write_one(&self, ptr: *mut u8, idx: usize) {
        Self::write(ptr, idx, 1);
    }

    unsafe fn copy_element(&self, src: *const u8, byte_offset: isize, dst: *mut u8, idx: usize) {
        Self::write(dst, idx, Self::read(src, byte_offset));
    }

    unsafe fn unary_op(&self, op: UnaryOp, src: *const u8, byte_offset: isize, out: *mut u8, idx: usize) {
        let v = Self::read(src, byte_offset);
        let result = match op {
            UnaryOp::Neg => 0u16.wrapping_sub(v),
            UnaryOp::Abs => v,
            UnaryOp::Sqrt => (v as f64).sqrt() as u16,
            UnaryOp::Exp => (v as f64).exp() as u16,
            UnaryOp::Log => (v as f64).ln() as u16,
            UnaryOp::Log10 => (v as f64).log10() as u16,
            UnaryOp::Log2 => (v as f64).log2() as u16,
            UnaryOp::Sin => (v as f64).sin() as u16,
            UnaryOp::Cos => (v as f64).cos() as u16,
            UnaryOp::Tan => (v as f64).tan() as u16,
            UnaryOp::Sinh => (v as f64).sinh() as u16,
            UnaryOp::Cosh => (v as f64).cosh() as u16,
            UnaryOp::Tanh => (v as f64).tanh() as u16,
            UnaryOp::Floor => v,
            UnaryOp::Ceil => v,
            UnaryOp::Arcsin => (v as f64).asin() as u16,
            UnaryOp::Arccos => (v as f64).acos() as u16,
            UnaryOp::Arctan => (v as f64).atan() as u16,
            UnaryOp::Sign => if v > 0 { 1 } else { 0 },
            UnaryOp::Isnan => 0,
            UnaryOp::Isinf => 0,
            UnaryOp::Isfinite => 1,
        };
        Self::write(out, idx, result);
    }

    unsafe fn binary_op(&self, op: BinaryOp, a: *const u8, a_offset: isize, b: *const u8, b_offset: isize, out: *mut u8, idx: usize) {
        let av = Self::read(a, a_offset);
        let bv = Self::read(b, b_offset);
        let result = match op {
            BinaryOp::Add => av.wrapping_add(bv),
            BinaryOp::Sub => av.wrapping_sub(bv),
            BinaryOp::Mul => av.wrapping_mul(bv),
            BinaryOp::Div => if bv != 0 { av / bv } else { 0 },
            BinaryOp::Pow => av.wrapping_pow(bv as u32),
            BinaryOp::Mod => if bv != 0 { av % bv } else { 0 },
            BinaryOp::FloorDiv => if bv != 0 { av / bv } else { 0 },
            BinaryOp::Maximum => av.max(bv),
            BinaryOp::Minimum => av.min(bv),
        };
        Self::write(out, idx, result);
    }

    unsafe fn reduce_init(&self, op: ReduceOp, out: *mut u8, idx: usize) {
        let v = match op {
            ReduceOp::Sum => 0,
            ReduceOp::Prod => 1,
            ReduceOp::Max => u16::MIN,
            ReduceOp::Min => u16::MAX,
        };
        Self::write(out, idx, v);
    }

    unsafe fn reduce_acc(&self, op: ReduceOp, acc: *mut u8, idx: usize, val: *const u8, byte_offset: isize) {
        let a = Self::read(acc as *const u8, (idx * 2) as isize);
        let v = Self::read(val, byte_offset);
        let result = match op {
            ReduceOp::Sum => a.wrapping_add(v),
            ReduceOp::Prod => a.wrapping_mul(v),
            ReduceOp::Max => a.max(v),
            ReduceOp::Min => a.min(v),
        };
        Self::write(acc, idx, result);
    }

    unsafe fn format_element(&self, ptr: *const u8, byte_offset: isize) -> String {
        format!("{}", Self::read(ptr, byte_offset))
    }

    unsafe fn compare_elements(&self, a: *const u8, a_offset: isize, b: *const u8, b_offset: isize) -> Ordering {
        Self::read(a, a_offset).cmp(&Self::read(b, b_offset))
    }

    unsafe fn is_truthy(&self, ptr: *const u8, byte_offset: isize) -> bool {
        Self::read(ptr, byte_offset) != 0
    }

    unsafe fn write_f64(&self, ptr: *mut u8, idx: usize, val: f64) -> bool {
        Self::write(ptr, idx, val as u16);
        true
    }

    unsafe fn read_f64(&self, ptr: *const u8, byte_offset: isize) -> Option<f64> {
        Some(Self::read(ptr, byte_offset) as f64)
    }

    unsafe fn write_complex(&self, ptr: *mut u8, idx: usize, real: f64, _imag: f64) -> bool {
        Self::write(ptr, idx, real as u16);
        true
    }

    unsafe fn read_complex(&self, ptr: *const u8, byte_offset: isize) -> Option<(f64, f64)> {
        Some((Self::read(ptr, byte_offset) as f64, 0.0))
    }
}
