//! Uint32 dtype implementation.

use super::{BinaryOp, DTypeKind, DTypeOps, ReduceOp, UnaryOp};
use std::cmp::Ordering;

/// Uint32 dtype operations.
pub(super) struct Uint32Ops;

impl Uint32Ops {
    #[inline]
    unsafe fn read(ptr: *const u8, byte_offset: isize) -> u32 {
        *(ptr.offset(byte_offset) as *const u32)
    }

    #[inline]
    unsafe fn write(ptr: *mut u8, idx: usize, val: u32) {
        *(ptr as *mut u32).add(idx) = val;
    }
}

impl DTypeOps for Uint32Ops {
    fn kind(&self) -> DTypeKind { DTypeKind::Uint32 }
    fn itemsize(&self) -> usize { 4 }
    fn typestr(&self) -> &'static str { "<u4" }
    fn format_char(&self) -> &'static str { "I" }
    fn name(&self) -> &'static str { "uint32" }
    fn promotion_priority(&self) -> u8 { 60 }

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
            UnaryOp::Neg => 0u32.wrapping_sub(v),
            UnaryOp::Abs => v,
            UnaryOp::Sqrt => (v as f64).sqrt() as u32,
            UnaryOp::Exp => (v as f64).exp() as u32,
            UnaryOp::Log => (v as f64).ln() as u32,
            UnaryOp::Sin => (v as f64).sin() as u32,
            UnaryOp::Cos => (v as f64).cos() as u32,
            UnaryOp::Tan => (v as f64).tan() as u32,
            UnaryOp::Floor => v,
            UnaryOp::Ceil => v,
            UnaryOp::Arcsin => (v as f64).asin() as u32,
            UnaryOp::Arccos => (v as f64).acos() as u32,
            UnaryOp::Arctan => (v as f64).atan() as u32,
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
            BinaryOp::Pow => av.wrapping_pow(bv),
            BinaryOp::Mod => if bv != 0 { av % bv } else { 0 },
            BinaryOp::FloorDiv => if bv != 0 { av / bv } else { 0 },
        };
        Self::write(out, idx, result);
    }

    unsafe fn reduce_init(&self, op: ReduceOp, out: *mut u8, idx: usize) {
        let v = match op {
            ReduceOp::Sum => 0,
            ReduceOp::Prod => 1,
            ReduceOp::Max => u32::MIN,
            ReduceOp::Min => u32::MAX,
        };
        Self::write(out, idx, v);
    }

    unsafe fn reduce_acc(&self, op: ReduceOp, acc: *mut u8, idx: usize, val: *const u8, byte_offset: isize) {
        let a = Self::read(acc as *const u8, (idx * 4) as isize);
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
        Self::write(ptr, idx, val as u32);
        true
    }

    unsafe fn read_f64(&self, ptr: *const u8, byte_offset: isize) -> Option<f64> {
        Some(Self::read(ptr, byte_offset) as f64)
    }

    unsafe fn write_complex(&self, ptr: *mut u8, idx: usize, real: f64, _imag: f64) -> bool {
        Self::write(ptr, idx, real as u32);
        true
    }

    unsafe fn read_complex(&self, ptr: *const u8, byte_offset: isize) -> Option<(f64, f64)> {
        Some((Self::read(ptr, byte_offset) as f64, 0.0))
    }
}
