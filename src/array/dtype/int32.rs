//! Int32 dtype implementation.

use super::{BinaryOp, DTypeKind, DTypeOps, ReduceOp, UnaryOp};
use std::cmp::Ordering;

/// Int32 dtype operations.
pub(super) struct Int32Ops;

impl Int32Ops {
    #[inline]
    unsafe fn read(ptr: *const u8, byte_offset: isize) -> i32 {
        *(ptr.offset(byte_offset) as *const i32)
    }

    #[inline]
    unsafe fn write(ptr: *mut u8, idx: usize, val: i32) {
        *(ptr as *mut i32).add(idx) = val;
    }
}

impl DTypeOps for Int32Ops {
    fn kind(&self) -> DTypeKind { DTypeKind::Int32 }
    fn itemsize(&self) -> usize { 4 }
    fn typestr(&self) -> &'static str { "<i4" }
    fn format_char(&self) -> &'static str { "i" }
    fn name(&self) -> &'static str { "int32" }
    fn promotion_priority(&self) -> u8 { 70 }

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
            UnaryOp::Neg => -v,
            UnaryOp::Abs => v.abs(),
            UnaryOp::Sqrt => (v as f64).sqrt() as i32,
            UnaryOp::Exp => (v as f64).exp() as i32,
            UnaryOp::Log => (v as f64).ln() as i32,
            UnaryOp::Log10 => (v as f64).log10() as i32,
            UnaryOp::Log2 => (v as f64).log2() as i32,
            UnaryOp::Sin => (v as f64).sin() as i32,
            UnaryOp::Cos => (v as f64).cos() as i32,
            UnaryOp::Tan => (v as f64).tan() as i32,
            UnaryOp::Sinh => (v as f64).sinh() as i32,
            UnaryOp::Cosh => (v as f64).cosh() as i32,
            UnaryOp::Tanh => (v as f64).tanh() as i32,
            UnaryOp::Floor => v,
            UnaryOp::Ceil => v,
            UnaryOp::Arcsin => (v as f64).asin() as i32,
            UnaryOp::Arccos => (v as f64).acos() as i32,
            UnaryOp::Arctan => (v as f64).atan() as i32,
            UnaryOp::Sign => if v > 0 { 1 } else if v < 0 { -1 } else { 0 },
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
            BinaryOp::Pow => if bv >= 0 { av.wrapping_pow(bv as u32) } else { 0 },
            BinaryOp::Mod => if bv != 0 { av % bv } else { 0 },
            BinaryOp::FloorDiv => if bv != 0 { av.div_euclid(bv) } else { 0 },
            BinaryOp::Maximum => av.max(bv),
            BinaryOp::Minimum => av.min(bv),
        };
        Self::write(out, idx, result);
    }

    unsafe fn reduce_init(&self, op: ReduceOp, out: *mut u8, idx: usize) {
        let v = match op {
            ReduceOp::Sum => 0,
            ReduceOp::Prod => 1,
            ReduceOp::Max => i32::MIN,
            ReduceOp::Min => i32::MAX,
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
        Self::write(ptr, idx, val as i32);
        true
    }

    unsafe fn read_f64(&self, ptr: *const u8, byte_offset: isize) -> Option<f64> {
        Some(Self::read(ptr, byte_offset) as f64)
    }

    unsafe fn write_complex(&self, ptr: *mut u8, idx: usize, real: f64, _imag: f64) -> bool {
        Self::write(ptr, idx, real as i32);
        true
    }

    unsafe fn read_complex(&self, ptr: *const u8, byte_offset: isize) -> Option<(f64, f64)> {
        Some((Self::read(ptr, byte_offset) as f64, 0.0))
    }
}
