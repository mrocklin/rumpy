//! Float32 dtype implementation.

use super::{BinaryOp, DTypeKind, DTypeOps, ReduceOp, UnaryOp};
use std::cmp::Ordering;

/// Float32 dtype operations.
pub(super) struct Float32Ops;

impl Float32Ops {
    #[inline]
    unsafe fn read(ptr: *const u8, byte_offset: isize) -> f32 {
        *(ptr.offset(byte_offset) as *const f32)
    }

    #[inline]
    unsafe fn write(ptr: *mut u8, idx: usize, val: f32) {
        *(ptr as *mut f32).add(idx) = val;
    }
}

impl DTypeOps for Float32Ops {
    fn kind(&self) -> DTypeKind { DTypeKind::Float32 }
    fn itemsize(&self) -> usize { 4 }
    fn typestr(&self) -> &'static str { "<f4" }
    fn format_char(&self) -> &'static str { "f" }
    fn name(&self) -> &'static str { "float32" }
    fn promotion_priority(&self) -> u8 { 90 }

    unsafe fn write_zero(&self, ptr: *mut u8, idx: usize) {
        Self::write(ptr, idx, 0.0);
    }

    unsafe fn write_one(&self, ptr: *mut u8, idx: usize) {
        Self::write(ptr, idx, 1.0);
    }

    unsafe fn copy_element(&self, src: *const u8, byte_offset: isize, dst: *mut u8, idx: usize) {
        Self::write(dst, idx, Self::read(src, byte_offset));
    }

    unsafe fn unary_op(&self, op: UnaryOp, src: *const u8, byte_offset: isize, out: *mut u8, idx: usize) {
        let v = Self::read(src, byte_offset);
        let result = match op {
            UnaryOp::Neg => -v,
            UnaryOp::Abs => v.abs(),
            UnaryOp::Sqrt => v.sqrt(),
            UnaryOp::Exp => v.exp(),
            UnaryOp::Log => v.ln(),
            UnaryOp::Sin => v.sin(),
            UnaryOp::Cos => v.cos(),
            UnaryOp::Tan => v.tan(),
        };
        Self::write(out, idx, result);
    }

    unsafe fn binary_op(&self, op: BinaryOp, a: *const u8, a_offset: isize, b: *const u8, b_offset: isize, out: *mut u8, idx: usize) {
        let av = Self::read(a, a_offset);
        let bv = Self::read(b, b_offset);
        let result = match op {
            BinaryOp::Add => av + bv,
            BinaryOp::Sub => av - bv,
            BinaryOp::Mul => av * bv,
            BinaryOp::Div => av / bv,
        };
        Self::write(out, idx, result);
    }

    unsafe fn reduce_init(&self, op: ReduceOp, out: *mut u8, idx: usize) {
        let v = match op {
            ReduceOp::Sum => 0.0,
            ReduceOp::Prod => 1.0,
            ReduceOp::Max => f32::NEG_INFINITY,
            ReduceOp::Min => f32::INFINITY,
        };
        Self::write(out, idx, v);
    }

    unsafe fn reduce_acc(&self, op: ReduceOp, acc: *mut u8, idx: usize, val: *const u8, byte_offset: isize) {
        let a = Self::read(acc as *const u8, (idx * 4) as isize);
        let v = Self::read(val, byte_offset);
        let result = match op {
            ReduceOp::Sum => a + v,
            ReduceOp::Prod => a * v,
            ReduceOp::Max => a.max(v),
            ReduceOp::Min => a.min(v),
        };
        Self::write(acc, idx, result);
    }

    unsafe fn format_element(&self, ptr: *const u8, byte_offset: isize) -> String {
        let v = Self::read(ptr, byte_offset);
        let s = format!("{:.8}", v);
        s.trim_end_matches('0').to_string()
    }

    unsafe fn compare_elements(&self, a: *const u8, a_offset: isize, b: *const u8, b_offset: isize) -> Ordering {
        let av = Self::read(a, a_offset);
        let bv = Self::read(b, b_offset);
        av.partial_cmp(&bv).unwrap_or(Ordering::Equal)
    }

    unsafe fn is_truthy(&self, ptr: *const u8, byte_offset: isize) -> bool {
        Self::read(ptr, byte_offset) != 0.0
    }

    unsafe fn write_f64(&self, ptr: *mut u8, idx: usize, val: f64) -> bool {
        Self::write(ptr, idx, val as f32);
        true
    }

    unsafe fn read_f64(&self, ptr: *const u8, byte_offset: isize) -> Option<f64> {
        Some(Self::read(ptr, byte_offset) as f64)
    }

    unsafe fn write_complex(&self, ptr: *mut u8, idx: usize, real: f64, _imag: f64) -> bool {
        Self::write(ptr, idx, real as f32);
        true
    }

    unsafe fn read_complex(&self, ptr: *const u8, byte_offset: isize) -> Option<(f64, f64)> {
        Some((Self::read(ptr, byte_offset) as f64, 0.0))
    }
}
