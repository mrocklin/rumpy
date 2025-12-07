//! Bool dtype implementation.

use super::{BinaryOp, DTypeKind, DTypeOps, ReduceOp, UnaryOp};
use std::cmp::Ordering;

/// Bool dtype operations.
pub(super) struct BoolOps;

impl BoolOps {
    #[inline]
    unsafe fn read(ptr: *const u8, byte_offset: isize) -> bool {
        *ptr.offset(byte_offset) != 0
    }

    #[inline]
    unsafe fn write(ptr: *mut u8, idx: usize, val: bool) {
        *ptr.add(idx) = val as u8;
    }
}

impl DTypeOps for BoolOps {
    fn kind(&self) -> DTypeKind { DTypeKind::Bool }
    fn itemsize(&self) -> usize { 1 }
    fn typestr(&self) -> &'static str { "|b1" }
    fn format_char(&self) -> &'static str { "?" }
    fn name(&self) -> &'static str { "bool" }
    fn promotion_priority(&self) -> u8 { 10 }

    unsafe fn write_zero(&self, ptr: *mut u8, idx: usize) {
        Self::write(ptr, idx, false);
    }

    unsafe fn write_one(&self, ptr: *mut u8, idx: usize) {
        Self::write(ptr, idx, true);
    }

    unsafe fn copy_element(&self, src: *const u8, byte_offset: isize, dst: *mut u8, idx: usize) {
        Self::write(dst, idx, Self::read(src, byte_offset));
    }

    unsafe fn unary_op(&self, op: UnaryOp, src: *const u8, byte_offset: isize, out: *mut u8, idx: usize) {
        let v = Self::read(src, byte_offset);
        let result = match op {
            UnaryOp::Neg => !v,  // logical not for bool
            UnaryOp::Abs => v,
            _ => v,  // transcendental ops make no sense for bool
        };
        Self::write(out, idx, result);
    }

    unsafe fn binary_op(&self, op: BinaryOp, a: *const u8, a_offset: isize, b: *const u8, b_offset: isize, out: *mut u8, idx: usize) {
        let av = Self::read(a, a_offset);
        let bv = Self::read(b, b_offset);
        let result = match op {
            BinaryOp::Add => av || bv,  // OR
            BinaryOp::Sub => av && !bv,  // AND NOT
            BinaryOp::Mul => av && bv,  // AND
            BinaryOp::Div => av,  // division not meaningful
            BinaryOp::Pow => av,  // not meaningful
            BinaryOp::Mod => av,  // not meaningful
            BinaryOp::FloorDiv => av,  // not meaningful
            BinaryOp::Maximum | BinaryOp::FMax => av || bv,  // max of bool is OR
            BinaryOp::Minimum | BinaryOp::FMin => av && bv,  // min of bool is AND
            // Float ops: treat bool as 0.0/1.0, return bool
            BinaryOp::Arctan2 | BinaryOp::Hypot => av || bv,
            BinaryOp::Copysign => av,
            BinaryOp::Logaddexp | BinaryOp::Logaddexp2 => av || bv,
            BinaryOp::Nextafter => bv,
        };
        Self::write(out, idx, result);
    }

    unsafe fn reduce_init(&self, op: ReduceOp, out: *mut u8, idx: usize) {
        let v = match op {
            ReduceOp::Sum => false,  // any() identity
            ReduceOp::Prod => true,  // all() identity
            ReduceOp::Max => false,
            ReduceOp::Min => true,
        };
        Self::write(out, idx, v);
    }

    unsafe fn reduce_acc(&self, op: ReduceOp, acc: *mut u8, idx: usize, val: *const u8, byte_offset: isize) {
        let a = Self::read(acc as *const u8, idx as isize);
        let v = Self::read(val, byte_offset);
        let result = match op {
            ReduceOp::Sum => a || v,  // any
            ReduceOp::Prod => a && v,  // all
            ReduceOp::Max => a || v,
            ReduceOp::Min => a && v,
        };
        Self::write(acc, idx, result);
    }

    unsafe fn format_element(&self, ptr: *const u8, byte_offset: isize) -> String {
        if Self::read(ptr, byte_offset) { "True".to_string() } else { "False".to_string() }
    }

    unsafe fn compare_elements(&self, a: *const u8, a_offset: isize, b: *const u8, b_offset: isize) -> Ordering {
        let av = Self::read(a, a_offset) as u8;
        let bv = Self::read(b, b_offset) as u8;
        av.cmp(&bv)
    }

    unsafe fn is_truthy(&self, ptr: *const u8, byte_offset: isize) -> bool {
        Self::read(ptr, byte_offset)
    }

    unsafe fn write_f64(&self, ptr: *mut u8, idx: usize, val: f64) -> bool {
        Self::write(ptr, idx, val != 0.0);
        true
    }

    unsafe fn read_f64(&self, ptr: *const u8, byte_offset: isize) -> Option<f64> {
        Some(if Self::read(ptr, byte_offset) { 1.0 } else { 0.0 })
    }

    unsafe fn write_f64_at_byte_offset(&self, ptr: *mut u8, byte_offset: isize, val: f64) -> bool {
        *(ptr.offset(byte_offset)) = if val != 0.0 { 1 } else { 0 };
        true
    }

    unsafe fn write_complex(&self, ptr: *mut u8, idx: usize, real: f64, imag: f64) -> bool {
        Self::write(ptr, idx, real != 0.0 || imag != 0.0);
        true
    }

    unsafe fn read_complex(&self, ptr: *const u8, byte_offset: isize) -> Option<(f64, f64)> {
        Some((if Self::read(ptr, byte_offset) { 1.0 } else { 0.0 }, 0.0))
    }
}
