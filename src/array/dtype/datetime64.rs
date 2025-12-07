//! DateTime64 dtype implementation - a parametric dtype.

use super::{BinaryOp, DTypeKind, DTypeOps, ReduceOp, UnaryOp};
use std::cmp::Ordering;

/// Time units for datetime64.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TimeUnit {
    Nanoseconds,
    Microseconds,
    Milliseconds,
    Seconds,
}

/// DateTime64 dtype operations.
pub(super) struct DateTime64Ops {
    pub(super) unit: TimeUnit,
}

impl DateTime64Ops {
    #[inline]
    unsafe fn read(ptr: *const u8, byte_offset: isize) -> i64 {
        *(ptr.offset(byte_offset) as *const i64)
    }

    #[inline]
    unsafe fn write(ptr: *mut u8, idx: usize, val: i64) {
        *(ptr as *mut i64).add(idx) = val;
    }
}

impl DTypeOps for DateTime64Ops {
    fn kind(&self) -> DTypeKind { DTypeKind::DateTime64(self.unit) }
    fn itemsize(&self) -> usize { 8 }

    fn typestr(&self) -> &'static str {
        match self.unit {
            TimeUnit::Nanoseconds => "<M8[ns]",
            TimeUnit::Microseconds => "<M8[us]",
            TimeUnit::Milliseconds => "<M8[ms]",
            TimeUnit::Seconds => "<M8[s]",
        }
    }

    fn format_char(&self) -> &'static str { "q" } // int64 underlying

    fn name(&self) -> &'static str {
        match self.unit {
            TimeUnit::Nanoseconds => "datetime64[ns]",
            TimeUnit::Microseconds => "datetime64[us]",
            TimeUnit::Milliseconds => "datetime64[ms]",
            TimeUnit::Seconds => "datetime64[s]",
        }
    }

    fn promotion_priority(&self) -> u8 { 50 } // Doesn't promote with numerics

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
        // Most unary ops don't make sense for datetime, just copy
        let result = match op {
            UnaryOp::Neg => -v,
            UnaryOp::Abs => v.abs(),
            _ => v,
        };
        Self::write(out, idx, result);
    }

    unsafe fn binary_op(&self, op: BinaryOp, a: *const u8, a_offset: isize, b: *const u8, b_offset: isize, out: *mut u8, idx: usize) {
        let av = Self::read(a, a_offset);
        let bv = Self::read(b, b_offset);
        // Only add/sub make sense for datetime (as timedelta ops)
        let result = match op {
            BinaryOp::Add => av.wrapping_add(bv),
            BinaryOp::Sub => av.wrapping_sub(bv),
            BinaryOp::Mul => av.wrapping_mul(bv),
            BinaryOp::Div => if bv != 0 { av / bv } else { 0 },
            BinaryOp::Pow => if bv >= 0 { av.wrapping_pow(bv as u32) } else { 0 },
            BinaryOp::Mod => if bv != 0 { av % bv } else { 0 },
            BinaryOp::FloorDiv => if bv != 0 { av.div_euclid(bv) } else { 0 },
            BinaryOp::Maximum | BinaryOp::FMax => av.max(bv),
            BinaryOp::Minimum | BinaryOp::FMin => av.min(bv),
            // Float ops don't make sense for datetime; just return av
            BinaryOp::Arctan2 | BinaryOp::Hypot | BinaryOp::Copysign
            | BinaryOp::Logaddexp | BinaryOp::Logaddexp2 => av,
            BinaryOp::Nextafter => if av < bv { av.wrapping_add(1) } else if av > bv { av.wrapping_sub(1) } else { bv },
        };
        Self::write(out, idx, result);
    }

    unsafe fn reduce_init(&self, op: ReduceOp, out: *mut u8, idx: usize) {
        let v = match op {
            ReduceOp::Sum => 0,
            ReduceOp::Prod => 1,
            ReduceOp::Max => i64::MIN,
            ReduceOp::Min => i64::MAX,
        };
        Self::write(out, idx, v);
    }

    unsafe fn reduce_acc(&self, op: ReduceOp, acc: *mut u8, idx: usize, val: *const u8, byte_offset: isize) {
        let a = Self::read(acc as *const u8, (idx * 8) as isize);
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
        // Just show the raw integer value for now
        format!("{}", Self::read(ptr, byte_offset))
    }

    unsafe fn compare_elements(&self, a: *const u8, a_offset: isize, b: *const u8, b_offset: isize) -> Ordering {
        Self::read(a, a_offset).cmp(&Self::read(b, b_offset))
    }

    unsafe fn is_truthy(&self, ptr: *const u8, byte_offset: isize) -> bool {
        Self::read(ptr, byte_offset) != 0
    }

    unsafe fn write_f64(&self, ptr: *mut u8, idx: usize, val: f64) -> bool {
        Self::write(ptr, idx, val as i64);
        true
    }

    unsafe fn read_f64(&self, ptr: *const u8, byte_offset: isize) -> Option<f64> {
        Some(Self::read(ptr, byte_offset) as f64)
    }

    unsafe fn write_f64_at_byte_offset(&self, ptr: *mut u8, byte_offset: isize, val: f64) -> bool {
        *(ptr.offset(byte_offset) as *mut i64) = val as i64;
        true
    }

    unsafe fn write_complex(&self, ptr: *mut u8, idx: usize, real: f64, _imag: f64) -> bool {
        Self::write(ptr, idx, real as i64);
        true
    }

    unsafe fn read_complex(&self, ptr: *const u8, byte_offset: isize) -> Option<(f64, f64)> {
        Some((Self::read(ptr, byte_offset) as f64, 0.0))
    }
}
