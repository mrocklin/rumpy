//! Int64 dtype implementation.

use super::{DTypeKind, DTypeOps};

/// Int64 dtype operations.
pub(super) struct Int64Ops;

impl DTypeOps for Int64Ops {
    fn kind(&self) -> DTypeKind { DTypeKind::Int64 }
    fn itemsize(&self) -> usize { 8 }
    fn typestr(&self) -> &'static str { "<i8" }
    fn format_char(&self) -> &'static str { "q" }
    fn name(&self) -> &'static str { "int64" }
    fn promotion_priority(&self) -> u8 { 80 }

    unsafe fn read_element(&self, ptr: *const u8, byte_offset: isize) -> f64 {
        *(ptr.offset(byte_offset) as *const i64) as f64
    }

    unsafe fn write_element(&self, ptr: *mut u8, idx: usize, val: f64) {
        *(ptr as *mut i64).add(idx) = val as i64;
    }
}
