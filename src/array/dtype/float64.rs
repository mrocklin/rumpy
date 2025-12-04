//! Float64 dtype implementation.

use super::{DTypeKind, DTypeOps};

/// Float64 dtype operations.
pub(super) struct Float64Ops;

impl DTypeOps for Float64Ops {
    fn kind(&self) -> DTypeKind { DTypeKind::Float64 }
    fn itemsize(&self) -> usize { 8 }
    fn typestr(&self) -> &'static str { "<f8" }
    fn format_char(&self) -> &'static str { "d" }
    fn name(&self) -> &'static str { "float64" }
    fn promotion_priority(&self) -> u8 { 100 }

    unsafe fn read_element(&self, ptr: *const u8, byte_offset: isize) -> f64 {
        *(ptr.offset(byte_offset) as *const f64)
    }

    unsafe fn write_element(&self, ptr: *mut u8, idx: usize, val: f64) {
        *(ptr as *mut f64).add(idx) = val;
    }
}
