//! Uint64 dtype implementation.

use super::{DTypeKind, DTypeOps};

/// Uint64 dtype operations.
pub(super) struct Uint64Ops;

impl DTypeOps for Uint64Ops {
    fn kind(&self) -> DTypeKind { DTypeKind::Uint64 }
    fn itemsize(&self) -> usize { 8 }
    fn typestr(&self) -> &'static str { "<u8" }
    fn format_char(&self) -> &'static str { "Q" }
    fn name(&self) -> &'static str { "uint64" }
    fn promotion_priority(&self) -> u8 { 70 }

    unsafe fn read_element(&self, ptr: *const u8, byte_offset: isize) -> f64 {
        *(ptr.offset(byte_offset) as *const u64) as f64
    }

    unsafe fn write_element(&self, ptr: *mut u8, idx: usize, val: f64) {
        *(ptr as *mut u64).add(idx) = val as u64;
    }
}
