//! Uint32 dtype implementation.

use super::{DTypeKind, DTypeOps};

/// Uint32 dtype operations.
pub(super) struct Uint32Ops;

impl DTypeOps for Uint32Ops {
    fn kind(&self) -> DTypeKind { DTypeKind::Uint32 }
    fn itemsize(&self) -> usize { 4 }
    fn typestr(&self) -> &'static str { "<u4" }
    fn format_char(&self) -> &'static str { "I" }
    fn name(&self) -> &'static str { "uint32" }
    fn promotion_priority(&self) -> u8 { 60 }

    unsafe fn read_element(&self, ptr: *const u8, byte_offset: isize) -> f64 {
        *(ptr.offset(byte_offset) as *const u32) as f64
    }

    unsafe fn write_element(&self, ptr: *mut u8, idx: usize, val: f64) {
        *(ptr as *mut u32).add(idx) = val as u32;
    }
}
