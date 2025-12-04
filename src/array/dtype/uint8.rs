//! Uint8 dtype implementation.

use super::{DTypeKind, DTypeOps};

/// Uint8 dtype operations.
pub(super) struct Uint8Ops;

impl DTypeOps for Uint8Ops {
    fn kind(&self) -> DTypeKind { DTypeKind::Uint8 }
    fn itemsize(&self) -> usize { 1 }
    fn typestr(&self) -> &'static str { "|u1" }
    fn format_char(&self) -> &'static str { "B" }
    fn name(&self) -> &'static str { "uint8" }
    fn promotion_priority(&self) -> u8 { 30 }

    unsafe fn read_element(&self, ptr: *const u8, byte_offset: isize) -> f64 {
        *ptr.offset(byte_offset) as f64
    }

    unsafe fn write_element(&self, ptr: *mut u8, idx: usize, val: f64) {
        *ptr.add(idx) = val as u8;
    }
}
