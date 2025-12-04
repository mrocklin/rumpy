//! Bool dtype implementation.

use super::DTypeOps;

/// Bool dtype operations.
pub(super) struct BoolOps;

impl DTypeOps for BoolOps {
    fn itemsize(&self) -> usize { 1 }
    fn typestr(&self) -> &'static str { "|b1" }
    fn format_char(&self) -> &'static str { "?" }
    fn name(&self) -> &'static str { "bool" }
    fn promotion_priority(&self) -> u8 { 10 }

    unsafe fn read_element(&self, ptr: *const u8, byte_offset: isize) -> f64 {
        *ptr.offset(byte_offset) as f64
    }

    unsafe fn write_element(&self, ptr: *mut u8, idx: usize, val: f64) {
        *ptr.add(idx) = (val != 0.0) as u8;
    }
}
