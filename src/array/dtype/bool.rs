//! Bool dtype implementation.

use super::{DTypeKind, DTypeOps};

/// Bool dtype operations.
pub(super) struct BoolOps;

impl DTypeOps for BoolOps {
    fn kind(&self) -> DTypeKind { DTypeKind::Bool }
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

    fn format_element(&self, val: f64) -> String {
        if val != 0.0 { "True".to_string() } else { "False".to_string() }
    }
}
