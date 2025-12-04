//! Int32 dtype implementation.

use super::DTypeOps;

/// Int32 dtype operations.
pub(super) struct Int32Ops;

impl DTypeOps for Int32Ops {
    fn itemsize(&self) -> usize { 4 }
    fn typestr(&self) -> &'static str { "<i4" }
    fn format_char(&self) -> &'static str { "i" }
    fn name(&self) -> &'static str { "int32" }
    fn promotion_priority(&self) -> u8 { 70 }

    unsafe fn read_element(&self, ptr: *const u8, byte_offset: isize) -> f64 {
        *(ptr.offset(byte_offset) as *const i32) as f64
    }

    unsafe fn write_element(&self, ptr: *mut u8, idx: usize, val: f64) {
        *(ptr as *mut i32).add(idx) = val as i32;
    }
}
