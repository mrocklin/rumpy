//! Float32 dtype implementation.

use super::DTypeOps;

/// Float32 dtype operations.
pub(super) struct Float32Ops;

impl DTypeOps for Float32Ops {
    fn itemsize(&self) -> usize { 4 }
    fn typestr(&self) -> &'static str { "<f4" }
    fn format_char(&self) -> &'static str { "f" }
    fn name(&self) -> &'static str { "float32" }
    fn promotion_priority(&self) -> u8 { 90 }

    unsafe fn read_element(&self, ptr: *const u8, byte_offset: isize) -> f64 {
        *(ptr.offset(byte_offset) as *const f32) as f64
    }

    unsafe fn write_element(&self, ptr: *mut u8, idx: usize, val: f64) {
        *(ptr as *mut f32).add(idx) = val as f32;
    }
}
