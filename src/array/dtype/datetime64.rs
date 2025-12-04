//! DateTime64 dtype implementation - a parametric dtype.

use super::{DTypeKind, DTypeOps};

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

    unsafe fn read_element(&self, ptr: *const u8, byte_offset: isize) -> f64 {
        *(ptr.offset(byte_offset) as *const i64) as f64
    }

    unsafe fn write_element(&self, ptr: *mut u8, idx: usize, val: f64) {
        *(ptr as *mut i64).add(idx) = val as i64;
    }
}
