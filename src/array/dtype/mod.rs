//! Extensible data type system for rumpy arrays.
//!
//! Each dtype is a unit struct implementing `DTypeOps`. The `DType` enum
//! provides a lightweight handle that delegates to trait objects.

mod bool;
mod float32;
mod float64;
mod int32;
mod int64;

use self::bool::BoolOps;
use float32::Float32Ops;
use float64::Float64Ops;
use int32::Int32Ops;
use int64::Int64Ops;

/// Trait defining all dtype-specific behavior.
///
/// Implement this trait to add a new dtype. Then add a variant to `DType`
/// and a single match arm in `DType::ops()`.
pub trait DTypeOps: Send + Sync + 'static {
    // === Metadata ===

    /// Size of one element in bytes.
    fn itemsize(&self) -> usize;

    /// NumPy type string for __array_interface__ (e.g., "<f8").
    fn typestr(&self) -> &'static str;

    /// Buffer protocol format character (e.g., "d" for f64).
    fn format_char(&self) -> &'static str;

    /// Human-readable name (e.g., "float64").
    fn name(&self) -> &'static str;

    // === Element access ===

    /// Read element from buffer at byte offset, returning as f64.
    ///
    /// # Safety
    /// Caller must ensure ptr + byte_offset is valid and aligned.
    unsafe fn read_element(&self, ptr: *const u8, byte_offset: isize) -> f64;

    /// Write f64 value to buffer at element index (not byte offset).
    ///
    /// # Safety
    /// Caller must ensure ptr points to valid memory for idx elements.
    unsafe fn write_element(&self, ptr: *mut u8, idx: usize, val: f64);

    // === Value creation ===

    /// The zero value for this dtype (as f64).
    fn zero_value(&self) -> f64 {
        0.0
    }

    /// The one value for this dtype (as f64).
    fn one_value(&self) -> f64 {
        1.0
    }

    // === Type promotion ===

    /// Priority for type promotion. Higher priority wins.
    /// Float64=100, Float32=90, Int64=80, Int32=70, Bool=10
    fn promotion_priority(&self) -> u8;
}

/// Data types supported by rumpy arrays.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DType {
    Float32,
    Float64,
    Int32,
    Int64,
    Bool,
}

impl DType {
    /// Get the ops trait object for this dtype.
    ///
    /// This is the single dispatch point - all dtype-specific behavior
    /// goes through here.
    pub fn ops(&self) -> &'static dyn DTypeOps {
        match self {
            DType::Float32 => &Float32Ops,
            DType::Float64 => &Float64Ops,
            DType::Int32 => &Int32Ops,
            DType::Int64 => &Int64Ops,
            DType::Bool => &BoolOps,
        }
    }

    /// Size in bytes of one element.
    #[inline]
    pub fn itemsize(&self) -> usize {
        self.ops().itemsize()
    }

    /// NumPy type string for __array_interface__.
    #[inline]
    pub fn typestr(&self) -> &'static str {
        self.ops().typestr()
    }

    /// Type character for buffer protocol format.
    #[inline]
    pub fn format_char(&self) -> &'static str {
        self.ops().format_char()
    }

    /// Parse dtype from string (numpy-style).
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "float32" | "f4" | "<f4" => Some(DType::Float32),
            "float64" | "f8" | "<f8" | "float" => Some(DType::Float64),
            "int32" | "i4" | "<i4" => Some(DType::Int32),
            "int64" | "i8" | "<i8" | "int" => Some(DType::Int64),
            "bool" | "?" | "|b1" => Some(DType::Bool),
            _ => None,
        }
    }
}

/// Promote two dtypes to their common type.
/// Uses priority-based promotion: higher priority wins.
pub fn promote_dtype(a: DType, b: DType) -> DType {
    if a.ops().promotion_priority() >= b.ops().promotion_priority() {
        a
    } else {
        b
    }
}
