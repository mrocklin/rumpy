//! Extensible data type system for rumpy arrays.
//!
//! `DType` wraps `Arc<dyn DTypeOps>`, enabling parametric types like datetime[ns].

mod bool;
mod datetime64;
mod float32;
mod float64;
mod int32;
mod int64;
mod uint8;
mod uint32;
mod uint64;

use self::bool::BoolOps;
use datetime64::DateTime64Ops;
pub use datetime64::TimeUnit;
use float32::Float32Ops;
use float64::Float64Ops;
use int32::Int32Ops;
use int64::Int64Ops;
use uint8::Uint8Ops;
use uint32::Uint32Ops;
use uint64::Uint64Ops;

use std::sync::Arc;
use std::hash::{Hash, Hasher};

/// Identifies the kind of dtype for equality/hashing.
/// Parametric types include their parameters.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum DTypeKind {
    Float32,
    Float64,
    Int32,
    Int64,
    Uint8,
    Uint32,
    Uint64,
    Bool,
    DateTime64(TimeUnit),
}

/// Trait defining all dtype-specific behavior.
///
/// Implement this trait to add a new dtype.
pub trait DTypeOps: Send + Sync + 'static {
    /// The kind of this dtype (for equality/hashing).
    fn kind(&self) -> DTypeKind;

    /// Size of one element in bytes.
    fn itemsize(&self) -> usize;

    /// NumPy type string for __array_interface__ (e.g., "<f8").
    fn typestr(&self) -> &'static str;

    /// Buffer protocol format character (e.g., "d" for f64).
    fn format_char(&self) -> &'static str;

    /// Human-readable name (e.g., "float64").
    fn name(&self) -> &'static str;

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

    /// The zero value for this dtype (as f64).
    fn zero_value(&self) -> f64 { 0.0 }

    /// The one value for this dtype (as f64).
    fn one_value(&self) -> f64 { 1.0 }

    /// Priority for type promotion. Higher priority wins.
    fn promotion_priority(&self) -> u8;
}

/// Data type descriptor wrapping trait object.
///
/// Supports parametric types (e.g., datetime[ns]) via Arc<dyn DTypeOps>.
#[derive(Clone)]
pub struct DType(Arc<dyn DTypeOps>);

impl DType {
    /// Create from any DTypeOps implementation.
    pub fn new<T: DTypeOps>(ops: T) -> Self {
        DType(Arc::new(ops))
    }

    /// Get the underlying ops trait object.
    #[inline]
    pub fn ops(&self) -> &dyn DTypeOps {
        &*self.0
    }

    /// Get the kind for pattern matching.
    #[inline]
    pub fn kind(&self) -> DTypeKind {
        self.0.kind()
    }

    // === Convenience constructors ===

    pub fn float32() -> Self { DType(Arc::new(Float32Ops)) }
    pub fn float64() -> Self { DType(Arc::new(Float64Ops)) }
    pub fn int32() -> Self { DType(Arc::new(Int32Ops)) }
    pub fn int64() -> Self { DType(Arc::new(Int64Ops)) }
    pub fn uint8() -> Self { DType(Arc::new(Uint8Ops)) }
    pub fn uint32() -> Self { DType(Arc::new(Uint32Ops)) }
    pub fn uint64() -> Self { DType(Arc::new(Uint64Ops)) }
    pub fn bool() -> Self { DType(Arc::new(BoolOps)) }
    pub fn datetime64(unit: TimeUnit) -> Self { DType(Arc::new(DateTime64Ops { unit })) }
    pub fn datetime64_ns() -> Self { Self::datetime64(TimeUnit::Nanoseconds) }
    pub fn datetime64_us() -> Self { Self::datetime64(TimeUnit::Microseconds) }
    pub fn datetime64_ms() -> Self { Self::datetime64(TimeUnit::Milliseconds) }
    pub fn datetime64_s() -> Self { Self::datetime64(TimeUnit::Seconds) }

    // === Delegated methods ===

    #[inline]
    pub fn itemsize(&self) -> usize { self.0.itemsize() }

    #[inline]
    pub fn typestr(&self) -> &'static str { self.0.typestr() }

    #[inline]
    pub fn format_char(&self) -> &'static str { self.0.format_char() }

    /// Parse dtype from string (numpy-style).
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "float32" | "f4" | "<f4" => Some(Self::float32()),
            "float64" | "f8" | "<f8" | "float" => Some(Self::float64()),
            "int32" | "i4" | "<i4" => Some(Self::int32()),
            "int64" | "i8" | "<i8" | "int" => Some(Self::int64()),
            "uint8" | "u1" | "|u1" => Some(Self::uint8()),
            "uint32" | "u4" | "<u4" => Some(Self::uint32()),
            "uint64" | "u8" | "<u8" => Some(Self::uint64()),
            "bool" | "?" | "|b1" => Some(Self::bool()),
            "datetime64[ns]" | "<M8[ns]" => Some(Self::datetime64_ns()),
            "datetime64[us]" | "<M8[us]" => Some(Self::datetime64_us()),
            "datetime64[ms]" | "<M8[ms]" => Some(Self::datetime64_ms()),
            "datetime64[s]" | "<M8[s]" => Some(Self::datetime64_s()),
            _ => None,
        }
    }
}

impl PartialEq for DType {
    fn eq(&self, other: &Self) -> bool {
        self.0.kind() == other.0.kind()
    }
}

impl Eq for DType {}

impl Hash for DType {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.kind().hash(state);
    }
}

impl std::fmt::Debug for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DType({})", self.0.name())
    }
}

/// Promote two dtypes to their common type.
pub fn promote_dtype(a: &DType, b: &DType) -> DType {
    if a.ops().promotion_priority() >= b.ops().promotion_priority() {
        a.clone()
    } else {
        b.clone()
    }
}
