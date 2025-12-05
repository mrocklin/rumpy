//! Extensible data type system for rumpy arrays.
//!
//! `DType` wraps `Arc<dyn DTypeOps>`, enabling parametric types like datetime[ns].
//!
//! Operations work directly on buffers - each dtype uses its native type internally.
//! There is no universal Rust value type; Python interop (PyObject) is handled separately.

mod bool;
mod complex128;
mod datetime64;
mod float32;
mod float64;
mod int32;
mod int64;
mod uint8;
mod uint32;
mod uint64;

use self::bool::BoolOps;
use complex128::Complex128Ops;
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
    Complex128,
}

/// Unary operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UnaryOp {
    Neg,
    Abs,
    Sqrt,
    Exp,
    Log,
    Sin,
    Cos,
    Tan,
}

/// Binary operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
}

/// Reduce operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ReduceOp {
    Sum,
    Prod,
    Max,
    Min,
}

/// Trait defining all dtype-specific behavior.
///
/// Operations work directly on buffers - each dtype uses its native type internally.
/// Implement this trait to add a new dtype.
pub trait DTypeOps: Send + Sync + 'static {
    // === Metadata ===

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

    /// Priority for type promotion. Higher priority wins.
    fn promotion_priority(&self) -> u8;

    /// Whether this is an integer type.
    fn is_integer(&self) -> bool {
        matches!(self.kind(), DTypeKind::Int32 | DTypeKind::Int64 |
                 DTypeKind::Uint8 | DTypeKind::Uint32 | DTypeKind::Uint64)
    }

    // === Buffer operations ===

    /// Write zero value at element index.
    ///
    /// # Safety
    /// Caller must ensure ptr points to valid memory for idx elements.
    unsafe fn write_zero(&self, ptr: *mut u8, idx: usize);

    /// Write one value at element index.
    ///
    /// # Safety
    /// Caller must ensure ptr points to valid memory for idx elements.
    unsafe fn write_one(&self, ptr: *mut u8, idx: usize);

    /// Copy element from src (at byte_offset) to dst (at element index).
    ///
    /// # Safety
    /// Both pointers must be valid.
    unsafe fn copy_element(&self, src: *const u8, byte_offset: isize, dst: *mut u8, idx: usize);

    /// Apply unary operation: out[idx] = op(src at byte_offset).
    ///
    /// # Safety
    /// Both pointers must be valid.
    unsafe fn unary_op(&self, op: UnaryOp, src: *const u8, byte_offset: isize, out: *mut u8, idx: usize);

    /// Apply binary operation: out[idx] = a op b.
    ///
    /// # Safety
    /// All pointers must be valid.
    unsafe fn binary_op(&self, op: BinaryOp, a: *const u8, a_offset: isize, b: *const u8, b_offset: isize, out: *mut u8, idx: usize);

    /// Write reduction identity value at element index.
    ///
    /// # Safety
    /// Pointer must be valid.
    unsafe fn reduce_init(&self, op: ReduceOp, out: *mut u8, idx: usize);

    /// Accumulate: acc[idx] = op(acc[idx], val at byte_offset).
    ///
    /// # Safety
    /// Both pointers must be valid.
    unsafe fn reduce_acc(&self, op: ReduceOp, acc: *mut u8, idx: usize, val: *const u8, byte_offset: isize);

    /// Format element at byte_offset as string for repr/str.
    ///
    /// # Safety
    /// Pointer must be valid.
    unsafe fn format_element(&self, ptr: *const u8, byte_offset: isize) -> String;

    /// Compare two elements for ordering (for sort/argmax/etc).
    /// Returns Ordering.
    ///
    /// # Safety
    /// Both pointers must be valid.
    unsafe fn compare_elements(&self, a: *const u8, a_offset: isize, b: *const u8, b_offset: isize) -> std::cmp::Ordering;

    /// Check if element is "truthy" (nonzero). Used for boolean indexing.
    ///
    /// # Safety
    /// Pointer must be valid.
    unsafe fn is_truthy(&self, ptr: *const u8, byte_offset: isize) -> bool;

    // === Scalar conversion (for Python interop, implemented in python module) ===
    // These are handled separately to avoid PyO3 dependency in core dtype module.

    /// Write a scalar f64 value at element index.
    /// Used for creating arrays from Python floats/ints.
    /// Returns false if this dtype doesn't support f64 conversion.
    ///
    /// # Safety
    /// Pointer must be valid.
    unsafe fn write_f64(&self, ptr: *mut u8, idx: usize, val: f64) -> bool {
        let _ = (ptr, idx, val);
        false
    }

    /// Read element as f64 (if possible).
    /// Returns None if this dtype doesn't support f64 conversion.
    ///
    /// # Safety
    /// Pointer must be valid.
    unsafe fn read_f64(&self, ptr: *const u8, byte_offset: isize) -> Option<f64> {
        let _ = (ptr, byte_offset);
        None
    }

    /// Write a complex value (real, imag) at element index.
    /// Returns false if this dtype doesn't support complex conversion.
    ///
    /// # Safety
    /// Pointer must be valid.
    unsafe fn write_complex(&self, ptr: *mut u8, idx: usize, real: f64, imag: f64) -> bool {
        let _ = (ptr, idx, real, imag);
        false
    }

    /// Read element as complex (real, imag) if possible.
    ///
    /// # Safety
    /// Pointer must be valid.
    unsafe fn read_complex(&self, ptr: *const u8, byte_offset: isize) -> Option<(f64, f64)> {
        let _ = (ptr, byte_offset);
        None
    }
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
    pub fn complex128() -> Self { DType(Arc::new(Complex128Ops)) }

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
            "complex128" | "c16" | "<c16" => Some(Self::complex128()),
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

/// Promote two dtypes to their common type (NumPy-compatible).
///
/// Rules:
/// 1. Same type -> same type
/// 2. Complex involved -> complex128
/// 3. Float + int -> float64 if int >= 32 bits and float == float32
/// 4. Float + float -> higher precision
/// 5. Int + int -> type that can hold both
/// 6. Bool acts like int8
pub fn promote_dtype(a: &DType, b: &DType) -> DType {
    use DTypeKind::*;

    let ak = a.kind();
    let bk = b.kind();

    // Same type
    if ak == bk {
        return a.clone();
    }

    // Complex always wins
    if matches!(ak, Complex128) || matches!(bk, Complex128) {
        return DType::complex128();
    }

    // Datetime doesn't promote with other types
    if matches!(ak, DateTime64(_)) || matches!(bk, DateTime64(_)) {
        // Return one of them; ops will likely fail anyway
        return a.clone();
    }

    // Helper to check if dtype is float32
    let is_f32 = |k: &DTypeKind| matches!(k, Float32);
    let is_f64 = |k: &DTypeKind| matches!(k, Float64);
    let is_float = |k: &DTypeKind| matches!(k, Float32 | Float64);

    // Helper to check if dtype is "large" int (32+ bits)
    let is_large_int = |k: &DTypeKind| matches!(k, Int32 | Int64 | Uint32 | Uint64);

    // Float + int combinations
    if is_float(&ak) || is_float(&bk) {
        // One is float, check if int can safely cast
        let (float_kind, int_kind) = if is_float(&ak) { (&ak, &bk) } else { (&bk, &ak) };

        // int32/int64/uint32/uint64 + float32 -> float64
        if is_f32(float_kind) && is_large_int(int_kind) {
            return DType::float64();
        }

        // Otherwise, use higher priority (float64 > float32 > ints)
        if is_f64(float_kind) || is_f64(int_kind) {
            return DType::float64();
        }
        if is_f32(float_kind) {
            return DType::float32();
        }
    }

    // Both are integers (or bool)
    // Use priority system for int+int
    if a.ops().promotion_priority() >= b.ops().promotion_priority() {
        a.clone()
    } else {
        b.clone()
    }
}
