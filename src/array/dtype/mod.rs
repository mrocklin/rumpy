//! Extensible data type system for rumpy arrays.
//!
//! `DType` wraps `Arc<dyn DTypeOps>`, enabling parametric types like datetime[ns].
//!
//! Operations work directly on buffers - each dtype uses its native type internally.
//! There is no universal Rust value type; Python interop (PyObject) is handled separately.

#![allow(clippy::too_many_arguments)]

mod macros;
mod floats;
mod integers;
mod float16;
mod bool;
mod complex64;
mod complex128;
mod datetime64;

use self::bool::BoolOps;
use complex64::Complex64Ops;
use complex128::Complex128Ops;
use datetime64::DateTime64Ops;
pub use datetime64::TimeUnit;
use float16::Float16Ops;
use floats::{Float32Ops, Float64Ops};
use integers::{Int16Ops, Int32Ops, Int64Ops, Uint8Ops, Uint16Ops, Uint32Ops, Uint64Ops};

use std::sync::Arc;
use std::hash::{Hash, Hasher};
use std::str::FromStr;

/// Identifies the kind of dtype for equality/hashing.
/// Parametric types include their parameters.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum DTypeKind {
    Float16,
    Float32,
    Float64,
    Int16,
    Int32,
    Int64,
    Uint8,
    Uint16,
    Uint32,
    Uint64,
    Bool,
    DateTime64(TimeUnit),
    Complex64,
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
    Log10,
    Log2,
    Sin,
    Cos,
    Tan,
    Sinh,
    Cosh,
    Tanh,
    Floor,
    Ceil,
    Arcsin,
    Arccos,
    Arctan,
    Sign,
    Isnan,
    Isinf,
    Isfinite,
}

/// Binary operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Mod,
    FloorDiv,
    Maximum,
    Minimum,
    // Stream 2: Binary Math Operations
    Arctan2,
    Hypot,
    FMax,      // max ignoring NaN
    FMin,      // min ignoring NaN
    Copysign,
    Logaddexp,
    Logaddexp2,
    Nextafter,
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
        matches!(self.kind(), DTypeKind::Int16 | DTypeKind::Int32 | DTypeKind::Int64 |
                 DTypeKind::Uint8 | DTypeKind::Uint16 | DTypeKind::Uint32 | DTypeKind::Uint64)
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

    /// Write f64 at a given byte offset (for strided array access).
    /// Returns false if this dtype doesn't support f64 conversion.
    ///
    /// # Safety
    /// Pointer must be valid.
    unsafe fn write_f64_at_byte_offset(&self, ptr: *mut u8, byte_offset: isize, val: f64) -> bool {
        let _ = (ptr, byte_offset, val);
        false
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

    pub fn float16() -> Self { DType(Arc::new(Float16Ops)) }
    pub fn float32() -> Self { DType(Arc::new(Float32Ops)) }
    pub fn float64() -> Self { DType(Arc::new(Float64Ops)) }
    pub fn int16() -> Self { DType(Arc::new(Int16Ops)) }
    pub fn int32() -> Self { DType(Arc::new(Int32Ops)) }
    pub fn int64() -> Self { DType(Arc::new(Int64Ops)) }
    pub fn uint8() -> Self { DType(Arc::new(Uint8Ops)) }
    pub fn uint16() -> Self { DType(Arc::new(Uint16Ops)) }
    pub fn uint32() -> Self { DType(Arc::new(Uint32Ops)) }
    pub fn uint64() -> Self { DType(Arc::new(Uint64Ops)) }
    pub fn bool() -> Self { DType(Arc::new(BoolOps)) }
    pub fn datetime64(unit: TimeUnit) -> Self { DType(Arc::new(DateTime64Ops { unit })) }
    pub fn datetime64_ns() -> Self { Self::datetime64(TimeUnit::Nanoseconds) }
    pub fn datetime64_us() -> Self { Self::datetime64(TimeUnit::Microseconds) }
    pub fn datetime64_ms() -> Self { Self::datetime64(TimeUnit::Milliseconds) }
    pub fn datetime64_s() -> Self { Self::datetime64(TimeUnit::Seconds) }
    pub fn complex64() -> Self { DType(Arc::new(Complex64Ops)) }
    pub fn complex128() -> Self { DType(Arc::new(Complex128Ops)) }

    // === Delegated methods ===

    #[inline]
    pub fn itemsize(&self) -> usize { self.0.itemsize() }

    #[inline]
    pub fn typestr(&self) -> &'static str { self.0.typestr() }

    #[inline]
    pub fn format_char(&self) -> &'static str { self.0.format_char() }

    /// Parse dtype from string (numpy-style).
    pub fn parse(s: &str) -> Option<Self> {
        s.parse().ok()
    }
}

impl FromStr for DType {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "float16" | "f2" | "<f2" => Ok(Self::float16()),
            "float32" | "f4" | "<f4" => Ok(Self::float32()),
            "float64" | "f8" | "<f8" | "float" => Ok(Self::float64()),
            "int16" | "i2" | "<i2" => Ok(Self::int16()),
            "int32" | "i4" | "<i4" => Ok(Self::int32()),
            "int64" | "i8" | "<i8" | "int" => Ok(Self::int64()),
            "uint8" | "u1" | "|u1" => Ok(Self::uint8()),
            "uint16" | "u2" | "<u2" => Ok(Self::uint16()),
            "uint32" | "u4" | "<u4" => Ok(Self::uint32()),
            "uint64" | "u8" | "<u8" => Ok(Self::uint64()),
            "bool" | "?" | "|b1" => Ok(Self::bool()),
            "datetime64[ns]" | "<M8[ns]" => Ok(Self::datetime64_ns()),
            "datetime64[us]" | "<M8[us]" => Ok(Self::datetime64_us()),
            "datetime64[ms]" | "<M8[ms]" => Ok(Self::datetime64_ms()),
            "datetime64[s]" | "<M8[s]" => Ok(Self::datetime64_s()),
            "complex64" | "c8" | "<c8" => Ok(Self::complex64()),
            "complex128" | "c16" | "<c16" => Ok(Self::complex128()),
            _ => Err(()),
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

    // Complex types promote to higher-precision complex
    let is_complex = |k: &DTypeKind| matches!(k, Complex64 | Complex128);
    if is_complex(&ak) || is_complex(&bk) {
        // If either is complex128, result is complex128
        if matches!(ak, Complex128) || matches!(bk, Complex128) {
            return DType::complex128();
        }
        // Both are complex64 or one is complex64 + real type
        // complex64 + float64 -> complex128, complex64 + anything else -> complex64
        if matches!(ak, Float64) || matches!(bk, Float64) {
            return DType::complex128();
        }
        if matches!(ak, Int64 | Uint64 | Int32 | Uint32) || matches!(bk, Int64 | Uint64 | Int32 | Uint32) {
            return DType::complex128();
        }
        return DType::complex64();
    }

    // Datetime doesn't promote with other types
    if matches!(ak, DateTime64(_)) || matches!(bk, DateTime64(_)) {
        // Return one of them; ops will likely fail anyway
        return a.clone();
    }

    // Helper to check if dtype is float
    let is_f16 = |k: &DTypeKind| matches!(k, Float16);
    let is_f32 = |k: &DTypeKind| matches!(k, Float32);
    let is_f64 = |k: &DTypeKind| matches!(k, Float64);
    let is_float = |k: &DTypeKind| matches!(k, Float16 | Float32 | Float64);

    // Helper to check int size categories
    let is_large_int = |k: &DTypeKind| matches!(k, Int32 | Int64 | Uint32 | Uint64);  // 4+ bytes
    let is_medium_int = |k: &DTypeKind| matches!(k, Int16 | Uint16);  // 2 bytes

    // Float + int combinations
    if is_float(&ak) || is_float(&bk) {
        // One is float, check if int can safely cast
        let (float_kind, int_kind) = if is_float(&ak) { (&ak, &bk) } else { (&bk, &ak) };

        // int32/int64/uint32/uint64 + float32 -> float64
        if is_f32(float_kind) && is_large_int(int_kind) {
            return DType::float64();
        }
        // int32/int64/uint32/uint64 + float16 -> float64
        if is_f16(float_kind) && is_large_int(int_kind) {
            return DType::float64();
        }
        // int16/uint16 + float16 -> float32 (medium ints exceed float16 precision)
        if is_f16(float_kind) && is_medium_int(int_kind) {
            return DType::float32();
        }

        // Float + float: use higher precision
        if is_f64(float_kind) || is_f64(int_kind) {
            return DType::float64();
        }
        if is_f32(float_kind) || is_f32(int_kind) {
            return DType::float32();
        }
        if is_f16(float_kind) {
            return DType::float16();
        }
    }

    // Both are integers (or bool)
    // NumPy promotes signed+unsigned to a type that can hold both ranges
    let is_signed = |k: &DTypeKind| matches!(k, Int16 | Int32 | Int64);
    let is_unsigned = |k: &DTypeKind| matches!(k, Uint8 | Uint16 | Uint32 | Uint64);

    if (is_signed(&ak) && is_unsigned(&bk)) || (is_unsigned(&ak) && is_signed(&bk)) {
        let (signed_dtype, unsigned_dtype) = if is_signed(&ak) { (a, b) } else { (b, a) };
        let signed_size = signed_dtype.ops().itemsize();
        let unsigned_size = unsigned_dtype.ops().itemsize();

        // If signed type is larger than unsigned, it can hold unsigned range
        if signed_size > unsigned_size {
            return signed_dtype.clone();
        }
        // Same size or unsigned larger: need next larger signed type
        let max_size = signed_size.max(unsigned_size);
        return match max_size {
            1 => DType::int16(),   // uint8 + int8 -> int16
            2 => DType::int32(),   // uint16 + int16 -> int32
            4 => DType::int64(),   // uint32 + int32 -> int64
            _ => DType::float64(), // uint64 + int64 -> float64 (no int can hold both)
        };
    }

    // Same sign integers: use priority system
    if a.ops().promotion_priority() >= b.ops().promotion_priority() {
        a.clone()
    } else {
        b.clone()
    }
}
