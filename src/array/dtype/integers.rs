//! Integer dtype implementations (signed and unsigned).

use super::macros::{impl_signed_int_dtype, impl_unsigned_int_dtype};
use super::{BinaryOp, DTypeKind, DTypeOps, ReduceOp, UnaryOp};

// Signed integers
impl_signed_int_dtype!(Int64Ops, i64, 8, Int64, "int64", "<i8", "q", 80);
impl_signed_int_dtype!(Int32Ops, i32, 4, Int32, "int32", "<i4", "i", 70);
impl_signed_int_dtype!(Int16Ops, i16, 2, Int16, "int16", "<i2", "h", 60);

// Unsigned integers
impl_unsigned_int_dtype!(Uint64Ops, u64, 8, Uint64, "uint64", "<u8", "Q", 75);
impl_unsigned_int_dtype!(Uint32Ops, u32, 4, Uint32, "uint32", "<u4", "I", 65);
impl_unsigned_int_dtype!(Uint16Ops, u16, 2, Uint16, "uint16", "<u2", "H", 55);
impl_unsigned_int_dtype!(Uint8Ops, u8, 1, Uint8, "uint8", "|u1", "B", 30);
