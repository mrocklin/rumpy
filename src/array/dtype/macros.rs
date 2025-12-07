//! Macros for reducing dtype implementation boilerplate.

/// Implement DTypeOps for a floating-point type.
macro_rules! impl_float_dtype {
    ($struct_name:ident, $T:ty, $size:expr, $kind:ident, $name:expr, $typestr:expr, $format:expr, $priority:expr) => {
        pub(super) struct $struct_name;

        impl $struct_name {
            #[inline]
            unsafe fn read(ptr: *const u8, byte_offset: isize) -> $T {
                *(ptr.offset(byte_offset) as *const $T)
            }

            #[inline]
            unsafe fn write(ptr: *mut u8, idx: usize, val: $T) {
                *(ptr as *mut $T).add(idx) = val;
            }

            #[inline]
            unsafe fn write_at_offset(ptr: *mut u8, byte_offset: isize, val: $T) {
                *(ptr.offset(byte_offset) as *mut $T) = val;
            }
        }

        impl DTypeOps for $struct_name {
            fn kind(&self) -> DTypeKind { DTypeKind::$kind }
            fn itemsize(&self) -> usize { $size }
            fn typestr(&self) -> &'static str { $typestr }
            fn format_char(&self) -> &'static str { $format }
            fn name(&self) -> &'static str { $name }
            fn promotion_priority(&self) -> u8 { $priority }

            unsafe fn write_zero(&self, ptr: *mut u8, idx: usize) {
                Self::write(ptr, idx, 0.0);
            }

            unsafe fn write_one(&self, ptr: *mut u8, idx: usize) {
                Self::write(ptr, idx, 1.0);
            }

            unsafe fn copy_element(&self, src: *const u8, byte_offset: isize, dst: *mut u8, idx: usize) {
                Self::write(dst, idx, Self::read(src, byte_offset));
            }

            unsafe fn unary_op(&self, op: UnaryOp, src: *const u8, byte_offset: isize, out: *mut u8, idx: usize) {
                let v = Self::read(src, byte_offset);
                let result = match op {
                    UnaryOp::Neg => -v,
                    UnaryOp::Abs => v.abs(),
                    UnaryOp::Sqrt => v.sqrt(),
                    UnaryOp::Exp => v.exp(),
                    UnaryOp::Log => v.ln(),
                    UnaryOp::Log10 => v.log10(),
                    UnaryOp::Log2 => v.log2(),
                    UnaryOp::Sin => v.sin(),
                    UnaryOp::Cos => v.cos(),
                    UnaryOp::Tan => v.tan(),
                    UnaryOp::Sinh => v.sinh(),
                    UnaryOp::Cosh => v.cosh(),
                    UnaryOp::Tanh => v.tanh(),
                    UnaryOp::Floor => v.floor(),
                    UnaryOp::Ceil => v.ceil(),
                    UnaryOp::Arcsin => v.asin(),
                    UnaryOp::Arccos => v.acos(),
                    UnaryOp::Arctan => v.atan(),
                    UnaryOp::Sign => if v > 0.0 { 1.0 } else if v < 0.0 { -1.0 } else { 0.0 },
                    UnaryOp::Isnan => if v.is_nan() { 1.0 } else { 0.0 },
                    UnaryOp::Isinf => if v.is_infinite() { 1.0 } else { 0.0 },
                    UnaryOp::Isfinite => if v.is_finite() { 1.0 } else { 0.0 },

                    UnaryOp::Square => v * v,
                    UnaryOp::Positive => v,
                    UnaryOp::Reciprocal => 1.0 / v,
                    UnaryOp::Exp2 => (2.0 as $T).powf(v),
                    UnaryOp::Expm1 => v.exp_m1(),
                    UnaryOp::Log1p => v.ln_1p(),
                    UnaryOp::Cbrt => v.cbrt(),
                    UnaryOp::Trunc => v.trunc(),
                    UnaryOp::Rint => v.round(),
                    UnaryOp::Arcsinh => v.asinh(),
                    UnaryOp::Arccosh => v.acosh(),
                    UnaryOp::Arctanh => v.atanh(),
                    UnaryOp::Signbit => if v.is_sign_negative() { 1.0 } else { 0.0 },
                };
                Self::write(out, idx, result);
            }

            unsafe fn binary_op(&self, op: BinaryOp, a: *const u8, a_offset: isize, b: *const u8, b_offset: isize, out: *mut u8, idx: usize) {
                let av = Self::read(a, a_offset);
                let bv = Self::read(b, b_offset);
                let result = match op {
                    BinaryOp::Add => av + bv,
                    BinaryOp::Sub => av - bv,
                    BinaryOp::Mul => av * bv,
                    BinaryOp::Div => av / bv,
                    BinaryOp::Pow => av.powf(bv),
                    BinaryOp::Mod => av % bv,
                    BinaryOp::FloorDiv => (av / bv).floor(),
                    BinaryOp::Maximum => if av.is_nan() || bv.is_nan() { <$T>::NAN } else { av.max(bv) },
                    BinaryOp::Minimum => if av.is_nan() || bv.is_nan() { <$T>::NAN } else { av.min(bv) },
                };
                Self::write(out, idx, result);
            }

            unsafe fn reduce_init(&self, op: ReduceOp, out: *mut u8, idx: usize) {
                let v = match op {
                    ReduceOp::Sum => 0.0,
                    ReduceOp::Prod => 1.0,
                    ReduceOp::Max => <$T>::NEG_INFINITY,
                    ReduceOp::Min => <$T>::INFINITY,
                };
                Self::write(out, idx, v);
            }

            unsafe fn reduce_acc(&self, op: ReduceOp, acc: *mut u8, idx: usize, val: *const u8, byte_offset: isize) {
                let a = Self::read(acc as *const u8, (idx * $size) as isize);
                let v = Self::read(val, byte_offset);
                let result = match op {
                    ReduceOp::Sum => a + v,
                    ReduceOp::Prod => a * v,
                    ReduceOp::Max => a.max(v),
                    ReduceOp::Min => a.min(v),
                };
                Self::write(acc, idx, result);
            }

            unsafe fn format_element(&self, ptr: *const u8, byte_offset: isize) -> String {
                let v = Self::read(ptr, byte_offset);
                let s = format!("{:.8}", v);
                s.trim_end_matches('0').to_string()
            }

            unsafe fn compare_elements(&self, a: *const u8, a_offset: isize, b: *const u8, b_offset: isize) -> std::cmp::Ordering {
                let av = Self::read(a, a_offset);
                let bv = Self::read(b, b_offset);
                av.partial_cmp(&bv).unwrap_or(std::cmp::Ordering::Equal)
            }

            unsafe fn is_truthy(&self, ptr: *const u8, byte_offset: isize) -> bool {
                Self::read(ptr, byte_offset) != 0.0
            }

            unsafe fn write_f64(&self, ptr: *mut u8, idx: usize, val: f64) -> bool {
                Self::write(ptr, idx, val as $T);
                true
            }

            unsafe fn read_f64(&self, ptr: *const u8, byte_offset: isize) -> Option<f64> {
                Some(Self::read(ptr, byte_offset) as f64)
            }

            unsafe fn write_f64_at_byte_offset(&self, ptr: *mut u8, byte_offset: isize, val: f64) -> bool {
                Self::write_at_offset(ptr, byte_offset, val as $T);
                true
            }

            unsafe fn write_complex(&self, ptr: *mut u8, idx: usize, real: f64, _imag: f64) -> bool {
                Self::write(ptr, idx, real as $T);
                true
            }

            unsafe fn read_complex(&self, ptr: *const u8, byte_offset: isize) -> Option<(f64, f64)> {
                Some((Self::read(ptr, byte_offset) as f64, 0.0))
            }
        }
    };
}

/// Implement DTypeOps for a signed integer type.
macro_rules! impl_signed_int_dtype {
    ($struct_name:ident, $T:ty, $size:expr, $kind:ident, $name:expr, $typestr:expr, $format:expr, $priority:expr) => {
        pub(super) struct $struct_name;

        impl $struct_name {
            #[inline]
            unsafe fn read(ptr: *const u8, byte_offset: isize) -> $T {
                *(ptr.offset(byte_offset) as *const $T)
            }

            #[inline]
            unsafe fn write(ptr: *mut u8, idx: usize, val: $T) {
                *(ptr as *mut $T).add(idx) = val;
            }

            #[inline]
            unsafe fn write_at_offset(ptr: *mut u8, byte_offset: isize, val: $T) {
                *(ptr.offset(byte_offset) as *mut $T) = val;
            }
        }

        impl DTypeOps for $struct_name {
            fn kind(&self) -> DTypeKind { DTypeKind::$kind }
            fn itemsize(&self) -> usize { $size }
            fn typestr(&self) -> &'static str { $typestr }
            fn format_char(&self) -> &'static str { $format }
            fn name(&self) -> &'static str { $name }
            fn promotion_priority(&self) -> u8 { $priority }

            unsafe fn write_zero(&self, ptr: *mut u8, idx: usize) {
                Self::write(ptr, idx, 0);
            }

            unsafe fn write_one(&self, ptr: *mut u8, idx: usize) {
                Self::write(ptr, idx, 1);
            }

            unsafe fn copy_element(&self, src: *const u8, byte_offset: isize, dst: *mut u8, idx: usize) {
                Self::write(dst, idx, Self::read(src, byte_offset));
            }

            unsafe fn unary_op(&self, op: UnaryOp, src: *const u8, byte_offset: isize, out: *mut u8, idx: usize) {
                let v = Self::read(src, byte_offset);
                let result = match op {
                    UnaryOp::Neg => -v,
                    UnaryOp::Abs => v.abs(),
                    UnaryOp::Sqrt => (v as f64).sqrt() as $T,
                    UnaryOp::Exp => (v as f64).exp() as $T,
                    UnaryOp::Log => (v as f64).ln() as $T,
                    UnaryOp::Log10 => (v as f64).log10() as $T,
                    UnaryOp::Log2 => (v as f64).log2() as $T,
                    UnaryOp::Sin => (v as f64).sin() as $T,
                    UnaryOp::Cos => (v as f64).cos() as $T,
                    UnaryOp::Tan => (v as f64).tan() as $T,
                    UnaryOp::Sinh => (v as f64).sinh() as $T,
                    UnaryOp::Cosh => (v as f64).cosh() as $T,
                    UnaryOp::Tanh => (v as f64).tanh() as $T,
                    UnaryOp::Floor => v,
                    UnaryOp::Ceil => v,
                    UnaryOp::Arcsin => (v as f64).asin() as $T,
                    UnaryOp::Arccos => (v as f64).acos() as $T,
                    UnaryOp::Arctan => (v as f64).atan() as $T,
                    UnaryOp::Sign => if v > 0 { 1 } else if v < 0 { -1 } else { 0 },
                    UnaryOp::Isnan => 0,
                    UnaryOp::Isinf => 0,
                    UnaryOp::Isfinite => 1,

                    UnaryOp::Square => v.wrapping_mul(v),
                    UnaryOp::Positive => v,
                    UnaryOp::Reciprocal => if v != 0 { 1 / v } else { 0 },
                    UnaryOp::Exp2 => 2f64.powf(v as f64) as $T,
                    UnaryOp::Expm1 => (v as f64).exp_m1() as $T,
                    UnaryOp::Log1p => (v as f64).ln_1p() as $T,
                    UnaryOp::Cbrt => (v as f64).cbrt() as $T,
                    UnaryOp::Trunc => v,
                    UnaryOp::Rint => v,
                    UnaryOp::Arcsinh => (v as f64).asinh() as $T,
                    UnaryOp::Arccosh => (v as f64).acosh() as $T,
                    UnaryOp::Arctanh => (v as f64).atanh() as $T,
                    UnaryOp::Signbit => if v < 0 { 1 } else { 0 },
                };
                Self::write(out, idx, result);
            }

            unsafe fn binary_op(&self, op: BinaryOp, a: *const u8, a_offset: isize, b: *const u8, b_offset: isize, out: *mut u8, idx: usize) {
                let av = Self::read(a, a_offset);
                let bv = Self::read(b, b_offset);
                let result = match op {
                    BinaryOp::Add => av.wrapping_add(bv),
                    BinaryOp::Sub => av.wrapping_sub(bv),
                    BinaryOp::Mul => av.wrapping_mul(bv),
                    BinaryOp::Div => if bv != 0 { av / bv } else { 0 },
                    BinaryOp::Pow => if bv >= 0 { av.wrapping_pow(bv as u32) } else { 0 },
                    BinaryOp::Mod => if bv != 0 { av % bv } else { 0 },
                    BinaryOp::FloorDiv => if bv != 0 { av.div_euclid(bv) } else { 0 },
                    BinaryOp::Maximum => av.max(bv),
                    BinaryOp::Minimum => av.min(bv),
                };
                Self::write(out, idx, result);
            }

            unsafe fn reduce_init(&self, op: ReduceOp, out: *mut u8, idx: usize) {
                let v = match op {
                    ReduceOp::Sum => 0,
                    ReduceOp::Prod => 1,
                    ReduceOp::Max => <$T>::MIN,
                    ReduceOp::Min => <$T>::MAX,
                };
                Self::write(out, idx, v);
            }

            unsafe fn reduce_acc(&self, op: ReduceOp, acc: *mut u8, idx: usize, val: *const u8, byte_offset: isize) {
                let a = Self::read(acc as *const u8, (idx * $size) as isize);
                let v = Self::read(val, byte_offset);
                let result = match op {
                    ReduceOp::Sum => a.wrapping_add(v),
                    ReduceOp::Prod => a.wrapping_mul(v),
                    ReduceOp::Max => a.max(v),
                    ReduceOp::Min => a.min(v),
                };
                Self::write(acc, idx, result);
            }

            unsafe fn format_element(&self, ptr: *const u8, byte_offset: isize) -> String {
                format!("{}", Self::read(ptr, byte_offset))
            }

            unsafe fn compare_elements(&self, a: *const u8, a_offset: isize, b: *const u8, b_offset: isize) -> std::cmp::Ordering {
                Self::read(a, a_offset).cmp(&Self::read(b, b_offset))
            }

            unsafe fn is_truthy(&self, ptr: *const u8, byte_offset: isize) -> bool {
                Self::read(ptr, byte_offset) != 0
            }

            unsafe fn write_f64(&self, ptr: *mut u8, idx: usize, val: f64) -> bool {
                Self::write(ptr, idx, val as $T);
                true
            }

            unsafe fn read_f64(&self, ptr: *const u8, byte_offset: isize) -> Option<f64> {
                Some(Self::read(ptr, byte_offset) as f64)
            }

            unsafe fn write_f64_at_byte_offset(&self, ptr: *mut u8, byte_offset: isize, val: f64) -> bool {
                Self::write_at_offset(ptr, byte_offset, val as $T);
                true
            }

            unsafe fn write_complex(&self, ptr: *mut u8, idx: usize, real: f64, _imag: f64) -> bool {
                Self::write(ptr, idx, real as $T);
                true
            }

            unsafe fn read_complex(&self, ptr: *const u8, byte_offset: isize) -> Option<(f64, f64)> {
                Some((Self::read(ptr, byte_offset) as f64, 0.0))
            }
        }
    };
}

/// Implement DTypeOps for an unsigned integer type.
macro_rules! impl_unsigned_int_dtype {
    ($struct_name:ident, $T:ty, $size:expr, $kind:ident, $name:expr, $typestr:expr, $format:expr, $priority:expr) => {
        pub(super) struct $struct_name;

        impl $struct_name {
            #[inline]
            unsafe fn read(ptr: *const u8, byte_offset: isize) -> $T {
                *(ptr.offset(byte_offset) as *const $T)
            }

            #[inline]
            unsafe fn write(ptr: *mut u8, idx: usize, val: $T) {
                *(ptr as *mut $T).add(idx) = val;
            }

            #[inline]
            unsafe fn write_at_offset(ptr: *mut u8, byte_offset: isize, val: $T) {
                *(ptr.offset(byte_offset) as *mut $T) = val;
            }
        }

        impl DTypeOps for $struct_name {
            fn kind(&self) -> DTypeKind { DTypeKind::$kind }
            fn itemsize(&self) -> usize { $size }
            fn typestr(&self) -> &'static str { $typestr }
            fn format_char(&self) -> &'static str { $format }
            fn name(&self) -> &'static str { $name }
            fn promotion_priority(&self) -> u8 { $priority }

            unsafe fn write_zero(&self, ptr: *mut u8, idx: usize) {
                Self::write(ptr, idx, 0);
            }

            unsafe fn write_one(&self, ptr: *mut u8, idx: usize) {
                Self::write(ptr, idx, 1);
            }

            unsafe fn copy_element(&self, src: *const u8, byte_offset: isize, dst: *mut u8, idx: usize) {
                Self::write(dst, idx, Self::read(src, byte_offset));
            }

            unsafe fn unary_op(&self, op: UnaryOp, src: *const u8, byte_offset: isize, out: *mut u8, idx: usize) {
                let v = Self::read(src, byte_offset);
                let result = match op {
                    UnaryOp::Neg => (0 as $T).wrapping_sub(v),
                    UnaryOp::Abs => v,
                    UnaryOp::Sqrt => (v as f64).sqrt() as $T,
                    UnaryOp::Exp => (v as f64).exp() as $T,
                    UnaryOp::Log => (v as f64).ln() as $T,
                    UnaryOp::Log10 => (v as f64).log10() as $T,
                    UnaryOp::Log2 => (v as f64).log2() as $T,
                    UnaryOp::Sin => (v as f64).sin() as $T,
                    UnaryOp::Cos => (v as f64).cos() as $T,
                    UnaryOp::Tan => (v as f64).tan() as $T,
                    UnaryOp::Sinh => (v as f64).sinh() as $T,
                    UnaryOp::Cosh => (v as f64).cosh() as $T,
                    UnaryOp::Tanh => (v as f64).tanh() as $T,
                    UnaryOp::Floor => v,
                    UnaryOp::Ceil => v,
                    UnaryOp::Arcsin => (v as f64).asin() as $T,
                    UnaryOp::Arccos => (v as f64).acos() as $T,
                    UnaryOp::Arctan => (v as f64).atan() as $T,
                    UnaryOp::Sign => if v > 0 { 1 } else { 0 },
                    UnaryOp::Isnan => 0,
                    UnaryOp::Isinf => 0,
                    UnaryOp::Isfinite => 1,

                    UnaryOp::Square => v.wrapping_mul(v),
                    UnaryOp::Positive => v,
                    UnaryOp::Reciprocal => if v != 0 { 1 / v } else { 0 },
                    UnaryOp::Exp2 => 2f64.powf(v as f64) as $T,
                    UnaryOp::Expm1 => (v as f64).exp_m1() as $T,
                    UnaryOp::Log1p => (v as f64).ln_1p() as $T,
                    UnaryOp::Cbrt => (v as f64).cbrt() as $T,
                    UnaryOp::Trunc => v,
                    UnaryOp::Rint => v,
                    UnaryOp::Arcsinh => (v as f64).asinh() as $T,
                    UnaryOp::Arccosh => (v as f64).acosh() as $T,
                    UnaryOp::Arctanh => (v as f64).atanh() as $T,
                    UnaryOp::Signbit => 0,  // Unsigned types never have sign bit set
                };
                Self::write(out, idx, result);
            }

            unsafe fn binary_op(&self, op: BinaryOp, a: *const u8, a_offset: isize, b: *const u8, b_offset: isize, out: *mut u8, idx: usize) {
                let av = Self::read(a, a_offset);
                let bv = Self::read(b, b_offset);
                let result = match op {
                    BinaryOp::Add => av.wrapping_add(bv),
                    BinaryOp::Sub => av.wrapping_sub(bv),
                    BinaryOp::Mul => av.wrapping_mul(bv),
                    BinaryOp::Div => if bv != 0 { av / bv } else { 0 },
                    BinaryOp::Pow => av.wrapping_pow(bv as u32),
                    BinaryOp::Mod => if bv != 0 { av % bv } else { 0 },
                    BinaryOp::FloorDiv => if bv != 0 { av / bv } else { 0 },
                    BinaryOp::Maximum => av.max(bv),
                    BinaryOp::Minimum => av.min(bv),
                };
                Self::write(out, idx, result);
            }

            unsafe fn reduce_init(&self, op: ReduceOp, out: *mut u8, idx: usize) {
                let v = match op {
                    ReduceOp::Sum => 0,
                    ReduceOp::Prod => 1,
                    ReduceOp::Max => <$T>::MIN,
                    ReduceOp::Min => <$T>::MAX,
                };
                Self::write(out, idx, v);
            }

            unsafe fn reduce_acc(&self, op: ReduceOp, acc: *mut u8, idx: usize, val: *const u8, byte_offset: isize) {
                let a = Self::read(acc as *const u8, (idx * $size) as isize);
                let v = Self::read(val, byte_offset);
                let result = match op {
                    ReduceOp::Sum => a.wrapping_add(v),
                    ReduceOp::Prod => a.wrapping_mul(v),
                    ReduceOp::Max => a.max(v),
                    ReduceOp::Min => a.min(v),
                };
                Self::write(acc, idx, result);
            }

            unsafe fn format_element(&self, ptr: *const u8, byte_offset: isize) -> String {
                format!("{}", Self::read(ptr, byte_offset))
            }

            unsafe fn compare_elements(&self, a: *const u8, a_offset: isize, b: *const u8, b_offset: isize) -> std::cmp::Ordering {
                Self::read(a, a_offset).cmp(&Self::read(b, b_offset))
            }

            unsafe fn is_truthy(&self, ptr: *const u8, byte_offset: isize) -> bool {
                Self::read(ptr, byte_offset) != 0
            }

            unsafe fn write_f64(&self, ptr: *mut u8, idx: usize, val: f64) -> bool {
                Self::write(ptr, idx, val as $T);
                true
            }

            unsafe fn read_f64(&self, ptr: *const u8, byte_offset: isize) -> Option<f64> {
                Some(Self::read(ptr, byte_offset) as f64)
            }

            unsafe fn write_f64_at_byte_offset(&self, ptr: *mut u8, byte_offset: isize, val: f64) -> bool {
                Self::write_at_offset(ptr, byte_offset, val as $T);
                true
            }

            unsafe fn write_complex(&self, ptr: *mut u8, idx: usize, real: f64, _imag: f64) -> bool {
                Self::write(ptr, idx, real as $T);
                true
            }

            unsafe fn read_complex(&self, ptr: *const u8, byte_offset: isize) -> Option<(f64, f64)> {
                Some((Self::read(ptr, byte_offset) as f64, 0.0))
            }
        }
    };
}

pub(super) use impl_float_dtype;
pub(super) use impl_signed_int_dtype;
pub(super) use impl_unsigned_int_dtype;
