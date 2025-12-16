//! Special mathematical functions.
//!
//! Contains: sinc, i0 (Bessel), spacing, modf, frexp, ldexp, heaviside, gcd, lcm
//!
//! All functions operate natively in their input dtype (no f64 conversion).

use crate::array::dtype::DTypeKind;
use crate::array::{DType, RumpyArray};
use std::sync::Arc;

// ============================================================================
// Sinc: sin(pi*x) / (pi*x)
// ============================================================================

pub fn sinc(arr: &RumpyArray) -> RumpyArray {
    match arr.dtype().kind() {
        DTypeKind::Float32 => sinc_typed::<f32>(arr, DType::float32()),
        DTypeKind::Float64 => sinc_typed::<f64>(arr, DType::float64()),
        _ => {
            // Convert to f64 for other types
            let arr_f64 = arr.astype(DType::float64());
            sinc_typed::<f64>(&arr_f64, DType::float64())
        }
    }
}

fn sinc_typed<T: Float>(arr: &RumpyArray, dtype: DType) -> RumpyArray {
    let size = arr.size();
    if size == 0 {
        return RumpyArray::zeros(arr.shape().to_vec(), dtype);
    }

    let mut result = RumpyArray::zeros(arr.shape().to_vec(), dtype);
    let buffer = result.buffer_mut();
    let result_buffer = Arc::get_mut(buffer).expect("unique");
    let result_ptr = result_buffer.as_mut_ptr() as *mut T;

    let src_ptr = arr.data_ptr() as *const T;
    for i in 0..size {
        let x = unsafe { *src_ptr.add(i) };
        let y = if x == T::ZERO {
            T::ONE
        } else {
            let pi_x = T::PI * x;
            pi_x.sin() / pi_x
        };
        unsafe { *result_ptr.add(i) = y };
    }
    result
}

// ============================================================================
// I0: Modified Bessel function of the first kind, order 0
// ============================================================================

pub fn i0(arr: &RumpyArray) -> RumpyArray {
    match arr.dtype().kind() {
        DTypeKind::Float32 => i0_typed::<f32>(arr, DType::float32()),
        DTypeKind::Float64 => i0_typed::<f64>(arr, DType::float64()),
        _ => {
            let arr_f64 = arr.astype(DType::float64());
            i0_typed::<f64>(&arr_f64, DType::float64())
        }
    }
}

fn i0_typed<T: Float>(arr: &RumpyArray, dtype: DType) -> RumpyArray {
    let size = arr.size();
    if size == 0 {
        return RumpyArray::zeros(arr.shape().to_vec(), dtype);
    }

    let mut result = RumpyArray::zeros(arr.shape().to_vec(), dtype);
    let buffer = result.buffer_mut();
    let result_buffer = Arc::get_mut(buffer).expect("unique");
    let result_ptr = result_buffer.as_mut_ptr() as *mut T;

    let src_ptr = arr.data_ptr() as *const T;
    for i in 0..size {
        let x = unsafe { *src_ptr.add(i) };
        let y = i0_scalar(x);
        unsafe { *result_ptr.add(i) = y };
    }
    result
}

/// Chebyshev polynomial approximation for i0.
fn i0_scalar<T: Float>(x: T) -> T {
    let ax = x.abs();
    let threshold = T::from_f64(3.75);

    if ax < threshold {
        let t = (x / threshold).powi(2);
        T::ONE + t * (T::from_f64(3.5156229) + t * (T::from_f64(3.0899424) + t * (T::from_f64(1.2067492)
            + t * (T::from_f64(0.2659732) + t * (T::from_f64(0.0360768) + t * T::from_f64(0.0045813))))))
    } else {
        let t = threshold / ax;
        (ax.exp() / ax.sqrt()) * (T::from_f64(0.39894228) + t * (T::from_f64(0.01328592)
            + t * (T::from_f64(0.00225319) + t * (T::from_f64(-0.00157565) + t * (T::from_f64(0.00916281)
            + t * (T::from_f64(-0.02057706) + t * (T::from_f64(0.02635537) + t * (T::from_f64(-0.01647633)
            + t * T::from_f64(0.00392377)))))))))
    }
}

// ============================================================================
// Spacing: ULP distance
// ============================================================================

pub fn spacing(arr: &RumpyArray) -> RumpyArray {
    match arr.dtype().kind() {
        DTypeKind::Float32 => spacing_f32(arr),
        DTypeKind::Float64 => spacing_f64(arr),
        _ => {
            let arr_f64 = arr.astype(DType::float64());
            spacing_f64(&arr_f64)
        }
    }
}

fn spacing_f64(arr: &RumpyArray) -> RumpyArray {
    let size = arr.size();
    if size == 0 {
        return RumpyArray::zeros(arr.shape().to_vec(), DType::float64());
    }

    let mut result = RumpyArray::zeros(arr.shape().to_vec(), DType::float64());
    let buffer = result.buffer_mut();
    let result_buffer = Arc::get_mut(buffer).expect("unique");
    let result_ptr = result_buffer.as_mut_ptr() as *mut f64;

    let src_ptr = arr.data_ptr() as *const f64;
    for i in 0..size {
        let x = unsafe { *src_ptr.add(i) };
        let next = if x >= 0.0 {
            f64::from_bits(x.to_bits().wrapping_add(1))
        } else if x.to_bits() == 0x8000_0000_0000_0000 {
            f64::from_bits(1)
        } else {
            f64::from_bits(x.to_bits().wrapping_sub(1))
        };
        unsafe { *result_ptr.add(i) = (next - x).abs() };
    }
    result
}

fn spacing_f32(arr: &RumpyArray) -> RumpyArray {
    let size = arr.size();
    if size == 0 {
        return RumpyArray::zeros(arr.shape().to_vec(), DType::float32());
    }

    let mut result = RumpyArray::zeros(arr.shape().to_vec(), DType::float32());
    let buffer = result.buffer_mut();
    let result_buffer = Arc::get_mut(buffer).expect("unique");
    let result_ptr = result_buffer.as_mut_ptr() as *mut f32;

    let src_ptr = arr.data_ptr() as *const f32;
    for i in 0..size {
        let x = unsafe { *src_ptr.add(i) };
        let next = if x >= 0.0 {
            f32::from_bits(x.to_bits().wrapping_add(1))
        } else if x.to_bits() == 0x8000_0000 {
            f32::from_bits(1)
        } else {
            f32::from_bits(x.to_bits().wrapping_sub(1))
        };
        unsafe { *result_ptr.add(i) = (next - x).abs() };
    }
    result
}

// ============================================================================
// Modf: fractional and integer parts
// ============================================================================

pub fn modf(arr: &RumpyArray) -> (RumpyArray, RumpyArray) {
    match arr.dtype().kind() {
        DTypeKind::Float32 => modf_typed::<f32>(arr, DType::float32()),
        DTypeKind::Float64 => modf_typed::<f64>(arr, DType::float64()),
        _ => {
            let arr_f64 = arr.astype(DType::float64());
            modf_typed::<f64>(&arr_f64, DType::float64())
        }
    }
}

fn modf_typed<T: Float>(arr: &RumpyArray, dtype: DType) -> (RumpyArray, RumpyArray) {
    let size = arr.size();
    let shape = arr.shape().to_vec();

    if size == 0 {
        return (
            RumpyArray::zeros(shape.clone(), dtype.clone()),
            RumpyArray::zeros(shape, dtype),
        );
    }

    let mut frac = RumpyArray::zeros(shape.clone(), dtype.clone());
    let mut int_part = RumpyArray::zeros(shape, dtype);

    let frac_buffer = Arc::get_mut(frac.buffer_mut()).expect("unique");
    let int_buffer = Arc::get_mut(int_part.buffer_mut()).expect("unique");
    let frac_ptr = frac_buffer.as_mut_ptr() as *mut T;
    let int_ptr = int_buffer.as_mut_ptr() as *mut T;

    let src_ptr = arr.data_ptr() as *const T;
    for i in 0..size {
        let x = unsafe { *src_ptr.add(i) };
        let int_val = x.trunc();
        let frac_val = x - int_val;
        unsafe {
            *frac_ptr.add(i) = frac_val;
            *int_ptr.add(i) = int_val;
        }
    }
    (frac, int_part)
}

// ============================================================================
// Frexp: mantissa and exponent
// ============================================================================

pub fn frexp(arr: &RumpyArray) -> (RumpyArray, RumpyArray) {
    match arr.dtype().kind() {
        DTypeKind::Float32 => frexp_f32(arr),
        DTypeKind::Float64 => frexp_f64(arr),
        _ => {
            let arr_f64 = arr.astype(DType::float64());
            frexp_f64(&arr_f64)
        }
    }
}

fn frexp_f64(arr: &RumpyArray) -> (RumpyArray, RumpyArray) {
    let size = arr.size();
    let shape = arr.shape().to_vec();

    if size == 0 {
        return (
            RumpyArray::zeros(shape.clone(), DType::float64()),
            RumpyArray::zeros(shape, DType::int32()),
        );
    }

    let mut mant = RumpyArray::zeros(shape.clone(), DType::float64());
    let mut exp = RumpyArray::zeros(shape, DType::int32());

    let mant_buffer = Arc::get_mut(mant.buffer_mut()).expect("unique");
    let exp_buffer = Arc::get_mut(exp.buffer_mut()).expect("unique");
    let mant_ptr = mant_buffer.as_mut_ptr() as *mut f64;
    let exp_ptr = exp_buffer.as_mut_ptr() as *mut i32;

    let src_ptr = arr.data_ptr() as *const f64;
    for i in 0..size {
        let x = unsafe { *src_ptr.add(i) };
        let (m, e) = frexp_scalar_f64(x);
        unsafe {
            *mant_ptr.add(i) = m;
            *exp_ptr.add(i) = e;
        }
    }
    (mant, exp)
}

fn frexp_f32(arr: &RumpyArray) -> (RumpyArray, RumpyArray) {
    let size = arr.size();
    let shape = arr.shape().to_vec();

    if size == 0 {
        return (
            RumpyArray::zeros(shape.clone(), DType::float32()),
            RumpyArray::zeros(shape, DType::int32()),
        );
    }

    let mut mant = RumpyArray::zeros(shape.clone(), DType::float32());
    let mut exp = RumpyArray::zeros(shape, DType::int32());

    let mant_buffer = Arc::get_mut(mant.buffer_mut()).expect("unique");
    let exp_buffer = Arc::get_mut(exp.buffer_mut()).expect("unique");
    let mant_ptr = mant_buffer.as_mut_ptr() as *mut f32;
    let exp_ptr = exp_buffer.as_mut_ptr() as *mut i32;

    let src_ptr = arr.data_ptr() as *const f32;
    for i in 0..size {
        let x = unsafe { *src_ptr.add(i) };
        let (m, e) = frexp_scalar_f32(x);
        unsafe {
            *mant_ptr.add(i) = m;
            *exp_ptr.add(i) = e;
        }
    }
    (mant, exp)
}

fn frexp_scalar_f64(x: f64) -> (f64, i32) {
    if x == 0.0 || x.is_nan() || x.is_infinite() {
        return (x, 0);
    }
    let bits = x.to_bits();
    let sign = bits & 0x8000_0000_0000_0000;
    let mut exp = ((bits >> 52) & 0x7FF) as i32;

    if exp == 0 {
        // Subnormal
        let normalized = x * (1u64 << 52) as f64;
        let (m, e) = frexp_scalar_f64(normalized);
        return (m, e - 52);
    }

    exp -= 1022;
    let mantissa_bits = (sign | 0x3FE0_0000_0000_0000) | (bits & 0x000F_FFFF_FFFF_FFFF);
    (f64::from_bits(mantissa_bits), exp)
}

fn frexp_scalar_f32(x: f32) -> (f32, i32) {
    if x == 0.0 || x.is_nan() || x.is_infinite() {
        return (x, 0);
    }
    let bits = x.to_bits();
    let sign = bits & 0x8000_0000;
    let mut exp = ((bits >> 23) & 0xFF) as i32;

    if exp == 0 {
        // Subnormal
        let normalized = x * (1u32 << 23) as f32;
        let (m, e) = frexp_scalar_f32(normalized);
        return (m, e - 23);
    }

    exp -= 126;
    let mantissa_bits = (sign | 0x3F00_0000) | (bits & 0x007F_FFFF);
    (f32::from_bits(mantissa_bits), exp)
}

// ============================================================================
// Ldexp: x * 2^i
// ============================================================================

pub fn ldexp(x: &RumpyArray, i: &RumpyArray) -> RumpyArray {
    match x.dtype().kind() {
        DTypeKind::Float32 => ldexp_typed::<f32>(x, i, DType::float32()),
        DTypeKind::Float64 => ldexp_typed::<f64>(x, i, DType::float64()),
        _ => {
            let x_f64 = x.astype(DType::float64());
            ldexp_typed::<f64>(&x_f64, i, DType::float64())
        }
    }
}

fn ldexp_typed<T: Float>(x: &RumpyArray, i: &RumpyArray, dtype: DType) -> RumpyArray {
    let size = x.size();
    if size == 0 {
        return RumpyArray::zeros(x.shape().to_vec(), dtype);
    }

    let mut result = RumpyArray::zeros(x.shape().to_vec(), dtype);
    let buffer = result.buffer_mut();
    let result_buffer = Arc::get_mut(buffer).expect("unique");
    let result_ptr = result_buffer.as_mut_ptr() as *mut T;

    let x_ptr = x.data_ptr() as *const T;

    // Read exponents - they could be any integer type
    let i_vals: Vec<i32> = match i.dtype().kind() {
        DTypeKind::Int32 => {
            let ptr = i.data_ptr() as *const i32;
            (0..size).map(|idx| unsafe { *ptr.add(idx) }).collect()
        }
        DTypeKind::Int64 => {
            let ptr = i.data_ptr() as *const i64;
            (0..size).map(|idx| unsafe { *ptr.add(idx) as i32 }).collect()
        }
        _ => {
            // Fallback: use to_vec
            i.to_vec().iter().map(|&v| v as i32).collect()
        }
    };

    for (idx, &iv) in i_vals.iter().enumerate() {
        let xv = unsafe { *x_ptr.add(idx) };
        let y = xv * T::TWO.powi(iv);
        unsafe { *result_ptr.add(idx) = y };
    }
    result
}

// ============================================================================
// Heaviside step function
// ============================================================================

pub fn heaviside(x: &RumpyArray, h0: &RumpyArray) -> RumpyArray {
    match x.dtype().kind() {
        DTypeKind::Float32 => heaviside_typed::<f32>(x, h0, DType::float32()),
        DTypeKind::Float64 => heaviside_typed::<f64>(x, h0, DType::float64()),
        _ => {
            let x_f64 = x.astype(DType::float64());
            let h0_f64 = h0.astype(DType::float64());
            heaviside_typed::<f64>(&x_f64, &h0_f64, DType::float64())
        }
    }
}

fn heaviside_typed<T: Float>(x: &RumpyArray, h0: &RumpyArray, dtype: DType) -> RumpyArray {
    let size = x.size();
    if size == 0 {
        return RumpyArray::zeros(x.shape().to_vec(), dtype);
    }

    let mut result = RumpyArray::zeros(x.shape().to_vec(), dtype);
    let buffer = result.buffer_mut();
    let result_buffer = Arc::get_mut(buffer).expect("unique");
    let result_ptr = result_buffer.as_mut_ptr() as *mut T;

    let x_ptr = x.data_ptr() as *const T;
    let h0_ptr = h0.data_ptr() as *const T;

    for i in 0..size {
        let xv = unsafe { *x_ptr.add(i) };
        let h0v = unsafe { *h0_ptr.add(i) };
        let y = if xv < T::ZERO {
            T::ZERO
        } else if xv == T::ZERO {
            h0v
        } else {
            T::ONE
        };
        unsafe { *result_ptr.add(i) = y };
    }
    result
}

// ============================================================================
// GCD: Greatest common divisor
// ============================================================================

pub fn gcd(a: &RumpyArray, b: &RumpyArray) -> RumpyArray {
    let kind = a.dtype().kind();
    match kind {
        DTypeKind::Int64 => gcd_typed::<i64>(a, b, DType::int64()),
        DTypeKind::Int32 => gcd_typed::<i32>(a, b, DType::int32()),
        DTypeKind::Int16 => gcd_typed::<i16>(a, b, DType::int16()),
        DTypeKind::Int8 => gcd_typed::<i8>(a, b, DType::int8()),
        DTypeKind::Uint64 => gcd_typed_unsigned::<u64>(a, b, DType::uint64()),
        DTypeKind::Uint32 => gcd_typed_unsigned::<u32>(a, b, DType::uint32()),
        DTypeKind::Uint16 => gcd_typed_unsigned::<u16>(a, b, DType::uint16()),
        DTypeKind::Uint8 => gcd_typed_unsigned::<u8>(a, b, DType::uint8()),
        _ => {
            // Fallback for float: convert to i64
            let a_i64 = a.astype(DType::int64());
            let b_i64 = b.astype(DType::int64());
            gcd_typed::<i64>(&a_i64, &b_i64, DType::int64())
        }
    }
}

fn gcd_typed<T: SignedInt>(a: &RumpyArray, b: &RumpyArray, dtype: DType) -> RumpyArray {
    let size = a.size();
    if size == 0 {
        return RumpyArray::zeros(a.shape().to_vec(), dtype);
    }

    let mut result = RumpyArray::zeros(a.shape().to_vec(), dtype);
    let buffer = result.buffer_mut();
    let result_buffer = Arc::get_mut(buffer).expect("unique");
    let result_ptr = result_buffer.as_mut_ptr() as *mut T;

    let a_ptr = a.data_ptr() as *const T;
    let b_ptr = b.data_ptr() as *const T;

    for i in 0..size {
        let av = unsafe { *a_ptr.add(i) };
        let bv = unsafe { *b_ptr.add(i) };
        let g = gcd_scalar_signed(av, bv);
        unsafe { *result_ptr.add(i) = g };
    }
    result
}

fn gcd_typed_unsigned<T: UnsignedInt>(a: &RumpyArray, b: &RumpyArray, dtype: DType) -> RumpyArray {
    let size = a.size();
    if size == 0 {
        return RumpyArray::zeros(a.shape().to_vec(), dtype);
    }

    let mut result = RumpyArray::zeros(a.shape().to_vec(), dtype);
    let buffer = result.buffer_mut();
    let result_buffer = Arc::get_mut(buffer).expect("unique");
    let result_ptr = result_buffer.as_mut_ptr() as *mut T;

    let a_ptr = a.data_ptr() as *const T;
    let b_ptr = b.data_ptr() as *const T;

    for i in 0..size {
        let av = unsafe { *a_ptr.add(i) };
        let bv = unsafe { *b_ptr.add(i) };
        let g = gcd_scalar_unsigned(av, bv);
        unsafe { *result_ptr.add(i) = g };
    }
    result
}

fn gcd_scalar_signed<T: SignedInt>(mut a: T, mut b: T) -> T {
    a = a.abs_val();
    b = b.abs_val();
    while b != T::ZERO {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

fn gcd_scalar_unsigned<T: UnsignedInt>(mut a: T, mut b: T) -> T {
    while b != T::ZERO {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

// ============================================================================
// LCM: Least common multiple
// ============================================================================

pub fn lcm(a: &RumpyArray, b: &RumpyArray) -> RumpyArray {
    let kind = a.dtype().kind();
    match kind {
        DTypeKind::Int64 => lcm_typed::<i64>(a, b, DType::int64()),
        DTypeKind::Int32 => lcm_typed::<i32>(a, b, DType::int32()),
        DTypeKind::Int16 => lcm_typed::<i16>(a, b, DType::int16()),
        DTypeKind::Int8 => lcm_typed::<i8>(a, b, DType::int8()),
        DTypeKind::Uint64 => lcm_typed_unsigned::<u64>(a, b, DType::uint64()),
        DTypeKind::Uint32 => lcm_typed_unsigned::<u32>(a, b, DType::uint32()),
        DTypeKind::Uint16 => lcm_typed_unsigned::<u16>(a, b, DType::uint16()),
        DTypeKind::Uint8 => lcm_typed_unsigned::<u8>(a, b, DType::uint8()),
        _ => {
            let a_i64 = a.astype(DType::int64());
            let b_i64 = b.astype(DType::int64());
            lcm_typed::<i64>(&a_i64, &b_i64, DType::int64())
        }
    }
}

fn lcm_typed<T: SignedInt>(a: &RumpyArray, b: &RumpyArray, dtype: DType) -> RumpyArray {
    let size = a.size();
    if size == 0 {
        return RumpyArray::zeros(a.shape().to_vec(), dtype);
    }

    let mut result = RumpyArray::zeros(a.shape().to_vec(), dtype);
    let buffer = result.buffer_mut();
    let result_buffer = Arc::get_mut(buffer).expect("unique");
    let result_ptr = result_buffer.as_mut_ptr() as *mut T;

    let a_ptr = a.data_ptr() as *const T;
    let b_ptr = b.data_ptr() as *const T;

    for i in 0..size {
        let av = unsafe { *a_ptr.add(i) };
        let bv = unsafe { *b_ptr.add(i) };
        let l = lcm_scalar_signed(av, bv);
        unsafe { *result_ptr.add(i) = l };
    }
    result
}

fn lcm_typed_unsigned<T: UnsignedInt>(a: &RumpyArray, b: &RumpyArray, dtype: DType) -> RumpyArray {
    let size = a.size();
    if size == 0 {
        return RumpyArray::zeros(a.shape().to_vec(), dtype);
    }

    let mut result = RumpyArray::zeros(a.shape().to_vec(), dtype);
    let buffer = result.buffer_mut();
    let result_buffer = Arc::get_mut(buffer).expect("unique");
    let result_ptr = result_buffer.as_mut_ptr() as *mut T;

    let a_ptr = a.data_ptr() as *const T;
    let b_ptr = b.data_ptr() as *const T;

    for i in 0..size {
        let av = unsafe { *a_ptr.add(i) };
        let bv = unsafe { *b_ptr.add(i) };
        let l = lcm_scalar_unsigned(av, bv);
        unsafe { *result_ptr.add(i) = l };
    }
    result
}

fn lcm_scalar_signed<T: SignedInt>(a: T, b: T) -> T {
    let a_abs = a.abs_val();
    let b_abs = b.abs_val();
    if a_abs == T::ZERO || b_abs == T::ZERO {
        T::ZERO
    } else {
        a_abs / gcd_scalar_signed(a_abs, b_abs) * b_abs
    }
}

fn lcm_scalar_unsigned<T: UnsignedInt>(a: T, b: T) -> T {
    if a == T::ZERO || b == T::ZERO {
        T::ZERO
    } else {
        a / gcd_scalar_unsigned(a, b) * b
    }
}

// ============================================================================
// Float trait for generic float operations
// ============================================================================

trait Float: Copy + PartialOrd + std::ops::Add<Output = Self> + std::ops::Mul<Output = Self> + std::ops::Div<Output = Self> + std::ops::Sub<Output = Self> {
    const ZERO: Self;
    const ONE: Self;
    const TWO: Self;
    const PI: Self;

    fn from_f64(v: f64) -> Self;
    fn sin(self) -> Self;
    fn exp(self) -> Self;
    fn sqrt(self) -> Self;
    fn abs(self) -> Self;
    fn trunc(self) -> Self;
    fn powi(self, n: i32) -> Self;
}

impl Float for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const TWO: Self = 2.0;
    const PI: Self = std::f32::consts::PI;

    #[inline] fn from_f64(v: f64) -> Self { v as f32 }
    #[inline] fn sin(self) -> Self { f32::sin(self) }
    #[inline] fn exp(self) -> Self { f32::exp(self) }
    #[inline] fn sqrt(self) -> Self { f32::sqrt(self) }
    #[inline] fn abs(self) -> Self { f32::abs(self) }
    #[inline] fn trunc(self) -> Self { f32::trunc(self) }
    #[inline] fn powi(self, n: i32) -> Self { f32::powi(self, n) }
}

impl Float for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const TWO: Self = 2.0;
    const PI: Self = std::f64::consts::PI;

    #[inline] fn from_f64(v: f64) -> Self { v }
    #[inline] fn sin(self) -> Self { f64::sin(self) }
    #[inline] fn exp(self) -> Self { f64::exp(self) }
    #[inline] fn sqrt(self) -> Self { f64::sqrt(self) }
    #[inline] fn abs(self) -> Self { f64::abs(self) }
    #[inline] fn trunc(self) -> Self { f64::trunc(self) }
    #[inline] fn powi(self, n: i32) -> Self { f64::powi(self, n) }
}

// ============================================================================
// Integer traits for generic integer operations
// ============================================================================

trait SignedInt: Copy + PartialEq + std::ops::Rem<Output = Self> + std::ops::Div<Output = Self> + std::ops::Mul<Output = Self> {
    const ZERO: Self;
    fn abs_val(self) -> Self;
}

impl SignedInt for i64 {
    const ZERO: Self = 0;
    #[inline] fn abs_val(self) -> Self { self.abs() }
}

impl SignedInt for i32 {
    const ZERO: Self = 0;
    #[inline] fn abs_val(self) -> Self { self.abs() }
}

impl SignedInt for i16 {
    const ZERO: Self = 0;
    #[inline] fn abs_val(self) -> Self { self.abs() }
}

impl SignedInt for i8 {
    const ZERO: Self = 0;
    #[inline] fn abs_val(self) -> Self { self.abs() }
}

trait UnsignedInt: Copy + PartialEq + std::ops::Rem<Output = Self> + std::ops::Div<Output = Self> + std::ops::Mul<Output = Self> {
    const ZERO: Self;
}

impl UnsignedInt for u64 { const ZERO: Self = 0; }
impl UnsignedInt for u32 { const ZERO: Self = 0; }
impl UnsignedInt for u16 { const ZERO: Self = 0; }
impl UnsignedInt for u8 { const ZERO: Self = 0; }
