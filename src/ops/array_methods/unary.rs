//! Unary math operations on RumpyArray (sqrt, exp, sin, etc.)

use crate::array::{DType, RumpyArray};
use crate::array::dtype::{DTypeKind, UnaryOp};
use crate::ops::{map_unary_op, UnaryOpError};
use std::sync::Arc;

// Note: half::f16 is imported where needed for Float16 handling

impl RumpyArray {
    // ========================================================================
    // Basic math ufuncs
    // ========================================================================

    /// Square root of each element.
    pub fn sqrt(&self) -> Result<RumpyArray, UnaryOpError> {
        map_unary_op(self, UnaryOp::Sqrt)
    }

    /// Exponential (e^x) of each element.
    pub fn exp(&self) -> Result<RumpyArray, UnaryOpError> {
        map_unary_op(self, UnaryOp::Exp)
    }

    /// Natural logarithm of each element.
    pub fn log(&self) -> Result<RumpyArray, UnaryOpError> {
        map_unary_op(self, UnaryOp::Log)
    }

    /// Sine of each element (radians).
    pub fn sin(&self) -> Result<RumpyArray, UnaryOpError> {
        map_unary_op(self, UnaryOp::Sin)
    }

    /// Cosine of each element (radians).
    pub fn cos(&self) -> Result<RumpyArray, UnaryOpError> {
        map_unary_op(self, UnaryOp::Cos)
    }

    /// Tangent of each element (radians).
    pub fn tan(&self) -> Result<RumpyArray, UnaryOpError> {
        map_unary_op(self, UnaryOp::Tan)
    }

    /// Floor of each element.
    pub fn floor(&self) -> Result<RumpyArray, UnaryOpError> {
        map_unary_op(self, UnaryOp::Floor)
    }

    /// Ceiling of each element.
    pub fn ceil(&self) -> Result<RumpyArray, UnaryOpError> {
        map_unary_op(self, UnaryOp::Ceil)
    }

    /// Inverse sine (arcsine) of each element.
    pub fn arcsin(&self) -> Result<RumpyArray, UnaryOpError> {
        map_unary_op(self, UnaryOp::Arcsin)
    }

    /// Inverse cosine (arccosine) of each element.
    pub fn arccos(&self) -> Result<RumpyArray, UnaryOpError> {
        map_unary_op(self, UnaryOp::Arccos)
    }

    /// Inverse tangent (arctangent) of each element.
    pub fn arctan(&self) -> Result<RumpyArray, UnaryOpError> {
        map_unary_op(self, UnaryOp::Arctan)
    }

    /// Base-10 logarithm of each element.
    pub fn log10(&self) -> Result<RumpyArray, UnaryOpError> {
        map_unary_op(self, UnaryOp::Log10)
    }

    /// Base-2 logarithm of each element.
    pub fn log2(&self) -> Result<RumpyArray, UnaryOpError> {
        map_unary_op(self, UnaryOp::Log2)
    }

    /// Hyperbolic sine of each element.
    pub fn sinh(&self) -> Result<RumpyArray, UnaryOpError> {
        map_unary_op(self, UnaryOp::Sinh)
    }

    /// Hyperbolic cosine of each element.
    pub fn cosh(&self) -> Result<RumpyArray, UnaryOpError> {
        map_unary_op(self, UnaryOp::Cosh)
    }

    /// Hyperbolic tangent of each element.
    pub fn tanh(&self) -> Result<RumpyArray, UnaryOpError> {
        map_unary_op(self, UnaryOp::Tanh)
    }

    /// Element-wise sign indication.
    pub fn sign(&self) -> Result<RumpyArray, UnaryOpError> {
        map_unary_op(self, UnaryOp::Sign)
    }

    /// Test element-wise for NaN.
    pub fn isnan(&self) -> Result<RumpyArray, UnaryOpError> {
        map_unary_op(self, UnaryOp::Isnan)
    }

    /// Test element-wise for infinity.
    pub fn isinf(&self) -> Result<RumpyArray, UnaryOpError> {
        map_unary_op(self, UnaryOp::Isinf)
    }

    /// Test element-wise for finiteness.
    pub fn isfinite(&self) -> Result<RumpyArray, UnaryOpError> {
        map_unary_op(self, UnaryOp::Isfinite)
    }

    /// Square of each element (x^2).
    pub fn square(&self) -> Result<RumpyArray, UnaryOpError> {
        map_unary_op(self, UnaryOp::Square)
    }

    /// Return a copy of the array (positive identity).
    pub fn positive(&self) -> Result<RumpyArray, UnaryOpError> {
        map_unary_op(self, UnaryOp::Positive)
    }

    /// Reciprocal of each element (1/x).
    pub fn reciprocal(&self) -> Result<RumpyArray, UnaryOpError> {
        map_unary_op(self, UnaryOp::Reciprocal)
    }

    /// Base-2 exponential (2^x) of each element.
    pub fn exp2(&self) -> Result<RumpyArray, UnaryOpError> {
        map_unary_op(self, UnaryOp::Exp2)
    }

    /// exp(x) - 1 for each element (more precise for small x).
    pub fn expm1(&self) -> Result<RumpyArray, UnaryOpError> {
        map_unary_op(self, UnaryOp::Expm1)
    }

    /// log(1 + x) for each element (more precise for small x).
    pub fn log1p(&self) -> Result<RumpyArray, UnaryOpError> {
        map_unary_op(self, UnaryOp::Log1p)
    }

    /// Cube root of each element.
    pub fn cbrt(&self) -> Result<RumpyArray, UnaryOpError> {
        map_unary_op(self, UnaryOp::Cbrt)
    }

    /// Truncate each element to integer towards zero.
    pub fn trunc(&self) -> Result<RumpyArray, UnaryOpError> {
        map_unary_op(self, UnaryOp::Trunc)
    }

    /// Round each element to nearest integer.
    pub fn rint(&self) -> Result<RumpyArray, UnaryOpError> {
        map_unary_op(self, UnaryOp::Rint)
    }

    /// Inverse hyperbolic sine of each element.
    pub fn arcsinh(&self) -> Result<RumpyArray, UnaryOpError> {
        map_unary_op(self, UnaryOp::Arcsinh)
    }

    /// Inverse hyperbolic cosine of each element.
    pub fn arccosh(&self) -> Result<RumpyArray, UnaryOpError> {
        map_unary_op(self, UnaryOp::Arccosh)
    }

    /// Inverse hyperbolic tangent of each element.
    pub fn arctanh(&self) -> Result<RumpyArray, UnaryOpError> {
        map_unary_op(self, UnaryOp::Arctanh)
    }

    /// Returns True where the sign bit is set (negative).
    pub fn signbit(&self) -> Result<RumpyArray, UnaryOpError> {
        map_unary_op(self, UnaryOp::Signbit)
    }

    // ========================================================================
    // Complex number operations
    // ========================================================================

    /// Return the real part of the array.
    /// For complex arrays, extracts the real component.
    /// For real arrays, returns a copy.
    pub fn real(&self) -> RumpyArray {
        let kind = self.dtype().kind();
        let src_ptr = self.data_ptr();
        let dtype = self.dtype();
        let ops = dtype.ops();

        match kind {
            DTypeKind::Complex128 => {
                let mut result = RumpyArray::zeros(self.shape().to_vec(), DType::float64());
                let result_buffer = Arc::get_mut(result.buffer_mut()).expect("unique");
                let result_ptr = result_buffer.as_mut_ptr();
                for (i, offset) in self.iter_offsets().enumerate() {
                    let (re, _im) = unsafe { ops.read_complex(src_ptr, offset) }.unwrap_or((0.0, 0.0));
                    unsafe { *(result_ptr as *mut f64).add(i) = re; }
                }
                result
            }
            DTypeKind::Complex64 => {
                let mut result = RumpyArray::zeros(self.shape().to_vec(), DType::float32());
                let result_buffer = Arc::get_mut(result.buffer_mut()).expect("unique");
                let result_ptr = result_buffer.as_mut_ptr();
                for (i, offset) in self.iter_offsets().enumerate() {
                    let (re, _im) = unsafe { ops.read_complex(src_ptr, offset) }.unwrap_or((0.0, 0.0));
                    unsafe { *(result_ptr as *mut f32).add(i) = re as f32; }
                }
                result
            }
            _ => self.copy()
        }
    }

    /// Return the imaginary part of the array.
    /// For complex arrays, extracts the imaginary component.
    /// For real arrays, returns zeros.
    pub fn imag(&self) -> RumpyArray {
        let kind = self.dtype().kind();
        let src_ptr = self.data_ptr();
        let dtype = self.dtype();
        let ops = dtype.ops();

        match kind {
            DTypeKind::Complex128 => {
                let mut result = RumpyArray::zeros(self.shape().to_vec(), DType::float64());
                let result_buffer = Arc::get_mut(result.buffer_mut()).expect("unique");
                let result_ptr = result_buffer.as_mut_ptr();
                for (i, offset) in self.iter_offsets().enumerate() {
                    let (_re, im) = unsafe { ops.read_complex(src_ptr, offset) }.unwrap_or((0.0, 0.0));
                    unsafe { *(result_ptr as *mut f64).add(i) = im; }
                }
                result
            }
            DTypeKind::Complex64 => {
                let mut result = RumpyArray::zeros(self.shape().to_vec(), DType::float32());
                let result_buffer = Arc::get_mut(result.buffer_mut()).expect("unique");
                let result_ptr = result_buffer.as_mut_ptr();
                for (i, offset) in self.iter_offsets().enumerate() {
                    let (_re, im) = unsafe { ops.read_complex(src_ptr, offset) }.unwrap_or((0.0, 0.0));
                    unsafe { *(result_ptr as *mut f32).add(i) = im as f32; }
                }
                result
            }
            _ => RumpyArray::zeros(self.shape().to_vec(), self.dtype())
        }
    }

    /// Return the complex conjugate of the array.
    /// For complex arrays, negates the imaginary component.
    /// For real arrays, returns a copy.
    pub fn conj(&self) -> RumpyArray {
        let kind = self.dtype().kind();
        let src_ptr = self.data_ptr();
        let dtype = self.dtype();
        let ops = dtype.ops();

        match kind {
            DTypeKind::Complex128 => {
                let mut result = RumpyArray::zeros(self.shape().to_vec(), DType::complex128());
                let result_buffer = Arc::get_mut(result.buffer_mut()).expect("unique");
                let result_ptr = result_buffer.as_mut_ptr();
                for (i, offset) in self.iter_offsets().enumerate() {
                    let (re, im) = unsafe { ops.read_complex(src_ptr, offset) }.unwrap_or((0.0, 0.0));
                    unsafe {
                        let ptr = (result_ptr as *mut f64).add(i * 2);
                        *ptr = re;
                        *ptr.add(1) = -im;
                    }
                }
                result
            }
            DTypeKind::Complex64 => {
                let mut result = RumpyArray::zeros(self.shape().to_vec(), DType::complex64());
                let result_buffer = Arc::get_mut(result.buffer_mut()).expect("unique");
                let result_ptr = result_buffer.as_mut_ptr();
                for (i, offset) in self.iter_offsets().enumerate() {
                    let (re, im) = unsafe { ops.read_complex(src_ptr, offset) }.unwrap_or((0.0, 0.0));
                    unsafe {
                        let ptr = (result_ptr as *mut f32).add(i * 2);
                        *ptr = re as f32;
                        *ptr.add(1) = -im as f32;
                    }
                }
                result
            }
            _ => self.copy()
        }
    }

    /// Replace NaN with zero and infinity with large finite numbers.
    ///
    /// Returns an array with the same shape where:
    /// - NaN is replaced with `nan` (default 0.0)
    /// - positive infinity is replaced with `posinf` (default a large positive number)
    /// - negative infinity is replaced with `neginf` (default a large negative number)
    pub fn nan_to_num(&self, nan: f64, posinf: Option<f64>, neginf: Option<f64>) -> RumpyArray {
        let kind = self.dtype().kind();
        let src_ptr = self.data_ptr();
        let dtype = self.dtype();
        let ops = dtype.ops();

        macro_rules! nan_to_num_impl {
            ($T:ty, $dtype:expr) => {{
                let max_val = <$T>::MAX;
                let min_val = <$T>::MIN;
                let pos = posinf.map(|v| v as $T).unwrap_or(max_val);
                let neg = neginf.map(|v| v as $T).unwrap_or(min_val);
                let nan_val = nan as $T;

                let mut result = RumpyArray::zeros(self.shape().to_vec(), $dtype);
                let result_buffer = Arc::get_mut(result.buffer_mut()).expect("unique");
                let result_ptr = result_buffer.as_mut_ptr();
                for (i, offset) in self.iter_offsets().enumerate() {
                    let v: $T = unsafe { *(src_ptr.offset(offset) as *const $T) };
                    let out = if v.is_nan() {
                        nan_val
                    } else if v.is_infinite() {
                        if v > 0.0 { pos } else { neg }
                    } else {
                        v
                    };
                    unsafe { *(result_ptr as *mut $T).add(i) = out; }
                }
                result
            }};
        }

        match kind {
            DTypeKind::Float64 => nan_to_num_impl!(f64, DType::float64()),
            DTypeKind::Float32 => nan_to_num_impl!(f32, DType::float32()),
            DTypeKind::Float16 => {
                use half::f16;
                let pos = posinf.map(f16::from_f64).unwrap_or(f16::MAX);
                let neg = neginf.map(f16::from_f64).unwrap_or(f16::MIN);
                let nan_val = f16::from_f64(nan);

                let mut result = RumpyArray::zeros(self.shape().to_vec(), DType::float16());
                let result_buffer = Arc::get_mut(result.buffer_mut()).expect("unique");
                let result_ptr = result_buffer.as_mut_ptr();
                for (i, offset) in self.iter_offsets().enumerate() {
                    let v = unsafe { *(src_ptr.offset(offset) as *const f16) };
                    let out = if v.is_nan() {
                        nan_val
                    } else if v.is_infinite() {
                        if v.to_f32() > 0.0 { pos } else { neg }
                    } else {
                        v
                    };
                    unsafe { *(result_ptr as *mut f16).add(i) = out; }
                }
                result
            }
            DTypeKind::Complex128 => {
                let max_val = f64::MAX;
                let min_val = f64::MIN;
                let pos = posinf.unwrap_or(max_val);
                let neg = neginf.unwrap_or(min_val);

                let mut result = RumpyArray::zeros(self.shape().to_vec(), DType::complex128());
                let result_buffer = Arc::get_mut(result.buffer_mut()).expect("unique");
                let result_ptr = result_buffer.as_mut_ptr();
                for (i, offset) in self.iter_offsets().enumerate() {
                    let (re, im) = unsafe { ops.read_complex(src_ptr, offset) }.unwrap_or((0.0, 0.0));
                    let out_re = if re.is_nan() { nan } else if re.is_infinite() { if re > 0.0 { pos } else { neg } } else { re };
                    let out_im = if im.is_nan() { nan } else if im.is_infinite() { if im > 0.0 { pos } else { neg } } else { im };
                    unsafe {
                        let ptr = (result_ptr as *mut f64).add(i * 2);
                        *ptr = out_re;
                        *ptr.add(1) = out_im;
                    }
                }
                result
            }
            DTypeKind::Complex64 => {
                let max_val = f32::MAX;
                let min_val = f32::MIN;
                let pos = posinf.map(|v| v as f32).unwrap_or(max_val);
                let neg = neginf.map(|v| v as f32).unwrap_or(min_val);
                let nan_val = nan as f32;

                let mut result = RumpyArray::zeros(self.shape().to_vec(), DType::complex64());
                let result_buffer = Arc::get_mut(result.buffer_mut()).expect("unique");
                let result_ptr = result_buffer.as_mut_ptr();
                for (i, offset) in self.iter_offsets().enumerate() {
                    let (re, im) = unsafe { ops.read_complex(src_ptr, offset) }.unwrap_or((0.0, 0.0));
                    let (re, im) = (re as f32, im as f32);
                    let out_re = if re.is_nan() { nan_val } else if re.is_infinite() { if re > 0.0 { pos } else { neg } } else { re };
                    let out_im = if im.is_nan() { nan_val } else if im.is_infinite() { if im > 0.0 { pos } else { neg } } else { im };
                    unsafe {
                        let ptr = (result_ptr as *mut f32).add(i * 2);
                        *ptr = out_re;
                        *ptr.add(1) = out_im;
                    }
                }
                result
            }
            // For integer types, no NaN or Inf, just return a copy
            _ => self.copy()
        }
    }
}
