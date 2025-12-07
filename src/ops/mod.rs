//! Element-wise operations (ufunc-style).

#![allow(clippy::new_without_default)]

pub mod dot;
pub mod fft;
pub mod gufunc;
pub mod indexing;
pub mod inner;
pub mod linalg;
pub mod matmul;
pub mod outer;
pub mod registry;
pub mod solve;

use crate::array::{broadcast_shapes, increment_indices, promote_dtype, DType, RumpyArray};
use crate::array::dtype::{DTypeKind, UnaryOp, ReduceOp};
use registry::registry;
use std::sync::Arc;

// Re-export BinaryOp from dtype module
pub use crate::array::dtype::BinaryOp;

/// Error type for binary operations.
#[derive(Debug, Clone)]
pub enum BinaryOpError {
    /// Shapes cannot be broadcast together
    ShapeMismatch,
    /// Operation not supported for these dtypes
    UnsupportedDtype,
}

/// Error type for unary operations.
#[derive(Debug, Clone)]
pub enum UnaryOpError {
    /// Operation not supported for this dtype
    UnsupportedDtype,
}

/// Comparison operation types.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum ComparisonOp {
    Gt,  // >
    Lt,  // <
    Ge,  // >=
    Le,  // <=
    Eq,  // ==
    Ne,  // !=
}

// ============================================================================
// Core ufunc machinery
// ============================================================================

/// Compute byte offset from indices and strides.
#[inline]
fn linear_offset(indices: &[usize], strides: &[isize]) -> isize {
    indices.iter().zip(strides).map(|(&i, &s)| i as isize * s).sum()
}

/// Two-pass variance for contiguous f64 data - vectorizable.
/// Pass 1: compute mean, Pass 2: compute sum of squared deviations.
#[inline]
fn variance_f64_contiguous(ptr: *const f64, size: usize) -> f64 {
    // Pass 1: mean (sum / n) - vectorizable
    let mut sum = 0.0;
    for i in 0..size {
        sum += unsafe { *ptr.add(i) };
    }
    let mean = sum / size as f64;

    // Pass 2: sum of squared deviations - vectorizable
    let mut sum_sq = 0.0;
    for i in 0..size {
        let x = unsafe { *ptr.add(i) };
        let diff = x - mean;
        sum_sq += diff * diff;
    }
    sum_sq / size as f64
}

/// Check if a unary op is transcendental (always returns float).
fn is_transcendental(op: UnaryOp) -> bool {
    matches!(op, UnaryOp::Exp | UnaryOp::Log | UnaryOp::Log10 | UnaryOp::Log2 | UnaryOp::Sqrt |
                 UnaryOp::Sin | UnaryOp::Cos | UnaryOp::Tan |
                 UnaryOp::Sinh | UnaryOp::Cosh | UnaryOp::Tanh |
                 UnaryOp::Arcsin | UnaryOp::Arccos | UnaryOp::Arctan)
}

/// Apply a unary operation element-wise, returning a new array.
fn map_unary_op(arr: &RumpyArray, op: UnaryOp) -> Result<RumpyArray, UnaryOpError> {
    use crate::array::dtype::DTypeKind;
    use crate::array::DType;

    let kind = arr.dtype().kind();

    // Transcendentals on non-float types: promote based on itemsize (like NumPy)
    // 1 byte -> float16, 2 bytes -> float32, 4+ bytes -> float64
    if is_transcendental(op) && !matches!(kind, DTypeKind::Float16 | DTypeKind::Float64 | DTypeKind::Float32 | DTypeKind::Complex64 | DTypeKind::Complex128) {
        let itemsize = arr.dtype().itemsize();
        let target_dtype = if itemsize <= 1 {
            DType::float16()
        } else if itemsize <= 2 {
            DType::float32()
        } else {
            DType::float64()
        };
        let arr_float = arr.astype(target_dtype);
        return map_unary_op(&arr_float, op);
    }

    let dtype = arr.dtype();

    // Validate unsupported operations
    if matches!(kind, DTypeKind::Complex64 | DTypeKind::Complex128) {
        match op {
            UnaryOp::Floor | UnaryOp::Ceil => return Err(UnaryOpError::UnsupportedDtype),
            _ => {}
        }
    }

    let mut result = RumpyArray::zeros(arr.shape().to_vec(), dtype.clone());
    let size = arr.size();
    if size == 0 {
        return Ok(result);
    }

    let buffer = result.buffer_mut();
    let result_buffer = Arc::get_mut(buffer).expect("buffer must be unique");
    let result_ptr = result_buffer.as_mut_ptr();
    let src_ptr = arr.data_ptr();
    let itemsize = dtype.itemsize() as isize;

    // Try registry first
    {
        let reg = registry().read().unwrap();
        if let Some((loop_fn, _)) = reg.lookup_unary(op, kind.clone()) {
            if arr.is_c_contiguous() {
                // Fast path: call loop once for entire array
                let strides = (itemsize, itemsize);
                unsafe { loop_fn(src_ptr, result_ptr, size, strides); }
            } else {
                // Strided path: call loop once per innermost row with actual strides
                let ndim = arr.ndim();
                let src_strides = arr.strides();

                if ndim == 0 {
                    // Scalar
                    unsafe { loop_fn(src_ptr, result_ptr, 1, (itemsize, itemsize)); }
                } else if ndim == 1 {
                    // 1D: single call with actual stride
                    unsafe { loop_fn(src_ptr, result_ptr, size, (src_strides[0], itemsize)); }
                } else {
                    // nD: iterate over all but last dimension, call once per row
                    let inner_size = arr.shape()[ndim - 1];
                    let inner_stride = src_strides[ndim - 1];
                    let outer_shape = &arr.shape()[..ndim - 1];
                    let outer_size: usize = outer_shape.iter().product();

                    let mut outer_indices = vec![0usize; ndim - 1];
                    for i in 0..outer_size {
                        let src_offset = linear_offset(&outer_indices, src_strides);
                        unsafe {
                            loop_fn(
                                src_ptr.offset(src_offset),
                                result_ptr.offset(i as isize * inner_size as isize * itemsize),
                                inner_size,
                                (inner_stride, itemsize),
                            );
                        }
                        increment_indices(&mut outer_indices, outer_shape);
                    }
                }
            }
            return Ok(result);
        }
    }

    // Fallback: trait-based dispatch
    let ops = dtype.ops();
    for (i, offset) in arr.iter_offsets().enumerate() {
        unsafe { ops.unary_op(op, src_ptr, offset, result_ptr, i); }
    }
    Ok(result)
}

/// Check if dtype is an integer type.
fn is_integer_kind(kind: &crate::array::dtype::DTypeKind) -> bool {
    use crate::array::dtype::DTypeKind;
    matches!(kind, DTypeKind::Int16 | DTypeKind::Int32 | DTypeKind::Int64 |
             DTypeKind::Uint8 | DTypeKind::Uint16 | DTypeKind::Uint32 | DTypeKind::Uint64)
}

/// Apply a binary operation element-wise with broadcasting.
fn map_binary_op(a: &RumpyArray, b: &RumpyArray, op: BinaryOp) -> Result<RumpyArray, BinaryOpError> {
    // Delegate to inplace version with no output buffer
    map_binary_op_inplace(a, b, op, None)
}

/// Apply a binary operation with optional in-place output buffer reuse.
///
/// If `out` is Some and its buffer can be reused (via Arc::get_mut), writes
/// result there. Otherwise allocates a new array.
///
/// This is the core of temporary elision: when the caller detects an ephemeral
/// intermediate (Python refcount=1), it can pass ownership here for reuse.
pub fn map_binary_op_inplace(
    a: &RumpyArray,
    b: &RumpyArray,
    op: BinaryOp,
    out: Option<RumpyArray>,
) -> Result<RumpyArray, BinaryOpError> {
    use crate::array::dtype::DTypeKind;
    use crate::array::DType;

    // Handle division type promotion (same as map_binary_op)
    if op == BinaryOp::Div {
        let a_int = is_integer_kind(&a.dtype().kind());
        let b_int = is_integer_kind(&b.dtype().kind());
        if a_int && b_int {
            let a_f64 = a.astype(DType::float64());
            let b_f64 = b.astype(DType::float64());
            return map_binary_op_inplace(&a_f64, &b_f64, op, None); // Can't reuse int buffer for float
        } else if a_int || b_int {
            let int_to_float = |arr: &RumpyArray| -> RumpyArray {
                let itemsize = arr.dtype().itemsize();
                let target = if itemsize <= 1 {
                    DType::float16()
                } else if itemsize <= 2 {
                    DType::float32()
                } else {
                    DType::float64()
                };
                arr.astype(target)
            };
            let a_float = if a_int { int_to_float(a) } else { a.clone() };
            let b_float = if b_int { int_to_float(b) } else { b.clone() };
            return map_binary_op_inplace(&a_float, &b_float, op, None);
        }
    }

    let out_shape = broadcast_shapes(a.shape(), b.shape()).ok_or(BinaryOpError::ShapeMismatch)?;
    let a_bc = a.broadcast_to(&out_shape).ok_or(BinaryOpError::ShapeMismatch)?;
    let b_bc = b.broadcast_to(&out_shape).ok_or(BinaryOpError::ShapeMismatch)?;

    // Validate datetime operations
    let a_is_datetime = matches!(a_bc.dtype().kind(), DTypeKind::DateTime64(_));
    let b_is_datetime = matches!(b_bc.dtype().kind(), DTypeKind::DateTime64(_));
    if a_is_datetime || b_is_datetime {
        match op {
            BinaryOp::Add if a_is_datetime && b_is_datetime => return Err(BinaryOpError::UnsupportedDtype),
            BinaryOp::Mul | BinaryOp::Div | BinaryOp::Pow | BinaryOp::Mod | BinaryOp::FloorDiv => return Err(BinaryOpError::UnsupportedDtype),
            _ => {}
        }
    }

    let result_dtype = promote_dtype(&a_bc.dtype(), &b_bc.dtype());

    // Try to reuse the output buffer if provided and compatible
    let mut result = if let Some(mut out_arr) = out {
        // Check if we can actually get mutable access to the buffer
        let buffer = out_arr.buffer_mut();
        if Arc::get_mut(buffer).is_some() {
            // Buffer is uniquely owned, we can reuse it
            out_arr
        } else {
            // Buffer is shared (shouldn't happen if caller checked, but be safe)
            RumpyArray::zeros(out_shape.clone(), result_dtype.clone())
        }
    } else {
        RumpyArray::zeros(out_shape.clone(), result_dtype.clone())
    };

    let size = result.size();
    if size == 0 {
        return Ok(result);
    }

    let buffer = result.buffer_mut();
    let result_buffer = Arc::get_mut(buffer).expect("buffer must be unique");
    let result_ptr = result_buffer.as_mut_ptr();
    let a_ptr = a_bc.data_ptr();
    let b_ptr = b_bc.data_ptr();
    let result_dtype_ref = result.dtype();
    let result_ops = result_dtype_ref.ops();

    let a_kind = a_bc.dtype().kind();
    let b_kind = b_bc.dtype().kind();

    let mut indices = vec![0usize; out_shape.len()];
    let itemsize = result_dtype_ref.itemsize() as isize;

    // Try registry first for same-type operations
    if a_kind == b_kind {
        let reg = registry().read().unwrap();
        if let Some((loop_fn, _)) = reg.lookup_binary(op, a_kind.clone(), b_kind) {
            let a_contig = a_bc.is_c_contiguous() && a.shape() == out_shape.as_slice();
            let b_contig = b_bc.is_c_contiguous() && b.shape() == out_shape.as_slice();

            if a_contig && b_contig {
                let strides = (itemsize, itemsize, itemsize);
                unsafe { loop_fn(a_ptr, b_ptr, result_ptr, size, strides); }
            } else {
                let ndim = out_shape.len();
                let a_strides = a_bc.strides();
                let b_strides = b_bc.strides();

                if ndim == 0 {
                    unsafe { loop_fn(a_ptr, b_ptr, result_ptr, 1, (itemsize, itemsize, itemsize)); }
                } else if ndim == 1 {
                    unsafe { loop_fn(a_ptr, b_ptr, result_ptr, size, (a_strides[0], b_strides[0], itemsize)); }
                } else {
                    let inner_size = out_shape[ndim - 1];
                    let a_inner_stride = a_strides[ndim - 1];
                    let b_inner_stride = b_strides[ndim - 1];
                    let outer_shape = &out_shape[..ndim - 1];
                    let outer_size: usize = outer_shape.iter().product();

                    let mut outer_indices = vec![0usize; ndim - 1];
                    for i in 0..outer_size {
                        let a_offset = linear_offset(&outer_indices, a_strides);
                        let b_offset = linear_offset(&outer_indices, b_strides);
                        unsafe {
                            loop_fn(
                                a_ptr.offset(a_offset),
                                b_ptr.offset(b_offset),
                                result_ptr.offset(i as isize * inner_size as isize * itemsize),
                                inner_size,
                                (a_inner_stride, b_inner_stride, itemsize),
                            );
                        }
                        increment_indices(&mut outer_indices, outer_shape);
                    }
                }
            }
            return Ok(result);
        }
    }

    // Fallback: trait-based dispatch
    let same_dtype = a_bc.dtype() == b_bc.dtype() && a_bc.dtype() == result.dtype();

    if same_dtype {
        for i in 0..size {
            let a_offset = a_bc.byte_offset_for(&indices);
            let b_offset = b_bc.byte_offset_for(&indices);
            unsafe { result_ops.binary_op(op, a_ptr, a_offset, b_ptr, b_offset, result_ptr, i); }
            increment_indices(&mut indices, &out_shape);
        }
    } else {
        let a_dtype = a_bc.dtype();
        let a_ops = a_dtype.ops();
        let b_dtype = b_bc.dtype();
        let b_ops = b_dtype.ops();
        let result_kind = result.dtype().kind();
        let result_is_complex = matches!(result_kind, crate::array::dtype::DTypeKind::Complex64 | crate::array::dtype::DTypeKind::Complex128);

        if result_is_complex {
            for i in 0..size {
                let a_offset = a_bc.byte_offset_for(&indices);
                let b_offset = b_bc.byte_offset_for(&indices);
                let av = unsafe { a_ops.read_complex(a_ptr, a_offset).unwrap_or((0.0, 0.0)) };
                let bv = unsafe { b_ops.read_complex(b_ptr, b_offset).unwrap_or((0.0, 0.0)) };

                let result_val = match op {
                    BinaryOp::Add => (av.0 + bv.0, av.1 + bv.1),
                    BinaryOp::Sub => (av.0 - bv.0, av.1 - bv.1),
                    BinaryOp::Mul => (av.0 * bv.0 - av.1 * bv.1, av.0 * bv.1 + av.1 * bv.0),
                    BinaryOp::Div => {
                        let denom = bv.0 * bv.0 + bv.1 * bv.1;
                        if denom != 0.0 {
                            ((av.0 * bv.0 + av.1 * bv.1) / denom, (av.1 * bv.0 - av.0 * bv.1) / denom)
                        } else {
                            (f64::NAN, f64::NAN)
                        }
                    }
                    BinaryOp::Pow => {
                        if av.0 == 0.0 && av.1 == 0.0 {
                            if bv.0 > 0.0 { (0.0, 0.0) } else { (f64::NAN, f64::NAN) }
                        } else {
                            let mag_a = (av.0 * av.0 + av.1 * av.1).sqrt();
                            let ln_r = mag_a.ln();
                            let ln_i = av.1.atan2(av.0);
                            let prod_r = bv.0 * ln_r - bv.1 * ln_i;
                            let prod_i = bv.0 * ln_i + bv.1 * ln_r;
                            let exp_r = prod_r.exp();
                            (exp_r * prod_i.cos(), exp_r * prod_i.sin())
                        }
                    }
                    BinaryOp::Mod | BinaryOp::FloorDiv => (f64::NAN, f64::NAN),
                    BinaryOp::Maximum => if av.0.is_nan() || av.1.is_nan() || bv.0.is_nan() || bv.1.is_nan() {
                        (f64::NAN, f64::NAN)
                    } else if av.0 > bv.0 || (av.0 == bv.0 && av.1 >= bv.1) { av } else { bv },
                    BinaryOp::Minimum => if av.0.is_nan() || av.1.is_nan() || bv.0.is_nan() || bv.1.is_nan() {
                        (f64::NAN, f64::NAN)
                    } else if av.0 < bv.0 || (av.0 == bv.0 && av.1 <= bv.1) { av } else { bv },
                    // Stream 2: Binary math ops for complex fallback
                    BinaryOp::Arctan2 => {
                        // atan2 for complex: atan(y/x) in complex domain
                        let denom = bv.0 * bv.0 + bv.1 * bv.1;
                        if denom == 0.0 { (f64::NAN, f64::NAN) } else {
                            let div_r = (av.0 * bv.0 + av.1 * bv.1) / denom;
                            let div_i = (av.1 * bv.0 - av.0 * bv.1) / denom;
                            (div_i.atan2(div_r), 0.0)  // Simplified
                        }
                    }
                    BinaryOp::Hypot => ((av.0 * av.0 + av.1 * av.1 + bv.0 * bv.0 + bv.1 * bv.1).sqrt(), 0.0),
                    BinaryOp::FMax => {
                        let a_nan = av.0.is_nan() || av.1.is_nan();
                        let b_nan = bv.0.is_nan() || bv.1.is_nan();
                        if a_nan && b_nan { (f64::NAN, f64::NAN) }
                        else if a_nan { bv }
                        else if b_nan { av }
                        else if av.0 > bv.0 || (av.0 == bv.0 && av.1 >= bv.1) { av } else { bv }
                    }
                    BinaryOp::FMin => {
                        let a_nan = av.0.is_nan() || av.1.is_nan();
                        let b_nan = bv.0.is_nan() || bv.1.is_nan();
                        if a_nan && b_nan { (f64::NAN, f64::NAN) }
                        else if a_nan { bv }
                        else if b_nan { av }
                        else if av.0 < bv.0 || (av.0 == bv.0 && av.1 <= bv.1) { av } else { bv }
                    }
                    BinaryOp::Copysign => (av.0.copysign(bv.0), av.1.copysign(bv.1)),
                    BinaryOp::Logaddexp | BinaryOp::Logaddexp2 => (f64::NAN, f64::NAN),
                    BinaryOp::Nextafter => bv,
                };

                unsafe { result_ops.write_complex(result_ptr, i, result_val.0, result_val.1); }
                increment_indices(&mut indices, &out_shape);
            }
        } else {
            for i in 0..size {
                let a_offset = a_bc.byte_offset_for(&indices);
                let b_offset = b_bc.byte_offset_for(&indices);
                let av = unsafe { a_ops.read_f64(a_ptr, a_offset).unwrap_or(0.0) };
                let bv = unsafe { b_ops.read_f64(b_ptr, b_offset).unwrap_or(0.0) };

                let result_val = match op {
                    BinaryOp::Add => av + bv,
                    BinaryOp::Sub => av - bv,
                    BinaryOp::Mul => av * bv,
                    BinaryOp::Div => if bv != 0.0 { av / bv } else { f64::NAN },
                    BinaryOp::Pow => av.powf(bv),
                    BinaryOp::Mod => av % bv,
                    BinaryOp::FloorDiv => (av / bv).floor(),
                    BinaryOp::Maximum => if av.is_nan() || bv.is_nan() { f64::NAN } else { av.max(bv) },
                    BinaryOp::Minimum => if av.is_nan() || bv.is_nan() { f64::NAN } else { av.min(bv) },
                    // Stream 2: Binary math ops fallback
                    BinaryOp::Arctan2 => av.atan2(bv),
                    BinaryOp::Hypot => av.hypot(bv),
                    BinaryOp::FMax => if bv.is_nan() { av } else if av.is_nan() { bv } else { av.max(bv) },
                    BinaryOp::FMin => if bv.is_nan() { av } else if av.is_nan() { bv } else { av.min(bv) },
                    BinaryOp::Copysign => av.copysign(bv),
                    BinaryOp::Logaddexp => {
                        let m = av.max(bv);
                        if m.is_infinite() { m } else { m + (1.0 + (-(av - bv).abs()).exp()).ln() }
                    }
                    BinaryOp::Logaddexp2 => {
                        let m = av.max(bv);
                        let ln2 = std::f64::consts::LN_2;
                        if m.is_infinite() { m } else { m + ((1.0 + (-(av - bv).abs() * ln2).exp()).ln() / ln2) }
                    }
                    BinaryOp::Nextafter => {
                        if av.is_nan() || bv.is_nan() { f64::NAN }
                        else if av == bv { bv }
                        else if av < bv {
                            let bits = av.to_bits();
                            f64::from_bits(if av >= 0.0 { bits + 1 } else { bits - 1 })
                        } else {
                            let bits = av.to_bits();
                            f64::from_bits(if av > 0.0 { bits - 1 } else { bits + 1 })
                        }
                    }
                };

                unsafe { result_ops.write_f64(result_ptr, i, result_val); }
                increment_indices(&mut indices, &out_shape);
            }
        }
    }
    Ok(result)
}

/// Apply a comparison function element-wise, returning bool array.
/// Note: comparison still uses f64 for now, since ordering on complex is tricky.
fn map_compare_op(a: &RumpyArray, b: &RumpyArray, op: ComparisonOp) -> Option<RumpyArray> {
    use crate::array::promote_dtype;

    let out_shape = broadcast_shapes(a.shape(), b.shape())?;

    // Promote to common dtype for comparison (like binary ops)
    let common_dtype = promote_dtype(&a.dtype(), &b.dtype());
    let a_promoted = if a.dtype() != common_dtype { a.astype(common_dtype.clone()) } else { a.clone() };
    let b_promoted = if b.dtype() != common_dtype { b.astype(common_dtype.clone()) } else { b.clone() };

    let a_bc = a_promoted.broadcast_to(&out_shape)?;
    let b_bc = b_promoted.broadcast_to(&out_shape)?;

    let mut result = RumpyArray::zeros(out_shape.clone(), DType::bool());
    let size = result.size();
    if size == 0 {
        return Some(result);
    }

    let buffer = result.buffer_mut();
    let result_buffer = Arc::get_mut(buffer).expect("buffer must be unique");
    let result_ptr = result_buffer.as_mut_ptr();
    let a_ptr = a_bc.data_ptr();
    let b_ptr = b_bc.data_ptr();
    let common_kind = common_dtype.kind();

    // Try registry for typed comparison loops (fast path for contiguous arrays)
    {
        let reg = registry().read().unwrap();
        if let Some(loop_fn) = reg.lookup_compare(op, common_kind) {
            let itemsize = common_dtype.itemsize() as isize;
            let a_contig = a_bc.is_c_contiguous();
            let b_contig = b_bc.is_c_contiguous();
            let a_same_shape = a_promoted.shape() == out_shape.as_slice();
            let b_same_shape = b_promoted.shape() == out_shape.as_slice();
            let b_is_scalar = b_promoted.size() == 1;

            // Fast path: contiguous a, and b is either contiguous same-shape or scalar
            let b_ok = (b_same_shape && b_contig) || b_is_scalar;
            if a_contig && a_same_shape && b_ok {
                let b_stride = if b_is_scalar { 0 } else { itemsize };
                unsafe { loop_fn(a_ptr, b_ptr, result_ptr, size, (itemsize, b_stride, 1)); }
                return Some(result);
            }
            // Non-contiguous falls through to generic path below
        }
    }

    // Generic path: works for any dtype/stride combination via get_element
    let f: fn(f64, f64) -> bool = match op {
        ComparisonOp::Gt => |a, b| a > b,
        ComparisonOp::Lt => |a, b| a < b,
        ComparisonOp::Ge => |a, b| a >= b,
        ComparisonOp::Le => |a, b| a <= b,
        ComparisonOp::Eq => |a, b| a == b,
        ComparisonOp::Ne => |a, b| a != b,
    };

    let mut indices = vec![0usize; out_shape.len()];
    for i in 0..size {
        let cmp_result = f(a_bc.get_element(&indices), b_bc.get_element(&indices));
        unsafe { *result_ptr.add(i) = if cmp_result { 1 } else { 0 }; }
        increment_indices(&mut indices, &out_shape);
    }
    Some(result)
}

/// Reduce array along all axes, returning a 0-d array.
fn reduce_all_op(arr: &RumpyArray, op: ReduceOp) -> RumpyArray {
    let mut result = RumpyArray::zeros(vec![1], arr.dtype());
    let size = arr.size();

    let buffer = result.buffer_mut();
    let result_buffer = Arc::get_mut(buffer).expect("buffer must be unique");
    let result_ptr = result_buffer.as_mut_ptr();
    let dtype = arr.dtype();
    let kind = dtype.kind();
    let itemsize = dtype.itemsize() as isize;

    // Try strided loops first (SIMD-optimized for contiguous arrays)
    {
        let reg = registry().read().unwrap();
        if arr.is_c_contiguous() {
            if let Some((init_fn, loop_fn, _)) = reg.lookup_reduce_strided(op, kind.clone()) {
                unsafe { init_fn(result_ptr, 0); }
                if size == 0 {
                    return result;
                }
                unsafe { loop_fn(result_ptr, arr.data_ptr(), size, itemsize); }
                return result;
            }
        }

        // Per-element path (for non-contiguous or missing strided loop)
        if let Some((init_fn, acc_fn, _)) = reg.lookup_reduce(op, kind.clone()) {
            unsafe { init_fn(result_ptr, 0); }
            if size == 0 {
                return result;
            }
            let src_ptr = arr.data_ptr();
            for offset in arr.iter_offsets() {
                unsafe { acc_fn(result_ptr, 0, src_ptr, offset); }
            }
            return result;
        }
    }

    // Fallback: trait-based dispatch
    let ops = dtype.ops();
    unsafe { ops.reduce_init(op, result_ptr, 0); }
    if size == 0 {
        return result;
    }

    let src_ptr = arr.data_ptr();
    for offset in arr.iter_offsets() {
        unsafe { ops.reduce_acc(op, result_ptr, 0, src_ptr, offset); }
    }
    result
}

/// Get reduction result as f64 (for backwards compatibility).
fn reduce_all_f64(arr: &RumpyArray, op: ReduceOp) -> f64 {
    let result = reduce_all_op(arr, op);
    result.get_element(&[0])
}

/// Reduce array along a specific axis.
///
/// Uses strided reduce loops from the registry for efficient axis reductions.
/// Each loop handles both contiguous (SIMD) and strided cases internally.
/// See designs/backstride-iteration.md for design rationale.
fn reduce_axis_op(arr: &RumpyArray, axis: usize, op: ReduceOp) -> RumpyArray {
    // Output shape: remove the reduction axis
    let mut out_shape: Vec<usize> = arr.shape().to_vec();
    let axis_len = out_shape.remove(axis);

    if out_shape.is_empty() {
        out_shape = vec![1]; // Scalar result wrapped in 1D array
    }

    let dtype = arr.dtype();
    let kind = dtype.kind();
    let itemsize = dtype.itemsize();
    let axis_stride = arr.strides()[axis];
    let src_ptr = arr.data_ptr();

    let mut result = RumpyArray::zeros(out_shape, arr.dtype());
    let out_size = result.size();

    if out_size == 0 || axis_len == 0 {
        return result;
    }

    let buffer = result.buffer_mut();
    let result_buffer = Arc::get_mut(buffer).expect("buffer must be unique");
    let result_ptr = result_buffer.as_mut_ptr();

    // Try registry lookups (single lock acquisition)
    let reg = registry().read().unwrap();

    // Prefer strided loops (process N elements at once)
    if let Some((init_fn, loop_fn, _)) = reg.lookup_reduce_strided(op, kind.clone()) {
        for i in 0..out_size {
            unsafe { init_fn(result_ptr, i); }
        }
        for (i, base_offset) in arr.axis_offsets(axis).enumerate() {
            let src_start = unsafe { src_ptr.offset(base_offset) };
            let acc_ptr = unsafe { result_ptr.add(i * itemsize) };
            unsafe { loop_fn(acc_ptr, src_start, axis_len, axis_stride); }
        }
        return result;
    }

    // Fallback to per-element loops
    if let Some((init_fn, acc_fn, _)) = reg.lookup_reduce(op, kind.clone()) {
        for i in 0..out_size {
            unsafe { init_fn(result_ptr, i); }
        }
        for (i, base_offset) in arr.axis_offsets(axis).enumerate() {
            let mut ptr = unsafe { src_ptr.offset(base_offset) };
            for _ in 0..axis_len {
                unsafe { acc_fn(result_ptr, i, ptr, 0); }
                ptr = unsafe { ptr.offset(axis_stride) };
            }
        }
        return result;
    }

    drop(reg); // Release lock before trait dispatch

    // Trait-based fallback
    let ops = dtype.ops();
    for i in 0..out_size {
        unsafe { ops.reduce_init(op, result_ptr, i); }
    }
    for (i, base_offset) in arr.axis_offsets(axis).enumerate() {
        let mut ptr = unsafe { src_ptr.offset(base_offset) };
        for _ in 0..axis_len {
            unsafe { ops.reduce_acc(op, result_ptr, i, ptr, 0); }
            ptr = unsafe { ptr.offset(axis_stride) };
        }
    }

    result
}

// Type promotion uses crate::array::promote_dtype

// ============================================================================
// diff helper functions
// ============================================================================

/// Fast 1D contiguous diff using registry's Sub loop.
/// result[i] = src[i+1] - src[i]
#[inline]
fn diff_1d_contiguous(src_ptr: *const u8, result_ptr: *mut u8, n: usize, dtype: &DType) {
    let itemsize = dtype.itemsize() as isize;

    // Try registry's vectorized Sub loop
    if let Some((loop_fn, _)) = registry().read().unwrap()
        .lookup_binary(BinaryOp::Sub, dtype.kind(), dtype.kind())
    {
        unsafe {
            loop_fn(
                src_ptr.offset(itemsize),  // a = src[i+1]
                src_ptr,                    // b = src[i]
                result_ptr,
                n,
                (itemsize, itemsize, itemsize),
            );
        }
    } else {
        // Fallback for unsupported dtypes
        let ops = dtype.ops();
        for i in 0..n {
            unsafe {
                let v1 = ops.read_f64(src_ptr, (i as isize) * itemsize).unwrap_or(0.0);
                let v2 = ops.read_f64(src_ptr, (i as isize + 1) * itemsize).unwrap_or(0.0);
                ops.write_f64(result_ptr, i, v2 - v1);
            }
        }
    }
}

/// Strided diff for N-D arrays along arbitrary axis.
/// Element-by-element via DTypeOps (no vectorization benefit for strided access).
fn diff_strided(
    src: &RumpyArray,
    result: &RumpyArray,
    axis_stride: isize,
    src_ptr: *const u8,
    result_ptr: *mut u8,
    dtype: &DType,
) {
    let out_shape = result.shape();
    let out_size: usize = out_shape.iter().product();
    let ops = dtype.ops();

    let mut out_indices = vec![0usize; src.ndim()];
    for i in 0..out_size {
        let offset1 = src.byte_offset_for(&out_indices);
        unsafe {
            let v1 = ops.read_f64(src_ptr, offset1).unwrap_or(0.0);
            let v2 = ops.read_f64(src_ptr, offset1 + axis_stride).unwrap_or(0.0);
            ops.write_f64(result_ptr, i, v2 - v1);
        }
        increment_indices(&mut out_indices, out_shape);
    }
}

// ============================================================================
// Public API using ufunc machinery
// ============================================================================

impl RumpyArray {
    /// Element-wise binary operation with broadcasting.
    pub fn binary_op(&self, other: &RumpyArray, op: BinaryOp) -> Result<RumpyArray, BinaryOpError> {
        map_binary_op(self, other, op)
    }

    /// Element-wise operation with scalar (arr op scalar).
    pub fn scalar_op(&self, scalar: f64, op: BinaryOp) -> RumpyArray {
        // Create a scalar array and use binary_op with broadcasting
        let scalar_arr = RumpyArray::full(vec![1], scalar, self.dtype());
        self.binary_op(&scalar_arr, op).expect("scalar broadcast always works")
    }

    /// Scalar on left side (scalar op arr).
    pub fn rscalar_op(&self, scalar: f64, op: BinaryOp) -> RumpyArray {
        let scalar_arr = RumpyArray::full(vec![1], scalar, self.dtype());
        scalar_arr.binary_op(self, op).expect("scalar broadcast always works")
    }

    /// Element-wise comparison with broadcasting.
    pub fn compare(&self, other: &RumpyArray, op: ComparisonOp) -> Option<RumpyArray> {
        map_compare_op(self, other, op)
    }

    /// Scalar comparison (arr op scalar).
    pub fn compare_scalar(&self, scalar: f64, op: ComparisonOp) -> RumpyArray {
        // Create scalar array and use broadcasting
        let scalar_arr = RumpyArray::full(vec![1], scalar, DType::float64());
        self.compare(&scalar_arr, op).expect("scalar broadcast always succeeds")
    }

    /// Negate each element.
    pub fn neg(&self) -> Result<RumpyArray, UnaryOpError> {
        map_unary_op(self, UnaryOp::Neg)
    }

    /// Absolute value of each element.
    pub fn abs(&self) -> Result<RumpyArray, UnaryOpError> {
        map_unary_op(self, UnaryOp::Abs)
    }

    /// Sum all elements.
    pub fn sum(&self) -> f64 {
        reduce_all_f64(self, ReduceOp::Sum)
    }

    /// Sum along axis.
    pub fn sum_axis(&self, axis: usize) -> RumpyArray {
        reduce_axis_op(self, axis, ReduceOp::Sum)
    }

    /// Product of all elements.
    pub fn prod(&self) -> f64 {
        reduce_all_f64(self, ReduceOp::Prod)
    }

    /// Product along axis.
    pub fn prod_axis(&self, axis: usize) -> RumpyArray {
        reduce_axis_op(self, axis, ReduceOp::Prod)
    }

    /// Maximum element.
    pub fn max(&self) -> f64 {
        reduce_all_f64(self, ReduceOp::Max)
    }

    /// Maximum along axis.
    pub fn max_axis(&self, axis: usize) -> RumpyArray {
        reduce_axis_op(self, axis, ReduceOp::Max)
    }

    /// Minimum element.
    pub fn min(&self) -> f64 {
        reduce_all_f64(self, ReduceOp::Min)
    }

    /// Minimum along axis.
    pub fn min_axis(&self, axis: usize) -> RumpyArray {
        reduce_axis_op(self, axis, ReduceOp::Min)
    }

    /// Mean of all elements.
    pub fn mean(&self) -> f64 {
        if self.size() == 0 {
            return f64::NAN;
        }
        self.sum() / self.size() as f64
    }

    /// Mean along axis.
    pub fn mean_axis(&self, axis: usize) -> RumpyArray {
        let sum = self.sum_axis(axis);
        let count = self.shape()[axis] as f64;
        // Divide each element by count
        let count_arr = RumpyArray::full(vec![1], count, sum.dtype());
        sum.binary_op(&count_arr, BinaryOp::Div).expect("broadcast works")
    }

    /// Variance of all elements.
    /// Uses two-pass algorithm (mean then sum of squared deviations) for vectorization.
    pub fn var(&self) -> f64 {
        let size = self.size();
        if size == 0 {
            return f64::NAN;
        }

        // Fast path for contiguous f64
        use crate::array::dtype::DTypeKind;
        if self.is_c_contiguous() && self.dtype().kind() == DTypeKind::Float64 {
            let ptr = self.data_ptr() as *const f64;
            return variance_f64_contiguous(ptr, size);
        }

        // General strided path: two-pass for numerical stability
        let mean = self.mean();
        let ptr = self.data_ptr();
        let dtype = self.dtype();
        let ops = dtype.ops();

        let mut sum_sq = 0.0;
        for offset in self.iter_offsets() {
            let x = unsafe { ops.read_f64(ptr, offset) }.unwrap_or(0.0);
            let diff = x - mean;
            sum_sq += diff * diff;
        }
        sum_sq / size as f64
    }

    /// Variance along axis (second central moment).
    pub fn var_axis(&self, axis: usize) -> RumpyArray {
        self.moment_axis(2, axis)
    }

    /// Central moment of order k for all elements.
    /// moment(k=2) == variance, moment(k=3)/std^3 == skewness, etc.
    pub fn moment(&self, k: usize) -> f64 {
        let size = self.size();
        if size == 0 {
            return f64::NAN;
        }
        let ptr = self.data_ptr();
        let dtype = self.dtype();
        let ops = dtype.ops();

        // Two-pass: compute mean, then sum of (x - mean)^k
        let mean = self.mean();
        let mut sum_mk = 0.0;
        for offset in self.iter_offsets() {
            let x = unsafe { ops.read_f64(ptr, offset) }.unwrap_or(0.0);
            sum_mk += (x - mean).powi(k as i32);
        }
        sum_mk / size as f64
    }

    /// Central moment along axis.
    /// Uses vectorized operations: (x - mean)^k summed along axis.
    pub fn moment_axis(&self, k: usize, axis: usize) -> RumpyArray {
        let axis_len = self.shape()[axis];
        if axis_len == 0 {
            let mut out_shape: Vec<usize> = self.shape().to_vec();
            out_shape.remove(axis);
            if out_shape.is_empty() {
                out_shape = vec![1];
            }
            return RumpyArray::zeros(out_shape, DType::float64());
        }

        // Compute mean and broadcast back to original shape
        let mean = self.mean_axis(axis);
        let mean_expanded = mean.expand_dims(axis).expect("expand_dims succeeds");

        // Compute deviations: x - mean (uses vectorized binary op with broadcasting)
        let diff = self.binary_op(&mean_expanded, BinaryOp::Sub).expect("broadcast succeeds");

        // Raise to power k using vectorized ops
        let powered = if k == 2 {
            // Special case: x^2 = x * x (faster than pow)
            diff.binary_op(&diff, BinaryOp::Mul).expect("same shape")
        } else {
            // General case: use pow
            let k_arr = RumpyArray::full(vec![1], k as f64, diff.dtype());
            diff.binary_op(&k_arr, BinaryOp::Pow).expect("broadcast works")
        };

        // Sum along axis and divide by count (uses efficient reduce_axis_op)
        let sum = powered.sum_axis(axis);
        let count_arr = RumpyArray::full(vec![1], axis_len as f64, sum.dtype());
        sum.binary_op(&count_arr, BinaryOp::Div).expect("broadcast works")
    }

    /// Skewness of all elements (Fisher's definition: m3 / m2^1.5).
    pub fn skew(&self) -> f64 {
        let m2 = self.moment(2);
        let m3 = self.moment(3);
        if m2 == 0.0 {
            return 0.0;
        }
        m3 / m2.powf(1.5)
    }

    /// Skewness along axis.
    pub fn skew_axis(&self, axis: usize) -> RumpyArray {
        let m2 = self.moment_axis(2, axis);
        let m3 = self.moment_axis(3, axis);
        // m3 / m2^1.5
        let m2_pow = map_unary_op(&m2, UnaryOp::Sqrt).expect("sqrt works");
        let m2_pow = m2.binary_op(&m2_pow, BinaryOp::Mul).expect("broadcast works");
        m3.binary_op(&m2_pow, BinaryOp::Div).expect("broadcast works")
    }

    /// Kurtosis of all elements (Fisher's definition: m4 / m2^2 - 3).
    pub fn kurtosis(&self) -> f64 {
        let m2 = self.moment(2);
        let m4 = self.moment(4);
        if m2 == 0.0 {
            return 0.0;
        }
        m4 / (m2 * m2) - 3.0
    }

    /// Kurtosis along axis.
    pub fn kurtosis_axis(&self, axis: usize) -> RumpyArray {
        let m2 = self.moment_axis(2, axis);
        let m4 = self.moment_axis(4, axis);
        // m4 / m2^2 - 3
        let m2_sq = m2.binary_op(&m2, BinaryOp::Mul).expect("broadcast works");
        let ratio = m4.binary_op(&m2_sq, BinaryOp::Div).expect("broadcast works");
        let three = RumpyArray::full(vec![1], 3.0, ratio.dtype());
        ratio.binary_op(&three, BinaryOp::Sub).expect("broadcast works")
    }

    /// Standard deviation of all elements.
    pub fn std(&self) -> f64 {
        self.var().sqrt()
    }

    /// Standard deviation along axis.
    pub fn std_axis(&self, axis: usize) -> RumpyArray {
        map_unary_op(&self.var_axis(axis), UnaryOp::Sqrt).expect("sqrt always succeeds on numeric types")
    }

    /// Index of maximum element (flattened).
    pub fn argmax(&self) -> usize {
        let size = self.size();
        if size == 0 {
            return 0;
        }
        let ptr = self.data_ptr();
        let dtype = self.dtype();
        let ops = dtype.ops();
        let mut max_val = f64::NEG_INFINITY;
        let mut max_idx = 0;
        for (i, offset) in self.iter_offsets().enumerate() {
            let val = unsafe { ops.read_f64(ptr, offset) }.unwrap_or(0.0);
            if val > max_val {
                max_val = val;
                max_idx = i;
            }
        }
        max_idx
    }

    /// Index of minimum element (flattened).
    pub fn argmin(&self) -> usize {
        let size = self.size();
        if size == 0 {
            return 0;
        }
        let ptr = self.data_ptr();
        let dtype = self.dtype();
        let ops = dtype.ops();
        let mut min_val = f64::INFINITY;
        let mut min_idx = 0;
        for (i, offset) in self.iter_offsets().enumerate() {
            let val = unsafe { ops.read_f64(ptr, offset) }.unwrap_or(0.0);
            if val < min_val {
                min_val = val;
                min_idx = i;
            }
        }
        min_idx
    }

    /// Index of maximum element along axis.
    pub fn argmax_axis(&self, axis: usize) -> RumpyArray {
        self.arg_reduce_axis(axis, |a, b| a > b)
    }

    /// Index of minimum element along axis.
    pub fn argmin_axis(&self, axis: usize) -> RumpyArray {
        self.arg_reduce_axis(axis, |a, b| a < b)
    }

    /// Helper for argmax_axis/argmin_axis.
    fn arg_reduce_axis<F>(&self, axis: usize, is_better: F) -> RumpyArray
    where
        F: Fn(f64, f64) -> bool,
    {
        let shape = self.shape();
        let axis_len = shape[axis];

        // Output shape: remove the axis dimension
        let mut out_shape: Vec<usize> = shape[..axis].to_vec();
        out_shape.extend_from_slice(&shape[axis + 1..]);
        if out_shape.is_empty() {
            out_shape = vec![1];
        }

        let out_size: usize = out_shape.iter().product();
        let mut result = RumpyArray::zeros(out_shape.clone(), DType::int64());

        if out_size == 0 || axis_len == 0 {
            return result;
        }

        let buffer = result.buffer_mut();
        let result_buffer = Arc::get_mut(buffer).expect("buffer must be unique");
        let result_ptr = result_buffer.as_mut_ptr();
        let int64_dtype = DType::int64();
        let ops = int64_dtype.ops();

        let mut outer_indices = vec![0usize; out_shape.len()];
        for out_i in 0..out_size {
            // Build indices for the input array
            let mut in_indices: Vec<usize> = outer_indices[..axis.min(outer_indices.len())].to_vec();
            in_indices.push(0);
            if axis < self.ndim() - 1 && outer_indices.len() > axis {
                in_indices.extend_from_slice(&outer_indices[axis..]);
            }

            // Find best index along axis
            let mut best_val = self.get_element(&in_indices);
            let mut best_idx: i64 = 0;
            for j in 1..axis_len {
                in_indices[axis] = j;
                let val = self.get_element(&in_indices);
                if is_better(val, best_val) {
                    best_val = val;
                    best_idx = j as i64;
                }
            }

            unsafe { ops.write_f64(result_ptr, out_i, best_idx as f64); }
            increment_indices(&mut outer_indices, &out_shape);
        }
        result
    }

    /// Collect all elements into a Vec (flattened, row-major order).
    fn to_vec(&self) -> Vec<f64> {
        let size = self.size();
        if size == 0 {
            return Vec::new();
        }

        // Fast path for contiguous f64 arrays
        if self.is_c_contiguous() && self.dtype().kind() == DTypeKind::Float64 {
            let ptr = self.data_ptr() as *const f64;
            let mut values = Vec::with_capacity(size);
            unsafe {
                values.set_len(size);
                std::ptr::copy_nonoverlapping(ptr, values.as_mut_ptr(), size);
            }
            return values;
        }

        // Slow path for non-contiguous or non-f64
        let ptr = self.data_ptr();
        let dtype = self.dtype();
        let ops = dtype.ops();
        let mut values = Vec::with_capacity(size);
        for offset in self.iter_offsets() {
            values.push(unsafe { ops.read_f64(ptr, offset) }.unwrap_or(0.0));
        }
        values
    }

    // sort and argsort with axis parameter are defined later

    /// Return unique sorted values.
    pub fn unique(&self) -> RumpyArray {
        let mut values = self.to_vec();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        values.dedup_by(|a, b| (*a - *b).abs() < f64::EPSILON);
        RumpyArray::from_vec(values, self.dtype())
    }

    /// Return indices of non-zero elements (tuple of arrays, one per dimension).
    pub fn nonzero(&self) -> Vec<RumpyArray> {
        let ndim = self.ndim();
        let mut indices_per_dim: Vec<Vec<i64>> = vec![Vec::new(); ndim];

        let mut indices = vec![0usize; ndim];
        let size = self.size();
        for _ in 0..size {
            let val = self.get_element(&indices);
            if val != 0.0 {
                for (dim, &idx) in indices.iter().enumerate() {
                    indices_per_dim[dim].push(idx as i64);
                }
            }
            increment_indices(&mut indices, self.shape());
        }

        indices_per_dim
            .into_iter()
            .map(|v| {
                let data: Vec<f64> = v.into_iter().map(|x| x as f64).collect();
                RumpyArray::from_vec(data, DType::int64())
            })
            .collect()
    }

    // Math ufuncs

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

    /// Return the real part of the array.
    /// For complex arrays, extracts the real component.
    /// For real arrays, returns a copy.
    pub fn real(&self) -> RumpyArray {
        use crate::array::dtype::DTypeKind;

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
        use crate::array::dtype::DTypeKind;

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
            _ => RumpyArray::zeros(self.shape().to_vec(), self.dtype().clone())
        }
    }

    /// Return the complex conjugate of the array.
    /// For complex arrays, negates the imaginary component.
    /// For real arrays, returns a copy.
    pub fn conj(&self) -> RumpyArray {
        use crate::array::dtype::DTypeKind;

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
                        *(result_ptr as *mut f64).add(i * 2) = re;
                        *(result_ptr as *mut f64).add(i * 2 + 1) = -im;
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
                        *(result_ptr as *mut f32).add(i * 2) = re as f32;
                        *(result_ptr as *mut f32).add(i * 2 + 1) = -im as f32;
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
        use crate::array::dtype::DTypeKind;

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
                let pos = posinf.map(|v| f16::from_f64(v)).unwrap_or(f16::MAX);
                let neg = neginf.map(|v| f16::from_f64(v)).unwrap_or(f16::MIN);
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
                        *(result_ptr as *mut f64).add(i * 2) = out_re;
                        *(result_ptr as *mut f64).add(i * 2 + 1) = out_im;
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
                        *(result_ptr as *mut f32).add(i * 2) = out_re;
                        *(result_ptr as *mut f32).add(i * 2 + 1) = out_im;
                    }
                }
                result
            }
            // For integer types, no NaN or Inf, just return a copy
            _ => self.copy()
        }
    }

    /// Extract diagonal from a 2D array.
    pub fn diagonal(&self) -> RumpyArray {
        assert!(self.ndim() >= 2, "diagonal requires at least 2D array");
        let shape = self.shape();
        let n = shape[0].min(shape[1]);
        let dtype = self.dtype().clone();

        let mut result = RumpyArray::zeros(vec![n], dtype.clone());
        if n == 0 {
            return result;
        }

        let result_buffer = Arc::get_mut(result.buffer_mut()).expect("unique");
        let result_ptr = result_buffer.as_mut_ptr();
        let ops = dtype.ops();

        for i in 0..n {
            let mut indices = vec![0usize; self.ndim()];
            indices[0] = i;
            indices[1] = i;
            let byte_offset = self.byte_offset_for(&indices);
            unsafe { ops.copy_element(self.data_ptr(), byte_offset, result_ptr, i); }
        }
        result
    }

    /// Return sum of diagonal elements (trace).
    pub fn trace(&self) -> f64 {
        assert!(self.ndim() >= 2, "trace requires at least 2D array");
        let shape = self.shape();
        let n = shape[0].min(shape[1]);

        let mut sum = 0.0;
        for i in 0..n {
            let mut indices = vec![0usize; self.ndim()];
            indices[0] = i;
            indices[1] = i;
            sum += self.get_element(&indices);
        }
        sum
    }

    /// Swap two axes of the array.
    pub fn swapaxes(&self, axis1: usize, axis2: usize) -> RumpyArray {
        let ndim = self.ndim();
        assert!(axis1 < ndim, "axis1 out of bounds");
        assert!(axis2 < ndim, "axis2 out of bounds");

        if axis1 == axis2 {
            return self.copy();
        }

        // Build permutation: swap axis1 and axis2
        let mut axes: Vec<usize> = (0..ndim).collect();
        axes.swap(axis1, axis2);

        self.transpose_axes(&axes)
    }

    /// Sort array along an axis.
    pub fn sort(&self, axis: Option<usize>) -> RumpyArray {
        // If axis is None, flatten and sort
        let axis = match axis {
            None => {
                let flat = self.reshape(vec![self.size()]).expect("flatten should work");
                return flat.sort(Some(0));
            }
            Some(ax) => ax,
        };
        assert!(axis < self.ndim(), "axis out of bounds");

        let shape = self.shape();
        let axis_len = shape[axis];
        let dtype = self.dtype().clone();

        let mut result = self.copy();
        if axis_len <= 1 {
            return result;
        }

        // For each position along other axes, sort the elements along the target axis
        let mut out_shape: Vec<usize> = shape[..axis].to_vec();
        out_shape.extend_from_slice(&shape[axis + 1..]);
        if out_shape.is_empty() {
            out_shape = vec![1];
        }
        let out_size: usize = out_shape.iter().product();

        let ops = dtype.ops();
        let result_buffer = Arc::get_mut(result.buffer_mut()).expect("unique");
        let result_ptr = result_buffer.as_mut_ptr();

        let mut out_indices = vec![0usize; out_shape.len()];
        for _ in 0..out_size {
            // Collect values along axis
            let mut values: Vec<(f64, usize)> = Vec::with_capacity(axis_len);
            for k in 0..axis_len {
                let mut in_indices: Vec<usize> = out_indices[..axis.min(out_indices.len())].to_vec();
                in_indices.push(k);
                if axis < self.ndim() - 1 {
                    in_indices.extend_from_slice(&out_indices[axis..]);
                }
                values.push((self.get_element(&in_indices), k));
            }

            // Sort by value
            values.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

            // Write sorted values back
            for (k, (val, _)) in values.iter().enumerate() {
                let mut in_indices: Vec<usize> = out_indices[..axis.min(out_indices.len())].to_vec();
                in_indices.push(k);
                if axis < self.ndim() - 1 {
                    in_indices.extend_from_slice(&out_indices[axis..]);
                }
                let byte_offset = result.byte_offset_for(&in_indices);
                let linear_idx = byte_offset as usize / dtype.itemsize();
                unsafe { ops.write_f64(result_ptr, linear_idx, *val); }
            }

            increment_indices(&mut out_indices, &out_shape);
        }
        result
    }

    /// Return indices that would sort the array along an axis.
    pub fn argsort(&self, axis: Option<usize>) -> RumpyArray {
        // If axis is None, flatten and argsort
        let axis = match axis {
            None => {
                let flat = self.reshape(vec![self.size()]).expect("flatten should work");
                return flat.argsort(Some(0));
            }
            Some(ax) => ax,
        };
        assert!(axis < self.ndim(), "axis out of bounds");

        let shape = self.shape();
        let axis_len = shape[axis];

        let mut result = RumpyArray::zeros(shape.to_vec(), DType::int64());
        if axis_len <= 1 {
            // For single element, index is 0
            return result;
        }

        let mut out_shape: Vec<usize> = shape[..axis].to_vec();
        out_shape.extend_from_slice(&shape[axis + 1..]);
        if out_shape.is_empty() {
            out_shape = vec![1];
        }
        let out_size: usize = out_shape.iter().product();

        let result_buffer = Arc::get_mut(result.buffer_mut()).expect("unique");
        let result_ptr = result_buffer.as_mut_ptr() as *mut i64;

        let mut out_indices = vec![0usize; out_shape.len()];
        for _ in 0..out_size {
            // Collect values along axis with their indices
            let mut values: Vec<(f64, usize)> = Vec::with_capacity(axis_len);
            for k in 0..axis_len {
                let mut in_indices: Vec<usize> = out_indices[..axis.min(out_indices.len())].to_vec();
                in_indices.push(k);
                if axis < self.ndim() - 1 {
                    in_indices.extend_from_slice(&out_indices[axis..]);
                }
                values.push((self.get_element(&in_indices), k));
            }

            // Sort by value, keeping track of original indices
            values.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

            // Write sorted indices back
            for (k, (_, orig_idx)) in values.iter().enumerate() {
                let mut in_indices: Vec<usize> = out_indices[..axis.min(out_indices.len())].to_vec();
                in_indices.push(k);
                if axis < self.ndim() - 1 {
                    in_indices.extend_from_slice(&out_indices[axis..]);
                }
                let byte_offset = result.byte_offset_for(&in_indices);
                let linear_idx = byte_offset as usize / 8; // i64 is 8 bytes
                unsafe { *result_ptr.add(linear_idx) = *orig_idx as i64; }
            }

            increment_indices(&mut out_indices, &out_shape);
        }
        result
    }

    /// Count number of non-zero elements.
    pub fn count_nonzero(&self) -> usize {
        let size = self.size();
        if size == 0 {
            return 0;
        }
        let ptr = self.data_ptr();
        let dtype = self.dtype();
        let ops = dtype.ops();
        let mut count = 0;
        for offset in self.iter_offsets() {
            if let Some(val) = unsafe { ops.read_f64(ptr, offset) } {
                if val != 0.0 {
                    count += 1;
                }
            }
        }
        count
    }

    /// Calculate the n-th discrete difference along the given axis.
    pub fn diff(&self, n: usize, axis: usize) -> RumpyArray {
        assert!(axis < self.ndim(), "axis out of bounds");

        if n == 0 {
            return self.copy();
        }

        let axis_len = self.shape()[axis];
        if axis_len <= n {
            // Result has size 0 along axis
            let mut new_shape = self.shape().to_vec();
            new_shape[axis] = 0;
            return RumpyArray::zeros(new_shape, self.dtype().clone());
        }

        // Single diff: result[i] = input[i+1] - input[i]
        let mut new_shape = self.shape().to_vec();
        new_shape[axis] = axis_len - 1;

        let dtype = self.dtype().clone();
        let mut result = RumpyArray::zeros(new_shape.clone(), dtype.clone());

        let out_size: usize = new_shape.iter().product();
        if out_size == 0 {
            return result;
        }

        let result_buffer = Arc::get_mut(result.buffer_mut()).expect("unique");
        let result_ptr = result_buffer.as_mut_ptr();
        let src_ptr = self.data_ptr();
        let axis_stride = self.strides()[axis];

        // Fast path for 1D contiguous case
        if self.ndim() == 1 && self.is_c_contiguous() {
            diff_1d_contiguous(src_ptr, result_ptr, out_size, &dtype);
        } else {
            // General strided case
            diff_strided(self, &result, axis_stride, src_ptr, result_ptr, &dtype);
        }

        // Apply recursively for n > 1
        if n > 1 {
            result.diff(n - 1, axis)
        } else {
            result
        }
    }

    /// Test if all elements evaluate to True.
    pub fn all(&self) -> bool {
        let size = self.size();
        if size == 0 {
            return true; // numpy convention: empty array is all True
        }
        let ptr = self.data_ptr();
        let dtype = self.dtype();
        let ops = dtype.ops();
        for offset in self.iter_offsets() {
            if let Some(val) = unsafe { ops.read_f64(ptr, offset) } {
                if val == 0.0 {
                    return false;
                }
            }
        }
        true
    }

    /// Test if all elements along axis evaluate to True.
    pub fn all_axis(&self, axis: usize) -> RumpyArray {
        let mut out_shape: Vec<usize> = self.shape().to_vec();
        let axis_len = out_shape.remove(axis);
        if out_shape.is_empty() {
            out_shape = vec![1];
        }

        let mut result = RumpyArray::zeros(out_shape.clone(), DType::bool());
        let out_size = result.size();
        if out_size == 0 || axis_len == 0 {
            return result;
        }

        let bool_dtype = DType::bool();
        let buffer = result.buffer_mut();
        let result_buffer = Arc::get_mut(buffer).expect("buffer must be unique");
        let result_ptr = result_buffer.as_mut_ptr();
        let ops = bool_dtype.ops();

        let mut out_indices = vec![0usize; out_shape.len()];
        for i in 0..out_size {
            let mut in_indices: Vec<usize> = out_indices[..axis.min(out_indices.len())].to_vec();
            in_indices.push(0);
            if axis < self.ndim() - 1 {
                in_indices.extend_from_slice(&out_indices[axis..]);
            }

            let mut result_val = true;
            for j in 0..axis_len {
                in_indices[axis] = j;
                if self.get_element(&in_indices) == 0.0 {
                    result_val = false;
                    break;
                }
            }

            unsafe { ops.write_f64(result_ptr, i, if result_val { 1.0 } else { 0.0 }); }
            increment_indices(&mut out_indices, &out_shape);
        }

        result
    }

    /// Test if any element evaluates to True.
    pub fn any(&self) -> bool {
        let size = self.size();
        if size == 0 {
            return false; // numpy convention: empty array is all False
        }
        let ptr = self.data_ptr();
        let dtype = self.dtype();
        let ops = dtype.ops();
        for offset in self.iter_offsets() {
            if let Some(val) = unsafe { ops.read_f64(ptr, offset) } {
                if val != 0.0 {
                    return true;
                }
            }
        }
        false
    }

    /// Test if any element along axis evaluates to True.
    pub fn any_axis(&self, axis: usize) -> RumpyArray {
        let mut out_shape: Vec<usize> = self.shape().to_vec();
        let axis_len = out_shape.remove(axis);
        if out_shape.is_empty() {
            out_shape = vec![1];
        }

        let mut result = RumpyArray::zeros(out_shape.clone(), DType::bool());
        let out_size = result.size();
        if out_size == 0 || axis_len == 0 {
            return result;
        }

        let bool_dtype = DType::bool();
        let buffer = result.buffer_mut();
        let result_buffer = Arc::get_mut(buffer).expect("buffer must be unique");
        let result_ptr = result_buffer.as_mut_ptr();
        let ops = bool_dtype.ops();

        let mut out_indices = vec![0usize; out_shape.len()];
        for i in 0..out_size {
            let mut in_indices: Vec<usize> = out_indices[..axis.min(out_indices.len())].to_vec();
            in_indices.push(0);
            if axis < self.ndim() - 1 {
                in_indices.extend_from_slice(&out_indices[axis..]);
            }

            let mut result_val = false;
            for j in 0..axis_len {
                in_indices[axis] = j;
                if self.get_element(&in_indices) != 0.0 {
                    result_val = true;
                    break;
                }
            }

            unsafe { ops.write_f64(result_ptr, i, if result_val { 1.0 } else { 0.0 }); }
            increment_indices(&mut out_indices, &out_shape);
        }

        result
    }

    /// Clip values to a range.
    pub fn clip(&self, a_min: Option<f64>, a_max: Option<f64>) -> RumpyArray {
        let dtype = self.dtype();
        let mut result = RumpyArray::zeros(self.shape().to_vec(), dtype.clone());
        let size = result.size();
        if size == 0 {
            return result;
        }

        let buffer = result.buffer_mut();
        let result_buffer = Arc::get_mut(buffer).expect("buffer must be unique");
        let result_ptr = result_buffer.as_mut_ptr();
        let ops = dtype.ops();
        let src_ptr = self.data_ptr();

        for (i, offset) in self.iter_offsets().enumerate() {
            let mut val = unsafe { ops.read_f64(src_ptr, offset) }.unwrap_or(0.0);
            if let Some(min) = a_min {
                if val < min {
                    val = min;
                }
            }
            if let Some(max) = a_max {
                if val > max {
                    val = max;
                }
            }
            unsafe { ops.write_f64(result_ptr, i, val); }
        }

        result
    }

    /// Round to the given number of decimals.
    pub fn round(&self, decimals: i32) -> RumpyArray {
        let dtype = self.dtype();
        let mut result = RumpyArray::zeros(self.shape().to_vec(), dtype.clone());
        let size = result.size();
        if size == 0 {
            return result;
        }

        let scale = 10.0_f64.powi(decimals);
        let buffer = result.buffer_mut();
        let result_buffer = Arc::get_mut(buffer).expect("buffer must be unique");
        let result_ptr = result_buffer.as_mut_ptr();
        let ops = dtype.ops();
        let src_ptr = self.data_ptr();

        for (i, offset) in self.iter_offsets().enumerate() {
            let val = unsafe { ops.read_f64(src_ptr, offset) }.unwrap_or(0.0);
            let rounded = (val * scale).round() / scale;
            unsafe { ops.write_f64(result_ptr, i, rounded); }
        }

        result
    }

    /// Generic cumulative operation along axis (or flattened if axis is None).
    fn cumulative_op<F>(&self, axis: Option<usize>, identity: f64, op: F) -> RumpyArray
    where
        F: Fn(f64, f64) -> f64,
    {
        match axis {
            None => {
                let size = self.size();
                let mut result = RumpyArray::zeros(vec![size], self.dtype());
                if size == 0 {
                    return result;
                }

                let dtype = self.dtype();
                let buffer = result.buffer_mut();
                let result_buffer = Arc::get_mut(buffer).expect("buffer must be unique");
                let result_ptr = result_buffer.as_mut_ptr();
                let ops = dtype.ops();
                let src_ptr = self.data_ptr();

                let mut acc = identity;
                for (i, offset) in self.iter_offsets().enumerate() {
                    let val = unsafe { ops.read_f64(src_ptr, offset) }.unwrap_or(0.0);
                    acc = op(acc, val);
                    unsafe { ops.write_f64(result_ptr, i, acc); }
                }
                result
            }
            Some(axis) => {
                let shape = self.shape().to_vec();
                let dtype = self.dtype();
                let mut result = RumpyArray::zeros(shape.clone(), dtype.clone());
                let size = result.size();
                if size == 0 {
                    return result;
                }

                let buffer = result.buffer_mut();
                let result_buffer = Arc::get_mut(buffer).expect("buffer must be unique");
                let result_ptr = result_buffer.as_mut_ptr();
                let ops = dtype.ops();

                let axis_len = shape[axis];
                let outer_size = size / axis_len;

                let mut outer_shape: Vec<usize> = shape[..axis].to_vec();
                outer_shape.extend_from_slice(&shape[axis + 1..]);
                if outer_shape.is_empty() {
                    outer_shape = vec![1];
                }

                let mut outer_indices = vec![0usize; outer_shape.len()];
                for _ in 0..outer_size {
                    let mut in_indices: Vec<usize> = outer_indices[..axis.min(outer_indices.len())].to_vec();
                    in_indices.push(0);
                    if axis < self.ndim() - 1 && outer_indices.len() > axis {
                        in_indices.extend_from_slice(&outer_indices[axis..]);
                    } else if axis < self.ndim() - 1 {
                        in_indices.extend_from_slice(&outer_indices[..]);
                    }

                    let mut acc = identity;
                    for j in 0..axis_len {
                        in_indices[axis] = j;
                        acc = op(acc, self.get_element(&in_indices));
                        let flat_idx = self.flat_index_for(&in_indices);
                        unsafe { ops.write_f64(result_ptr, flat_idx, acc); }
                    }
                    increment_indices(&mut outer_indices, &outer_shape);
                }
                result
            }
        }
    }

    /// Cumulative sum along axis (or flattened if axis is None).
    pub fn cumsum(&self, axis: Option<usize>) -> RumpyArray {
        self.cumulative_op(axis, 0.0, |acc, x| acc + x)
    }

    /// Cumulative product along axis (or flattened if axis is None).
    pub fn cumprod(&self, axis: Option<usize>) -> RumpyArray {
        self.cumulative_op(axis, 1.0, |acc, x| acc * x)
    }

    /// Calculate flat index for given n-dimensional indices (C-order).
    fn flat_index_for(&self, indices: &[usize]) -> usize {
        let shape = self.shape();
        let mut flat = 0;
        let mut stride = 1;
        for i in (0..indices.len()).rev() {
            flat += indices[i] * stride;
            stride *= shape[i];
        }
        flat
    }

    // ========================================================================
    // NaN-aware reductions
    // ========================================================================
    //
    // TODO(perf): Axis reductions use get_element() which computes offsets per-element.
    // For better performance, refactor to use:
    // - axis_offsets() to iterate outer dimensions
    // - Registry-registered ReduceLoopFn for inner loops with contiguous fast path
    // See designs/iteration-performance.md and designs/backstride-iteration.md

    /// Helper: iterate non-NaN values, returning (value, was_found) for full reductions.
    fn nan_reduce_full<F>(&self, init: f64, mut f: F) -> (f64, bool)
    where
        F: FnMut(f64, f64) -> f64,
    {
        if self.size() == 0 {
            return (init, false);
        }
        let ptr = self.data_ptr();
        let dtype = self.dtype();
        let ops = dtype.ops();
        let mut acc = init;
        let mut found = false;
        for offset in self.iter_offsets() {
            let val = unsafe { ops.read_f64(ptr, offset) }.unwrap_or(0.0);
            if !val.is_nan() {
                acc = f(acc, val);
                found = true;
            }
        }
        (acc, found)
    }

    /// Helper: axis reduction ignoring NaN values.
    fn nan_reduce_axis<F>(&self, axis: usize, result_dtype: DType, mut reduce_fn: F) -> RumpyArray
    where
        F: FnMut(&mut [usize], usize, usize) -> f64,
    {
        let mut out_shape: Vec<usize> = self.shape().to_vec();
        let axis_len = out_shape.remove(axis);
        if out_shape.is_empty() {
            out_shape = vec![1];
        }

        let mut result = RumpyArray::zeros(out_shape.clone(), result_dtype.clone());
        let out_size = result.size();
        if out_size == 0 || axis_len == 0 {
            return result;
        }

        let buffer = result.buffer_mut();
        let result_buffer = Arc::get_mut(buffer).expect("buffer must be unique");
        let result_ptr = result_buffer.as_mut_ptr();
        let ops = result_dtype.ops();

        let mut out_indices = vec![0usize; out_shape.len()];
        for i in 0..out_size {
            // Build input indices with placeholder for axis
            let mut in_indices: Vec<usize> = out_indices[..axis.min(out_indices.len())].to_vec();
            in_indices.push(0);
            if axis < self.ndim() - 1 {
                in_indices.extend_from_slice(&out_indices[axis..]);
            }

            let val = reduce_fn(&mut in_indices, axis, axis_len);
            unsafe { ops.write_f64(result_ptr, i, val); }
            increment_indices(&mut out_indices, &out_shape);
        }
        result
    }

    /// Sum all elements, ignoring NaN values.
    pub fn nansum(&self) -> f64 {
        self.nan_reduce_full(0.0, |acc, v| acc + v).0
    }

    /// Sum along axis, ignoring NaN values.
    pub fn nansum_axis(&self, axis: usize) -> RumpyArray {
        self.nan_reduce_axis(axis, self.dtype(), |indices, ax, len| {
            let mut sum = 0.0;
            for j in 0..len {
                indices[ax] = j;
                let val = self.get_element(indices);
                if !val.is_nan() {
                    sum += val;
                }
            }
            sum
        })
    }

    /// Product of all elements, ignoring NaN values.
    pub fn nanprod(&self) -> f64 {
        self.nan_reduce_full(1.0, |acc, v| acc * v).0
    }

    /// Product along axis, ignoring NaN values.
    pub fn nanprod_axis(&self, axis: usize) -> RumpyArray {
        self.nan_reduce_axis(axis, self.dtype(), |indices, ax, len| {
            let mut prod = 1.0;
            for j in 0..len {
                indices[ax] = j;
                let val = self.get_element(indices);
                if !val.is_nan() {
                    prod *= val;
                }
            }
            prod
        })
    }

    /// Mean of all elements, ignoring NaN values.
    pub fn nanmean(&self) -> f64 {
        if self.size() == 0 {
            return f64::NAN;
        }
        let ptr = self.data_ptr();
        let dtype = self.dtype();
        let ops = dtype.ops();
        let mut sum = 0.0;
        let mut count = 0usize;
        for offset in self.iter_offsets() {
            let val = unsafe { ops.read_f64(ptr, offset) }.unwrap_or(0.0);
            if !val.is_nan() {
                sum += val;
                count += 1;
            }
        }
        if count == 0 { f64::NAN } else { sum / count as f64 }
    }

    /// Mean along axis, ignoring NaN values.
    pub fn nanmean_axis(&self, axis: usize) -> RumpyArray {
        self.nan_reduce_axis(axis, DType::float64(), |indices, ax, len| {
            let mut sum = 0.0;
            let mut count = 0usize;
            for j in 0..len {
                indices[ax] = j;
                let val = self.get_element(indices);
                if !val.is_nan() {
                    sum += val;
                    count += 1;
                }
            }
            if count == 0 { f64::NAN } else { sum / count as f64 }
        })
    }

    /// Variance of all elements, ignoring NaN values.
    pub fn nanvar(&self) -> f64 {
        let mean = self.nanmean();
        if mean.is_nan() {
            return f64::NAN;
        }
        let ptr = self.data_ptr();
        let dtype = self.dtype();
        let ops = dtype.ops();
        let mut sum_sq = 0.0;
        let mut count = 0usize;
        for offset in self.iter_offsets() {
            let val = unsafe { ops.read_f64(ptr, offset) }.unwrap_or(0.0);
            if !val.is_nan() {
                let diff = val - mean;
                sum_sq += diff * diff;
                count += 1;
            }
        }
        sum_sq / count as f64
    }

    /// Variance along axis, ignoring NaN values.
    pub fn nanvar_axis(&self, axis: usize) -> RumpyArray {
        self.nan_reduce_axis(axis, DType::float64(), |indices, ax, len| {
            // Two-pass: mean then variance
            let mut sum = 0.0;
            let mut count = 0usize;
            for j in 0..len {
                indices[ax] = j;
                let val = self.get_element(indices);
                if !val.is_nan() {
                    sum += val;
                    count += 1;
                }
            }
            if count == 0 {
                return f64::NAN;
            }
            let mean = sum / count as f64;
            let mut sum_sq = 0.0;
            for j in 0..len {
                indices[ax] = j;
                let val = self.get_element(indices);
                if !val.is_nan() {
                    let diff = val - mean;
                    sum_sq += diff * diff;
                }
            }
            sum_sq / count as f64
        })
    }

    /// Standard deviation of all elements, ignoring NaN values.
    pub fn nanstd(&self) -> f64 {
        self.nanvar().sqrt()
    }

    /// Standard deviation along axis, ignoring NaN values.
    pub fn nanstd_axis(&self, axis: usize) -> RumpyArray {
        map_unary_op(&self.nanvar_axis(axis), UnaryOp::Sqrt)
            .expect("sqrt always succeeds on numeric types")
    }

    /// Minimum of all elements, ignoring NaN values.
    pub fn nanmin(&self) -> f64 {
        let (val, found) = self.nan_reduce_full(f64::INFINITY, |acc, v| acc.min(v));
        if found { val } else { f64::NAN }
    }

    /// Minimum along axis, ignoring NaN values.
    pub fn nanmin_axis(&self, axis: usize) -> RumpyArray {
        self.nan_reduce_axis(axis, self.dtype(), |indices, ax, len| {
            let mut min_val = f64::INFINITY;
            let mut found = false;
            for j in 0..len {
                indices[ax] = j;
                let val = self.get_element(indices);
                if !val.is_nan() {
                    min_val = min_val.min(val);
                    found = true;
                }
            }
            if found { min_val } else { f64::NAN }
        })
    }

    /// Maximum of all elements, ignoring NaN values.
    pub fn nanmax(&self) -> f64 {
        let (val, found) = self.nan_reduce_full(f64::NEG_INFINITY, |acc, v| acc.max(v));
        if found { val } else { f64::NAN }
    }

    /// Maximum along axis, ignoring NaN values.
    pub fn nanmax_axis(&self, axis: usize) -> RumpyArray {
        self.nan_reduce_axis(axis, self.dtype(), |indices, ax, len| {
            let mut max_val = f64::NEG_INFINITY;
            let mut found = false;
            for j in 0..len {
                indices[ax] = j;
                let val = self.get_element(indices);
                if !val.is_nan() {
                    max_val = max_val.max(val);
                    found = true;
                }
            }
            if found { max_val } else { f64::NAN }
        })
    }

    /// Index of minimum element, ignoring NaN values.
    /// Returns None if all elements are NaN.
    pub fn nanargmin(&self) -> Option<usize> {
        if self.size() == 0 {
            return None;
        }
        let ptr = self.data_ptr();
        let dtype = self.dtype();
        let ops = dtype.ops();
        let mut min_val = f64::INFINITY;
        let mut min_idx: Option<usize> = None;
        for (i, offset) in self.iter_offsets().enumerate() {
            let val = unsafe { ops.read_f64(ptr, offset) }.unwrap_or(0.0);
            if !val.is_nan() && val < min_val {
                min_val = val;
                min_idx = Some(i);
            }
        }
        min_idx
    }

    /// Index of minimum element along axis, ignoring NaN values.
    pub fn nanargmin_axis(&self, axis: usize) -> RumpyArray {
        self.nan_arg_reduce_axis(axis, |a, b| a < b)
    }

    /// Index of maximum element, ignoring NaN values.
    /// Returns None if all elements are NaN.
    pub fn nanargmax(&self) -> Option<usize> {
        if self.size() == 0 {
            return None;
        }
        let ptr = self.data_ptr();
        let dtype = self.dtype();
        let ops = dtype.ops();
        let mut max_val = f64::NEG_INFINITY;
        let mut max_idx: Option<usize> = None;
        for (i, offset) in self.iter_offsets().enumerate() {
            let val = unsafe { ops.read_f64(ptr, offset) }.unwrap_or(0.0);
            if !val.is_nan() && val > max_val {
                max_val = val;
                max_idx = Some(i);
            }
        }
        max_idx
    }

    /// Index of maximum element along axis, ignoring NaN values.
    pub fn nanargmax_axis(&self, axis: usize) -> RumpyArray {
        self.nan_arg_reduce_axis(axis, |a, b| a > b)
    }

    /// Helper for nanargmax_axis/nanargmin_axis.
    fn nan_arg_reduce_axis<F>(&self, axis: usize, is_better: F) -> RumpyArray
    where
        F: Fn(f64, f64) -> bool,
    {
        self.nan_reduce_axis(axis, DType::int64(), |indices, ax, len| {
            let mut best_val: Option<f64> = None;
            let mut best_idx: i64 = 0;
            for j in 0..len {
                indices[ax] = j;
                let val = self.get_element(indices);
                if !val.is_nan() {
                    match best_val {
                        None => {
                            best_val = Some(val);
                            best_idx = j as i64;
                        }
                        Some(bv) if is_better(val, bv) => {
                            best_val = Some(val);
                            best_idx = j as i64;
                        }
                        _ => {}
                    }
                }
            }
            best_idx as f64
        })
    }

    /// Convert array to nested Python lists.
    pub fn to_pylist(&self, py: pyo3::Python<'_>) -> pyo3::PyResult<pyo3::PyObject> {
        use pyo3::types::PyList;

        fn build_list(
            arr: &RumpyArray,
            py: pyo3::Python<'_>,
            depth: usize,
            indices: &mut Vec<usize>,
        ) -> pyo3::PyResult<pyo3::PyObject> {
            use pyo3::IntoPyObject;
            if depth == arr.ndim() {
                // Base case: return scalar
                let val = arr.get_element(indices);
                return Ok(val.into_pyobject(py)?.into_any().unbind());
            }

            // Build list for this dimension
            let dim_size = arr.shape()[depth];
            let mut items = Vec::with_capacity(dim_size);
            for i in 0..dim_size {
                indices[depth] = i;
                items.push(build_list(arr, py, depth + 1, indices)?);
            }
            let list = PyList::new(py, items)?;
            Ok(list.into_pyobject(py)?.into_any().unbind())
        }

        let mut indices = vec![0usize; self.ndim()];
        build_list(self, py, 0, &mut indices)
    }

    /// Median of all elements (flattened).
    /// Uses select_nth_unstable for O(n) instead of O(n log n) sort.
    pub fn median(&self) -> f64 {
        let size = self.size();
        if size == 0 {
            return f64::NAN;
        }
        let mut values = self.to_vec();
        let mid = size / 2;

        // select_nth_unstable partitions so values[mid] is the correct element
        values.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        if size % 2 == 0 {
            // For even length, need max of left partition (which is unsorted)
            let left_max = values[..mid].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            (left_max + values[mid]) / 2.0
        } else {
            values[mid]
        }
    }

    /// Median along axis.
    pub fn median_axis(&self, axis: usize) -> RumpyArray {
        let shape = self.shape();
        let ndim = self.ndim();
        let axis_len = shape[axis];

        // Output shape: remove the axis dimension
        let mut out_shape: Vec<usize> = shape[..axis].to_vec();
        out_shape.extend_from_slice(&shape[axis + 1..]);
        if out_shape.is_empty() {
            out_shape = vec![1];
        }

        let out_size: usize = out_shape.iter().product();
        let mut result = RumpyArray::zeros(out_shape.clone(), DType::float64());

        if out_size == 0 || axis_len == 0 {
            return result;
        }

        let buffer = result.buffer_mut();
        let result_buffer = Arc::get_mut(buffer).expect("buffer must be unique");
        let result_ptr = result_buffer.as_mut_ptr() as *mut f64;

        let mut outer_indices = vec![0usize; out_shape.len()];
        for out_i in 0..out_size {
            // Collect values along axis
            let mut values = Vec::with_capacity(axis_len);
            for k in 0..axis_len {
                // Build input indices: insert k at position `axis`
                let mut in_indices = Vec::with_capacity(ndim);
                in_indices.extend_from_slice(&outer_indices[..axis]);
                in_indices.push(k);
                in_indices.extend_from_slice(&outer_indices[axis..]);
                values.push(self.get_element(&in_indices));
            }

            // Sort and compute median
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let median = if axis_len.is_multiple_of(2) {
                (values[axis_len / 2 - 1] + values[axis_len / 2]) / 2.0
            } else {
                values[axis_len / 2]
            };

            unsafe { *result_ptr.add(out_i) = median; }
            increment_indices(&mut outer_indices, &out_shape);
        }
        result
    }

    /// Peak-to-peak (max - min) of all elements.
    pub fn ptp(&self) -> f64 {
        if self.size() == 0 {
            return f64::NAN;
        }
        self.max() - self.min()
    }

    /// Peak-to-peak along axis.
    pub fn ptp_axis(&self, axis: usize) -> RumpyArray {
        let max_arr = self.max_axis(axis);
        let min_arr = self.min_axis(axis);
        max_arr.binary_op(&min_arr, BinaryOp::Sub).expect("shapes match")
    }

    /// Weighted average of all elements.
    /// If weights is None, computes simple mean.
    pub fn average(&self, weights: Option<&RumpyArray>) -> f64 {
        match weights {
            None => self.mean(),
            Some(w) => {
                // weighted_sum / sum_of_weights
                let product = self.binary_op(w, BinaryOp::Mul).expect("broadcast works");
                let weighted_sum = product.sum();
                let total_weight = w.sum();
                if total_weight == 0.0 {
                    f64::NAN
                } else {
                    weighted_sum / total_weight
                }
            }
        }
    }

    /// Weighted average along axis.
    pub fn average_axis(&self, axis: usize, weights: Option<&RumpyArray>) -> RumpyArray {
        match weights {
            None => self.mean_axis(axis),
            Some(w) => {
                // Broadcast weights and compute weighted mean
                let product = self.binary_op(w, BinaryOp::Mul).expect("broadcast works");
                let weighted_sum = product.sum_axis(axis);
                let total_weight = w.sum_axis(axis);
                weighted_sum.binary_op(&total_weight, BinaryOp::Div).expect("shapes match")
            }
        }
    }
}

/// Compute histogram of 1D array.
/// Returns (counts, bin_edges) tuple.
pub fn histogram(arr: &RumpyArray, bins: usize, range: Option<(f64, f64)>) -> (RumpyArray, RumpyArray) {
    let values = arr.to_vec();

    let (min_val, max_val) = range.unwrap_or_else(|| {
        if values.is_empty() {
            (0.0, 1.0)
        } else {
            let min_v = values.iter().cloned().fold(f64::INFINITY, f64::min);
            let max_v = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            (min_v, max_v)
        }
    });

    // Create bin edges
    let mut bin_edges = Vec::with_capacity(bins + 1);
    let bin_width = (max_val - min_val) / bins as f64;
    for i in 0..=bins {
        bin_edges.push(min_val + i as f64 * bin_width);
    }

    // Count values in each bin
    let mut counts = vec![0i64; bins];
    for &val in &values {
        if val >= min_val && val <= max_val {
            let mut bin_idx = ((val - min_val) / bin_width) as usize;
            // Handle edge case: value == max_val goes in last bin
            if bin_idx >= bins {
                bin_idx = bins - 1;
            }
            counts[bin_idx] += 1;
        }
    }

    let counts_data: Vec<f64> = counts.iter().map(|&c| c as f64).collect();
    let counts_arr = RumpyArray::from_vec(counts_data, DType::int64());
    let edges_arr = RumpyArray::from_vec(bin_edges, DType::float64());

    (counts_arr, edges_arr)
}

/// Compute covariance matrix.
/// x should be 2D with each row representing a variable and each column an observation.
/// Returns a 2D array where element [i,j] is cov(row_i, row_j).
pub fn cov(arr: &RumpyArray, ddof: usize) -> RumpyArray {
    // For 1D input, compute variance
    if arr.ndim() == 1 {
        let n = arr.size();
        if n <= ddof {
            return RumpyArray::from_vec(vec![f64::NAN], DType::float64());
        }
        let mean = arr.mean();
        // Use to_vec for fast access, then compute variance
        let values = arr.to_vec();
        let sum_sq: f64 = values.iter().map(|&v| (v - mean) * (v - mean)).sum();
        let var = sum_sq / (n - ddof) as f64;
        return RumpyArray::from_vec(vec![var], DType::float64());
    }

    // 2D case: each row is a variable
    let shape = arr.shape();
    let nvars = shape[0];  // number of variables (rows)
    let nobs = shape[1];   // number of observations (columns)

    if nobs <= ddof {
        // Return NaN matrix
        return RumpyArray::full(vec![nvars, nvars], f64::NAN, DType::float64());
    }

    // Compute means for each variable and center the data
    let means = arr.mean_axis(1);
    let means_broadcasted = means.reshape(vec![nvars, 1]).unwrap();
    let centered = arr.binary_op(&means_broadcasted, BinaryOp::Sub).expect("broadcast works");

    // Compute covariance: centered @ centered.T / (n - ddof)
    let centered_t = centered.transpose();
    let cov_unnorm = matmul::matmul(&centered, &centered_t).expect("matmul works");

    // Divide by (n - ddof)
    let divisor = RumpyArray::full(vec![1], (nobs - ddof) as f64, DType::float64());
    cov_unnorm.binary_op(&divisor, BinaryOp::Div).expect("broadcast works")
}

/// Compute Pearson correlation coefficient matrix.
/// Input should be 2D with each row representing a variable.
pub fn corrcoef(arr: &RumpyArray) -> RumpyArray {
    // For 1D input, return 1x1 matrix with 1.0 (perfect correlation with itself)
    if arr.ndim() == 1 {
        return RumpyArray::from_vec(vec![1.0], DType::float64()).reshape(vec![1, 1]).unwrap();
    }

    // Compute covariance matrix first
    let cov_matrix = cov(arr, 1);
    let shape = cov_matrix.shape();
    let nvars = shape[0];

    // Extract diagonal (standard deviations)
    let diag = cov_matrix.diagonal();

    // Compute correlation: corr[i,j] = cov[i,j] / (std[i] * std[j])
    let mut result = RumpyArray::zeros(vec![nvars, nvars], DType::float64());
    let buffer = result.buffer_mut();
    let result_buffer = Arc::get_mut(buffer).expect("buffer must be unique");
    let result_ptr = result_buffer.as_mut_ptr() as *mut f64;

    for i in 0..nvars {
        let std_i = diag.get_element(&[i]).sqrt();
        for j in 0..nvars {
            let std_j = diag.get_element(&[j]).sqrt();
            let cov_ij = cov_matrix.get_element(&[i, j]);
            let corr = if std_i == 0.0 || std_j == 0.0 {
                if i == j { 1.0 } else { f64::NAN }
            } else {
                cov_ij / (std_i * std_j)
            };
            unsafe { *result_ptr.add(i * nvars + j) = corr; }
        }
    }

    result
}

/// Conditional selection: where(condition, x, y).
/// Returns elements from x where condition is true, else from y.
/// All three arrays are broadcast together.
pub fn where_select(condition: &RumpyArray, x: &RumpyArray, y: &RumpyArray) -> Option<RumpyArray> {
    // Broadcast all three shapes together
    let shape_cx = broadcast_shapes(condition.shape(), x.shape())?;
    let out_shape = broadcast_shapes(&shape_cx, y.shape())?;

    let cond = condition.broadcast_to(&out_shape)?;
    let x_bc = x.broadcast_to(&out_shape)?;
    let y_bc = y.broadcast_to(&out_shape)?;

    // Result dtype is promoted from x and y
    let result_dtype = promote_dtype(&x_bc.dtype(), &y_bc.dtype());
    let mut result = RumpyArray::zeros(out_shape.clone(), result_dtype);
    let size = result.size();

    if size == 0 {
        return Some(result);
    }

    let buffer = result.buffer_mut();
    let result_buffer = Arc::get_mut(buffer).expect("buffer must be unique");
    let result_ptr = result_buffer.as_mut_ptr();
    let result_dtype_ref = result.dtype();
    let result_ops = result_dtype_ref.ops();

    let cond_ptr = cond.data_ptr();
    let cond_dtype = cond.dtype();
    let cond_ops = cond_dtype.ops();
    let x_ptr = x_bc.data_ptr();
    let y_ptr = y_bc.data_ptr();

    // Check if all dtypes match for direct copy
    let same_dtype = x_bc.dtype() == y_bc.dtype() && x_bc.dtype() == result.dtype();

    let mut indices = vec![0usize; out_shape.len()];

    if same_dtype {
        for i in 0..size {
            let cond_offset = cond.byte_offset_for(&indices);
            let is_true = unsafe { cond_ops.is_truthy(cond_ptr, cond_offset) };
            if is_true {
                let x_offset = x_bc.byte_offset_for(&indices);
                unsafe { result_ops.copy_element(x_ptr, x_offset, result_ptr, i); }
            } else {
                let y_offset = y_bc.byte_offset_for(&indices);
                unsafe { result_ops.copy_element(y_ptr, y_offset, result_ptr, i); }
            }
            increment_indices(&mut indices, &out_shape);
        }
    } else {
        // Different dtypes: read as f64, write as f64
        let x_dtype = x_bc.dtype();
        let x_ops = x_dtype.ops();
        let y_dtype = y_bc.dtype();
        let y_ops = y_dtype.ops();

        for i in 0..size {
            let cond_offset = cond.byte_offset_for(&indices);
            let is_true = unsafe { cond_ops.is_truthy(cond_ptr, cond_offset) };
            let val = if is_true {
                let x_offset = x_bc.byte_offset_for(&indices);
                unsafe { x_ops.read_f64(x_ptr, x_offset).unwrap_or(0.0) }
            } else {
                let y_offset = y_bc.byte_offset_for(&indices);
                unsafe { y_ops.read_f64(y_ptr, y_offset).unwrap_or(0.0) }
            };
            unsafe { result_ops.write_f64(result_ptr, i, val); }
            increment_indices(&mut indices, &out_shape);
        }
    }

    Some(result)
}

// ============================================================================
// Binary broadcast helper
// ============================================================================

/// Set up broadcasting for a binary operation, returning broadcasted arrays and output.
/// Returns (a_broadcasted, b_broadcasted, result_array, output_shape).
fn broadcast_binary_setup(
    a: &RumpyArray,
    b: &RumpyArray,
    out_dtype: DType,
) -> Option<(RumpyArray, RumpyArray, RumpyArray, Vec<usize>)> {
    let out_shape = broadcast_shapes(a.shape(), b.shape())?;
    let a_bc = a.broadcast_to(&out_shape)?;
    let b_bc = b.broadcast_to(&out_shape)?;
    let result = RumpyArray::zeros(out_shape.clone(), out_dtype);
    Some((a_bc, b_bc, result, out_shape))
}

/// Apply a binary function element-wise, writing bool results.
/// The function receives (a_value, b_value) as f64 and returns bool.
pub fn map_binary_to_bool<F>(a: &RumpyArray, b: &RumpyArray, f: F) -> Option<RumpyArray>
where
    F: Fn(f64, f64) -> bool,
{
    let (a_bc, b_bc, mut result, out_shape) = broadcast_binary_setup(a, b, DType::bool())?;
    let size = result.size();
    if size == 0 {
        return Some(result);
    }

    let buffer = result.buffer_mut();
    let result_buffer = Arc::get_mut(buffer).expect("buffer must be unique");
    let result_ptr = result_buffer.as_mut_ptr();

    let mut indices = vec![0usize; out_shape.len()];
    for i in 0..size {
        let a_val = a_bc.get_element(&indices);
        let b_val = b_bc.get_element(&indices);
        unsafe { *result_ptr.add(i) = if f(a_val, b_val) { 1 } else { 0 }; }
        increment_indices(&mut indices, &out_shape);
    }
    Some(result)
}

// ============================================================================
// Logical operations (element-wise boolean logic)
// ============================================================================

/// Apply a binary operation on truthiness values, returning a bool array.
fn map_binary_logical<F>(a: &RumpyArray, b: &RumpyArray, f: F) -> Option<RumpyArray>
where
    F: Fn(bool, bool) -> bool,
{
    let (a_bc, b_bc, mut result, out_shape) = broadcast_binary_setup(a, b, DType::bool())?;
    let size = result.size();
    if size == 0 {
        return Some(result);
    }

    let buffer = result.buffer_mut();
    let result_buffer = Arc::get_mut(buffer).expect("buffer must be unique");
    let result_ptr = result_buffer.as_mut_ptr();
    let a_ptr = a_bc.data_ptr();
    let b_ptr = b_bc.data_ptr();
    let a_dtype = a_bc.dtype();
    let b_dtype = b_bc.dtype();
    let a_ops = a_dtype.ops();
    let b_ops = b_dtype.ops();

    let mut indices = vec![0usize; out_shape.len()];
    for i in 0..size {
        let a_offset = a_bc.byte_offset_for(&indices);
        let b_offset = b_bc.byte_offset_for(&indices);
        let a_truthy = unsafe { a_ops.is_truthy(a_ptr, a_offset) };
        let b_truthy = unsafe { b_ops.is_truthy(b_ptr, b_offset) };
        unsafe { *result_ptr.add(i) = if f(a_truthy, b_truthy) { 1 } else { 0 }; }
        increment_indices(&mut indices, &out_shape);
    }
    Some(result)
}

/// Apply a unary operation on truthiness values, returning a bool array.
fn map_unary_logical<F>(a: &RumpyArray, f: F) -> RumpyArray
where
    F: Fn(bool) -> bool,
{
    let mut result = RumpyArray::zeros(a.shape().to_vec(), DType::bool());
    let size = result.size();
    if size == 0 {
        return result;
    }

    let buffer = result.buffer_mut();
    let result_buffer = Arc::get_mut(buffer).expect("buffer must be unique");
    let result_ptr = result_buffer.as_mut_ptr();
    let a_ptr = a.data_ptr();
    let a_dtype = a.dtype();
    let a_ops = a_dtype.ops();

    for (i, offset) in a.iter_offsets().enumerate() {
        let a_truthy = unsafe { a_ops.is_truthy(a_ptr, offset) };
        unsafe { *result_ptr.add(i) = if f(a_truthy) { 1 } else { 0 }; }
    }
    result
}

/// Element-wise logical AND.
pub fn logical_and(a: &RumpyArray, b: &RumpyArray) -> Option<RumpyArray> {
    map_binary_logical(a, b, |x, y| x && y)
}

/// Element-wise logical OR.
pub fn logical_or(a: &RumpyArray, b: &RumpyArray) -> Option<RumpyArray> {
    map_binary_logical(a, b, |x, y| x || y)
}

/// Element-wise logical XOR.
pub fn logical_xor(a: &RumpyArray, b: &RumpyArray) -> Option<RumpyArray> {
    map_binary_logical(a, b, |x, y| x != y)
}

/// Element-wise logical NOT.
pub fn logical_not(a: &RumpyArray) -> RumpyArray {
    map_unary_logical(a, |x| !x)
}

// ============================================================================
// Comparison functions (element-wise, return bool array)
// ============================================================================

/// Element-wise equality test.
pub fn equal(a: &RumpyArray, b: &RumpyArray) -> Option<RumpyArray> {
    map_compare_op(a, b, ComparisonOp::Eq)
}

/// Element-wise not-equal test.
pub fn not_equal(a: &RumpyArray, b: &RumpyArray) -> Option<RumpyArray> {
    map_compare_op(a, b, ComparisonOp::Ne)
}

/// Element-wise less-than test.
pub fn less(a: &RumpyArray, b: &RumpyArray) -> Option<RumpyArray> {
    map_compare_op(a, b, ComparisonOp::Lt)
}

/// Element-wise less-than-or-equal test.
pub fn less_equal(a: &RumpyArray, b: &RumpyArray) -> Option<RumpyArray> {
    map_compare_op(a, b, ComparisonOp::Le)
}

/// Element-wise greater-than test.
pub fn greater(a: &RumpyArray, b: &RumpyArray) -> Option<RumpyArray> {
    map_compare_op(a, b, ComparisonOp::Gt)
}

/// Element-wise greater-than-or-equal test.
pub fn greater_equal(a: &RumpyArray, b: &RumpyArray) -> Option<RumpyArray> {
    map_compare_op(a, b, ComparisonOp::Ge)
}

/// Element-wise approximate equality test: |a - b| <= atol + rtol * |b|.
/// Handles NaN and infinity according to NumPy semantics.
pub fn isclose(a: &RumpyArray, b: &RumpyArray, rtol: f64, atol: f64, equal_nan: bool) -> Option<RumpyArray> {
    map_binary_to_bool(a, b, |x, y| {
        if x.is_nan() && y.is_nan() {
            equal_nan
        } else if x.is_nan() || y.is_nan() {
            false
        } else if x.is_infinite() || y.is_infinite() {
            x == y  // Both must be same infinity
        } else {
            (x - y).abs() <= atol + rtol * y.abs()
        }
    })
}

/// Test if all elements are approximately equal.
pub fn allclose(a: &RumpyArray, b: &RumpyArray, rtol: f64, atol: f64, equal_nan: bool) -> Option<bool> {
    let close = isclose(a, b, rtol, atol, equal_nan)?;
    Some(close.all())
}

/// Test if two arrays have the same shape and elements.
pub fn array_equal(a: &RumpyArray, b: &RumpyArray) -> bool {
    if a.shape() != b.shape() {
        return false;
    }
    equal(a, b).map(|r| r.all()).unwrap_or(false)
}

// ============================================================================
// Bitwise operations (dtype-preserving for integers/bool)
// ============================================================================

use crate::array::dtype::BitwiseOp;

/// Apply a binary bitwise operation element-wise, preserving dtype.
fn map_binary_bitwise(a: &RumpyArray, b: &RumpyArray, op: BitwiseOp) -> Option<RumpyArray> {
    let out_dtype = promote_dtype(&a.dtype(), &b.dtype());
    let (a_bc, b_bc, mut result, out_shape) = broadcast_binary_setup(a, b, out_dtype.clone())?;
    let size = result.size();
    if size == 0 {
        return Some(result);
    }

    let buffer = result.buffer_mut();
    let result_buffer = Arc::get_mut(buffer).expect("buffer must be unique");
    let result_ptr = result_buffer.as_mut_ptr();
    let a_ptr = a_bc.data_ptr();
    let b_ptr = b_bc.data_ptr();
    let itemsize = out_dtype.itemsize() as isize;

    // Try registry first for contiguous same-type operations
    let a_kind = a.dtype().kind();
    let b_kind = b.dtype().kind();
    if a_kind == b_kind {
        let reg = registry().read().unwrap();
        if let Some(loop_fn) = reg.lookup_bitwise_binary(op, a_kind.clone()) {
            let a_contig = a_bc.is_c_contiguous() && a.shape() == out_shape.as_slice();
            let b_contig = b_bc.is_c_contiguous() && b.shape() == out_shape.as_slice();

            if a_contig && b_contig {
                // Fast path: both contiguous, same shape as output
                let strides = (itemsize, itemsize, itemsize);
                unsafe { loop_fn(a_ptr, b_ptr, result_ptr, size, strides); }
                return Some(result);
            }
            // Strided but still use registry loop for vectorization
            drop(reg);
            return map_binary_bitwise_strided(a_bc, b_bc, result, out_shape, op, loop_fn, itemsize);
        }
    }

    // Fallback to trait dispatch
    let ops = out_dtype.ops();
    let mut indices = vec![0usize; out_shape.len()];
    for i in 0..size {
        let a_offset = a_bc.byte_offset_for(&indices);
        let b_offset = b_bc.byte_offset_for(&indices);
        let ok = unsafe { ops.bitwise_op(op, a_ptr, a_offset, b_ptr, b_offset, result_ptr, i) };
        if !ok {
            return None;
        }
        increment_indices(&mut indices, &out_shape);
    }
    Some(result)
}

/// Strided bitwise binary using registry loop.
fn map_binary_bitwise_strided(
    a_bc: RumpyArray,
    b_bc: RumpyArray,
    mut result: RumpyArray,
    out_shape: Vec<usize>,
    _op: BitwiseOp,
    loop_fn: registry::BinaryLoopFn,
    itemsize: isize,
) -> Option<RumpyArray> {
    let size = result.size();
    let buffer = result.buffer_mut();
    let result_buffer = Arc::get_mut(buffer).expect("buffer must be unique");
    let result_ptr = result_buffer.as_mut_ptr();

    // Process in contiguous chunks along the last axis when possible
    let ndim = out_shape.len();
    if ndim > 0 {
        let inner_size = out_shape[ndim - 1];
        let a_inner_contig = a_bc.strides().last().copied().unwrap_or(0) == itemsize;
        let b_inner_contig = b_bc.strides().last().copied().unwrap_or(0) == itemsize;

        if a_inner_contig && b_inner_contig && inner_size > 0 {
            let outer_size = size / inner_size;
            let mut indices = vec![0usize; ndim];
            for _ in 0..outer_size {
                let a_offset = a_bc.byte_offset_for(&indices);
                let b_offset = b_bc.byte_offset_for(&indices);
                let out_offset = (0..ndim).fold(0isize, |acc, d| {
                    acc + (indices[d] as isize) * result.strides()[d]
                });
                unsafe {
                    loop_fn(
                        a_bc.data_ptr().offset(a_offset),
                        b_bc.data_ptr().offset(b_offset),
                        result_ptr.offset(out_offset),
                        inner_size,
                        (itemsize, itemsize, itemsize),
                    );
                }
                // Advance outer indices
                for d in (0..ndim - 1).rev() {
                    indices[d] += 1;
                    if indices[d] < out_shape[d] {
                        break;
                    }
                    indices[d] = 0;
                }
                indices[ndim - 1] = 0;
            }
            return Some(result);
        }
    }

    // Fully strided fallback
    let a_ptr = a_bc.data_ptr();
    let b_ptr = b_bc.data_ptr();
    let mut indices = vec![0usize; ndim];
    for i in 0..size {
        let a_offset = a_bc.byte_offset_for(&indices);
        let b_offset = b_bc.byte_offset_for(&indices);
        unsafe {
            loop_fn(
                a_ptr.offset(a_offset),
                b_ptr.offset(b_offset),
                result_ptr.add(i * itemsize as usize),
                1,
                (itemsize, itemsize, itemsize),
            );
        }
        increment_indices(&mut indices, &out_shape);
    }
    Some(result)
}

/// Apply bitwise NOT element-wise, preserving dtype.
fn map_unary_bitwise_not(a: &RumpyArray) -> Option<RumpyArray> {
    let mut result = RumpyArray::zeros(a.shape().to_vec(), a.dtype().clone());
    let size = result.size();
    if size == 0 {
        return Some(result);
    }

    let out_dtype = result.dtype().clone();
    let buffer = result.buffer_mut();
    let result_buffer = Arc::get_mut(buffer).expect("buffer must be unique");
    let result_ptr = result_buffer.as_mut_ptr();
    let a_ptr = a.data_ptr();
    let itemsize = out_dtype.itemsize() as isize;

    // Try registry first for contiguous operations
    let a_kind = a.dtype().kind();
    let reg = registry().read().unwrap();
    if let Some(loop_fn) = reg.lookup_bitwise_not(a_kind.clone()) {
        if a.is_c_contiguous() {
            // Fast path: contiguous
            let strides = (itemsize, itemsize);
            unsafe { loop_fn(a_ptr, result_ptr, size, strides); }
            return Some(result);
        }
        // Strided
        drop(reg);
        let mut indices = vec![0usize; a.shape().len()];
        for i in 0..size {
            let a_offset = a.byte_offset_for(&indices);
            unsafe {
                loop_fn(
                    a_ptr.offset(a_offset),
                    result_ptr.add(i * itemsize as usize),
                    1,
                    (itemsize, itemsize),
                );
            }
            increment_indices(&mut indices, a.shape());
        }
        return Some(result);
    }
    drop(reg);

    // Fallback to trait dispatch
    let ops = out_dtype.ops();
    let mut indices = vec![0usize; a.shape().len()];
    for i in 0..size {
        let a_offset = a.byte_offset_for(&indices);
        let ok = unsafe { ops.bitwise_not(a_ptr, a_offset, result_ptr, i) };
        if !ok {
            return None;
        }
        increment_indices(&mut indices, a.shape());
    }
    Some(result)
}

/// Element-wise bitwise AND.
pub fn bitwise_and(a: &RumpyArray, b: &RumpyArray) -> Option<RumpyArray> {
    map_binary_bitwise(a, b, BitwiseOp::And)
}

/// Element-wise bitwise OR.
pub fn bitwise_or(a: &RumpyArray, b: &RumpyArray) -> Option<RumpyArray> {
    map_binary_bitwise(a, b, BitwiseOp::Or)
}

/// Element-wise bitwise XOR.
pub fn bitwise_xor(a: &RumpyArray, b: &RumpyArray) -> Option<RumpyArray> {
    map_binary_bitwise(a, b, BitwiseOp::Xor)
}

/// Element-wise bitwise NOT (invert).
pub fn bitwise_not(a: &RumpyArray) -> Option<RumpyArray> {
    map_unary_bitwise_not(a)
}

/// Element-wise left shift.
pub fn left_shift(a: &RumpyArray, b: &RumpyArray) -> Option<RumpyArray> {
    map_binary_bitwise(a, b, BitwiseOp::LeftShift)
}

/// Element-wise right shift.
pub fn right_shift(a: &RumpyArray, b: &RumpyArray) -> Option<RumpyArray> {
    map_binary_bitwise(a, b, BitwiseOp::RightShift)
}
