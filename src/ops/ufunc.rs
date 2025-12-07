//! Core ufunc machinery for element-wise operations and reductions.
//!
//! This module provides the fundamental building blocks:
//! - `map_unary_op`: Apply unary operation element-wise
//! - `map_binary_op`: Apply binary operation with broadcasting
//! - `map_compare_op`: Element-wise comparison returning bool array
//! - `reduce_all_op`: Reduce all elements to scalar
//! - `reduce_axis_op`: Reduce along a specific axis
//!
//! All functions use registry dispatch for optimized type-specific loops,
//! with trait-based fallback for unsupported types.

use crate::array::{broadcast_shapes, increment_indices, promote_dtype, DType, RumpyArray};
use crate::array::dtype::{DTypeKind, UnaryOp, ReduceOp};
use crate::ops::registry::registry;
use crate::ops::{BinaryOp, BinaryOpError, UnaryOpError, ComparisonOp};
use std::sync::Arc;

// ============================================================================
// Helper functions
// ============================================================================

/// Compute byte offset from indices and strides.
#[inline]
pub(crate) fn linear_offset(indices: &[usize], strides: &[isize]) -> isize {
    indices.iter().zip(strides).map(|(&i, &s)| i as isize * s).sum()
}

/// Two-pass variance for contiguous f64 data - vectorizable.
/// Pass 1: compute mean, Pass 2: compute sum of squared deviations.
#[inline]
pub(crate) fn variance_f64_contiguous(ptr: *const f64, size: usize) -> f64 {
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

/// Check if dtype is an integer type.
pub(crate) fn is_integer_kind(kind: &DTypeKind) -> bool {
    matches!(kind, DTypeKind::Int16 | DTypeKind::Int32 | DTypeKind::Int64 |
             DTypeKind::Uint8 | DTypeKind::Uint16 | DTypeKind::Uint32 | DTypeKind::Uint64)
}

// ============================================================================
// Unary operations
// ============================================================================

/// Apply a unary operation element-wise, returning a new array.
pub fn map_unary_op(arr: &RumpyArray, op: UnaryOp) -> Result<RumpyArray, UnaryOpError> {
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

// ============================================================================
// Binary operations
// ============================================================================

/// Apply a binary operation element-wise with broadcasting.
pub fn map_binary_op(a: &RumpyArray, b: &RumpyArray, op: BinaryOp) -> Result<RumpyArray, BinaryOpError> {
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
    // Handle division type promotion
    if op == BinaryOp::Div {
        let a_int = is_integer_kind(&a.dtype().kind());
        let b_int = is_integer_kind(&b.dtype().kind());
        if a_int && b_int {
            let a_f64 = a.astype(DType::float64());
            let b_f64 = b.astype(DType::float64());
            return map_binary_op_inplace(&a_f64, &b_f64, op, None);
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
        let buffer = out_arr.buffer_mut();
        if Arc::get_mut(buffer).is_some() {
            out_arr
        } else {
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
        let result_is_complex = matches!(result_kind, DTypeKind::Complex64 | DTypeKind::Complex128);

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
                    BinaryOp::Arctan2 => {
                        let denom = bv.0 * bv.0 + bv.1 * bv.1;
                        if denom == 0.0 { (f64::NAN, f64::NAN) } else {
                            let div_r = (av.0 * bv.0 + av.1 * bv.1) / denom;
                            let div_i = (av.1 * bv.0 - av.0 * bv.1) / denom;
                            (div_i.atan2(div_r), 0.0)
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

// ============================================================================
// Comparison operations
// ============================================================================

/// Apply a comparison function element-wise, returning bool array.
/// Note: comparison still uses f64 for now, since ordering on complex is tricky.
pub fn map_compare_op(a: &RumpyArray, b: &RumpyArray, op: ComparisonOp) -> Option<RumpyArray> {
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

// ============================================================================
// Reduction operations
// ============================================================================

/// Reduce array along all axes, returning a 0-d array.
pub fn reduce_all_op(arr: &RumpyArray, op: ReduceOp) -> RumpyArray {
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
pub fn reduce_all_f64(arr: &RumpyArray, op: ReduceOp) -> f64 {
    let result = reduce_all_op(arr, op);
    result.get_element(&[0])
}

/// Reduce array along a specific axis.
///
/// Uses strided reduce loops from the registry for efficient axis reductions.
/// Each loop handles both contiguous (SIMD) and strided cases internally.
/// See designs/backstride-iteration.md for design rationale.
pub fn reduce_axis_op(arr: &RumpyArray, axis: usize, op: ReduceOp) -> RumpyArray {
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
