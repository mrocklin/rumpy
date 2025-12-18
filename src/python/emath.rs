//! Python bindings for emath submodule (numpy.emath compatibility).
//!
//! Math functions with automatic complex domain extension.
//! These functions return complex results when real input would give NaN.

use pyo3::prelude::*;
use crate::array::dtype::{DType, DTypeKind, BinaryOp};
use crate::array::RumpyArray;
use crate::python::PyRumpyArray;
use crate::python::shape::to_rumpy_array;
use crate::array::manipulation::broadcast_shapes;

/// Get the appropriate complex dtype for a given dtype.
/// float32 -> complex64, float64/int -> complex128
fn get_complex_dtype(dtype: &DType) -> DType {
    match dtype.kind() {
        DTypeKind::Complex64 => DType::complex64(),
        DTypeKind::Complex128 => DType::complex128(),
        DTypeKind::Float32 => DType::complex64(),
        _ => DType::complex128(), // float64, int, etc.
    }
}

/// Check if an array contains any values that would need complex results.
/// For sqrt/log: any negative values
fn has_negative_values(arr: &RumpyArray) -> bool {
    let size = arr.size();
    if size == 0 {
        return false;
    }
    let ptr = arr.data_ptr();
    for offset in arr.iter_offsets() {
        unsafe {
            if let Some(val) = arr.dtype().ops().read_f64(ptr, offset) {
                if val < 0.0 {
                    return true;
                }
            } else {
                // Complex type - check real part
                if let Some((real, _)) = arr.dtype().ops().read_complex(ptr, offset) {
                    if real < 0.0 {
                        return true;
                    }
                }
            }
        }
    }
    false
}

/// Check if any values are outside [-1, 1] (for arccos/arcsin).
fn has_values_outside_unit(arr: &RumpyArray) -> bool {
    let size = arr.size();
    if size == 0 {
        return false;
    }
    let ptr = arr.data_ptr();
    for offset in arr.iter_offsets() {
        unsafe {
            if let Some(val) = arr.dtype().ops().read_f64(ptr, offset) {
                if val < -1.0 || val > 1.0 {
                    return true;
                }
            }
        }
    }
    false
}

/// Check if any values are outside (-1, 1) (for arctanh).
fn has_values_outside_open_unit(arr: &RumpyArray) -> bool {
    let size = arr.size();
    if size == 0 {
        return false;
    }
    let ptr = arr.data_ptr();
    for offset in arr.iter_offsets() {
        unsafe {
            if let Some(val) = arr.dtype().ops().read_f64(ptr, offset) {
                if val <= -1.0 || val >= 1.0 {
                    return true;
                }
            }
        }
    }
    false
}

/// Apply unary operation, converting to complex if needed.
fn apply_sqrt_with_complex(arr: &RumpyArray) -> RumpyArray {
    let is_complex = matches!(arr.dtype().kind(), DTypeKind::Complex64 | DTypeKind::Complex128);

    if is_complex || has_negative_values(arr) {
        let complex_dtype = get_complex_dtype(&arr.dtype());
        let complex_arr = arr.astype(complex_dtype);
        complex_arr.sqrt().unwrap_or_else(|_| complex_arr)
    } else {
        arr.sqrt().unwrap_or_else(|_| arr.clone())
    }
}

fn apply_log_with_complex(arr: &RumpyArray) -> RumpyArray {
    let is_complex = matches!(arr.dtype().kind(), DTypeKind::Complex64 | DTypeKind::Complex128);

    if is_complex || has_negative_values(arr) {
        let complex_dtype = get_complex_dtype(&arr.dtype());
        let complex_arr = arr.astype(complex_dtype);
        complex_arr.log().unwrap_or_else(|_| complex_arr)
    } else {
        arr.log().unwrap_or_else(|_| arr.clone())
    }
}

fn apply_log2_with_complex(arr: &RumpyArray) -> RumpyArray {
    let is_complex = matches!(arr.dtype().kind(), DTypeKind::Complex64 | DTypeKind::Complex128);

    if is_complex || has_negative_values(arr) {
        let complex_dtype = get_complex_dtype(&arr.dtype());
        let complex_arr = arr.astype(complex_dtype);
        complex_arr.log2().unwrap_or_else(|_| complex_arr)
    } else {
        arr.log2().unwrap_or_else(|_| arr.clone())
    }
}

fn apply_log10_with_complex(arr: &RumpyArray) -> RumpyArray {
    let is_complex = matches!(arr.dtype().kind(), DTypeKind::Complex64 | DTypeKind::Complex128);

    if is_complex || has_negative_values(arr) {
        let complex_dtype = get_complex_dtype(&arr.dtype());
        let complex_arr = arr.astype(complex_dtype);
        complex_arr.log10().unwrap_or_else(|_| complex_arr)
    } else {
        arr.log10().unwrap_or_else(|_| arr.clone())
    }
}

fn apply_arccos_with_complex(arr: &RumpyArray) -> RumpyArray {
    let is_complex = matches!(arr.dtype().kind(), DTypeKind::Complex64 | DTypeKind::Complex128);

    if is_complex || has_values_outside_unit(arr) {
        let complex_dtype = get_complex_dtype(&arr.dtype());
        let complex_arr = arr.astype(complex_dtype);
        complex_arr.arccos().unwrap_or_else(|_| complex_arr)
    } else {
        arr.arccos().unwrap_or_else(|_| arr.clone())
    }
}

fn apply_arcsin_with_complex(arr: &RumpyArray) -> RumpyArray {
    let is_complex = matches!(arr.dtype().kind(), DTypeKind::Complex64 | DTypeKind::Complex128);

    if is_complex || has_values_outside_unit(arr) {
        let complex_dtype = get_complex_dtype(&arr.dtype());
        let complex_arr = arr.astype(complex_dtype);
        complex_arr.arcsin().unwrap_or_else(|_| complex_arr)
    } else {
        arr.arcsin().unwrap_or_else(|_| arr.clone())
    }
}

fn apply_arctanh_with_complex(arr: &RumpyArray) -> RumpyArray {
    let is_complex = matches!(arr.dtype().kind(), DTypeKind::Complex64 | DTypeKind::Complex128);

    if is_complex || has_values_outside_open_unit(arr) {
        let complex_dtype = get_complex_dtype(&arr.dtype());
        let complex_arr = arr.astype(complex_dtype);
        complex_arr.arctanh().unwrap_or_else(|_| complex_arr)
    } else {
        arr.arctanh().unwrap_or_else(|_| arr.clone())
    }
}

// ============================================================================
// emath functions
// ============================================================================

/// Square root with complex extension.
/// Returns complex results for negative inputs.
#[pyfunction]
pub fn sqrt(x: &Bound<'_, pyo3::PyAny>) -> PyResult<PyRumpyArray> {
    let arr = to_rumpy_array(x)?;
    Ok(PyRumpyArray::new(apply_sqrt_with_complex(&arr)))
}

/// Natural logarithm with complex extension.
/// Returns complex results for negative inputs.
#[pyfunction]
pub fn log(x: &Bound<'_, pyo3::PyAny>) -> PyResult<PyRumpyArray> {
    let arr = to_rumpy_array(x)?;
    Ok(PyRumpyArray::new(apply_log_with_complex(&arr)))
}

/// Base-2 logarithm with complex extension.
/// Returns complex results for negative inputs.
#[pyfunction]
pub fn log2(x: &Bound<'_, pyo3::PyAny>) -> PyResult<PyRumpyArray> {
    let arr = to_rumpy_array(x)?;
    Ok(PyRumpyArray::new(apply_log2_with_complex(&arr)))
}

/// Base-10 logarithm with complex extension.
/// Returns complex results for negative inputs.
#[pyfunction]
pub fn log10(x: &Bound<'_, pyo3::PyAny>) -> PyResult<PyRumpyArray> {
    let arr = to_rumpy_array(x)?;
    Ok(PyRumpyArray::new(apply_log10_with_complex(&arr)))
}

/// Logarithm with arbitrary base.
/// Returns complex results for negative base or x.
#[pyfunction]
pub fn logn(n: &Bound<'_, pyo3::PyAny>, x: &Bound<'_, pyo3::PyAny>) -> PyResult<PyRumpyArray> {
    // logn(n, x) = log(x) / log(n)
    let n_arr = to_rumpy_array(n)?;
    let x_arr = to_rumpy_array(x)?;

    let needs_complex = has_negative_values(&n_arr) || has_negative_values(&x_arr);
    let is_complex_n = matches!(n_arr.dtype().kind(), DTypeKind::Complex64 | DTypeKind::Complex128);
    let is_complex_x = matches!(x_arr.dtype().kind(), DTypeKind::Complex64 | DTypeKind::Complex128);

    let (log_x, log_n) = if needs_complex || is_complex_n || is_complex_x {
        // Work in complex domain
        let complex_dtype = DType::complex128();
        let n_complex = n_arr.astype(complex_dtype.clone());
        let x_complex = x_arr.astype(complex_dtype);
        (x_complex.log().unwrap_or_else(|_| x_complex.clone()),
         n_complex.log().unwrap_or_else(|_| n_complex.clone()))
    } else {
        (x_arr.log().unwrap_or_else(|_| x_arr.clone()),
         n_arr.log().unwrap_or_else(|_| n_arr.clone()))
    };

    // Broadcast shapes and divide
    let out_shape = broadcast_shapes(log_x.shape(), log_n.shape())
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("cannot broadcast shapes"))?;

    let log_x_bc = log_x.broadcast_to(&out_shape)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("broadcast failed"))?;
    let log_n_bc = log_n.broadcast_to(&out_shape)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("broadcast failed"))?;

    log_x_bc.binary_op(&log_n_bc, BinaryOp::Div)
        .map(PyRumpyArray::new)
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("logn: division failed"))
}

/// Power with complex extension.
/// Returns complex results for negative base with fractional exponent.
#[pyfunction]
pub fn power(x: &Bound<'_, pyo3::PyAny>, p: &Bound<'_, pyo3::PyAny>) -> PyResult<PyRumpyArray> {
    let x_arr = to_rumpy_array(x)?;
    let p_arr = to_rumpy_array(p)?;

    // Check if we need complex: negative x with non-integer p
    let needs_complex = check_power_needs_complex(&x_arr, &p_arr);
    let is_complex_x = matches!(x_arr.dtype().kind(), DTypeKind::Complex64 | DTypeKind::Complex128);
    let is_complex_p = matches!(p_arr.dtype().kind(), DTypeKind::Complex64 | DTypeKind::Complex128);

    let (x_use, p_use) = if needs_complex || is_complex_x || is_complex_p {
        let complex_dtype = DType::complex128();
        (x_arr.astype(complex_dtype.clone()), p_arr.astype(complex_dtype))
    } else {
        // Keep original dtypes for real case - but promote to float64 for consistency
        let float_dtype = DType::float64();
        (x_arr.astype(float_dtype.clone()), p_arr.astype(float_dtype))
    };

    // Broadcast shapes and apply power
    let out_shape = broadcast_shapes(x_use.shape(), p_use.shape())
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("cannot broadcast shapes"))?;

    let x_bc = x_use.broadcast_to(&out_shape)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("broadcast failed"))?;
    let p_bc = p_use.broadcast_to(&out_shape)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("broadcast failed"))?;

    x_bc.binary_op(&p_bc, BinaryOp::Pow)
        .map(PyRumpyArray::new)
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("power: operation failed"))
}

/// Check if power operation needs complex domain.
fn check_power_needs_complex(x: &RumpyArray, p: &RumpyArray) -> bool {
    // If x has negative values and p has non-integer values, need complex
    let x_size = x.size();
    let p_size = p.size();
    if x_size == 0 || p_size == 0 {
        return false;
    }

    let x_ptr = x.data_ptr();
    let p_ptr = p.data_ptr();

    // Check if x has any negative values
    let mut has_negative_x = false;
    for offset in x.iter_offsets() {
        unsafe {
            if let Some(val) = x.dtype().ops().read_f64(x_ptr, offset) {
                if val < 0.0 {
                    has_negative_x = true;
                    break;
                }
            }
        }
    }

    if !has_negative_x {
        return false;
    }

    // Check if p has any non-integer values
    for offset in p.iter_offsets() {
        unsafe {
            if let Some(val) = p.dtype().ops().read_f64(p_ptr, offset) {
                if val != val.floor() {
                    return true;
                }
            }
        }
    }

    false
}

/// Inverse cosine with complex extension.
/// Returns complex results for |x| > 1.
#[pyfunction]
pub fn arccos(x: &Bound<'_, pyo3::PyAny>) -> PyResult<PyRumpyArray> {
    let arr = to_rumpy_array(x)?;
    Ok(PyRumpyArray::new(apply_arccos_with_complex(&arr)))
}

/// Inverse sine with complex extension.
/// Returns complex results for |x| > 1.
#[pyfunction]
pub fn arcsin(x: &Bound<'_, pyo3::PyAny>) -> PyResult<PyRumpyArray> {
    let arr = to_rumpy_array(x)?;
    Ok(PyRumpyArray::new(apply_arcsin_with_complex(&arr)))
}

/// Inverse hyperbolic tangent with complex extension.
/// Returns complex results for |x| >= 1.
#[pyfunction]
pub fn arctanh(x: &Bound<'_, pyo3::PyAny>) -> PyResult<PyRumpyArray> {
    let arr = to_rumpy_array(x)?;
    Ok(PyRumpyArray::new(apply_arctanh_with_complex(&arr)))
}

// ============================================================================
// Submodule registration
// ============================================================================

pub fn register_submodule(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let emath_module = PyModule::new(parent.py(), "emath")?;
    emath_module.add_function(wrap_pyfunction!(sqrt, &emath_module)?)?;
    emath_module.add_function(wrap_pyfunction!(log, &emath_module)?)?;
    emath_module.add_function(wrap_pyfunction!(log2, &emath_module)?)?;
    emath_module.add_function(wrap_pyfunction!(log10, &emath_module)?)?;
    emath_module.add_function(wrap_pyfunction!(logn, &emath_module)?)?;
    emath_module.add_function(wrap_pyfunction!(power, &emath_module)?)?;
    emath_module.add_function(wrap_pyfunction!(arccos, &emath_module)?)?;
    emath_module.add_function(wrap_pyfunction!(arcsin, &emath_module)?)?;
    emath_module.add_function(wrap_pyfunction!(arctanh, &emath_module)?)?;
    parent.add_submodule(&emath_module)?;
    Ok(())
}
