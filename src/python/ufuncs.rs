// Python bindings for ufunc operations (element-wise math).

use pyo3::prelude::*;

use crate::array::{DType, RumpyArray};
use super::PyRumpyArray;

/// Result type for ufuncs that can return scalar or array.
pub enum UnaryResult {
    Scalar(f64),
    Array(PyRumpyArray),
}

impl<'py> IntoPyObject<'py> for UnaryResult {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        match self {
            UnaryResult::Scalar(v) => Ok(v.into_pyobject(py)?.into_any()),
            UnaryResult::Array(arr) => Ok(arr.into_pyobject(py)?.into_any()),
        }
    }
}

/// Apply a unary ufunc to either scalar or array input.
fn apply_unary<F, G>(x: &Bound<'_, PyAny>, scalar_op: F, array_op: G) -> PyResult<UnaryResult>
where
    F: FnOnce(f64) -> f64,
    G: FnOnce(&RumpyArray) -> Result<RumpyArray, crate::ops::UnaryOpError>,
{
    // Try array first
    if let Ok(arr) = x.extract::<PyRef<'_, PyRumpyArray>>() {
        return array_op(&arr.inner)
            .map(|a| UnaryResult::Array(PyRumpyArray::new(a)))
            .map_err(|_| pyo3::exceptions::PyTypeError::new_err("ufunc not supported for this dtype"));
    }
    // Try scalar
    if let Ok(scalar) = x.extract::<f64>() {
        return Ok(UnaryResult::Scalar(scalar_op(scalar)));
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "input must be ndarray or number",
    ))
}

// ============================================================================
// Unary math functions
// ============================================================================

#[pyfunction]
pub fn sqrt(x: &Bound<'_, PyAny>) -> PyResult<UnaryResult> {
    apply_unary(x, |v| v.sqrt(), |a| a.sqrt())
}

#[pyfunction]
pub fn exp(x: &Bound<'_, PyAny>) -> PyResult<UnaryResult> {
    apply_unary(x, |v| v.exp(), |a| a.exp())
}

#[pyfunction]
pub fn log(x: &Bound<'_, PyAny>) -> PyResult<UnaryResult> {
    apply_unary(x, |v| v.ln(), |a| a.log())
}

#[pyfunction]
pub fn sin(x: &Bound<'_, PyAny>) -> PyResult<UnaryResult> {
    apply_unary(x, |v| v.sin(), |a| a.sin())
}

#[pyfunction]
pub fn cos(x: &Bound<'_, PyAny>) -> PyResult<UnaryResult> {
    apply_unary(x, |v| v.cos(), |a| a.cos())
}

#[pyfunction]
pub fn tan(x: &Bound<'_, PyAny>) -> PyResult<UnaryResult> {
    apply_unary(x, |v| v.tan(), |a| a.tan())
}

#[pyfunction]
pub fn floor(x: &Bound<'_, PyAny>) -> PyResult<UnaryResult> {
    apply_unary(x, |v| v.floor(), |a| a.floor())
}

#[pyfunction]
pub fn ceil(x: &Bound<'_, PyAny>) -> PyResult<UnaryResult> {
    apply_unary(x, |v| v.ceil(), |a| a.ceil())
}

#[pyfunction]
pub fn arcsin(x: &Bound<'_, PyAny>) -> PyResult<UnaryResult> {
    apply_unary(x, |v| v.asin(), |a| a.arcsin())
}

#[pyfunction]
pub fn arccos(x: &Bound<'_, PyAny>) -> PyResult<UnaryResult> {
    apply_unary(x, |v| v.acos(), |a| a.arccos())
}

#[pyfunction]
pub fn arctan(x: &Bound<'_, PyAny>) -> PyResult<UnaryResult> {
    apply_unary(x, |v| v.atan(), |a| a.arctan())
}

#[pyfunction]
pub fn log10(x: &Bound<'_, PyAny>) -> PyResult<UnaryResult> {
    apply_unary(x, |v| v.log10(), |a| a.log10())
}

#[pyfunction]
pub fn log2(x: &Bound<'_, PyAny>) -> PyResult<UnaryResult> {
    apply_unary(x, |v| v.log2(), |a| a.log2())
}

#[pyfunction]
pub fn sinh(x: &Bound<'_, PyAny>) -> PyResult<UnaryResult> {
    apply_unary(x, |v| v.sinh(), |a| a.sinh())
}

#[pyfunction]
pub fn cosh(x: &Bound<'_, PyAny>) -> PyResult<UnaryResult> {
    apply_unary(x, |v| v.cosh(), |a| a.cosh())
}

#[pyfunction]
pub fn tanh(x: &Bound<'_, PyAny>) -> PyResult<UnaryResult> {
    apply_unary(x, |v| v.tanh(), |a| a.tanh())
}

#[pyfunction]
pub fn sign(x: &Bound<'_, PyAny>) -> PyResult<UnaryResult> {
    apply_unary(x, |v| if v > 0.0 { 1.0 } else if v < 0.0 { -1.0 } else { 0.0 }, |a| a.sign())
}

#[pyfunction]
pub fn isnan(x: &Bound<'_, PyAny>) -> PyResult<UnaryResult> {
    apply_unary(x, |v| if v.is_nan() { 1.0 } else { 0.0 }, |a| a.isnan())
}

#[pyfunction]
pub fn isinf(x: &Bound<'_, PyAny>) -> PyResult<UnaryResult> {
    apply_unary(x, |v| if v.is_infinite() { 1.0 } else { 0.0 }, |a| a.isinf())
}

#[pyfunction]
pub fn isfinite(x: &Bound<'_, PyAny>) -> PyResult<UnaryResult> {
    apply_unary(x, |v| if v.is_finite() { 1.0 } else { 0.0 }, |a| a.isfinite())
}

#[pyfunction]
pub fn isneginf(x: &Bound<'_, PyAny>) -> PyResult<UnaryResult> {
    apply_unary(x, |v| if v == f64::NEG_INFINITY { 1.0 } else { 0.0 }, |a| a.isneginf())
}

#[pyfunction]
pub fn isposinf(x: &Bound<'_, PyAny>) -> PyResult<UnaryResult> {
    apply_unary(x, |v| if v == f64::INFINITY { 1.0 } else { 0.0 }, |a| a.isposinf())
}

#[pyfunction]
pub fn isreal(x: &Bound<'_, PyAny>) -> PyResult<UnaryResult> {
    apply_unary(x, |_| 1.0, |a| a.isreal())  // scalars are always real
}

#[pyfunction]
pub fn iscomplex(x: &Bound<'_, PyAny>) -> PyResult<UnaryResult> {
    apply_unary(x, |_| 0.0, |a| a.iscomplex())  // f64 scalars are never complex
}

/// Check if the array has a real-valued dtype (not complex).
#[pyfunction]
pub fn isrealobj(x: &Bound<'_, PyAny>) -> PyResult<bool> {
    use crate::array::dtype::DTypeKind;
    if let Ok(arr) = x.extract::<PyRef<'_, PyRumpyArray>>() {
        let kind = arr.inner.dtype().kind();
        Ok(!matches!(kind, DTypeKind::Complex64 | DTypeKind::Complex128))
    } else {
        // Python scalars: check type name
        let type_name = x.get_type().name()?;
        Ok(type_name != "complex")
    }
}

/// Check if the array has a complex dtype.
#[pyfunction]
pub fn iscomplexobj(x: &Bound<'_, PyAny>) -> PyResult<bool> {
    use crate::array::dtype::DTypeKind;
    if let Ok(arr) = x.extract::<PyRef<'_, PyRumpyArray>>() {
        let kind = arr.inner.dtype().kind();
        Ok(matches!(kind, DTypeKind::Complex64 | DTypeKind::Complex128))
    } else {
        // Python scalars: check type name
        let type_name = x.get_type().name()?;
        Ok(type_name == "complex")
    }
}

/// Check if two arrays share memory.
#[pyfunction]
pub fn shares_memory(a: PyRef<'_, PyRumpyArray>, b: PyRef<'_, PyRumpyArray>) -> bool {
    a.inner.shares_buffer_with(&b.inner)
}

/// Check if two arrays might share memory (same as shares_memory for rumpy).
#[pyfunction]
pub fn may_share_memory(a: PyRef<'_, PyRumpyArray>, b: PyRef<'_, PyRumpyArray>) -> bool {
    // In NumPy, may_share_memory is more conservative (can return true even if no overlap).
    // In rumpy, we use exact check since our buffers are Arc-managed.
    a.inner.shares_buffer_with(&b.inner)
}

/// Check if the input is a scalar (int, float, complex, str, bool).
/// Arrays (including 0-d arrays) and sequences are NOT scalars.
#[pyfunction]
pub fn isscalar(x: &Bound<'_, PyAny>) -> PyResult<bool> {
    // If it's a rumpy array, not a scalar
    if x.extract::<PyRef<'_, PyRumpyArray>>().is_ok() {
        return Ok(false);
    }
    // If it's a numpy array, not a scalar
    let type_name = x.get_type().name()?.to_string();
    if type_name == "ndarray" {
        return Ok(false);
    }
    // Check for Python scalar types
    Ok(matches!(type_name.as_str(), "int" | "float" | "complex" | "str" | "bool" | "bytes"))
}

/// Return the number of dimensions of the input.
#[pyfunction]
pub fn ndim(x: &Bound<'_, PyAny>) -> PyResult<usize> {
    use super::creation::asarray;
    // If it's already a rumpy array
    if let Ok(arr) = x.extract::<PyRef<'_, PyRumpyArray>>() {
        return Ok(arr.inner.ndim());
    }
    // Try to convert to array and get ndim
    let arr = asarray(x.py(), x, None)?;
    Ok(arr.inner.ndim())
}

/// Return the total number of elements in the input.
#[pyfunction]
pub fn size(x: &Bound<'_, PyAny>) -> PyResult<usize> {
    use super::creation::asarray;
    // If it's already a rumpy array
    if let Ok(arr) = x.extract::<PyRef<'_, PyRumpyArray>>() {
        return Ok(arr.inner.size());
    }
    // Try to convert to array and get size
    let arr = asarray(x.py(), x, None)?;
    Ok(arr.inner.size())
}

/// Return the shape of the input as a tuple.
#[pyfunction]
pub fn shape(py: Python<'_>, x: &Bound<'_, PyAny>) -> PyResult<Py<pyo3::types::PyTuple>> {
    use pyo3::types::PyTuple;
    use super::creation::asarray;
    // If it's already a rumpy array
    if let Ok(arr) = x.extract::<PyRef<'_, PyRumpyArray>>() {
        let shape_vec: Vec<usize> = arr.inner.shape().to_vec();
        return Ok(PyTuple::new(py, shape_vec)?.unbind());
    }
    // Try to convert to array and get shape
    let arr = asarray(x.py(), x, None)?;
    let shape_vec: Vec<usize> = arr.inner.shape().to_vec();
    Ok(PyTuple::new(py, shape_vec)?.unbind())
}

#[pyfunction]
pub fn abs(x: &Bound<'_, PyAny>) -> PyResult<UnaryResult> {
    apply_unary(x, |v| v.abs(), |a| a.abs())
}

#[pyfunction]
pub fn square(x: &Bound<'_, PyAny>) -> PyResult<UnaryResult> {
    apply_unary(x, |v| v * v, |a| a.square())
}

#[pyfunction]
pub fn positive(x: &Bound<'_, PyAny>) -> PyResult<UnaryResult> {
    apply_unary(x, |v| v, |a| a.positive())
}

#[pyfunction]
pub fn negative(x: &Bound<'_, PyAny>) -> PyResult<UnaryResult> {
    apply_unary(x, |v| -v, |a| a.neg())
}

#[pyfunction]
pub fn reciprocal(x: &Bound<'_, PyAny>) -> PyResult<UnaryResult> {
    apply_unary(x, |v| 1.0 / v, |a| a.reciprocal())
}

#[pyfunction]
pub fn exp2(x: &Bound<'_, PyAny>) -> PyResult<UnaryResult> {
    apply_unary(x, |v| 2.0_f64.powf(v), |a| a.exp2())
}

#[pyfunction]
pub fn expm1(x: &Bound<'_, PyAny>) -> PyResult<UnaryResult> {
    apply_unary(x, |v| v.exp_m1(), |a| a.expm1())
}

#[pyfunction]
pub fn log1p(x: &Bound<'_, PyAny>) -> PyResult<UnaryResult> {
    apply_unary(x, |v| v.ln_1p(), |a| a.log1p())
}

#[pyfunction]
pub fn cbrt(x: &Bound<'_, PyAny>) -> PyResult<UnaryResult> {
    apply_unary(x, |v| v.cbrt(), |a| a.cbrt())
}

#[pyfunction]
pub fn trunc(x: &Bound<'_, PyAny>) -> PyResult<UnaryResult> {
    apply_unary(x, |v| v.trunc(), |a| a.trunc())
}

/// Alias for trunc - round toward zero.
#[pyfunction]
pub fn fix(x: &Bound<'_, PyAny>) -> PyResult<UnaryResult> {
    trunc(x)
}

#[pyfunction]
pub fn rint(x: &Bound<'_, PyAny>) -> PyResult<UnaryResult> {
    apply_unary(x, |v| v.round(), |a| a.rint())
}

#[pyfunction]
pub fn arcsinh(x: &Bound<'_, PyAny>) -> PyResult<UnaryResult> {
    apply_unary(x, |v| v.asinh(), |a| a.arcsinh())
}

#[pyfunction]
pub fn arccosh(x: &Bound<'_, PyAny>) -> PyResult<UnaryResult> {
    apply_unary(x, |v| v.acosh(), |a| a.arccosh())
}

#[pyfunction]
pub fn arctanh(x: &Bound<'_, PyAny>) -> PyResult<UnaryResult> {
    apply_unary(x, |v| v.atanh(), |a| a.arctanh())
}

#[pyfunction]
pub fn signbit(x: &Bound<'_, PyAny>) -> PyResult<UnaryResult> {
    apply_unary(x, |v| if v.is_sign_negative() { 1.0 } else { 0.0 }, |a| a.signbit())
}

#[pyfunction]
#[pyo3(signature = (x, nan=None, posinf=None, neginf=None))]
pub fn nan_to_num(
    x: &PyRumpyArray,
    nan: Option<f64>,
    posinf: Option<f64>,
    neginf: Option<f64>,
) -> PyRumpyArray {
    let nan_val = nan.unwrap_or(0.0);
    PyRumpyArray::new(x.inner.nan_to_num(nan_val, posinf, neginf))
}

// ============================================================================
// Binary math functions
// ============================================================================

/// Apply a binary ufunc to either array or scalar inputs.
fn apply_binary_ufunc(
    x1: &Bound<'_, PyAny>,
    x2: &Bound<'_, PyAny>,
    op: crate::array::dtype::BinaryOp,
) -> PyResult<PyRumpyArray> {
    use crate::ops::BinaryOpError;

    // Try array-array first
    let arr1 = if let Ok(arr) = x1.extract::<PyRef<'_, PyRumpyArray>>() {
        arr.inner.clone()
    } else if let Ok(scalar) = x1.extract::<f64>() {
        RumpyArray::full(vec![1], scalar, DType::float64())
    } else {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "operand must be ndarray or number",
        ));
    };

    let arr2 = if let Ok(arr) = x2.extract::<PyRef<'_, PyRumpyArray>>() {
        arr.inner.clone()
    } else if let Ok(scalar) = x2.extract::<f64>() {
        RumpyArray::full(vec![1], scalar, DType::float64())
    } else {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "operand must be ndarray or number",
        ));
    };

    arr1.binary_op(&arr2, op)
        .map(PyRumpyArray::new)
        .map_err(|e| match e {
            BinaryOpError::ShapeMismatch => {
                pyo3::exceptions::PyValueError::new_err("operands have incompatible shapes")
            }
            BinaryOpError::UnsupportedDtype => {
                pyo3::exceptions::PyTypeError::new_err("operation not supported for these dtypes")
            }
        })
}

#[pyfunction]
pub fn maximum(x1: &Bound<'_, PyAny>, x2: &Bound<'_, PyAny>) -> PyResult<PyRumpyArray> {
    apply_binary_ufunc(x1, x2, crate::array::dtype::BinaryOp::Maximum)
}

#[pyfunction]
pub fn minimum(x1: &Bound<'_, PyAny>, x2: &Bound<'_, PyAny>) -> PyResult<PyRumpyArray> {
    apply_binary_ufunc(x1, x2, crate::array::dtype::BinaryOp::Minimum)
}

#[pyfunction]
pub fn add(x1: &Bound<'_, PyAny>, x2: &Bound<'_, PyAny>) -> PyResult<PyRumpyArray> {
    apply_binary_ufunc(x1, x2, crate::array::dtype::BinaryOp::Add)
}

#[pyfunction]
pub fn subtract(x1: &Bound<'_, PyAny>, x2: &Bound<'_, PyAny>) -> PyResult<PyRumpyArray> {
    apply_binary_ufunc(x1, x2, crate::array::dtype::BinaryOp::Sub)
}

#[pyfunction]
pub fn multiply(x1: &Bound<'_, PyAny>, x2: &Bound<'_, PyAny>) -> PyResult<PyRumpyArray> {
    apply_binary_ufunc(x1, x2, crate::array::dtype::BinaryOp::Mul)
}

#[pyfunction]
pub fn divide(x1: &Bound<'_, PyAny>, x2: &Bound<'_, PyAny>) -> PyResult<PyRumpyArray> {
    apply_binary_ufunc(x1, x2, crate::array::dtype::BinaryOp::Div)
}

#[pyfunction]
pub fn power(x1: &Bound<'_, PyAny>, x2: &Bound<'_, PyAny>) -> PyResult<PyRumpyArray> {
    apply_binary_ufunc(x1, x2, crate::array::dtype::BinaryOp::Pow)
}

#[pyfunction]
pub fn floor_divide(x1: &Bound<'_, PyAny>, x2: &Bound<'_, PyAny>) -> PyResult<PyRumpyArray> {
    apply_binary_ufunc(x1, x2, crate::array::dtype::BinaryOp::FloorDiv)
}

#[pyfunction]
pub fn remainder(x1: &Bound<'_, PyAny>, x2: &Bound<'_, PyAny>) -> PyResult<PyRumpyArray> {
    apply_binary_ufunc(x1, x2, crate::array::dtype::BinaryOp::Mod)
}

#[pyfunction]
pub fn arctan2(y: &Bound<'_, PyAny>, x: &Bound<'_, PyAny>) -> PyResult<PyRumpyArray> {
    apply_binary_ufunc(y, x, crate::array::dtype::BinaryOp::Arctan2)
}

#[pyfunction]
pub fn hypot(x1: &Bound<'_, PyAny>, x2: &Bound<'_, PyAny>) -> PyResult<PyRumpyArray> {
    apply_binary_ufunc(x1, x2, crate::array::dtype::BinaryOp::Hypot)
}

#[pyfunction]
pub fn fmax(x1: &Bound<'_, PyAny>, x2: &Bound<'_, PyAny>) -> PyResult<PyRumpyArray> {
    apply_binary_ufunc(x1, x2, crate::array::dtype::BinaryOp::FMax)
}

#[pyfunction]
pub fn fmin(x1: &Bound<'_, PyAny>, x2: &Bound<'_, PyAny>) -> PyResult<PyRumpyArray> {
    apply_binary_ufunc(x1, x2, crate::array::dtype::BinaryOp::FMin)
}

#[pyfunction]
pub fn copysign(x1: &Bound<'_, PyAny>, x2: &Bound<'_, PyAny>) -> PyResult<PyRumpyArray> {
    apply_binary_ufunc(x1, x2, crate::array::dtype::BinaryOp::Copysign)
}

#[pyfunction]
pub fn logaddexp(x1: &Bound<'_, PyAny>, x2: &Bound<'_, PyAny>) -> PyResult<PyRumpyArray> {
    apply_binary_ufunc(x1, x2, crate::array::dtype::BinaryOp::Logaddexp)
}

#[pyfunction]
pub fn logaddexp2(x1: &Bound<'_, PyAny>, x2: &Bound<'_, PyAny>) -> PyResult<PyRumpyArray> {
    apply_binary_ufunc(x1, x2, crate::array::dtype::BinaryOp::Logaddexp2)
}

#[pyfunction]
pub fn nextafter(x1: &Bound<'_, PyAny>, x2: &Bound<'_, PyAny>) -> PyResult<PyRumpyArray> {
    apply_binary_ufunc(x1, x2, crate::array::dtype::BinaryOp::Nextafter)
}

// ============================================================================
// Angle conversion functions
// ============================================================================

#[pyfunction]
pub fn deg2rad(x: &Bound<'_, PyAny>) -> PyResult<UnaryResult> {
    let deg_to_rad = std::f64::consts::PI / 180.0;
    apply_unary(x, |v| v * deg_to_rad, |a| {
        // Promote to float64 if integer (numpy behavior)
        let arr = if a.dtype() != DType::float64() && a.dtype() != DType::float32() {
            a.astype(DType::float64())
        } else {
            a.clone()
        };
        Ok(arr.scalar_op(deg_to_rad, crate::array::dtype::BinaryOp::Mul))
    })
}

#[pyfunction]
pub fn rad2deg(x: &Bound<'_, PyAny>) -> PyResult<UnaryResult> {
    let rad_to_deg = 180.0 / std::f64::consts::PI;
    apply_unary(x, |v| v * rad_to_deg, |a| {
        // Promote to float64 if integer (numpy behavior)
        let arr = if a.dtype() != DType::float64() && a.dtype() != DType::float32() {
            a.astype(DType::float64())
        } else {
            a.clone()
        };
        Ok(arr.scalar_op(rad_to_deg, crate::array::dtype::BinaryOp::Mul))
    })
}

/// Alias for deg2rad.
#[pyfunction]
pub fn radians(x: &Bound<'_, PyAny>) -> PyResult<UnaryResult> {
    deg2rad(x)
}

/// Alias for rad2deg.
#[pyfunction]
pub fn degrees(x: &Bound<'_, PyAny>) -> PyResult<UnaryResult> {
    rad2deg(x)
}

// ============================================================================
// Complex number functions
// ============================================================================

#[pyfunction]
pub fn real(x: &PyRumpyArray) -> PyRumpyArray {
    PyRumpyArray::new(x.inner.real())
}

#[pyfunction]
pub fn imag(x: &PyRumpyArray) -> PyRumpyArray {
    PyRumpyArray::new(x.inner.imag())
}

#[pyfunction]
pub fn conj(x: &PyRumpyArray) -> PyRumpyArray {
    PyRumpyArray::new(x.inner.conj())
}

// ============================================================================
// Comparison operations
// ============================================================================

#[pyfunction]
pub fn logical_and(x1: &PyRumpyArray, x2: &PyRumpyArray) -> PyResult<PyRumpyArray> {
    crate::ops::logical_and(&x1.inner, &x2.inner)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("shapes not broadcastable"))
}

#[pyfunction]
pub fn logical_or(x1: &PyRumpyArray, x2: &PyRumpyArray) -> PyResult<PyRumpyArray> {
    crate::ops::logical_or(&x1.inner, &x2.inner)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("shapes not broadcastable"))
}

#[pyfunction]
pub fn logical_xor(x1: &PyRumpyArray, x2: &PyRumpyArray) -> PyResult<PyRumpyArray> {
    crate::ops::logical_xor(&x1.inner, &x2.inner)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("shapes not broadcastable"))
}

#[pyfunction]
pub fn logical_not(x: &PyRumpyArray) -> PyRumpyArray {
    PyRumpyArray::new(crate::ops::logical_not(&x.inner))
}

#[pyfunction]
pub fn equal(x1: &Bound<'_, pyo3::PyAny>, x2: &Bound<'_, pyo3::PyAny>) -> PyResult<PyRumpyArray> {
    let a = super::shape::to_rumpy_array(x1)?;
    let b = super::shape::to_rumpy_array(x2)?;
    crate::ops::equal(&a, &b)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("operands could not be broadcast together"))
}

#[pyfunction]
pub fn not_equal(x1: &Bound<'_, pyo3::PyAny>, x2: &Bound<'_, pyo3::PyAny>) -> PyResult<PyRumpyArray> {
    let a = super::shape::to_rumpy_array(x1)?;
    let b = super::shape::to_rumpy_array(x2)?;
    crate::ops::not_equal(&a, &b)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("operands could not be broadcast together"))
}

#[pyfunction]
pub fn less(x1: &Bound<'_, pyo3::PyAny>, x2: &Bound<'_, pyo3::PyAny>) -> PyResult<PyRumpyArray> {
    let a = super::shape::to_rumpy_array(x1)?;
    let b = super::shape::to_rumpy_array(x2)?;
    crate::ops::less(&a, &b)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("operands could not be broadcast together"))
}

#[pyfunction]
pub fn less_equal(x1: &Bound<'_, pyo3::PyAny>, x2: &Bound<'_, pyo3::PyAny>) -> PyResult<PyRumpyArray> {
    let a = super::shape::to_rumpy_array(x1)?;
    let b = super::shape::to_rumpy_array(x2)?;
    crate::ops::less_equal(&a, &b)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("operands could not be broadcast together"))
}

#[pyfunction]
pub fn greater(x1: &Bound<'_, pyo3::PyAny>, x2: &Bound<'_, pyo3::PyAny>) -> PyResult<PyRumpyArray> {
    let a = super::shape::to_rumpy_array(x1)?;
    let b = super::shape::to_rumpy_array(x2)?;
    crate::ops::greater(&a, &b)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("operands could not be broadcast together"))
}

#[pyfunction]
pub fn greater_equal(x1: &Bound<'_, pyo3::PyAny>, x2: &Bound<'_, pyo3::PyAny>) -> PyResult<PyRumpyArray> {
    let a = super::shape::to_rumpy_array(x1)?;
    let b = super::shape::to_rumpy_array(x2)?;
    crate::ops::greater_equal(&a, &b)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("operands could not be broadcast together"))
}

#[pyfunction]
#[pyo3(signature = (x1, x2, rtol=1e-5, atol=1e-8, equal_nan=false))]
pub fn isclose(
    x1: &PyRumpyArray,
    x2: &PyRumpyArray,
    rtol: f64,
    atol: f64,
    equal_nan: bool,
) -> PyResult<PyRumpyArray> {
    crate::ops::isclose(&x1.inner, &x2.inner, rtol, atol, equal_nan)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("shapes not broadcastable"))
}

/// Check if two arrays are element-wise equal within a tolerance.
#[pyfunction]
#[pyo3(signature = (a, b, rtol=1e-5, atol=1e-8, equal_nan=false))]
pub fn allclose(
    a: &Bound<'_, pyo3::PyAny>,
    b: &Bound<'_, pyo3::PyAny>,
    rtol: f64,
    atol: f64,
    equal_nan: bool,
) -> PyResult<bool> {
    let arr_a = super::shape::to_rumpy_array(a)?;
    let arr_b = super::shape::to_rumpy_array(b)?;
    crate::ops::allclose(&arr_a, &arr_b, rtol, atol, equal_nan)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("operands could not be broadcast together"))
}

#[pyfunction]
pub fn array_equal(x1: &PyRumpyArray, x2: &PyRumpyArray) -> bool {
    crate::ops::array_equal(&x1.inner, &x2.inner)
}

// ============================================================================
// Bitwise operations
// ============================================================================

#[pyfunction]
pub fn bitwise_and(x1: &PyRumpyArray, x2: &PyRumpyArray) -> PyResult<PyRumpyArray> {
    crate::ops::bitwise_and(&x1.inner, &x2.inner)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("bitwise_and requires integer arrays"))
}

#[pyfunction]
pub fn bitwise_or(x1: &PyRumpyArray, x2: &PyRumpyArray) -> PyResult<PyRumpyArray> {
    crate::ops::bitwise_or(&x1.inner, &x2.inner)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("bitwise_or requires integer arrays"))
}

#[pyfunction]
pub fn bitwise_xor(x1: &PyRumpyArray, x2: &PyRumpyArray) -> PyResult<PyRumpyArray> {
    crate::ops::bitwise_xor(&x1.inner, &x2.inner)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("bitwise_xor requires integer arrays"))
}

#[pyfunction]
pub fn bitwise_not(x: &PyRumpyArray) -> PyResult<PyRumpyArray> {
    crate::ops::bitwise_not(&x.inner)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("bitwise_not requires integer array"))
}

/// Alias for bitwise_not.
#[pyfunction]
pub fn invert(x: &PyRumpyArray) -> PyResult<PyRumpyArray> {
    bitwise_not(x)
}

#[pyfunction]
pub fn left_shift(x1: &PyRumpyArray, x2: &PyRumpyArray) -> PyResult<PyRumpyArray> {
    crate::ops::left_shift(&x1.inner, &x2.inner)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("left_shift requires integer arrays"))
}

#[pyfunction]
pub fn right_shift(x1: &PyRumpyArray, x2: &PyRumpyArray) -> PyResult<PyRumpyArray> {
    crate::ops::right_shift(&x1.inner, &x2.inner)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("right_shift requires integer arrays"))
}

// ============================================================================
// Misc element-wise operations
// ============================================================================

/// Clip (limit) values in an array.
#[pyfunction]
#[pyo3(signature = (x, a_min=None, a_max=None))]
pub fn clip(x: &PyRumpyArray, a_min: Option<f64>, a_max: Option<f64>) -> PyRumpyArray {
    PyRumpyArray::new(x.inner.clip(a_min, a_max))
}

#[pyfunction]
#[pyo3(signature = (x, decimals=0))]
pub fn round(x: &PyRumpyArray, decimals: i32) -> PyRumpyArray {
    PyRumpyArray::new(x.inner.round(decimals))
}

// ============================================================================
// Special functions
// ============================================================================

/// Compute the normalized sinc function: sin(pi*x) / (pi*x).
#[pyfunction]
pub fn sinc(x: &PyRumpyArray) -> PyRumpyArray {
    PyRumpyArray::new(crate::ops::sinc(&x.inner))
}

/// Compute the modified Bessel function of the first kind, order 0.
#[pyfunction]
pub fn i0(x: &PyRumpyArray) -> PyRumpyArray {
    PyRumpyArray::new(crate::ops::i0(&x.inner))
}

/// Compute spacing (ULP) for floating point values.
#[pyfunction]
pub fn spacing(x: &PyRumpyArray) -> PyRumpyArray {
    PyRumpyArray::new(crate::ops::spacing(&x.inner))
}

/// Compute modf: return (fractional part, integer part).
#[pyfunction]
pub fn modf(x: &PyRumpyArray) -> (PyRumpyArray, PyRumpyArray) {
    let (frac, int) = crate::ops::modf(&x.inner);
    (PyRumpyArray::new(frac), PyRumpyArray::new(int))
}

/// Compute frexp: decompose x into (mantissa, exponent) where x = mantissa * 2^exponent.
#[pyfunction]
pub fn frexp(x: &PyRumpyArray) -> (PyRumpyArray, PyRumpyArray) {
    let (mant, exp) = crate::ops::frexp(&x.inner);
    (PyRumpyArray::new(mant), PyRumpyArray::new(exp))
}

/// Compute ldexp: x * 2^i.
#[pyfunction]
pub fn ldexp(x: &PyRumpyArray, i: &PyRumpyArray) -> PyRumpyArray {
    PyRumpyArray::new(crate::ops::ldexp(&x.inner, &i.inner))
}

/// Compute the Heaviside step function.
#[pyfunction]
pub fn heaviside(x: &PyRumpyArray, h0: &PyRumpyArray) -> PyRumpyArray {
    PyRumpyArray::new(crate::ops::heaviside(&x.inner, &h0.inner))
}

/// Compute the greatest common divisor element-wise.
#[pyfunction]
pub fn gcd(x1: &PyRumpyArray, x2: &PyRumpyArray) -> PyRumpyArray {
    PyRumpyArray::new(crate::ops::gcd(&x1.inner, &x2.inner))
}

/// Compute the least common multiple element-wise.
#[pyfunction]
pub fn lcm(x1: &PyRumpyArray, x2: &PyRumpyArray) -> PyRumpyArray {
    PyRumpyArray::new(crate::ops::lcm(&x1.inner, &x2.inner))
}

// ============================================================================
// Convenience aliases (Stream 30)
// ============================================================================

/// Alias for abs - compute the absolute value element-wise.
#[pyfunction]
pub fn absolute(x: &Bound<'_, PyAny>) -> PyResult<UnaryResult> {
    abs(x)
}

/// Alias for conj - compute the complex conjugate element-wise.
#[pyfunction]
pub fn conjugate(x: &PyRumpyArray) -> PyRumpyArray {
    conj(x)
}

/// Alias for arcsin - compute the inverse sine element-wise.
#[pyfunction]
pub fn asin(x: &Bound<'_, PyAny>) -> PyResult<UnaryResult> {
    arcsin(x)
}

/// Alias for arccos - compute the inverse cosine element-wise.
#[pyfunction]
pub fn acos(x: &Bound<'_, PyAny>) -> PyResult<UnaryResult> {
    arccos(x)
}

/// Alias for arctan - compute the inverse tangent element-wise.
#[pyfunction]
pub fn atan(x: &Bound<'_, PyAny>) -> PyResult<UnaryResult> {
    arctan(x)
}

/// Alias for arcsinh - compute the inverse hyperbolic sine element-wise.
#[pyfunction]
pub fn asinh(x: &Bound<'_, PyAny>) -> PyResult<UnaryResult> {
    arcsinh(x)
}

/// Alias for arccosh - compute the inverse hyperbolic cosine element-wise.
#[pyfunction]
pub fn acosh(x: &Bound<'_, PyAny>) -> PyResult<UnaryResult> {
    arccosh(x)
}

/// Alias for arctanh - compute the inverse hyperbolic tangent element-wise.
#[pyfunction]
pub fn atanh(x: &Bound<'_, PyAny>) -> PyResult<UnaryResult> {
    arctanh(x)
}

/// Alias for power - raise elements to given powers element-wise.
#[pyfunction]
#[pyo3(name = "pow")]
pub fn pow_fn(x1: &Bound<'_, PyAny>, x2: &Bound<'_, PyAny>) -> PyResult<PyRumpyArray> {
    power(x1, x2)
}

/// Alias for remainder - compute element-wise remainder of division.
#[pyfunction]
#[pyo3(name = "mod")]
pub fn mod_fn(x1: &Bound<'_, PyAny>, x2: &Bound<'_, PyAny>) -> PyResult<PyRumpyArray> {
    remainder(x1, x2)
}

/// Compute the element-wise remainder (C-style fmod, sign of dividend preserved).
/// Note: In NumPy, fmod differs from mod/remainder. Our implementation uses C-style %.
#[pyfunction]
pub fn fmod(x1: &Bound<'_, PyAny>, x2: &Bound<'_, PyAny>) -> PyResult<PyRumpyArray> {
    remainder(x1, x2)
}

/// Alias for divide - true division element-wise.
#[pyfunction]
pub fn true_divide(x1: &Bound<'_, PyAny>, x2: &Bound<'_, PyAny>) -> PyResult<PyRumpyArray> {
    divide(x1, x2)
}

/// Compute absolute values (float-only in NumPy, but we accept all numeric).
/// Note: In NumPy, fabs only accepts real floats. We accept any numeric type.
#[pyfunction]
pub fn fabs(x: &Bound<'_, PyAny>) -> PyResult<UnaryResult> {
    abs(x)
}

// ============================================================================
// Stream 38: Additional Math
// ============================================================================

/// Element-wise power, always returning float64 (or complex128 for complex inputs).
/// Unlike regular power(), float_power() always produces floating point output.
/// Uses dtype-generic dispatch that converts inline without intermediate allocations.
#[pyfunction]
pub fn float_power(x1: &Bound<'_, PyAny>, x2: &Bound<'_, PyAny>) -> PyResult<PyRumpyArray> {
    use crate::ops::dispatch;

    // Extract arrays (convert scalars/lists to arrays)
    let arr1 = if let Ok(arr) = x1.extract::<PyRef<'_, PyRumpyArray>>() {
        arr.inner.clone()
    } else if let Ok(scalar) = x1.extract::<f64>() {
        RumpyArray::full(vec![1], scalar, DType::float64())
    } else if let Ok(list) = x1.extract::<Vec<f64>>() {
        let mut arr = RumpyArray::zeros(vec![list.len()], DType::float64());
        let ptr = arr.data_ptr_mut() as *mut f64;
        for (i, &v) in list.iter().enumerate() {
            unsafe { *ptr.add(i) = v; }
        }
        arr
    } else {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "operand must be ndarray, number, or sequence",
        ));
    };

    let arr2 = if let Ok(arr) = x2.extract::<PyRef<'_, PyRumpyArray>>() {
        arr.inner.clone()
    } else if let Ok(scalar) = x2.extract::<f64>() {
        RumpyArray::full(vec![1], scalar, DType::float64())
    } else if let Ok(list) = x2.extract::<Vec<f64>>() {
        let mut arr = RumpyArray::zeros(vec![list.len()], DType::float64());
        let ptr = arr.data_ptr_mut() as *mut f64;
        for (i, &v) in list.iter().enumerate() {
            unsafe { *ptr.add(i) = v; }
        }
        arr
    } else {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "operand must be ndarray, number, or sequence",
        ));
    };

    // Broadcast shapes
    let out_shape = crate::array::broadcast_shapes(arr1.shape(), arr2.shape())
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("operands have incompatible shapes"))?;

    // Promote dtypes to common type for same-type dispatch
    let promoted_dtype = crate::array::dtype::promote_dtype(&arr1.dtype(), &arr2.dtype());
    let arr1 = if arr1.dtype() != promoted_dtype { arr1.astype(promoted_dtype.clone()) } else { arr1 };
    let arr2 = if arr2.dtype() != promoted_dtype { arr2.astype(promoted_dtype) } else { arr2 };

    // Broadcast arrays
    let arr1 = arr1.broadcast_to(&out_shape)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("broadcast failed"))?;
    let arr2 = arr2.broadcast_to(&out_shape)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("broadcast failed"))?;

    // Use dtype-generic dispatch that outputs f64/complex128
    dispatch::dispatch_float_power(&arr1, &arr2, &out_shape)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyTypeError::new_err("float_power not supported for this dtype"))
}

/// Return quotient and remainder from floor division, element-wise.
/// Equivalent to (floor_divide(x1, x2), remainder(x1, x2)) but computed together.
#[pyfunction]
pub fn divmod_fn(
    py: Python<'_>,
    x1: &Bound<'_, PyAny>,
    x2: &Bound<'_, PyAny>,
) -> PyResult<Py<pyo3::types::PyTuple>> {
    let q = floor_divide(x1, x2)?;
    let r = remainder(x1, x2)?;
    Ok(pyo3::types::PyTuple::new(py, [q.into_pyobject(py)?, r.into_pyobject(py)?])?.unbind())
}
