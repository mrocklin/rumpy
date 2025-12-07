pub mod creation;
pub mod fft;
pub mod indexing;
pub mod linalg;
pub mod pyarray;
pub mod random;
pub mod reductions;
pub mod shape;
pub mod ufuncs;

use pyo3::prelude::*;

pub use pyarray::{parse_dtype, parse_shape, PyRumpyArray};

use crate::array::RumpyArray;

/// 1D discrete convolution.
#[pyfunction]
#[pyo3(signature = (a, v, mode="full"))]
pub fn convolve(a: &PyRumpyArray, v: &PyRumpyArray, mode: &str) -> PyResult<PyRumpyArray> {
    crate::array::convolve(&a.inner, &v.inner, mode)
        .map(PyRumpyArray::new)
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("convolve requires 1D arrays and valid mode")
        })
}

/// 1D cross-correlation.
#[pyfunction]
#[pyo3(signature = (a, v, mode="full"))]
pub fn correlate(a: &PyRumpyArray, v: &PyRumpyArray, mode: &str) -> PyResult<PyRumpyArray> {
    crate::ops::numerical::correlate(&a.inner, &v.inner, mode)
        .map(PyRumpyArray::new)
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("correlate requires 1D arrays and valid mode")
        })
}

/// Compute numerical gradient.
///
/// For N-D arrays, returns a list of gradients along each axis.
/// If axis is specified, returns a single gradient array.
#[pyfunction]
#[pyo3(signature = (f, *varargs, axis=None))]
pub fn gradient(
    f: &PyRumpyArray,
    varargs: &Bound<'_, pyo3::types::PyTuple>,
    axis: Option<isize>,
) -> PyResult<pyo3::PyObject> {
    use pyo3::types::PyList;
    use pyo3::IntoPyObject;

    let py = varargs.py();

    // Parse varargs for spacing
    let (spacing, coords): (Option<f64>, Option<RumpyArray>) = if varargs.is_empty() {
        (None, None)
    } else if varargs.len() == 1 {
        // Single argument: either scalar spacing or coordinate array
        let arg = varargs.get_item(0)?;
        if let Ok(scalar) = arg.extract::<f64>() {
            (Some(scalar), None)
        } else if let Ok(py_arr) = arg.downcast::<PyRumpyArray>() {
            (None, Some(py_arr.borrow().inner.clone()))
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "gradient spacing must be scalar or array"
            ));
        }
    } else {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "gradient: too many positional arguments"
        ));
    };

    // Normalize axis
    let axis_usize = axis.map(|a| {
        if a < 0 {
            (f.inner.ndim() as isize + a) as usize
        } else {
            a as usize
        }
    });

    // Use coordinate-based gradient if coords provided
    if let Some(ref coord_arr) = coords {
        let ax = axis_usize.unwrap_or(0);
        let result = crate::ops::numerical::gradient_with_coords(&f.inner, coord_arr, ax)
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("gradient: invalid axis or coordinates")
            })?;
        return Ok(PyRumpyArray::new(result).into_pyobject(py)?.into_any().unbind());
    }

    // Use scalar spacing gradient
    let results = crate::ops::numerical::gradient(&f.inner, spacing, axis_usize)
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("gradient: invalid input")
        })?;

    // If single axis or 1D, return single array
    if results.len() == 1 {
        Ok(PyRumpyArray::new(results.into_iter().next().unwrap()).into_pyobject(py)?.into_any().unbind())
    } else {
        // Return tuple of arrays for N-D case
        let py_arrays: Vec<_> = results.into_iter().map(PyRumpyArray::new).collect();
        let list = PyList::new(py, py_arrays)?;
        Ok(list.into_pyobject(py)?.into_any().unbind())
    }
}

/// Trapezoidal integration.
#[pyfunction]
#[pyo3(signature = (y, x=None, dx=1.0, axis=-1))]
pub fn trapezoid(
    y: &PyRumpyArray,
    x: Option<&PyRumpyArray>,
    dx: f64,
    axis: isize,
) -> PyResult<PyRumpyArray> {
    let x_inner = x.map(|a| &a.inner);
    crate::ops::numerical::trapezoid(&y.inner, x_inner, dx, axis)
        .map(PyRumpyArray::new)
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("trapezoid: invalid input")
        })
}

/// 1D linear interpolation.
#[pyfunction]
#[pyo3(signature = (x, xp, fp, left=None, right=None))]
pub fn interp(
    x: &PyRumpyArray,
    xp: &PyRumpyArray,
    fp: &PyRumpyArray,
    left: Option<f64>,
    right: Option<f64>,
) -> PyResult<PyRumpyArray> {
    crate::ops::numerical::interp(&x.inner, &xp.inner, &fp.inner, left, right)
        .map(PyRumpyArray::new)
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("interp: xp and fp must be 1D with same size")
        })
}

/// Matrix multiplication.
#[pyfunction]
pub fn matmul(a: &PyRumpyArray, b: &PyRumpyArray) -> PyResult<PyRumpyArray> {
    crate::ops::matmul::matmul(&a.inner, &b.inner)
        .map(PyRumpyArray::new)
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("matmul: incompatible shapes")
        })
}

/// Dot product with numpy semantics.
#[pyfunction]
pub fn dot(a: &PyRumpyArray, b: &PyRumpyArray) -> PyResult<PyRumpyArray> {
    crate::ops::dot::dot(&a.inner, &b.inner)
        .map(PyRumpyArray::new)
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("dot: incompatible shapes")
        })
}

/// Inner product of two arrays.
#[pyfunction]
pub fn inner(a: &PyRumpyArray, b: &PyRumpyArray) -> PyResult<PyRumpyArray> {
    crate::ops::inner::inner(&a.inner, &b.inner)
        .map(PyRumpyArray::new)
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("inner: incompatible shapes")
        })
}

/// Outer product of two arrays.
#[pyfunction]
pub fn outer(a: &PyRumpyArray, b: &PyRumpyArray) -> PyResult<PyRumpyArray> {
    crate::ops::outer::outer(&a.inner, &b.inner)
        .map(PyRumpyArray::new)
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("outer: incompatible shapes")
        })
}

/// Solve linear system Ax = b.
#[pyfunction]
pub fn solve(a: &PyRumpyArray, b: &PyRumpyArray) -> PyResult<PyRumpyArray> {
    crate::ops::solve::solve(&a.inner, &b.inner)
        .map(PyRumpyArray::new)
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("solve: invalid dimensions or singular matrix")
        })
}

/// Compute trace of a matrix (sum of diagonal elements).
#[pyfunction]
pub fn trace(a: &PyRumpyArray) -> PyResult<f64> {
    crate::ops::linalg::trace(&a.inner)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("trace requires 2D array"))
}

/// Compute determinant of a square matrix.
#[pyfunction]
pub fn det(a: &PyRumpyArray) -> PyResult<f64> {
    crate::ops::linalg::det(&a.inner)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("det requires square 2D array"))
}

/// Compute matrix/vector norm.
#[pyfunction]
#[pyo3(signature = (a, ord=None))]
pub fn norm(a: &PyRumpyArray, ord: Option<&str>) -> PyResult<f64> {
    crate::ops::linalg::norm(&a.inner, ord)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("unsupported norm type"))
}

/// QR decomposition: A = QR.
#[pyfunction]
pub fn qr(a: &PyRumpyArray) -> PyResult<(PyRumpyArray, PyRumpyArray)> {
    crate::ops::linalg::qr(&a.inner)
        .map(|(q, r)| (PyRumpyArray::new(q), PyRumpyArray::new(r)))
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("qr requires 2D array"))
}

/// SVD decomposition: A = U @ diag(S) @ Vt.
#[pyfunction]
pub fn svd(a: &PyRumpyArray) -> PyResult<(PyRumpyArray, PyRumpyArray, PyRumpyArray)> {
    crate::ops::linalg::svd(&a.inner)
        .map(|(u, s, vt)| (PyRumpyArray::new(u), PyRumpyArray::new(s), PyRumpyArray::new(vt)))
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("svd requires 2D array"))
}

/// Matrix inverse.
#[pyfunction]
pub fn inv(a: &PyRumpyArray) -> PyResult<PyRumpyArray> {
    crate::ops::linalg::inv(&a.inner)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("inv requires square 2D array"))
}

/// Eigendecomposition of symmetric matrix.
#[pyfunction]
pub fn eigh(a: &PyRumpyArray) -> PyResult<(PyRumpyArray, PyRumpyArray)> {
    crate::ops::linalg::eigh(&a.inner)
        .map(|(w, v)| (PyRumpyArray::new(w), PyRumpyArray::new(v)))
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("eigh requires square 2D array"))
}

/// Extract diagonal or construct diagonal matrix.
#[pyfunction]
pub fn diag(a: &PyRumpyArray) -> PyResult<PyRumpyArray> {
    crate::ops::linalg::diag(&a.inner)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("diag requires 1D or 2D array"))
}

/// Vector dot product (flattens arrays then computes inner product).
#[pyfunction]
pub fn vdot(a: &PyRumpyArray, b: &PyRumpyArray) -> PyResult<f64> {
    crate::ops::linalg::vdot(&a.inner, &b.inner)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("vdot: arrays must have same total size"))
}

/// Kronecker product of two arrays.
#[pyfunction]
pub fn kron(a: &PyRumpyArray, b: &PyRumpyArray) -> PyResult<PyRumpyArray> {
    crate::ops::linalg::kron(&a.inner, &b.inner)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("kron: unsupported dimensions"))
}

/// Cross product of two 3D vectors.
#[pyfunction]
pub fn cross(a: &PyRumpyArray, b: &PyRumpyArray) -> PyResult<PyRumpyArray> {
    crate::ops::linalg::cross(&a.inner, &b.inner)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("cross: requires 1D arrays of length 3"))
}

/// Tensor dot product over specified axes.
#[pyfunction]
#[pyo3(signature = (a, b, axes=None))]
pub fn tensordot(a: &PyRumpyArray, b: &PyRumpyArray, axes: Option<&Bound<'_, pyo3::PyAny>>) -> PyResult<PyRumpyArray> {
    // Parse axes - can be int or tuple of two lists (default is 2)
    let (a_axes, b_axes) = if let Some(axes) = axes {
        if let Ok(n) = axes.extract::<usize>() {
            // axes=n means last n axes of a and first n axes of b
            let a_axes: Vec<usize> = (a.inner.ndim().saturating_sub(n)..a.inner.ndim()).collect();
            let b_axes: Vec<usize> = (0..n).collect();
            (a_axes, b_axes)
        } else if let Ok((ax_a, ax_b)) = axes.extract::<(Vec<usize>, Vec<usize>)>() {
            (ax_a, ax_b)
        } else if let Ok((ax_a, ax_b)) = axes.extract::<(Vec<i64>, Vec<i64>)>() {
            // Handle negative indices
            let a_axes: Vec<usize> = ax_a.into_iter()
                .map(|x| if x < 0 { (a.inner.ndim() as i64 + x) as usize } else { x as usize })
                .collect();
            let b_axes: Vec<usize> = ax_b.into_iter()
                .map(|x| if x < 0 { (b.inner.ndim() as i64 + x) as usize } else { x as usize })
                .collect();
            (a_axes, b_axes)
        } else {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "axes must be integer or tuple of two lists"
            ));
        }
    } else {
        // Default: axes=2
        let n = 2;
        let a_axes: Vec<usize> = (a.inner.ndim().saturating_sub(n)..a.inner.ndim()).collect();
        let b_axes: Vec<usize> = (0..n).collect();
        (a_axes, b_axes)
    };

    crate::ops::linalg::tensordot(&a.inner, &b.inner, (a_axes, b_axes))
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("tensordot: incompatible dimensions"))
}

// ============================================================================
// Stream 17: Polynomial Operations
// ============================================================================

/// Evaluate polynomial at given points.
/// Coefficients are in descending order (highest degree first).
#[pyfunction]
pub fn polyval(p: &PyRumpyArray, x: &PyRumpyArray) -> PyRumpyArray {
    PyRumpyArray::new(crate::ops::polyval(&p.inner, &x.inner))
}

/// Compute polynomial derivative.
/// Returns coefficients of d^m/dx^m polynomial.
#[pyfunction]
#[pyo3(signature = (p, m=1))]
pub fn polyder(p: &PyRumpyArray, m: usize) -> PyRumpyArray {
    PyRumpyArray::new(crate::ops::polyder(&p.inner, m))
}

/// Compute polynomial integral.
/// Returns coefficients of m-th order antiderivative with integration constant(s) k.
#[pyfunction]
#[pyo3(signature = (p, m=1, k=None))]
pub fn polyint(p: &PyRumpyArray, m: usize, k: Option<&PyRumpyArray>) -> PyRumpyArray {
    let k_inner = k.map(|arr| &arr.inner);
    PyRumpyArray::new(crate::ops::polyint(&p.inner, m, k_inner))
}

/// Fit polynomial of degree deg to data points (x, y).
/// Returns coefficients in descending order (highest degree first).
#[pyfunction]
#[pyo3(signature = (x, y, deg, w=None))]
pub fn polyfit(x: &PyRumpyArray, y: &PyRumpyArray, deg: usize, w: Option<&PyRumpyArray>) -> PyResult<PyRumpyArray> {
    let w_inner = w.map(|arr| &arr.inner);
    crate::ops::polyfit(&x.inner, &y.inner, deg, w_inner)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("polyfit failed - check inputs"))
}

/// Find roots of polynomial.
/// Coefficients are in descending order (highest degree first).
#[pyfunction]
#[pyo3(name = "roots")]
pub fn roots_fn(p: &PyRumpyArray) -> PyResult<PyRumpyArray> {
    crate::ops::roots(&p.inner)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("roots computation failed"))
}

/// Register Python module contents.
pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRumpyArray>()?;
    // Register submodules
    random::register_submodule(m)?;
    fft::register_submodule(m)?;
    linalg::register_submodule(m)?;
    // Constructors (from creation module)
    m.add_function(wrap_pyfunction!(creation::zeros, m)?)?;
    m.add_function(wrap_pyfunction!(creation::ones, m)?)?;
    m.add_function(wrap_pyfunction!(creation::arange, m)?)?;
    m.add_function(wrap_pyfunction!(creation::linspace, m)?)?;
    m.add_function(wrap_pyfunction!(creation::eye, m)?)?;
    m.add_function(wrap_pyfunction!(creation::full, m)?)?;
    m.add_function(wrap_pyfunction!(creation::empty, m)?)?;
    m.add_function(wrap_pyfunction!(creation::zeros_like, m)?)?;
    m.add_function(wrap_pyfunction!(creation::ones_like, m)?)?;
    m.add_function(wrap_pyfunction!(creation::empty_like, m)?)?;
    m.add_function(wrap_pyfunction!(creation::full_like, m)?)?;
    m.add_function(wrap_pyfunction!(creation::identity, m)?)?;
    m.add_function(wrap_pyfunction!(creation::logspace, m)?)?;
    m.add_function(wrap_pyfunction!(creation::geomspace, m)?)?;
    m.add_function(wrap_pyfunction!(creation::tri, m)?)?;
    m.add_function(wrap_pyfunction!(creation::tril, m)?)?;
    m.add_function(wrap_pyfunction!(creation::triu, m)?)?;
    m.add_function(wrap_pyfunction!(creation::diagflat, m)?)?;
    m.add_function(wrap_pyfunction!(creation::meshgrid, m)?)?;
    m.add_function(wrap_pyfunction!(creation::indices, m)?)?;
    m.add_function(wrap_pyfunction!(creation::fromfunction, m)?)?;
    m.add_function(wrap_pyfunction!(creation::copy, m)?)?;
    m.add_function(wrap_pyfunction!(creation::asarray, m)?)?;
    m.add_function(wrap_pyfunction!(creation::array, m)?)?;
    // Reductions (from reductions module)
    m.add_function(wrap_pyfunction!(reductions::sum, m)?)?;
    m.add_function(wrap_pyfunction!(reductions::prod, m)?)?;
    m.add_function(wrap_pyfunction!(reductions::mean, m)?)?;
    m.add_function(wrap_pyfunction!(reductions::var, m)?)?;
    m.add_function(wrap_pyfunction!(reductions::std_fn, m)?)?;
    m.add_function(wrap_pyfunction!(reductions::max, m)?)?;
    m.add_function(wrap_pyfunction!(reductions::min, m)?)?;
    m.add_function(wrap_pyfunction!(reductions::argmax, m)?)?;
    m.add_function(wrap_pyfunction!(reductions::argmin, m)?)?;
    m.add_function(wrap_pyfunction!(reductions::nansum, m)?)?;
    m.add_function(wrap_pyfunction!(reductions::nanprod, m)?)?;
    m.add_function(wrap_pyfunction!(reductions::nanmean, m)?)?;
    m.add_function(wrap_pyfunction!(reductions::nanvar, m)?)?;
    m.add_function(wrap_pyfunction!(reductions::nanstd, m)?)?;
    m.add_function(wrap_pyfunction!(reductions::nanmin, m)?)?;
    m.add_function(wrap_pyfunction!(reductions::nanmax, m)?)?;
    m.add_function(wrap_pyfunction!(reductions::nanargmin, m)?)?;
    m.add_function(wrap_pyfunction!(reductions::nanargmax, m)?)?;
    // Math ufuncs (from ufuncs module)
    m.add_function(wrap_pyfunction!(ufuncs::sqrt, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::exp, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::log, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::sin, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::cos, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::tan, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::floor, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::ceil, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::arcsin, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::arccos, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::arctan, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::log10, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::log2, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::sinh, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::cosh, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::tanh, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::sign, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::isnan, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::isinf, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::isfinite, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::abs, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::square, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::positive, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::negative, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::reciprocal, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::exp2, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::expm1, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::log1p, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::cbrt, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::trunc, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::fix, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::rint, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::arcsinh, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::arccosh, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::arctanh, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::signbit, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::nan_to_num, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::maximum, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::minimum, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::add, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::subtract, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::multiply, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::divide, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::power, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::floor_divide, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::remainder, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::arctan2, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::hypot, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::fmax, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::fmin, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::copysign, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::logaddexp, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::logaddexp2, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::nextafter, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::deg2rad, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::rad2deg, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::radians, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::degrees, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::real, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::imag, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::conj, m)?)?;
    m.add_function(wrap_pyfunction!(shape::diagonal, m)?)?;
    m.add_function(wrap_pyfunction!(reductions::count_nonzero, m)?)?;
    m.add_function(wrap_pyfunction!(shape::swapaxes, m)?)?;
    m.add_function(wrap_pyfunction!(indexing::sort, m)?)?;
    m.add_function(wrap_pyfunction!(indexing::argsort, m)?)?;
    m.add_function(wrap_pyfunction!(indexing::partition, m)?)?;
    m.add_function(wrap_pyfunction!(indexing::argpartition, m)?)?;
    m.add_function(wrap_pyfunction!(indexing::lexsort, m)?)?;
    m.add_function(wrap_pyfunction!(reductions::diff, m)?)?;
    m.add_function(wrap_pyfunction!(reductions::all, m)?)?;
    m.add_function(wrap_pyfunction!(reductions::any, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::clip, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::round, m)?)?;
    m.add_function(wrap_pyfunction!(reductions::cumsum, m)?)?;
    m.add_function(wrap_pyfunction!(reductions::cumprod, m)?)?;
    // Conditional (from indexing module)
    m.add_function(wrap_pyfunction!(indexing::where_fn, m)?)?;
    // Logical operations (from ufuncs module)
    m.add_function(wrap_pyfunction!(ufuncs::logical_and, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::logical_or, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::logical_xor, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::logical_not, m)?)?;
    // Comparison operations (from ufuncs module)
    m.add_function(wrap_pyfunction!(ufuncs::equal, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::not_equal, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::less, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::less_equal, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::greater, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::greater_equal, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::isclose, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::allclose, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::array_equal, m)?)?;
    // Bitwise operations (from ufuncs module)
    m.add_function(wrap_pyfunction!(ufuncs::bitwise_and, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::bitwise_or, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::bitwise_xor, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::bitwise_not, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::invert, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::left_shift, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::right_shift, m)?)?;
    // Shape manipulation (from shape module)
    m.add_function(wrap_pyfunction!(shape::expand_dims, m)?)?;
    m.add_function(wrap_pyfunction!(shape::squeeze, m)?)?;
    m.add_function(wrap_pyfunction!(shape::flip, m)?)?;
    m.add_function(wrap_pyfunction!(shape::flipud, m)?)?;
    m.add_function(wrap_pyfunction!(shape::fliplr, m)?)?;
    m.add_function(wrap_pyfunction!(shape::reshape, m)?)?;
    m.add_function(wrap_pyfunction!(shape::ravel, m)?)?;
    m.add_function(wrap_pyfunction!(shape::flatten, m)?)?;
    m.add_function(wrap_pyfunction!(shape::transpose, m)?)?;
    m.add_function(wrap_pyfunction!(shape::atleast_1d, m)?)?;
    m.add_function(wrap_pyfunction!(shape::atleast_2d, m)?)?;
    m.add_function(wrap_pyfunction!(shape::atleast_3d, m)?)?;
    m.add_function(wrap_pyfunction!(shape::moveaxis, m)?)?;
    m.add_function(wrap_pyfunction!(shape::rollaxis, m)?)?;
    m.add_function(wrap_pyfunction!(shape::broadcast_to, m)?)?;
    m.add_function(wrap_pyfunction!(shape::broadcast_arrays, m)?)?;
    // Concatenation (from shape module)
    m.add_function(wrap_pyfunction!(shape::concatenate, m)?)?;
    m.add_function(wrap_pyfunction!(shape::stack, m)?)?;
    m.add_function(wrap_pyfunction!(shape::vstack, m)?)?;
    m.add_function(wrap_pyfunction!(shape::hstack, m)?)?;
    // Splitting (from shape module)
    m.add_function(wrap_pyfunction!(shape::split, m)?)?;
    m.add_function(wrap_pyfunction!(shape::array_split, m)?)?;
    m.add_function(wrap_pyfunction!(shape::hsplit, m)?)?;
    m.add_function(wrap_pyfunction!(shape::vsplit, m)?)?;
    m.add_function(wrap_pyfunction!(shape::dsplit, m)?)?;
    m.add_function(wrap_pyfunction!(shape::column_stack, m)?)?;
    m.add_function(wrap_pyfunction!(shape::row_stack, m)?)?;
    m.add_function(wrap_pyfunction!(shape::dstack, m)?)?;
    m.add_function(wrap_pyfunction!(shape::repeat, m)?)?;
    m.add_function(wrap_pyfunction!(shape::tile, m)?)?;
    m.add_function(wrap_pyfunction!(shape::append, m)?)?;
    m.add_function(wrap_pyfunction!(shape::insert, m)?)?;
    m.add_function(wrap_pyfunction!(shape::delete, m)?)?;
    m.add_function(wrap_pyfunction!(shape::pad, m)?)?;
    m.add_function(wrap_pyfunction!(shape::roll, m)?)?;
    m.add_function(wrap_pyfunction!(shape::rot90, m)?)?;
    m.add_function(wrap_pyfunction!(shape::unique, m)?)?;
    m.add_function(wrap_pyfunction!(shape::nonzero, m)?)?;
    // Counting and statistics (from reductions module)
    m.add_function(wrap_pyfunction!(reductions::bincount, m)?)?;
    m.add_function(wrap_pyfunction!(reductions::percentile, m)?)?;
    m.add_function(wrap_pyfunction!(reductions::quantile, m)?)?;
    m.add_function(wrap_pyfunction!(reductions::median, m)?)?;
    m.add_function(wrap_pyfunction!(reductions::average, m)?)?;
    m.add_function(wrap_pyfunction!(reductions::ptp, m)?)?;
    m.add_function(wrap_pyfunction!(reductions::histogram, m)?)?;
    m.add_function(wrap_pyfunction!(reductions::cov_fn, m)?)?;
    m.add_function(wrap_pyfunction!(reductions::corrcoef, m)?)?;
    // Signal processing
    m.add_function(wrap_pyfunction!(convolve, m)?)?;
    m.add_function(wrap_pyfunction!(correlate, m)?)?;
    // Numerical operations (Stream 16)
    m.add_function(wrap_pyfunction!(gradient, m)?)?;
    m.add_function(wrap_pyfunction!(trapezoid, m)?)?;
    m.add_function(wrap_pyfunction!(interp, m)?)?;
    // Linear algebra
    m.add_function(wrap_pyfunction!(matmul, m)?)?;
    m.add_function(wrap_pyfunction!(dot, m)?)?;
    m.add_function(wrap_pyfunction!(inner, m)?)?;
    m.add_function(wrap_pyfunction!(outer, m)?)?;
    m.add_function(wrap_pyfunction!(solve, m)?)?;
    m.add_function(wrap_pyfunction!(trace, m)?)?;
    m.add_function(wrap_pyfunction!(det, m)?)?;
    m.add_function(wrap_pyfunction!(norm, m)?)?;
    m.add_function(wrap_pyfunction!(qr, m)?)?;
    m.add_function(wrap_pyfunction!(svd, m)?)?;
    m.add_function(wrap_pyfunction!(inv, m)?)?;
    m.add_function(wrap_pyfunction!(eigh, m)?)?;
    m.add_function(wrap_pyfunction!(diag, m)?)?;
    m.add_function(wrap_pyfunction!(vdot, m)?)?;
    m.add_function(wrap_pyfunction!(kron, m)?)?;
    m.add_function(wrap_pyfunction!(cross, m)?)?;
    m.add_function(wrap_pyfunction!(tensordot, m)?)?;
    // Indexing operations (from indexing module)
    m.add_function(wrap_pyfunction!(indexing::take, m)?)?;
    m.add_function(wrap_pyfunction!(indexing::take_along_axis, m)?)?;
    m.add_function(wrap_pyfunction!(indexing::compress, m)?)?;
    m.add_function(wrap_pyfunction!(indexing::searchsorted, m)?)?;
    m.add_function(wrap_pyfunction!(indexing::argwhere, m)?)?;
    m.add_function(wrap_pyfunction!(indexing::flatnonzero, m)?)?;
    m.add_function(wrap_pyfunction!(indexing::put, m)?)?;
    m.add_function(wrap_pyfunction!(indexing::put_along_axis, m)?)?;
    m.add_function(wrap_pyfunction!(indexing::choose, m)?)?;
    // Set operations (from indexing module)
    m.add_function(wrap_pyfunction!(indexing::isin, m)?)?;
    m.add_function(wrap_pyfunction!(indexing::in1d, m)?)?;
    m.add_function(wrap_pyfunction!(indexing::intersect1d, m)?)?;
    m.add_function(wrap_pyfunction!(indexing::union1d, m)?)?;
    m.add_function(wrap_pyfunction!(indexing::setdiff1d, m)?)?;
    m.add_function(wrap_pyfunction!(indexing::setxor1d, m)?)?;
    // Stream 17: Polynomial operations
    m.add_function(wrap_pyfunction!(polyval, m)?)?;
    m.add_function(wrap_pyfunction!(polyder, m)?)?;
    m.add_function(wrap_pyfunction!(polyint, m)?)?;
    m.add_function(wrap_pyfunction!(polyfit, m)?)?;
    m.add_function(wrap_pyfunction!(roots_fn, m)?)?;
    // Dtype constants (as strings, compatible with our dtype= parameters)
    m.add("float32", "float32")?;
    m.add("float64", "float64")?;
    m.add("int16", "int16")?;
    m.add("int32", "int32")?;
    m.add("int64", "int64")?;
    m.add("uint8", "uint8")?;
    m.add("uint16", "uint16")?;
    m.add("uint32", "uint32")?;
    m.add("uint64", "uint64")?;
    m.add("bool_", "bool")?;  // bool_ to avoid Python keyword conflict
    m.add("complex64", "complex64")?;
    m.add("complex128", "complex128")?;
    // newaxis is None in numpy (used for broadcasting)
    m.add("newaxis", m.py().None())?;
    Ok(())
}
