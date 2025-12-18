pub mod char;
pub mod creation;
pub mod datetime;
pub mod dtype;
pub mod einsum;
pub mod emath;
pub mod fft;
pub mod functional;
pub mod indexing;
pub mod io;
pub mod linalg;
pub mod misc;
pub mod numerical;
pub mod poly;
pub mod pyarray;
pub mod random;
pub mod reductions;
pub mod shape;
pub mod ufuncs;

use pyo3::prelude::*;

pub use pyarray::{parse_dtype, parse_shape, PyRumpyArray};

/// Register Python module contents.
pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRumpyArray>()?;
    m.add_class::<pyarray::PyArrayFlags>()?;
    m.add_class::<pyarray::PyFlatIter>()?;
    // Register submodules
    random::register_submodule(m)?;
    fft::register_submodule(m)?;
    linalg::register_submodule(m)?;
    char::register_char_submodule(m)?;
    emath::register_submodule(m)?;
    // Register dtype functions (finfo, iinfo, promote_types, etc.)
    dtype::register_module(m)?;
    // Register einsum functions
    einsum::register_module(m)?;
    // Register datetime functions
    datetime::register_module(m)?;
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
    // Window functions
    m.add_function(wrap_pyfunction!(creation::bartlett, m)?)?;
    m.add_function(wrap_pyfunction!(creation::blackman, m)?)?;
    m.add_function(wrap_pyfunction!(creation::hamming, m)?)?;
    m.add_function(wrap_pyfunction!(creation::hanning, m)?)?;
    m.add_function(wrap_pyfunction!(creation::kaiser, m)?)?;
    // Memory layout utilities (from creation module)
    m.add_function(wrap_pyfunction!(creation::ascontiguousarray, m)?)?;
    m.add_function(wrap_pyfunction!(creation::asfortranarray, m)?)?;
    m.add_function(wrap_pyfunction!(creation::require, m)?)?;
    m.add_function(wrap_pyfunction!(creation::copyto, m)?)?;
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
    m.add_function(wrap_pyfunction!(ufuncs::isneginf, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::isposinf, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::isreal, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::iscomplex, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::isrealobj, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::iscomplexobj, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::shares_memory, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::may_share_memory, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::isscalar, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::ndim, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::size, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::shape, m)?)?;
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
    m.add_function(wrap_pyfunction!(ufuncs::clip, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::round, m)?)?;
    // Special functions (from ufuncs module)
    m.add_function(wrap_pyfunction!(ufuncs::sinc, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::i0, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::spacing, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::modf, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::frexp, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::ldexp, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::heaviside, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::gcd, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::lcm, m)?)?;
    // More reductions (from reductions module)
    m.add_function(wrap_pyfunction!(reductions::count_nonzero, m)?)?;
    m.add_function(wrap_pyfunction!(reductions::diff, m)?)?;
    m.add_function(wrap_pyfunction!(reductions::all, m)?)?;
    m.add_function(wrap_pyfunction!(reductions::any, m)?)?;
    m.add_function(wrap_pyfunction!(reductions::cumsum, m)?)?;
    m.add_function(wrap_pyfunction!(reductions::cumprod, m)?)?;
    // Sorting (from indexing module)
    m.add_function(wrap_pyfunction!(indexing::sort, m)?)?;
    m.add_function(wrap_pyfunction!(indexing::argsort, m)?)?;
    m.add_function(wrap_pyfunction!(indexing::partition, m)?)?;
    m.add_function(wrap_pyfunction!(indexing::argpartition, m)?)?;
    m.add_function(wrap_pyfunction!(indexing::lexsort, m)?)?;
    m.add_function(wrap_pyfunction!(indexing::where_fn, m)?)?;
    // More shape ops (from shape module)
    m.add_function(wrap_pyfunction!(shape::diagonal, m)?)?;
    m.add_function(wrap_pyfunction!(shape::swapaxes, m)?)?;
    // NumPy 2.0 aliases and broadcasting utilities (from shape module)
    m.add_function(wrap_pyfunction!(shape::broadcast_shapes, m)?)?;
    m.add_function(wrap_pyfunction!(shape::concat, m)?)?;
    m.add_function(wrap_pyfunction!(shape::permute_dims, m)?)?;
    m.add_function(wrap_pyfunction!(shape::matrix_transpose, m)?)?;
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
    m.add_function(wrap_pyfunction!(shape::unique_values, m)?)?;
    m.add_function(wrap_pyfunction!(shape::unique_counts, m)?)?;
    m.add_function(wrap_pyfunction!(shape::unique_inverse, m)?)?;
    m.add_function(wrap_pyfunction!(shape::unique_all, m)?)?;
    m.add_function(wrap_pyfunction!(shape::nonzero, m)?)?;
    // Counting and statistics (from reductions module)
    m.add_function(wrap_pyfunction!(reductions::bincount, m)?)?;
    m.add_function(wrap_pyfunction!(reductions::percentile, m)?)?;
    m.add_function(wrap_pyfunction!(reductions::quantile, m)?)?;
    m.add_function(wrap_pyfunction!(reductions::median, m)?)?;
    m.add_function(wrap_pyfunction!(reductions::average, m)?)?;
    m.add_function(wrap_pyfunction!(reductions::ptp, m)?)?;
    m.add_function(wrap_pyfunction!(reductions::histogram, m)?)?;
    m.add_function(wrap_pyfunction!(reductions::histogram_bin_edges, m)?)?;
    m.add_function(wrap_pyfunction!(reductions::histogram2d, m)?)?;
    m.add_function(wrap_pyfunction!(reductions::histogramdd, m)?)?;
    m.add_function(wrap_pyfunction!(reductions::cumulative_sum, m)?)?;
    m.add_function(wrap_pyfunction!(reductions::cumulative_prod, m)?)?;
    m.add_function(wrap_pyfunction!(reductions::cov_fn, m)?)?;
    m.add_function(wrap_pyfunction!(reductions::corrcoef, m)?)?;
    // Signal processing and numerical operations (from numerical module)
    m.add_function(wrap_pyfunction!(numerical::convolve, m)?)?;
    m.add_function(wrap_pyfunction!(numerical::correlate, m)?)?;
    m.add_function(wrap_pyfunction!(numerical::gradient, m)?)?;
    m.add_function(wrap_pyfunction!(numerical::trapezoid, m)?)?;
    m.add_function(wrap_pyfunction!(numerical::interp, m)?)?;
    // Linear algebra (from linalg module)
    m.add_function(wrap_pyfunction!(linalg::matmul, m)?)?;
    m.add_function(wrap_pyfunction!(linalg::dot, m)?)?;
    m.add_function(wrap_pyfunction!(linalg::inner, m)?)?;
    m.add_function(wrap_pyfunction!(linalg::outer, m)?)?;
    m.add_function(wrap_pyfunction!(linalg::solve, m)?)?;
    m.add_function(wrap_pyfunction!(linalg::trace, m)?)?;
    m.add_function(wrap_pyfunction!(linalg::det, m)?)?;
    m.add_function(wrap_pyfunction!(linalg::norm, m)?)?;
    m.add_function(wrap_pyfunction!(linalg::qr, m)?)?;
    m.add_function(wrap_pyfunction!(linalg::svd, m)?)?;
    m.add_function(wrap_pyfunction!(linalg::inv, m)?)?;
    m.add_function(wrap_pyfunction!(linalg::eigh, m)?)?;
    m.add_function(wrap_pyfunction!(linalg::diag, m)?)?;
    m.add_function(wrap_pyfunction!(linalg::vdot, m)?)?;
    m.add_function(wrap_pyfunction!(linalg::kron, m)?)?;
    m.add_function(wrap_pyfunction!(linalg::cross, m)?)?;
    m.add_function(wrap_pyfunction!(linalg::tensordot, m)?)?;
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
    // Index utilities (Stream 26)
    m.add_function(wrap_pyfunction!(indexing::unravel_index, m)?)?;
    m.add_function(wrap_pyfunction!(indexing::ravel_multi_index, m)?)?;
    m.add_function(wrap_pyfunction!(indexing::diag_indices, m)?)?;
    m.add_function(wrap_pyfunction!(indexing::diag_indices_from, m)?)?;
    m.add_function(wrap_pyfunction!(indexing::tril_indices, m)?)?;
    m.add_function(wrap_pyfunction!(indexing::triu_indices, m)?)?;
    m.add_function(wrap_pyfunction!(indexing::tril_indices_from, m)?)?;
    m.add_function(wrap_pyfunction!(indexing::triu_indices_from, m)?)?;
    m.add_function(wrap_pyfunction!(indexing::mask_indices, m)?)?;
    m.add_function(wrap_pyfunction!(indexing::digitize, m)?)?;
    m.add_function(wrap_pyfunction!(indexing::packbits, m)?)?;
    m.add_function(wrap_pyfunction!(indexing::unpackbits, m)?)?;
    // Polynomial operations (from poly module)
    m.add_function(wrap_pyfunction!(poly::polyval, m)?)?;
    m.add_function(wrap_pyfunction!(poly::polyder, m)?)?;
    m.add_function(wrap_pyfunction!(poly::polyint, m)?)?;
    m.add_function(wrap_pyfunction!(poly::polyfit, m)?)?;
    m.add_function(wrap_pyfunction!(poly::roots_fn, m)?)?;
    // I/O operations (from io module)
    m.add_function(wrap_pyfunction!(io::loadtxt, m)?)?;
    m.add_function(wrap_pyfunction!(io::savetxt, m)?)?;
    m.add_function(wrap_pyfunction!(io::genfromtxt, m)?)?;
    m.add_function(wrap_pyfunction!(io::save, m)?)?;
    m.add_function(wrap_pyfunction!(io::load, m)?)?;
    m.add_function(wrap_pyfunction!(io::savez, m)?)?;
    m.add_function(wrap_pyfunction!(io::savez_compressed, m)?)?;
    m.add_function(wrap_pyfunction!(io::frombuffer, m)?)?;
    m.add_function(wrap_pyfunction!(io::fromfile, m)?)?;
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
    // Mathematical constants (Stream 30)
    m.add("pi", std::f64::consts::PI)?;
    m.add("e", std::f64::consts::E)?;
    m.add("inf", f64::INFINITY)?;
    m.add("nan", f64::NAN)?;
    // Convenience aliases (Stream 30)
    m.add_function(wrap_pyfunction!(ufuncs::absolute, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::conjugate, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::asin, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::acos, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::atan, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::asinh, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::acosh, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::atanh, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::pow_fn, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::mod_fn, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::fmod, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::true_divide, m)?)?;
    m.add_function(wrap_pyfunction!(ufuncs::fabs, m)?)?;
    m.add_function(wrap_pyfunction!(reductions::amax, m)?)?;
    m.add_function(wrap_pyfunction!(reductions::amin, m)?)?;
    // Stream 38: Additional Math
    m.add_function(wrap_pyfunction!(ufuncs::float_power, m)?)?;
    m.add("divmod", wrap_pyfunction!(ufuncs::divmod_fn, m)?)?;
    m.add("euler_gamma", 0.5772156649015329_f64)?;
    // NaN-aware extensions (Stream 31)
    m.add_function(wrap_pyfunction!(reductions::nanmedian, m)?)?;
    m.add_function(wrap_pyfunction!(reductions::nanpercentile, m)?)?;
    m.add_function(wrap_pyfunction!(reductions::nanquantile, m)?)?;
    m.add_function(wrap_pyfunction!(reductions::nancumsum, m)?)?;
    m.add_function(wrap_pyfunction!(reductions::nancumprod, m)?)?;
    // Miscellaneous operations (Stream 32)
    m.add_function(wrap_pyfunction!(misc::resize, m)?)?;
    m.add_function(wrap_pyfunction!(misc::unstack, m)?)?;
    m.add_function(wrap_pyfunction!(misc::block, m)?)?;
    m.add_function(wrap_pyfunction!(misc::trim_zeros, m)?)?;
    m.add_function(wrap_pyfunction!(misc::extract, m)?)?;
    m.add_function(wrap_pyfunction!(misc::place, m)?)?;
    m.add_function(wrap_pyfunction!(misc::putmask, m)?)?;
    m.add_function(wrap_pyfunction!(misc::select, m)?)?;
    m.add_function(wrap_pyfunction!(misc::piecewise, m)?)?;
    m.add_function(wrap_pyfunction!(misc::ediff1d, m)?)?;
    m.add_function(wrap_pyfunction!(misc::unwrap, m)?)?;
    m.add_function(wrap_pyfunction!(misc::angle, m)?)?;
    m.add_function(wrap_pyfunction!(misc::real_if_close, m)?)?;
    // Functional programming operations (Stream 33)
    m.add_function(wrap_pyfunction!(functional::apply_along_axis, m)?)?;
    m.add_function(wrap_pyfunction!(functional::apply_over_axes, m)?)?;
    m.add_function(wrap_pyfunction!(functional::vectorize, m)?)?;
    m.add_function(wrap_pyfunction!(functional::frompyfunc, m)?)?;
    m.add_class::<functional::PyVectorize>()?;
    m.add_class::<functional::PyFrompyfunc>()?;
    // Index builders (Stream 34)
    m.add_function(wrap_pyfunction!(indexing::ix_, m)?)?;
    m.add_function(wrap_pyfunction!(indexing::fill_diagonal, m)?)?;
    m.add_function(wrap_pyfunction!(indexing::ndenumerate, m)?)?;
    m.add_function(wrap_pyfunction!(indexing::ndindex, m)?)?;
    m.add_class::<indexing::NdEnumerate>()?;
    m.add_class::<indexing::NdIndex>()?;
    m.add_class::<indexing::OGridClass>()?;
    m.add_class::<indexing::MGridClass>()?;
    // ogrid and mgrid are singleton instances (like numpy)
    m.add("ogrid", indexing::OGridClass)?;
    m.add("mgrid", indexing::MGridClass)?;
    Ok(())
}
