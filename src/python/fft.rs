//! Python bindings for FFT module.

use pyo3::prelude::*;
use crate::ops::fft as fft_ops;
use crate::python::PyRumpyArray;

/// 1D FFT.
#[pyfunction]
pub fn fft(arr: &PyRumpyArray) -> PyResult<PyRumpyArray> {
    fft_ops::fft(&arr.inner)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("fft requires 1D array"))
}

/// 1D inverse FFT.
#[pyfunction]
pub fn ifft(arr: &PyRumpyArray) -> PyResult<PyRumpyArray> {
    fft_ops::ifft(&arr.inner)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("ifft requires 1D array"))
}

/// 2D FFT.
#[pyfunction]
pub fn fft2(arr: &PyRumpyArray) -> PyResult<PyRumpyArray> {
    fft_ops::fft2(&arr.inner)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("fft2 requires 2D array"))
}

/// 2D inverse FFT.
#[pyfunction]
pub fn ifft2(arr: &PyRumpyArray) -> PyResult<PyRumpyArray> {
    fft_ops::ifft2(&arr.inner)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("ifft2 requires 2D array"))
}

/// Real FFT (for real input, returns only positive frequencies).
#[pyfunction]
pub fn rfft(arr: &PyRumpyArray) -> PyResult<PyRumpyArray> {
    fft_ops::rfft(&arr.inner)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("rfft requires 1D array"))
}

/// Inverse real FFT.
#[pyfunction]
pub fn irfft(arr: &PyRumpyArray) -> PyResult<PyRumpyArray> {
    fft_ops::irfft(&arr.inner)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("irfft requires 1D array"))
}

/// Shift zero-frequency component to center.
#[pyfunction]
pub fn fftshift(arr: &PyRumpyArray) -> PyResult<PyRumpyArray> {
    fft_ops::fftshift(&arr.inner)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("fftshift requires 1D or 2D array"))
}

/// Inverse of fftshift.
#[pyfunction]
pub fn ifftshift(arr: &PyRumpyArray) -> PyResult<PyRumpyArray> {
    fft_ops::ifftshift(&arr.inner)
        .map(PyRumpyArray::new)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("ifftshift requires 1D or 2D array"))
}

/// Return FFT sample frequencies.
#[pyfunction]
#[pyo3(signature = (n, d=1.0))]
pub fn fftfreq(n: usize, d: f64) -> PyRumpyArray {
    PyRumpyArray::new(fft_ops::fftfreq(n, d))
}

/// Return FFT sample frequencies for rfft.
#[pyfunction]
#[pyo3(signature = (n, d=1.0))]
pub fn rfftfreq(n: usize, d: f64) -> PyRumpyArray {
    PyRumpyArray::new(fft_ops::rfftfreq(n, d))
}

/// Register FFT submodule.
pub fn register_submodule(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let fft_module = PyModule::new(parent.py(), "fft")?;
    fft_module.add_function(wrap_pyfunction!(fft, &fft_module)?)?;
    fft_module.add_function(wrap_pyfunction!(ifft, &fft_module)?)?;
    fft_module.add_function(wrap_pyfunction!(fft2, &fft_module)?)?;
    fft_module.add_function(wrap_pyfunction!(ifft2, &fft_module)?)?;
    fft_module.add_function(wrap_pyfunction!(rfft, &fft_module)?)?;
    fft_module.add_function(wrap_pyfunction!(irfft, &fft_module)?)?;
    fft_module.add_function(wrap_pyfunction!(fftshift, &fft_module)?)?;
    fft_module.add_function(wrap_pyfunction!(ifftshift, &fft_module)?)?;
    fft_module.add_function(wrap_pyfunction!(fftfreq, &fft_module)?)?;
    fft_module.add_function(wrap_pyfunction!(rfftfreq, &fft_module)?)?;
    parent.add_submodule(&fft_module)?;
    Ok(())
}
