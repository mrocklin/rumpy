pub mod array;
pub mod capi;
pub mod ops;
pub mod python;
pub mod random;

use pyo3::prelude::*;

/// NumPy reimplementation in Rust.
#[pymodule]
fn rumpy(m: &Bound<'_, PyModule>) -> PyResult<()> {
    python::register_module(m)?;
    Ok(())
}
