pub mod array;
pub mod ops;
pub mod python;

use pyo3::prelude::*;

/// NumPy reimplementation in Rust.
#[pymodule]
fn rumpy(m: &Bound<'_, PyModule>) -> PyResult<()> {
    python::register_module(m)?;
    Ok(())
}
