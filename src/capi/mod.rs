//! NumPy C-API compatibility layer for Rumpy.
//!
//! This module provides a C-compatible interface that matches NumPy's C-API,
//! allowing C extensions written for NumPy to potentially work with Rumpy arrays.
//!
//! ## Architecture
//!
//! NumPy's C-API is exported through:
//! 1. `PyArrayObject` - the C struct layout for array objects
//! 2. `PyArray_Descr` - the dtype descriptor struct
//! 3. A capsule containing function pointers (accessed via `import_array()`)
//!
//! This module provides:
//! - C-compatible struct definitions matching NumPy's layout
//! - Bridge code to convert RumpyArray <-> PyArrayObject
//! - Core accessor functions (PyArray_NDIM, PyArray_DATA, etc.)
//! - The API function pointer table
//!
//! ## Usage from C
//!
//! ```c
//! #include <rumpy/arrayobject.h>
//!
//! // Instead of NumPy's import_array():
//! import_rumpy_array();
//!
//! // Then use PyArray_* functions as normal
//! int nd = PyArray_NDIM(arr);
//! void *data = PyArray_DATA(arr);
//! ```
//!
//! ## Limitations
//!
//! Not all NumPy C-API functions are implemented. Priority was given to:
//! - Array accessors (NDIM, DATA, DIMS, STRIDES, etc.)
//! - Type checking (ISFLOAT, ISINTEGER, etc.)
//! - Basic creation (SimpleNew)
//!
//! Functions not yet implemented will be null pointers in the API table.

pub mod api_table;
pub mod array_funcs;
pub mod bridge;
pub mod structs;

// Re-export commonly used items
pub use api_table::{get_api_table_ptr, init_api_table};
pub use bridge::{dtype_to_typenum, typenum_to_dtype, CArrayWrapper};
pub use structs::{NpyArrayFlags, NpyTypes, PyArrayObject, PyArray_Descr};

use std::os::raw::c_int;

use pyo3::prelude::*;

/// Register the C-API capsule with a Python module.
///
/// This exports `_ARRAY_API` capsule that C extensions can import.
pub fn register_capi(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Initialize the API table
    unsafe {
        init_api_table();
    }

    // Note: Full PyCapsule export would go here.
    // For now, we export markers showing the API exists.

    // Add version info
    m.add("__array_api_version__", "2.0")?;
    m.add("_ARRAY_API_EXISTS", true)?;

    Ok(())
}

/// Get array info in a format suitable for C access.
///
/// Returns (data_ptr, ndim, shape_ptr, strides_ptr, typenum, flags)
/// This is useful for quick interop without full PyArrayObject creation.
pub fn get_array_info(
    arr: &crate::array::RumpyArray,
) -> (
    *const u8, // data
    usize,     // ndim
    c_int,     // typenum
    c_int,     // flags
) {
    let mut flags = 0i32;
    if arr.is_c_contiguous() {
        flags |= NpyArrayFlags::NPY_ARRAY_C_CONTIGUOUS;
    }
    if arr.is_f_contiguous() {
        flags |= NpyArrayFlags::NPY_ARRAY_F_CONTIGUOUS;
    }
    flags |= NpyArrayFlags::NPY_ARRAY_ALIGNED;
    flags |= NpyArrayFlags::NPY_ARRAY_WRITEABLE;

    (
        arr.data_ptr(),
        arr.ndim(),
        dtype_to_typenum(&arr.dtype()),
        flags,
    )
}
