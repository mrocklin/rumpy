//! NumPy C-API function pointer table.
//!
//! NumPy exports its C-API through a capsule containing an array of function pointers.
//! Extension modules call `import_array()` which loads this table.
//!
//! Index numbers from numpy/_core/code_generators/numpy_api.py

use std::os::raw::{c_int, c_void};
use std::ptr;

use pyo3::ffi::Py_ssize_t;

use super::array_funcs::*;
use super::structs::{PyArrayObject, PyArray_Descr};

// =============================================================================
// API Table Index Numbers
// These must match NumPy's exactly for binary compatibility
// =============================================================================

/// API indices for core functions.
/// Source: numpy/_core/code_generators/numpy_api.py
pub mod api_index {
    pub const PYARRAY_TYPE: usize = 2;
    pub const PYARRAY_DESCR_TYPE: usize = 3;

    // Array creation
    pub const PYARRAY_NEWFROMDESCRTOR: usize = 94;
    pub const PYARRAY_SIMPLENEW: usize = 96;
    pub const PYARRAY_SIMPLENEWFROMDATA: usize = 97;

    // Dtype functions
    pub const PYARRAY_DESCRFROMTYPE: usize = 45;
    pub const PYARRAY_DESCRNEW: usize = 48;

    // Array accessor functions (though often inlined as macros)
    pub const PYARRAY_NDIM_FUNC: usize = 74;
    pub const PYARRAY_DIM_FUNC: usize = 75;
    pub const PYARRAY_STRIDE_FUNC: usize = 76;
    pub const PYARRAY_DATA_FUNC: usize = 77;

    // Conversion/utility
    pub const PYARRAY_COPY: usize = 105;
    pub const PYARRAY_ARANGE: usize = 109;
    pub const PYARRAY_ZEROS: usize = 111;
    pub const PYARRAY_EMPTY: usize = 112;

    // Size of the API table (must be at least this big)
    pub const API_TABLE_SIZE: usize = 400;
}

// =============================================================================
// Function pointer type aliases
// =============================================================================

// These match NumPy's function signatures exactly

/// PyArray_SimpleNew signature
pub type PyArray_SimpleNew_t =
    unsafe extern "C" fn(nd: c_int, dims: *const Py_ssize_t, typenum: c_int) -> *mut PyArrayObject;

/// PyArray_SimpleNewFromData signature
pub type PyArray_SimpleNewFromData_t = unsafe extern "C" fn(
    nd: c_int,
    dims: *const Py_ssize_t,
    typenum: c_int,
    data: *mut c_void,
) -> *mut PyArrayObject;

/// PyArray_DescrFromType signature
pub type PyArray_DescrFromType_t = unsafe extern "C" fn(typenum: c_int) -> *mut PyArray_Descr;

// =============================================================================
// API Table
// =============================================================================

/// The global API table.
/// This is populated at module init and exported via PyCapsule.
pub static mut RUMPY_API_TABLE: [*const c_void; api_index::API_TABLE_SIZE] =
    [ptr::null(); api_index::API_TABLE_SIZE];

/// Initialize the API table with function pointers.
///
/// # Safety
/// Must be called exactly once during module initialization.
pub unsafe fn init_api_table() {
    // Type objects (placeholders - would need actual PyTypeObject pointers)
    // RUMPY_API_TABLE[api_index::PYARRAY_TYPE] = &PyArray_Type as *const _ as *const c_void;
    // RUMPY_API_TABLE[api_index::PYARRAY_DESCR_TYPE] = &PyArrayDescr_Type as *const _ as *const c_void;

    // Accessor functions
    RUMPY_API_TABLE[api_index::PYARRAY_NDIM_FUNC] = PyArray_NDIM as *const c_void;
    RUMPY_API_TABLE[api_index::PYARRAY_DIM_FUNC] = PyArray_DIM as *const c_void;
    RUMPY_API_TABLE[api_index::PYARRAY_STRIDE_FUNC] = PyArray_STRIDE as *const c_void;
    RUMPY_API_TABLE[api_index::PYARRAY_DATA_FUNC] = PyArray_DATA as *const c_void;

    // Dtype functions
    // RUMPY_API_TABLE[api_index::PYARRAY_DESCRFROMTYPE] = PyArray_DescrFromType as *const c_void;

    // Array creation
    // RUMPY_API_TABLE[api_index::PYARRAY_SIMPLENEW] = PyArray_SimpleNew as *const c_void;
    // RUMPY_API_TABLE[api_index::PYARRAY_SIMPLENEWFROMDATA] = PyArray_SimpleNewFromData as *const c_void;
}

/// Get a pointer to the API table (for PyCapsule export).
pub fn get_api_table_ptr() -> *const *const c_void {
    unsafe { RUMPY_API_TABLE.as_ptr() }
}
