//! Core PyArray_* function implementations.
//!
//! These functions provide NumPy C-API compatible access to Rumpy arrays.

use std::os::raw::{c_char, c_int, c_void};
use std::ptr;

use pyo3::ffi::Py_ssize_t;

use super::structs::{NpyArrayFlags, NpyTypes, PyArrayObject, PyArray_Descr};

// =============================================================================
// Array accessor functions (inline in NumPy, but we export as functions too)
// =============================================================================

/// Get number of dimensions.
/// NumPy: PyArray_NDIM(arr)
#[no_mangle]
pub unsafe extern "C" fn PyArray_NDIM(arr: *mut PyArrayObject) -> c_int {
    if arr.is_null() {
        return 0;
    }
    (*arr).nd
}

/// Get pointer to raw data buffer.
/// NumPy: PyArray_DATA(arr)
#[no_mangle]
pub unsafe extern "C" fn PyArray_DATA(arr: *mut PyArrayObject) -> *mut c_void {
    if arr.is_null() {
        return ptr::null_mut();
    }
    (*arr).data as *mut c_void
}

/// Get pointer to raw data buffer as char*.
/// NumPy: PyArray_BYTES(arr)
#[no_mangle]
pub unsafe extern "C" fn PyArray_BYTES(arr: *mut PyArrayObject) -> *mut c_char {
    if arr.is_null() {
        return ptr::null_mut();
    }
    (*arr).data
}

/// Get pointer to dimensions (shape) array.
/// NumPy: PyArray_DIMS(arr)
#[no_mangle]
pub unsafe extern "C" fn PyArray_DIMS(arr: *mut PyArrayObject) -> *mut Py_ssize_t {
    if arr.is_null() {
        return ptr::null_mut();
    }
    (*arr).dimensions
}

/// Get pointer to strides array.
/// NumPy: PyArray_STRIDES(arr)
#[no_mangle]
pub unsafe extern "C" fn PyArray_STRIDES(arr: *mut PyArrayObject) -> *mut Py_ssize_t {
    if arr.is_null() {
        return ptr::null_mut();
    }
    (*arr).strides
}

/// Get single dimension (shape[i]).
/// NumPy: PyArray_DIM(arr, n)
#[no_mangle]
pub unsafe extern "C" fn PyArray_DIM(arr: *mut PyArrayObject, n: c_int) -> Py_ssize_t {
    if arr.is_null() {
        return 0;
    }
    let dims = (*arr).dimensions;
    if dims.is_null() || n < 0 || n >= (*arr).nd {
        return 0;
    }
    *dims.offset(n as isize)
}

/// Get single stride.
/// NumPy: PyArray_STRIDE(arr, n)
#[no_mangle]
pub unsafe extern "C" fn PyArray_STRIDE(arr: *mut PyArrayObject, n: c_int) -> Py_ssize_t {
    if arr.is_null() {
        return 0;
    }
    let strides = (*arr).strides;
    if strides.is_null() || n < 0 || n >= (*arr).nd {
        return 0;
    }
    *strides.offset(n as isize)
}

/// Get pointer to dtype descriptor.
/// NumPy: PyArray_DESCR(arr)
#[no_mangle]
pub unsafe extern "C" fn PyArray_DESCR(arr: *mut PyArrayObject) -> *mut PyArray_Descr {
    if arr.is_null() {
        return ptr::null_mut();
    }
    (*arr).descr
}

/// Get type number from array.
/// NumPy: PyArray_TYPE(arr)
#[no_mangle]
pub unsafe extern "C" fn PyArray_TYPE(arr: *mut PyArrayObject) -> c_int {
    if arr.is_null() {
        return -1;
    }
    let descr = (*arr).descr;
    if descr.is_null() {
        return -1;
    }
    (*descr).type_num
}

/// Get element size in bytes.
/// NumPy: PyArray_ITEMSIZE(arr)
#[no_mangle]
pub unsafe extern "C" fn PyArray_ITEMSIZE(arr: *mut PyArrayObject) -> c_int {
    if arr.is_null() {
        return 0;
    }
    let descr = (*arr).descr;
    if descr.is_null() {
        return 0;
    }
    (*descr).elsize as c_int
}

/// Get array flags.
/// NumPy: PyArray_FLAGS(arr)
#[no_mangle]
pub unsafe extern "C" fn PyArray_FLAGS(arr: *mut PyArrayObject) -> c_int {
    if arr.is_null() {
        return 0;
    }
    (*arr).flags
}

/// Check if array is C-contiguous.
/// NumPy: PyArray_ISCONTIGUOUS(arr)
#[no_mangle]
pub unsafe extern "C" fn PyArray_ISCONTIGUOUS(arr: *mut PyArrayObject) -> c_int {
    if arr.is_null() {
        return 0;
    }
    (((*arr).flags & NpyArrayFlags::NPY_ARRAY_C_CONTIGUOUS) != 0) as c_int
}

/// Check if array is F-contiguous.
/// NumPy: PyArray_ISFORTRAN(arr)
#[no_mangle]
pub unsafe extern "C" fn PyArray_ISFORTRAN(arr: *mut PyArrayObject) -> c_int {
    if arr.is_null() {
        return 0;
    }
    (((*arr).flags & NpyArrayFlags::NPY_ARRAY_F_CONTIGUOUS) != 0) as c_int
}

/// Check if array is writeable.
/// NumPy: PyArray_ISWRITEABLE(arr)
#[no_mangle]
pub unsafe extern "C" fn PyArray_ISWRITEABLE(arr: *mut PyArrayObject) -> c_int {
    if arr.is_null() {
        return 0;
    }
    (((*arr).flags & NpyArrayFlags::NPY_ARRAY_WRITEABLE) != 0) as c_int
}

/// Check if array is aligned.
/// NumPy: PyArray_ISALIGNED(arr)
#[no_mangle]
pub unsafe extern "C" fn PyArray_ISALIGNED(arr: *mut PyArrayObject) -> c_int {
    if arr.is_null() {
        return 0;
    }
    (((*arr).flags & NpyArrayFlags::NPY_ARRAY_ALIGNED) != 0) as c_int
}

// =============================================================================
// Size calculations
// =============================================================================

/// Get total number of elements.
/// NumPy: PyArray_SIZE(arr)
#[no_mangle]
pub unsafe extern "C" fn PyArray_SIZE(arr: *mut PyArrayObject) -> Py_ssize_t {
    if arr.is_null() {
        return 0;
    }
    let nd = (*arr).nd;
    if nd == 0 {
        return 1; // Scalar
    }
    let dims = (*arr).dimensions;
    if dims.is_null() {
        return 0;
    }
    let mut size: Py_ssize_t = 1;
    for i in 0..nd {
        size *= *dims.offset(i as isize);
    }
    size
}

/// Get total size in bytes.
/// NumPy: PyArray_NBYTES(arr)
#[no_mangle]
pub unsafe extern "C" fn PyArray_NBYTES(arr: *mut PyArrayObject) -> Py_ssize_t {
    PyArray_SIZE(arr) * (PyArray_ITEMSIZE(arr) as Py_ssize_t)
}

// =============================================================================
// Pointer access
// =============================================================================

/// Get pointer to element at 1D index.
/// NumPy: PyArray_GETPTR1(arr, i)
#[no_mangle]
pub unsafe extern "C" fn PyArray_GETPTR1(
    arr: *mut PyArrayObject,
    i: Py_ssize_t,
) -> *mut c_void {
    if arr.is_null() {
        return ptr::null_mut();
    }
    let data = (*arr).data;
    let strides = (*arr).strides;
    if data.is_null() || strides.is_null() {
        return ptr::null_mut();
    }
    data.offset(i * *strides) as *mut c_void
}

/// Get pointer to element at 2D index.
/// NumPy: PyArray_GETPTR2(arr, i, j)
#[no_mangle]
pub unsafe extern "C" fn PyArray_GETPTR2(
    arr: *mut PyArrayObject,
    i: Py_ssize_t,
    j: Py_ssize_t,
) -> *mut c_void {
    if arr.is_null() {
        return ptr::null_mut();
    }
    let data = (*arr).data;
    let strides = (*arr).strides;
    if data.is_null() || strides.is_null() {
        return ptr::null_mut();
    }
    data.offset(i * *strides + j * *strides.offset(1)) as *mut c_void
}

/// Get pointer to element at 3D index.
/// NumPy: PyArray_GETPTR3(arr, i, j, k)
#[no_mangle]
pub unsafe extern "C" fn PyArray_GETPTR3(
    arr: *mut PyArrayObject,
    i: Py_ssize_t,
    j: Py_ssize_t,
    k: Py_ssize_t,
) -> *mut c_void {
    if arr.is_null() {
        return ptr::null_mut();
    }
    let data = (*arr).data;
    let strides = (*arr).strides;
    if data.is_null() || strides.is_null() {
        return ptr::null_mut();
    }
    data.offset(
        i * *strides + j * *strides.offset(1) + k * *strides.offset(2),
    ) as *mut c_void
}

// =============================================================================
// Type checking helpers
// =============================================================================

/// Check if type number is an integer type.
#[no_mangle]
pub extern "C" fn PyTypeNum_ISINTEGER(type_num: c_int) -> c_int {
    matches!(
        type_num,
        NpyTypes::NPY_BYTE
            | NpyTypes::NPY_UBYTE
            | NpyTypes::NPY_SHORT
            | NpyTypes::NPY_USHORT
            | NpyTypes::NPY_INT
            | NpyTypes::NPY_UINT
            | NpyTypes::NPY_LONG
            | NpyTypes::NPY_ULONG
            | NpyTypes::NPY_LONGLONG
            | NpyTypes::NPY_ULONGLONG
    ) as c_int
}

/// Check if type number is a float type.
#[no_mangle]
pub extern "C" fn PyTypeNum_ISFLOAT(type_num: c_int) -> c_int {
    matches!(
        type_num,
        NpyTypes::NPY_HALF | NpyTypes::NPY_FLOAT | NpyTypes::NPY_DOUBLE | NpyTypes::NPY_LONGDOUBLE
    ) as c_int
}

/// Check if type number is a complex type.
#[no_mangle]
pub extern "C" fn PyTypeNum_ISCOMPLEX(type_num: c_int) -> c_int {
    matches!(
        type_num,
        NpyTypes::NPY_CFLOAT | NpyTypes::NPY_CDOUBLE | NpyTypes::NPY_CLONGDOUBLE
    ) as c_int
}

/// Check if type number is numeric (integer, float, or complex).
#[no_mangle]
pub extern "C" fn PyTypeNum_ISNUMBER(type_num: c_int) -> c_int {
    (PyTypeNum_ISINTEGER(type_num) != 0
        || PyTypeNum_ISFLOAT(type_num) != 0
        || PyTypeNum_ISCOMPLEX(type_num) != 0) as c_int
}

/// Check if type number is a signed integer.
#[no_mangle]
pub extern "C" fn PyTypeNum_ISSIGNED(type_num: c_int) -> c_int {
    matches!(
        type_num,
        NpyTypes::NPY_BYTE
            | NpyTypes::NPY_SHORT
            | NpyTypes::NPY_INT
            | NpyTypes::NPY_LONG
            | NpyTypes::NPY_LONGLONG
    ) as c_int
}

/// Check if type number is an unsigned integer.
#[no_mangle]
pub extern "C" fn PyTypeNum_ISUNSIGNED(type_num: c_int) -> c_int {
    matches!(
        type_num,
        NpyTypes::NPY_UBYTE
            | NpyTypes::NPY_USHORT
            | NpyTypes::NPY_UINT
            | NpyTypes::NPY_ULONG
            | NpyTypes::NPY_ULONGLONG
    ) as c_int
}

/// Check if type number is bool.
#[no_mangle]
pub extern "C" fn PyTypeNum_ISBOOL(type_num: c_int) -> c_int {
    (type_num == NpyTypes::NPY_BOOL) as c_int
}

// =============================================================================
// Array type checking (for arrays, not type numbers)
// =============================================================================

/// Check if array dtype is integer.
/// NumPy: PyArray_ISINTEGER(arr)
#[no_mangle]
pub unsafe extern "C" fn PyArray_ISINTEGER(arr: *mut PyArrayObject) -> c_int {
    PyTypeNum_ISINTEGER(PyArray_TYPE(arr))
}

/// Check if array dtype is float.
/// NumPy: PyArray_ISFLOAT(arr)
#[no_mangle]
pub unsafe extern "C" fn PyArray_ISFLOAT(arr: *mut PyArrayObject) -> c_int {
    PyTypeNum_ISFLOAT(PyArray_TYPE(arr))
}

/// Check if array dtype is complex.
/// NumPy: PyArray_ISCOMPLEX(arr)
#[no_mangle]
pub unsafe extern "C" fn PyArray_ISCOMPLEX(arr: *mut PyArrayObject) -> c_int {
    PyTypeNum_ISCOMPLEX(PyArray_TYPE(arr))
}

/// Check if array dtype is numeric.
/// NumPy: PyArray_ISNUMBER(arr)
#[no_mangle]
pub unsafe extern "C" fn PyArray_ISNUMBER(arr: *mut PyArrayObject) -> c_int {
    PyTypeNum_ISNUMBER(PyArray_TYPE(arr))
}

/// Check if array dtype is bool.
/// NumPy: PyArray_ISBOOL(arr)
#[no_mangle]
pub unsafe extern "C" fn PyArray_ISBOOL(arr: *mut PyArrayObject) -> c_int {
    PyTypeNum_ISBOOL(PyArray_TYPE(arr))
}
