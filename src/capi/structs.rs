//! C-compatible struct definitions matching NumPy's C-API layout.
//!
//! These structs must match NumPy 2.x's memory layout exactly for binary compatibility.
//! Reference: numpy/_core/include/numpy/ndarraytypes.h

use std::os::raw::{c_char, c_int, c_void};

use pyo3::ffi::{PyObject, PyTypeObject, Py_hash_t, Py_ssize_t};

/// NumPy type numbers (NPY_TYPES enum).
/// These must match numpy's type numbering exactly.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NpyTypes;

impl NpyTypes {
    pub const NPY_BOOL: c_int = 0;
    pub const NPY_BYTE: c_int = 1;
    pub const NPY_UBYTE: c_int = 2;
    pub const NPY_SHORT: c_int = 3;
    pub const NPY_USHORT: c_int = 4;
    pub const NPY_INT: c_int = 5;
    pub const NPY_UINT: c_int = 6;
    pub const NPY_LONG: c_int = 7;
    pub const NPY_ULONG: c_int = 8;
    pub const NPY_LONGLONG: c_int = 9;
    pub const NPY_ULONGLONG: c_int = 10;
    pub const NPY_FLOAT: c_int = 11;
    pub const NPY_DOUBLE: c_int = 12;
    pub const NPY_LONGDOUBLE: c_int = 13;
    pub const NPY_CFLOAT: c_int = 14;
    pub const NPY_CDOUBLE: c_int = 15;
    pub const NPY_CLONGDOUBLE: c_int = 16;
    pub const NPY_OBJECT: c_int = 17;
    pub const NPY_STRING: c_int = 18;
    pub const NPY_UNICODE: c_int = 19;
    pub const NPY_VOID: c_int = 20;
    pub const NPY_DATETIME: c_int = 21;
    pub const NPY_TIMEDELTA: c_int = 22;
    pub const NPY_HALF: c_int = 23;

    // Platform-dependent aliases
    // On LP64 (Linux, macOS 64-bit): long is 64-bit, so int64 = NPY_LONG
    // On LLP64 (Windows 64-bit): long is 32-bit, so int64 = NPY_LONGLONG
    pub const NPY_INT8: c_int = Self::NPY_BYTE;
    pub const NPY_UINT8: c_int = Self::NPY_UBYTE;
    pub const NPY_INT16: c_int = Self::NPY_SHORT;
    pub const NPY_UINT16: c_int = Self::NPY_USHORT;
    pub const NPY_INT32: c_int = Self::NPY_INT;
    pub const NPY_UINT32: c_int = Self::NPY_UINT;

    #[cfg(target_pointer_width = "64")]
    #[cfg(not(windows))]
    pub const NPY_INT64: c_int = Self::NPY_LONG; // LP64: long is 64-bit
    #[cfg(target_pointer_width = "64")]
    #[cfg(not(windows))]
    pub const NPY_UINT64: c_int = Self::NPY_ULONG;

    #[cfg(any(target_pointer_width = "32", windows))]
    pub const NPY_INT64: c_int = Self::NPY_LONGLONG; // LLP64 or 32-bit
    #[cfg(any(target_pointer_width = "32", windows))]
    pub const NPY_UINT64: c_int = Self::NPY_ULONGLONG;

    pub const NPY_FLOAT32: c_int = Self::NPY_FLOAT;
    pub const NPY_FLOAT64: c_int = Self::NPY_DOUBLE;
    pub const NPY_COMPLEX64: c_int = Self::NPY_CFLOAT;
    pub const NPY_COMPLEX128: c_int = Self::NPY_CDOUBLE;
    pub const NPY_FLOAT16: c_int = Self::NPY_HALF;

    #[cfg(target_pointer_width = "64")]
    #[cfg(not(windows))]
    pub const NPY_INTP: c_int = Self::NPY_LONG;
    #[cfg(target_pointer_width = "64")]
    #[cfg(not(windows))]
    pub const NPY_UINTP: c_int = Self::NPY_ULONG;

    #[cfg(any(target_pointer_width = "32", windows))]
    pub const NPY_INTP: c_int = Self::NPY_INT;
    #[cfg(any(target_pointer_width = "32", windows))]
    pub const NPY_UINTP: c_int = Self::NPY_UINT;
}

/// Array flags (NPY_ARRAY_* constants).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct NpyArrayFlags;

impl NpyArrayFlags {
    pub const NPY_ARRAY_C_CONTIGUOUS: c_int = 0x0001;
    pub const NPY_ARRAY_F_CONTIGUOUS: c_int = 0x0002;
    pub const NPY_ARRAY_OWNDATA: c_int = 0x0004;
    pub const NPY_ARRAY_ALIGNED: c_int = 0x0100;
    pub const NPY_ARRAY_WRITEABLE: c_int = 0x0400;
    pub const NPY_ARRAY_WRITEBACKIFCOPY: c_int = 0x2000;

    // Common combinations
    pub const NPY_ARRAY_BEHAVED: c_int = Self::NPY_ARRAY_ALIGNED | Self::NPY_ARRAY_WRITEABLE;
    pub const NPY_ARRAY_CARRAY: c_int =
        Self::NPY_ARRAY_C_CONTIGUOUS | Self::NPY_ARRAY_BEHAVED;
    pub const NPY_ARRAY_FARRAY: c_int =
        Self::NPY_ARRAY_F_CONTIGUOUS | Self::NPY_ARRAY_BEHAVED;
}

/// NumPy data type descriptor.
///
/// This matches NumPy 2.x's PyArray_Descr layout.
/// For NumPy 1.x compatibility, the legacy descriptor has additional fields.
#[repr(C)]
pub struct PyArray_Descr {
    /// PyObject header
    pub ob_refcnt: Py_ssize_t,
    pub ob_type: *mut PyTypeObject,

    /// Type object for array scalars of this type
    pub typeobj: *mut PyTypeObject,

    /// Kind character ('f' = float, 'i' = signed int, 'u' = unsigned int, etc.)
    pub kind: c_char,

    /// Unique type character (for backward compat)
    pub type_: c_char,

    /// Byte order: '<' = little-endian, '>' = big-endian, '=' = native, '|' = N/A
    pub byteorder: c_char,

    /// Unused, kept for ABI stability
    pub _former_flags: c_char,

    /// Type number (NPY_FLOAT64, NPY_INT32, etc.)
    pub type_num: c_int,

    /// Flags for this dtype instance
    pub flags: u64,

    /// Element size in bytes
    pub elsize: Py_ssize_t,

    /// Required alignment
    pub alignment: Py_ssize_t,

    /// Metadata dict (or NULL)
    pub metadata: *mut PyObject,

    /// Cached hash value
    pub hash: Py_hash_t,

    /// Reserved for future use
    pub reserved_null: [*mut c_void; 2],
}

/// NumPy array object - the core array structure.
///
/// This matches NumPy 2.x's PyArrayObject_fields layout.
/// C code accesses this through the PyArray_* accessor macros/functions.
#[repr(C)]
pub struct PyArrayObject {
    /// PyObject header
    pub ob_refcnt: Py_ssize_t,
    pub ob_type: *mut PyTypeObject,

    /// Pointer to raw data buffer
    pub data: *mut c_char,

    /// Number of dimensions
    pub nd: c_int,

    /// Shape: size in each dimension
    pub dimensions: *mut Py_ssize_t,

    /// Strides: bytes to jump per dimension
    pub strides: *mut Py_ssize_t,

    /// Base object (for views), or NULL if owns data
    pub base: *mut PyObject,

    /// Data type descriptor
    pub descr: *mut PyArray_Descr,

    /// Array flags (C_CONTIGUOUS, WRITEABLE, etc.)
    pub flags: c_int,

    /// Weak reference list
    pub weakreflist: *mut PyObject,

    // NumPy 1.20+ fields
    /// Private buffer info
    pub _buffer_info: *mut c_void,

    // NumPy 1.22+ fields
    /// Memory handler
    pub mem_handler: *mut PyObject,
}

/// Byte order markers
pub const NPY_LITTLE: c_char = b'<' as c_char;
pub const NPY_BIG: c_char = b'>' as c_char;
pub const NPY_NATIVE: c_char = b'=' as c_char;
pub const NPY_IGNORE: c_char = b'|' as c_char;

#[cfg(target_endian = "little")]
pub const NPY_NATBYTE: c_char = NPY_LITTLE;
#[cfg(target_endian = "big")]
pub const NPY_NATBYTE: c_char = NPY_BIG;

/// Get the native byte order character for the current platform.
#[inline]
pub fn npy_native_byteorder() -> c_char {
    NPY_NATBYTE
}

/// Kind characters for dtype
pub const NPY_BOOLLTR: c_char = b'b' as c_char;
pub const NPY_SIGNEDLTR: c_char = b'i' as c_char;
pub const NPY_UNSIGNEDLTR: c_char = b'u' as c_char;
pub const NPY_FLOATINGLTR: c_char = b'f' as c_char;
pub const NPY_COMPLEXLTR: c_char = b'c' as c_char;
