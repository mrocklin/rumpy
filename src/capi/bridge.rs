//! Bridge between Rumpy's RumpyArray and NumPy C-API compatible structures.
//!
//! This module handles the conversion between Rumpy's internal representation
//! and the C-compatible PyArrayObject layout that C extensions expect.

use std::alloc::{alloc, dealloc, Layout};
use std::os::raw::{c_char, c_int, c_void};
use std::ptr;

use pyo3::ffi::Py_ssize_t;

use crate::array::{DType, RumpyArray};
use crate::array::dtype::DTypeKind;

use super::structs::{
    NpyArrayFlags, NpyTypes, PyArrayObject, PyArray_Descr,
    NPY_BOOLLTR, NPY_COMPLEXLTR, NPY_FLOATINGLTR, NPY_SIGNEDLTR, NPY_UNSIGNEDLTR,
    npy_native_byteorder,
};

// =============================================================================
// DType <-> PyArray_Descr conversion
// =============================================================================

/// Convert Rumpy DType to NumPy type number.
pub fn dtype_to_typenum(dtype: &DType) -> c_int {
    match dtype.kind() {
        DTypeKind::Bool => NpyTypes::NPY_BOOL,
        DTypeKind::Int8 => NpyTypes::NPY_INT8,
        DTypeKind::Int16 => NpyTypes::NPY_INT16,
        DTypeKind::Int32 => NpyTypes::NPY_INT32,
        DTypeKind::Int64 => NpyTypes::NPY_INT64,  // Platform-dependent
        DTypeKind::Uint8 => NpyTypes::NPY_UINT8,
        DTypeKind::Uint16 => NpyTypes::NPY_UINT16,
        DTypeKind::Uint32 => NpyTypes::NPY_UINT32,
        DTypeKind::Uint64 => NpyTypes::NPY_UINT64,  // Platform-dependent
        DTypeKind::Float16 => NpyTypes::NPY_FLOAT16,
        DTypeKind::Float32 => NpyTypes::NPY_FLOAT32,
        DTypeKind::Float64 => NpyTypes::NPY_FLOAT64,
        DTypeKind::Complex64 => NpyTypes::NPY_COMPLEX64,
        DTypeKind::Complex128 => NpyTypes::NPY_COMPLEX128,
        DTypeKind::DateTime64(_) => NpyTypes::NPY_DATETIME,
    }
}

/// Convert NumPy type number to Rumpy DType.
pub fn typenum_to_dtype(typenum: c_int) -> Option<DType> {
    match typenum {
        NpyTypes::NPY_BOOL => Some(DType::bool()),
        NpyTypes::NPY_BYTE => Some(DType::int8()),
        NpyTypes::NPY_SHORT => Some(DType::int16()),
        NpyTypes::NPY_INT => Some(DType::int32()),
        NpyTypes::NPY_UBYTE => Some(DType::uint8()),
        NpyTypes::NPY_USHORT => Some(DType::uint16()),
        NpyTypes::NPY_UINT => Some(DType::uint32()),
        NpyTypes::NPY_HALF => Some(DType::float16()),
        NpyTypes::NPY_FLOAT => Some(DType::float32()),
        NpyTypes::NPY_DOUBLE => Some(DType::float64()),
        NpyTypes::NPY_CFLOAT => Some(DType::complex64()),
        NpyTypes::NPY_CDOUBLE => Some(DType::complex128()),
        // Platform-dependent 64-bit types
        NpyTypes::NPY_LONG => Some(DType::int64()),      // LP64: long is 64-bit
        NpyTypes::NPY_ULONG => Some(DType::uint64()),
        NpyTypes::NPY_LONGLONG => Some(DType::int64()),  // LLP64 or explicit
        NpyTypes::NPY_ULONGLONG => Some(DType::uint64()),
        _ => None,
    }
}

/// Get the kind character for a dtype.
pub fn dtype_to_kind(dtype: &DType) -> c_char {
    match dtype.kind() {
        DTypeKind::Bool => NPY_BOOLLTR,
        DTypeKind::Int8 | DTypeKind::Int16 | DTypeKind::Int32 | DTypeKind::Int64 => NPY_SIGNEDLTR,
        DTypeKind::Uint8 | DTypeKind::Uint16 | DTypeKind::Uint32 | DTypeKind::Uint64 => {
            NPY_UNSIGNEDLTR
        }
        DTypeKind::Float16 | DTypeKind::Float32 | DTypeKind::Float64 => NPY_FLOATINGLTR,
        DTypeKind::Complex64 | DTypeKind::Complex128 => NPY_COMPLEXLTR,
        DTypeKind::DateTime64(_) => b'M' as c_char, // datetime
    }
}

/// Create a PyArray_Descr from a DType.
///
/// # Safety
/// The returned descriptor must be freed with `free_descr`.
pub unsafe fn create_descr(dtype: &DType) -> *mut PyArray_Descr {
    let layout = Layout::new::<PyArray_Descr>();
    let ptr = alloc(layout) as *mut PyArray_Descr;
    if ptr.is_null() {
        return ptr::null_mut();
    }

    // Initialize the descriptor
    (*ptr).ob_refcnt = 1;
    (*ptr).ob_type = ptr::null_mut(); // Would need actual PyArrayDescr_Type
    (*ptr).typeobj = ptr::null_mut();
    (*ptr).kind = dtype_to_kind(dtype);
    (*ptr).type_ = 0; // Legacy field
    (*ptr).byteorder = npy_native_byteorder();
    (*ptr)._former_flags = 0;
    (*ptr).type_num = dtype_to_typenum(dtype);
    (*ptr).flags = 0;
    (*ptr).elsize = dtype.itemsize() as Py_ssize_t;
    (*ptr).alignment = dtype.itemsize() as Py_ssize_t; // Simplified
    (*ptr).metadata = ptr::null_mut();
    (*ptr).hash = -1;
    (*ptr).reserved_null = [ptr::null_mut(); 2];

    ptr
}

/// Free a PyArray_Descr created by `create_descr`.
///
/// # Safety
/// Must only be called on descriptors created by `create_descr`.
pub unsafe fn free_descr(descr: *mut PyArray_Descr) {
    if !descr.is_null() {
        let layout = Layout::new::<PyArray_Descr>();
        dealloc(descr as *mut u8, layout);
    }
}

// =============================================================================
// RumpyArray <-> PyArrayObject conversion
// =============================================================================

/// Storage for shape/strides that lives alongside the PyArrayObject.
/// This ensures the pointers remain valid while the C struct is in use.
struct ArrayStorage {
    shape: Vec<Py_ssize_t>,
    strides: Vec<Py_ssize_t>,
    descr: *mut PyArray_Descr,
    /// Keep the RumpyArray alive to prevent buffer deallocation
    _array: RumpyArray,
}

/// Wrapper that holds a PyArrayObject and its backing storage.
pub struct CArrayWrapper {
    /// The C-compatible array struct
    pub array: *mut PyArrayObject,
    /// Storage that must outlive the array pointer
    storage: Box<ArrayStorage>,
}

impl CArrayWrapper {
    /// Create a C-compatible array wrapper from a RumpyArray.
    ///
    /// The returned wrapper contains a PyArrayObject pointer that can be passed
    /// to C code expecting NumPy arrays. The wrapper must be kept alive while
    /// the pointer is in use.
    pub fn from_rumpy(arr: RumpyArray) -> Option<Self> {
        unsafe {
            // Allocate the PyArrayObject
            let layout = Layout::new::<PyArrayObject>();
            let array_ptr = alloc(layout) as *mut PyArrayObject;
            if array_ptr.is_null() {
                return None;
            }

            // Create descriptor
            let descr = create_descr(&arr.dtype());
            if descr.is_null() {
                dealloc(array_ptr as *mut u8, layout);
                return None;
            }

            // Convert shape and strides
            let shape: Vec<Py_ssize_t> = arr.shape().iter().map(|&s| s as Py_ssize_t).collect();
            let strides: Vec<Py_ssize_t> = arr.strides().iter().map(|&s| s as Py_ssize_t).collect();

            // Compute flags
            let mut flags = NpyArrayFlags::NPY_ARRAY_ALIGNED | NpyArrayFlags::NPY_ARRAY_WRITEABLE;
            if arr.is_c_contiguous() {
                flags |= NpyArrayFlags::NPY_ARRAY_C_CONTIGUOUS;
            }
            if arr.is_f_contiguous() {
                flags |= NpyArrayFlags::NPY_ARRAY_F_CONTIGUOUS;
            }

            // Create storage (must live as long as the wrapper)
            let mut storage = Box::new(ArrayStorage {
                shape,
                strides,
                descr,
                _array: arr,
            });

            // Initialize the PyArrayObject
            (*array_ptr).ob_refcnt = 1;
            (*array_ptr).ob_type = ptr::null_mut(); // Would need actual PyArray_Type
            (*array_ptr).data = storage._array.data_ptr() as *mut c_char;
            (*array_ptr).nd = storage._array.ndim() as c_int;
            (*array_ptr).dimensions = storage.shape.as_mut_ptr();
            (*array_ptr).strides = storage.strides.as_mut_ptr();
            (*array_ptr).base = ptr::null_mut();
            (*array_ptr).descr = storage.descr;
            (*array_ptr).flags = flags;
            (*array_ptr).weakreflist = ptr::null_mut();
            (*array_ptr)._buffer_info = ptr::null_mut();
            (*array_ptr).mem_handler = ptr::null_mut();

            Some(CArrayWrapper {
                array: array_ptr,
                storage,
            })
        }
    }

    /// Get the raw PyArrayObject pointer.
    ///
    /// The pointer is valid as long as this wrapper exists.
    pub fn as_ptr(&self) -> *mut PyArrayObject {
        self.array
    }

    /// Get the raw PyArrayObject pointer as a generic PyObject pointer.
    pub fn as_pyobject(&self) -> *mut pyo3::ffi::PyObject {
        self.array as *mut pyo3::ffi::PyObject
    }
}

impl Drop for CArrayWrapper {
    fn drop(&mut self) {
        unsafe {
            // Free the descriptor
            free_descr(self.storage.descr);

            // Free the PyArrayObject
            if !self.array.is_null() {
                let layout = Layout::new::<PyArrayObject>();
                dealloc(self.array as *mut u8, layout);
            }
        }
    }
}

// =============================================================================
// Array creation functions (C-API style)
// =============================================================================

/// Create a new array with uninitialized data.
///
/// This is the Rust implementation of PyArray_SimpleNew.
pub fn simple_new(shape: &[Py_ssize_t], typenum: c_int) -> Option<CArrayWrapper> {
    let dtype = typenum_to_dtype(typenum)?;
    let shape_usize: Vec<usize> = shape.iter().map(|&s| s as usize).collect();
    let arr = RumpyArray::zeros(shape_usize, dtype);
    CArrayWrapper::from_rumpy(arr)
}

/// Create a new array from existing data (view, no copy).
///
/// This is the Rust implementation of PyArray_SimpleNewFromData.
///
/// # Safety
/// The data pointer must remain valid for the lifetime of the returned array.
/// The caller is responsible for ensuring this.
pub unsafe fn simple_new_from_data(
    shape: &[Py_ssize_t],
    typenum: c_int,
    data: *mut c_void,
) -> Option<*mut PyArrayObject> {
    // This is complex because we need to create an array that views external data.
    // For now, we'll just return None as this requires deeper integration.
    // A full implementation would need to create a RumpyArray that wraps external memory.
    let _ = (shape, typenum, data);
    None
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_conversion_roundtrip() {
        let dtypes = vec![
            DType::bool(),
            DType::int8(),
            DType::int16(),
            DType::int32(),
            DType::int64(),
            DType::uint8(),
            DType::uint16(),
            DType::uint32(),
            DType::uint64(),
            DType::float32(),
            DType::float64(),
            DType::complex64(),
            DType::complex128(),
        ];

        for dtype in dtypes {
            let typenum = dtype_to_typenum(&dtype);
            let converted = typenum_to_dtype(typenum).unwrap();
            assert_eq!(dtype.kind(), converted.kind());
        }
    }

    #[test]
    fn test_carray_wrapper_basic() {
        let arr = RumpyArray::zeros(vec![3, 4], DType::float64());
        let wrapper = CArrayWrapper::from_rumpy(arr).unwrap();

        unsafe {
            let ptr = wrapper.as_ptr();
            assert!(!ptr.is_null());
            assert_eq!((*ptr).nd, 2);

            // Check dimensions
            let dims = (*ptr).dimensions;
            assert_eq!(*dims, 3);
            assert_eq!(*dims.offset(1), 4);

            // Check itemsize via descr
            let descr = (*ptr).descr;
            assert_eq!((*descr).elsize, 8); // float64
            assert_eq!((*descr).type_num, NpyTypes::NPY_DOUBLE);
        }
    }

    #[test]
    fn test_simple_new() {
        let shape: [Py_ssize_t; 2] = [2, 3];
        let wrapper = simple_new(&shape, NpyTypes::NPY_FLOAT).unwrap();

        unsafe {
            let ptr = wrapper.as_ptr();
            assert_eq!((*ptr).nd, 2);

            let descr = (*ptr).descr;
            assert_eq!((*descr).elsize, 4); // float32
        }
    }
}
