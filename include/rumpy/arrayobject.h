/*
 * Rumpy C-API Header
 *
 * NumPy-compatible C-API for Rumpy arrays.
 *
 * Usage:
 *   #include <rumpy/arrayobject.h>
 *
 *   // In module init:
 *   import_rumpy_array();
 *
 *   // Then use PyArray_* functions as with NumPy
 */

#ifndef RUMPY_ARRAYOBJECT_H
#define RUMPY_ARRAYOBJECT_H

#include <Python.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ==========================================================================
 * Type Numbers (must match NpyTypes in structs.rs)
 * ========================================================================== */

enum NPY_TYPES {
    NPY_BOOL = 0,
    NPY_BYTE = 1,
    NPY_UBYTE = 2,
    NPY_SHORT = 3,
    NPY_USHORT = 4,
    NPY_INT = 5,
    NPY_UINT = 6,
    NPY_LONG = 7,
    NPY_ULONG = 8,
    NPY_LONGLONG = 9,
    NPY_ULONGLONG = 10,
    NPY_FLOAT = 11,
    NPY_DOUBLE = 12,
    NPY_LONGDOUBLE = 13,
    NPY_CFLOAT = 14,
    NPY_CDOUBLE = 15,
    NPY_CLONGDOUBLE = 16,
    NPY_OBJECT = 17,
    NPY_STRING = 18,
    NPY_UNICODE = 19,
    NPY_VOID = 20,
    NPY_DATETIME = 21,
    NPY_TIMEDELTA = 22,
    NPY_HALF = 23,

    /* Aliases for sized types (64-bit platform) */
    NPY_INT8 = NPY_BYTE,
    NPY_UINT8 = NPY_UBYTE,
    NPY_INT16 = NPY_SHORT,
    NPY_UINT16 = NPY_USHORT,
    NPY_INT32 = NPY_INT,
    NPY_UINT32 = NPY_UINT,
    NPY_INT64 = NPY_LONGLONG,
    NPY_UINT64 = NPY_ULONGLONG,
    NPY_FLOAT32 = NPY_FLOAT,
    NPY_FLOAT64 = NPY_DOUBLE,
    NPY_COMPLEX64 = NPY_CFLOAT,
    NPY_COMPLEX128 = NPY_CDOUBLE,
    NPY_FLOAT16 = NPY_HALF,

    NPY_INTP = NPY_INT64,
    NPY_UINTP = NPY_UINT64
};

/* ==========================================================================
 * Array Flags
 * ========================================================================== */

#define NPY_ARRAY_C_CONTIGUOUS    0x0001
#define NPY_ARRAY_F_CONTIGUOUS    0x0002
#define NPY_ARRAY_OWNDATA         0x0004
#define NPY_ARRAY_ALIGNED         0x0100
#define NPY_ARRAY_WRITEABLE       0x0400
#define NPY_ARRAY_WRITEBACKIFCOPY 0x2000

/* Common combinations */
#define NPY_ARRAY_BEHAVED   (NPY_ARRAY_ALIGNED | NPY_ARRAY_WRITEABLE)
#define NPY_ARRAY_CARRAY    (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_BEHAVED)
#define NPY_ARRAY_FARRAY    (NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_BEHAVED)

/* Byte order */
#define NPY_LITTLE '<'
#define NPY_BIG    '>'
#define NPY_NATIVE '='
#define NPY_IGNORE '|'

#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
#define NPY_NATBYTE NPY_LITTLE
#else
#define NPY_NATBYTE NPY_BIG
#endif

/* ==========================================================================
 * Data Type Descriptor
 * ========================================================================== */

typedef struct _PyArray_Descr {
    PyObject_HEAD
    PyTypeObject *typeobj;
    char kind;
    char type;
    char byteorder;
    char _former_flags;
    int type_num;
    uint64_t flags;
    Py_ssize_t elsize;
    Py_ssize_t alignment;
    PyObject *metadata;
    Py_hash_t hash;
    void *reserved_null[2];
} PyArray_Descr;

/* ==========================================================================
 * Array Object Structure
 * ========================================================================== */

typedef struct tagPyArrayObject {
    PyObject_HEAD
    char *data;
    int nd;
    Py_ssize_t *dimensions;
    Py_ssize_t *strides;
    PyObject *base;
    PyArray_Descr *descr;
    int flags;
    PyObject *weakreflist;
    void *_buffer_info;
    PyObject *mem_handler;
} PyArrayObject;

/* For NumPy 1.x compatibility */
typedef PyArrayObject PyArrayObject_fields;

/* npy_intp is Py_ssize_t on most platforms */
typedef Py_ssize_t npy_intp;

/* ==========================================================================
 * Accessor Macros
 * These provide direct struct access (fast path)
 * ========================================================================== */

#define PyArray_NDIM(arr) (((PyArrayObject *)(arr))->nd)
#define PyArray_DATA(arr) ((void *)(((PyArrayObject *)(arr))->data))
#define PyArray_BYTES(arr) (((PyArrayObject *)(arr))->data)
#define PyArray_DIMS(arr) (((PyArrayObject *)(arr))->dimensions)
#define PyArray_STRIDES(arr) (((PyArrayObject *)(arr))->strides)
#define PyArray_DIM(arr, n) (((PyArrayObject *)(arr))->dimensions[(n)])
#define PyArray_STRIDE(arr, n) (((PyArrayObject *)(arr))->strides[(n)])
#define PyArray_DESCR(arr) (((PyArrayObject *)(arr))->descr)
#define PyArray_FLAGS(arr) (((PyArrayObject *)(arr))->flags)
#define PyArray_BASE(arr) (((PyArrayObject *)(arr))->base)

#define PyArray_TYPE(arr) (PyArray_DESCR(arr)->type_num)
#define PyArray_ITEMSIZE(arr) ((int)(PyArray_DESCR(arr)->elsize))

/* Flag checking macros */
#define PyArray_ISCONTIGUOUS(arr) ((PyArray_FLAGS(arr) & NPY_ARRAY_C_CONTIGUOUS) != 0)
#define PyArray_ISFORTRAN(arr)    ((PyArray_FLAGS(arr) & NPY_ARRAY_F_CONTIGUOUS) != 0)
#define PyArray_ISWRITEABLE(arr)  ((PyArray_FLAGS(arr) & NPY_ARRAY_WRITEABLE) != 0)
#define PyArray_ISALIGNED(arr)    ((PyArray_FLAGS(arr) & NPY_ARRAY_ALIGNED) != 0)

/* Pointer access macros */
#define PyArray_GETPTR1(arr, i) \
    ((void *)(PyArray_BYTES(arr) + (i) * PyArray_STRIDES(arr)[0]))

#define PyArray_GETPTR2(arr, i, j) \
    ((void *)(PyArray_BYTES(arr) + (i) * PyArray_STRIDES(arr)[0] + \
                                   (j) * PyArray_STRIDES(arr)[1]))

#define PyArray_GETPTR3(arr, i, j, k) \
    ((void *)(PyArray_BYTES(arr) + (i) * PyArray_STRIDES(arr)[0] + \
                                   (j) * PyArray_STRIDES(arr)[1] + \
                                   (k) * PyArray_STRIDES(arr)[2]))

/* Size macros */
static inline npy_intp PyArray_SIZE(PyArrayObject *arr) {
    npy_intp size = 1;
    int i;
    for (i = 0; i < arr->nd; i++) {
        size *= arr->dimensions[i];
    }
    return size;
}

#define PyArray_NBYTES(arr) (PyArray_ITEMSIZE(arr) * PyArray_SIZE((PyArrayObject*)(arr)))

/* ==========================================================================
 * Type Checking Macros
 * ========================================================================== */

#define PyTypeNum_ISBOOL(t)     ((t) == NPY_BOOL)
#define PyTypeNum_ISINTEGER(t)  ((t) >= NPY_BYTE && (t) <= NPY_ULONGLONG)
#define PyTypeNum_ISFLOAT(t)    ((t) == NPY_HALF || (t) == NPY_FLOAT || \
                                 (t) == NPY_DOUBLE || (t) == NPY_LONGDOUBLE)
#define PyTypeNum_ISCOMPLEX(t)  ((t) == NPY_CFLOAT || (t) == NPY_CDOUBLE || \
                                 (t) == NPY_CLONGDOUBLE)
#define PyTypeNum_ISNUMBER(t)   (PyTypeNum_ISINTEGER(t) || PyTypeNum_ISFLOAT(t) || \
                                 PyTypeNum_ISCOMPLEX(t))
#define PyTypeNum_ISSIGNED(t)   ((t) == NPY_BYTE || (t) == NPY_SHORT || \
                                 (t) == NPY_INT || (t) == NPY_LONG || \
                                 (t) == NPY_LONGLONG)
#define PyTypeNum_ISUNSIGNED(t) ((t) == NPY_UBYTE || (t) == NPY_USHORT || \
                                 (t) == NPY_UINT || (t) == NPY_ULONG || \
                                 (t) == NPY_ULONGLONG)

/* Array type checking */
#define PyArray_ISBOOL(arr)     PyTypeNum_ISBOOL(PyArray_TYPE(arr))
#define PyArray_ISINTEGER(arr)  PyTypeNum_ISINTEGER(PyArray_TYPE(arr))
#define PyArray_ISFLOAT(arr)    PyTypeNum_ISFLOAT(PyArray_TYPE(arr))
#define PyArray_ISCOMPLEX(arr)  PyTypeNum_ISCOMPLEX(PyArray_TYPE(arr))
#define PyArray_ISNUMBER(arr)   PyTypeNum_ISNUMBER(PyArray_TYPE(arr))

/* ==========================================================================
 * API Table (for functions that aren't simple macros)
 * ========================================================================== */

/* API table indices */
#define PYARRAY_TYPE_IDX           2
#define PYARRAY_DESCR_TYPE_IDX     3
#define PYARRAY_DESCRFROMTYPE_IDX  45
#define PYARRAY_NDIM_IDX           74
#define PYARRAY_DIM_IDX            75
#define PYARRAY_STRIDE_IDX         76
#define PYARRAY_DATA_IDX           77
#define PYARRAY_NEWFROMDESCR_IDX   94
#define PYARRAY_SIMPLENEW_IDX      96
#define PYARRAY_SIMPLENEWFROMDATA_IDX 97

#define API_TABLE_SIZE 400

/* Global API table pointer */
static void **RumpyArray_API = NULL;

/* ==========================================================================
 * Import Function
 * ========================================================================== */

/* Import the Rumpy array API */
static int import_rumpy_array(void) {
    PyObject *module = PyImport_ImportModule("rumpy");
    if (module == NULL) {
        PyErr_SetString(PyExc_ImportError, "Failed to import rumpy");
        return -1;
    }

    PyObject *c_api = PyObject_GetAttrString(module, "_ARRAY_API");
    if (c_api == NULL) {
        Py_DECREF(module);
        PyErr_SetString(PyExc_ImportError, "rumpy._ARRAY_API not found");
        return -1;
    }

    if (!PyCapsule_CheckExact(c_api)) {
        Py_DECREF(c_api);
        Py_DECREF(module);
        PyErr_SetString(PyExc_RuntimeError, "rumpy._ARRAY_API is not a capsule");
        return -1;
    }

    RumpyArray_API = (void **)PyCapsule_GetPointer(c_api, "rumpy._ARRAY_API");
    Py_DECREF(c_api);
    Py_DECREF(module);

    if (RumpyArray_API == NULL) {
        return -1;
    }

    return 0;
}

/* For drop-in NumPy compatibility */
#define import_array() import_rumpy_array()

/* ==========================================================================
 * Function Declarations (via API table or direct link)
 * ========================================================================== */

/* These can be called via the API table or linked directly if building
 * against librumpy */

/* Declared in Rust, available for direct linking */
extern int PyArray_NDIM(PyArrayObject *arr);
extern void* PyArray_DATA(PyArrayObject *arr);
extern Py_ssize_t* PyArray_DIMS(PyArrayObject *arr);
extern Py_ssize_t* PyArray_STRIDES(PyArrayObject *arr);
extern Py_ssize_t PyArray_DIM(PyArrayObject *arr, int n);
extern Py_ssize_t PyArray_STRIDE(PyArrayObject *arr, int n);
extern int PyArray_TYPE(PyArrayObject *arr);
extern int PyArray_ITEMSIZE(PyArrayObject *arr);
extern int PyArray_FLAGS(PyArrayObject *arr);
extern Py_ssize_t PyArray_SIZE(PyArrayObject *arr);
extern Py_ssize_t PyArray_NBYTES(PyArrayObject *arr);

/* Type checking functions */
extern int PyTypeNum_ISINTEGER(int typenum);
extern int PyTypeNum_ISFLOAT(int typenum);
extern int PyTypeNum_ISCOMPLEX(int typenum);
extern int PyTypeNum_ISNUMBER(int typenum);
extern int PyTypeNum_ISSIGNED(int typenum);
extern int PyTypeNum_ISUNSIGNED(int typenum);
extern int PyTypeNum_ISBOOL(int typenum);

extern int PyArray_ISCONTIGUOUS(PyArrayObject *arr);
extern int PyArray_ISFORTRAN(PyArrayObject *arr);
extern int PyArray_ISWRITEABLE(PyArrayObject *arr);
extern int PyArray_ISALIGNED(PyArrayObject *arr);

extern int PyArray_ISINTEGER(PyArrayObject *arr);
extern int PyArray_ISFLOAT(PyArrayObject *arr);
extern int PyArray_ISCOMPLEX(PyArrayObject *arr);
extern int PyArray_ISNUMBER(PyArrayObject *arr);
extern int PyArray_ISBOOL(PyArrayObject *arr);

/* Pointer access */
extern void* PyArray_GETPTR1(PyArrayObject *arr, Py_ssize_t i);
extern void* PyArray_GETPTR2(PyArrayObject *arr, Py_ssize_t i, Py_ssize_t j);
extern void* PyArray_GETPTR3(PyArrayObject *arr, Py_ssize_t i, Py_ssize_t j, Py_ssize_t k);

#ifdef __cplusplus
}
#endif

#endif /* RUMPY_ARRAYOBJECT_H */
