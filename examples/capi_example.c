/*
 * Example C extension using Rumpy's NumPy-compatible C-API.
 *
 * This demonstrates how a C extension can work with Rumpy arrays
 * using the same API it would use for NumPy.
 *
 * To compile (example, not tested):
 *   gcc -I../include -I$(python3-config --includes) \
 *       -shared -fPIC -o example.so capi_example.c \
 *       $(python3-config --ldflags)
 */

#include <rumpy/arrayobject.h>

/*
 * Example function: sum all elements in a float64 array.
 *
 * This uses PyArray_* accessors to work with the array in a
 * NumPy-compatible way.
 */
static double sum_array(PyArrayObject *arr) {
    /* Check that we got a contiguous float64 array */
    if (!PyArray_ISCONTIGUOUS(arr)) {
        return -1.0;  /* Error: not contiguous */
    }
    if (PyArray_TYPE(arr) != NPY_FLOAT64) {
        return -2.0;  /* Error: wrong dtype */
    }

    /* Get array info using standard accessors */
    int ndim = PyArray_NDIM(arr);
    npy_intp *dims = PyArray_DIMS(arr);
    double *data = (double *)PyArray_DATA(arr);

    /* Calculate total size */
    npy_intp size = 1;
    for (int i = 0; i < ndim; i++) {
        size *= dims[i];
    }

    /* Sum elements */
    double total = 0.0;
    for (npy_intp i = 0; i < size; i++) {
        total += data[i];
    }

    return total;
}

/*
 * Example function: element-wise multiply for 2D arrays.
 *
 * Demonstrates using GETPTR2 for strided access.
 */
static void multiply_2d(PyArrayObject *a, PyArrayObject *b, PyArrayObject *out) {
    npy_intp rows = PyArray_DIM(a, 0);
    npy_intp cols = PyArray_DIM(a, 1);

    for (npy_intp i = 0; i < rows; i++) {
        for (npy_intp j = 0; j < cols; j++) {
            double *pa = (double *)PyArray_GETPTR2(a, i, j);
            double *pb = (double *)PyArray_GETPTR2(b, i, j);
            double *po = (double *)PyArray_GETPTR2(out, i, j);
            *po = (*pa) * (*pb);
        }
    }
}

/*
 * Example Python function wrapper.
 *
 * This shows how you'd wrap the C function for Python.
 */
static PyObject *py_sum_array(PyObject *self, PyObject *args) {
    PyObject *arr_obj;

    if (!PyArg_ParseTuple(args, "O", &arr_obj)) {
        return NULL;
    }

    /* Check if it's an array */
    if (!PyArray_Check(arr_obj)) {
        PyErr_SetString(PyExc_TypeError, "Expected an array");
        return NULL;
    }

    double result = sum_array((PyArrayObject *)arr_obj);
    return PyFloat_FromDouble(result);
}

/*
 * Module initialization.
 *
 * For a real extension, you'd:
 * 1. Call import_array() (or import_rumpy_array())
 * 2. Define your method table
 * 3. Create the module
 */
static PyMethodDef example_methods[] = {
    {"sum_array", py_sum_array, METH_VARARGS, "Sum array elements"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef example_module = {
    PyModuleDef_HEAD_INIT,
    "capi_example",
    "Example C extension using Rumpy C-API",
    -1,
    example_methods
};

PyMODINIT_FUNC PyInit_capi_example(void) {
    /* Import the array API */
    if (import_array() < 0) {
        return NULL;
    }

    return PyModule_Create(&example_module);
}
