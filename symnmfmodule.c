#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include "symnmf.h"

/** Helper function to free 2D arrays **/
void Free_Double2DArray(double** array, int n) {
    for (int i = 0; i < n; i++) {
        free(array[i]);
    }
    free(array);
}

/** Helper function to convert Python list to C array **/
double** PyList_To_Double2DArray(PyObject* list_obj, int n, int d) {
    // Check if the object is a list
    if (!PyList_Check(list_obj)) {
        PyErr_SetString(PyExc_TypeError, "Expected a list object");
        return NULL;
    }

    // Check if the size matches 'n'
    if (PyList_Size(list_obj) != n) {
        PyErr_SetString(PyExc_ValueError, "Mismatch in number of rows");
        return NULL;
    }

    double** array = safe_malloc_matrix(n, d);
    if (!array) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory");
        return NULL;
    }

    for (int i = 0; i < n; i++) {
        PyObject* row_list = PyList_GetItem(list_obj, i);

        // Check if the object is a list and its size matches 'd'
        if (!PyList_Check(row_list) || PyList_Size(row_list) != d) {
            PyErr_SetString(PyExc_ValueError, "Expected a row of the correct size");
            // Don't forget to free the previously allocated memory
            for (int j = 0; j < i; j++) {
                free(array[j]);
            }
            free(array);
            return NULL;
        }

        for (int j = 0; j < d; j++) {
            PyObject* num = PyList_GetItem(row_list, j);
            array[i][j] = PyFloat_AsDouble(num);

            // Check for errors after attempting to convert to float
            if (PyErr_Occurred()) {
                // Free memory
                for (int k = 0; k <= i; k++) {
                    free(array[k]);
                }
                free(array);
                return NULL;
            }
        }
    }

    return array;
}


/** Helper function to convert C array to Python list **/
PyObject* Double2DArray_To_PyList(double** array, int n, int d) {
    PyObject* result_list = PyList_New(n);
    for (int i = 0; i < n; i++) {
        PyObject* row_list = PyList_New(d);
        for (int j = 0; j < d; j++) {
            PyList_SetItem(row_list, j, PyFloat_FromDouble(array[i][j]));
        }
        PyList_SetItem(result_list, i, row_list);
    }
    return result_list;
}

/** Wrapper for the sym function **/
static PyObject* py_sym(PyObject* self, PyObject* args) {
    PyObject* list_obj;
    int n, d;
    if (!PyArg_ParseTuple(args, "Oii", &list_obj, &n, &d)) return NULL;

    double** X = PyList_To_Double2DArray(list_obj, n, d);
    if (!X) return NULL;

    double** result = sym(X, n, d);
    if (!result) {
        Free_Double2DArray(X, n);
        return NULL;
    }

    PyObject* result_list = Double2DArray_To_PyList(result, n, d);
    Free_Double2DArray(X, n);
    Free_Double2DArray(result, n);

    return result_list;
}

/** Wrapper for the ddg function **/
static PyObject* py_ddg(PyObject* self, PyObject* args) {
    PyObject* list_obj;
    int n, d;
    if (!PyArg_ParseTuple(args, "Oii", &list_obj, &n, &d)) return NULL;

    double** X = PyList_To_Double2DArray(list_obj, n, d);
    if (!X) return NULL;

    double** result = ddg(X, n, d);
    if (!result) {
        Free_Double2DArray(X, n);
        return NULL;
    }

    PyObject* result_list = Double2DArray_To_PyList(result, n, d);
    Free_Double2DArray(X, n);
    Free_Double2DArray(result, n);

    return result_list;
}

/** Wrapper for the norm function **/
static PyObject* py_norm(PyObject* self, PyObject* args) {
    PyObject* list_obj;
    int n, d;
    if (!PyArg_ParseTuple(args, "Oii", &list_obj, &n, &d)) return NULL;

    double** X = PyList_To_Double2DArray(list_obj, n, d);
    if (!X) return NULL;

    double** result = norm(X, n, d);
    if (!result) {
        Free_Double2DArray(X, n);
        return NULL;
    }

    PyObject* result_list = Double2DArray_To_PyList(result, n, d);
    Free_Double2DArray(X, n);
    Free_Double2DArray(result, n);

    return result_list;
}

/** Wrapper for the symnmf function **/
static PyObject* py_symnmf(PyObject* self, PyObject* args) {
    PyObject *list_W, *list_H_initial;
    int n, k;
    if (!PyArg_ParseTuple(args, "OOii", &list_W, &list_H_initial, &n, &k)) return NULL;

    double** W = PyList_To_Double2DArray(list_W, n, k);
    if (!W) return NULL;

    double** H_initial = PyList_To_Double2DArray(list_H_initial, n, k);
    if (!H_initial) {
        Free_Double2DArray(W, n);
        return NULL;
    }

    double** result = symnmf(W, H_initial, n, k);
    if (!result) {
        Free_Double2DArray(W, n);
        Free_Double2DArray(H_initial, n);
        return NULL;
    }

    PyObject* result_list = Double2DArray_To_PyList(result, n, k);
    Free_Double2DArray(W, n);
    Free_Double2DArray(H_initial, n);
    Free_Double2DArray(result, n);

    return result_list;
}

/** Module's function definition structure **/
static PyMethodDef symnmfmethods[] = {
    {"sym", py_sym, METH_VARARGS, "Calculate sym matrix"},
    {"ddg", py_ddg, METH_VARARGS, "Calculate ddg matrix"},
    {"norm", py_norm, METH_VARARGS, "Normalize matrix"},
    {"symnmf", py_symnmf, METH_VARARGS, "Calculate symnmf"},
    {NULL, NULL, 0, NULL}
};

/** Module definition structure **/
static struct PyModuleDef symnmfmodule = {
    PyModuleDef_HEAD_INIT,
    "symnmf",      // name of module
    NULL,          // module documentation, may be NULL
    -1,            // size of per-interpreter state of the module, or -1 if the module keeps state in global variables
    symnmfmethods
};

/** Module initialization function **/
PyMODINIT_FUNC PyInit_symnmf(void) {
    return PyModule_Create(&symnmfmodule);
}

