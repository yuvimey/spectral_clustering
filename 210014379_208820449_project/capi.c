#define PY_SSIZE_T_CLEAN
#include "Python.h"
#include "capi.h"

/*To main.py - 
This file transports our program in C to
something that main.py can interpret and
return the final clusterings*/

static double* get_double_array(PyObject * list, int rows, int cols){
    /*For Python: returns a matrix from a Python list*/
	int i,j;
	double *new_list;
	new_list = new_malloc(rows * cols, sizeof(double));
	if(new_list == NULL){
		return NULL;
	}
	PyObject *row;
	PyObject *val;
	for(i = 0; i < rows; i++){
		row = PyList_GetItem(list, i);
		Py_INCREF(row);
		if (!PyList_Check(row)){
			printf("PyList_Check(row) failed\n");
			continue;
		}
		for(j = 0; j < cols; j++){
			val = PyList_GetItem(row, j);
			Py_INCREF(val);
			new_list[i*cols+j] = PyFloat_AsDouble(val);
			if (new_list[i*cols+j]  == -1 && PyErr_Occurred()){
				puts("Error occured parsing array from python to c. \n");
				new_free(new_list);
				return NULL;
			}
			Py_DECREF(val);
			val = NULL;
		}
		Py_DECREF(row);
		row = NULL;
	}
	return new_list;
}

static PyObject* create_pylist_from_array(int* array, int array_len){
    /*For Python: turns an array into a Python list*/
	int i;
    PyObject* pylist = PyList_New(array_len);
	if(pylist == NULL){
		return NULL;
	}
    for( i=0; i< array_len; i++){
        if(0 != PyList_SetItem(pylist, i, Py_BuildValue("i", array[i]))){
            fprintf(stderr,"bad PyList_SetItem value %d : %d\n",i,array[i]);
        }
    }
    return pylist;
}


static PyObject* clustering_capi(PyObject *self, PyObject *args){
    /*For Python: insert Normal Spectral Clustering into main.py*/
    int k, n, d, max_iter;
    double epsilon;
	int malloc_start = counter;
	int malloc_end;
	PyObject *obs_array_py;
    if(!PyArg_ParseTuple(args, "iiiifO", &k, &n, &d, &max_iter, &epsilon, &obs_array_py)) {
        return NULL;
    }
	double *obs_array = get_double_array(obs_array_py, n, d);
	if(obs_array == NULL){
		fprintf(stderr,"K means failed! exiting...\n");
		exit(1);
	}
	int* k_means_assign = k_means_pp(obs_array, k, n, d, max_iter, epsilon);
	new_free(obs_array);
	if(k_means_assign == NULL){
		fprintf(stderr,"K means failed! exiting...\n");
		exit(1);
	}
    PyObject *buildval = Py_BuildValue("O", create_pylist_from_array(k_means_assign, n));
	new_free(k_means_assign);
	if(buildval == NULL){
		fprintf(stderr,"K means failed! exiting...\n");
		exit(1);
	}
	malloc_end = counter;
	if(malloc_end - malloc_start != 0){
		fprintf(stderr,"ERROR: memory leak!\n");
		fprintf(stderr,"ERROR: allocation count: %d\n",(malloc_end - malloc_start));
	}
    return buildval;
}

static PyMethodDef methods[] = {
    /*For Python: define methods*/
    {"clustering_capi",
      (PyCFunction) clustering_capi,
      METH_VARARGS,
      PyDoc_STR("Calculating the means of K clusters")},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    /*For Python: define module*/
    PyModuleDef_HEAD_INIT,
    "capi",
    NULL,
    -1,
    methods
};

PyMODINIT_FUNC
PyInit_capi(void)
{
    /*For Python: create module*/
    PyObject *m;
    m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }
    return m;
}