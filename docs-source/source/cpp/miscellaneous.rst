Miscellaneous methods and definitions
=====================================

.. role:: raw-html(raw)
    :format: html


Making contiguous arrays
************************

In general, allocating arrays requires one to know the size of all dimensions of the
array, whereas de-allocating requires you to know the dimensions all but the inner
dimension size.

The canonical way to allocate and deallocate arrays (for e.g. floats) in C++ is as
follows:

.. code-block:: cpp

   float* array_1d = new float[number_of_elements]; 
   delete[] array_1d;

This is *totally fine* for 1 dimensional arrays. It can also be easily extended to 
higher dimensional arrays. In 2D, the implementation would look something like this:

.. code-block:: cpp

   float** array_2d = new float*[number_of_rows];
   for(int i = 0; i < number_of_rows; ++i)
       array_2d[i] = new float[number_of_columns];

Again, this is *fine* for 2 dimensional arrays. But not totally. What happens here is
that basically we run the code from the previous code block for 1D arrays as many times
as we have rows, so we allocate number_of_rows 1D arrays. This does however not
guarantee that all these arrays are contiguous in memory. They can be all over the
place. This might impact performance of the finite difference code.

What we want, is that the entire 2D (or higher D) array is in one piece in the computers
memory. To create space for the entire array, one can simply allocate a 1D array with as
many elements as the higher order dimensions array would be. This helps, because 1D
arrays are guaranteed to be contiguous.

.. code-block:: cpp

   float* fake_2d_which_is_actually_1d = new float[number_of_rows * number_of_columns]; 
   delete[] fake_2d_which_is_actually_1d;

However, this breaks the nice interface we would normally have for higher order arrays:

.. code-block:: cpp

   fake_2d_which_is_actually_1d[0][5]; // Does not work
   fake_2d_which_is_actually_1d[0 + 5 * number_of_rows]; // Does actually work

Here we see the need for linear indexing when we create arrays like this. To re-enable
the 'nice' behaviour of 2D arrays, one can still create a 2D array of the required type,
and do the pointer arithmetic manually, s.t. it refers to the locations in the large 1D 
array. That is what the allocate_array() functions do. They are implemented for 1
through 4 dimensional arrays.

Example usage
-------------

Below is an example on how to create, access and destroy a 1D array of float. Any other
type can be used, since the definitions are templated.

.. code-block:: cpp

    float *t;
    int nt = 100;
    allocate_array(t, nt);

    t[50] = 3.0;

    deallocate_array(t); // Remember to deallocate, otherwise you get memory leaks.

The same thing, but in 4 dimensions is given below.

.. code-block:: cpp

    float ****t;
    int nt = 100;
    int nx = 100;
    int ny = 100;
    int ns = 100;
    allocate_array(t, nt, nx, ny, ns);

    t[50][25][74][1] = 3.0;

    deallocate_array(t); // Remember to deallocate, otherwise you get memory leaks.

Allocation and deallocation functions
-------------------------------------

.. doxygenfunction:: allocate_array(T *&pointer, int dim1)
   :project: psvWave

.. doxygenfunction:: deallocate_array(T *&pointer)
   :project: psvWave

.. doxygenfunction:: allocate_array(T **&pointer, int dim1, int dim2)
   :project: psvWave

.. doxygenfunction:: deallocate_array(T **&pointer)
   :project: psvWave

.. doxygenfunction:: allocate_array(T ***&pointer, int dim1, int dim2, int dim3)
   :project: psvWave

.. doxygenfunction:: deallocate_array(T ***&pointer)
   :project: psvWave

.. doxygenfunction:: allocate_array(T ****&pointer, int dim1, int dim2, int dim3, int dim4)
   :project: psvWave

.. doxygenfunction:: deallocate_array(T ****&pointer)
   :project: psvWave


Parse functions
***************

.. doxygenfunction:: parse_string_to_vector
   :project: psvWave

.. doxygenfunction:: parse_string_to_nested_int_vector
   :project: psvWave



