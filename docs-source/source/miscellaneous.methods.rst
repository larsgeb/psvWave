Miscellaneous methods
=====================

.. role:: raw-html(raw)
    :format: html

Allocation and de-allocation functions
**************************************

In general, allocating arrays requires one to know the size of all dimensions of the array, whereas de-allocating requires you to know the all but
the inner dimension size.

.. doxygenfunction:: allocate_1d_array
   :project: forward-virieux

.. doxygenfunction:: deallocate_1d_array
   :project: forward-virieux

.. doxygenfunction:: allocate_2d_array
   :project: forward-virieux

.. doxygenfunction:: deallocate_2d_array
   :project: forward-virieux

.. doxygenfunction:: allocate_3d_array
   :project: forward-virieux

.. doxygenfunction:: deallocate_3d_array
   :project: forward-virieux

.. doxygenfunction:: allocate_4d_array
   :project: forward-virieux

.. doxygenfunction:: deallocate_4d_array
   :project: forward-virieux



Parse functions
***************

.. doxygenfunction:: parse_string_to_vector
   :project: forward-virieux

.. doxygenfunction:: parse_string_to_nested_int_vector
   :project: forward-virieux

Signal processing functions
***************************

.. doxygenfunction:: cross_correlate
   :project: forward-virieux



