Methods for performing FWI
==========================

Below is the documentation for the methods that are most relevant when performing 
full-waveform inversion using the psv package in C++. These are all methods of 
fdModel instances.

.. doxygenfunction:: forward_simulate
   :project: psvWave

.. doxygenfunction:: adjoint_simulate
   :project: psvWave

.. doxygenfunction:: write_receivers
   :project: psvWave

.. doxygenfunction:: write_sources
   :project: psvWave

.. doxygenfunction:: load_receivers
   :project: psvWave

.. doxygenfunction:: load_model
   :project: psvWave

.. doxygenfunction:: map_kernels_to_velocity
   :project: psvWave

.. doxygenfunction:: reset_kernels
   :project: psvWave

.. doxygenfunction:: update_from_velocity
   :project: psvWave

.. doxygenfunction:: calculate_l2_misfit
   :project: psvWave

.. doxygenfunction:: calculate_l2_adjoint_sources
   :project: psvWave

.. doxygenfunction:: run_model(bool, bool)
   :project: psvWave

