C++ API reference
=================

The true bulk fo this package is written in the class fdModel.
This class takes care of the wavefield simulations.
Additionally, it is able to backpropagate a wavefield.
By storing the forward propagation and correlating it during a backward (or adjoint) 
simulation, we can compute sensitivity kernels.

Typical usage should be done in Python. 
However, if you want to interface directly with the finite difference code, you can work
with the C++ API.
It gives direct access to the dynamical fields, sensitivity kernels, etc.
This C++ API is exposed in Python using PyBind11.

Here and there we need to do some type casting between Python and C++ objects. 
This additional layer is not documented in this reference (as it is not part of the C++
api).
It can be found in `src/psvWave.cpp`

Using the C++ API also requires the configuration files to be present at time of
instance construction. 
In other words, everytime we make an `fdModel` object, we need to have a properly
formatted `conf.ini` file.
How to create and use these is described in the Python API reference.

.. toctree::
   :maxdepth: 1
   :caption: Important parts

   fdmodel/index
   miscellaneous

Type definitions
################

Throughout the C++ code, we have a few important type definitions.
This makes working with varying precision and eigen types a little more concise.

.. doxygentypedef:: real_simulation
   :project: psvWave

.. doxygentypedef:: dynamic_vector
   :project: psvWave
