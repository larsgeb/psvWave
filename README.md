# forward-virieux

![Main GitHub Language](https://img.shields.io/github/languages/top/larsgeb/forward-virieux) [![Build Status](https://travis-ci.com/larsgeb/forward-virieux.svg?branch=master)](https://travis-ci.com/larsgeb/forward-virieux) ![License](https://img.shields.io/github/license/larsgeb/forward-virieux) ![](https://img.shields.io/pypi/dm/psvWave)

Parallel numerical FD simulation for P-SV wave, using the velocity-stress formulation on a staggered grid from Virieux 1986. Requires only OpenMP and the git subrepos (header-only). Used as a PDE-simulation code for [this publication](https://doi.org/10.1029/2019JB018428).

# Compilation for C++

```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make forward_test # or other targets
```

# Python interface

Compiling the CMake target `psvWave` creates a Python module for the current environments Python version. However, to find the right requirements, we first need to do two things:

1. Install PyBind, to interface Python to C++, using your favourite package manager;
1. Set relevant environment variables.

Before you continue, make sure you are in your desired Python environment, e.g. Conda or PyEnv. **Python 3.6** or higher is recommended. Also, the `python3-dev` package is needed for compilation.

### Installing PyBind11

Using your relevant package manager, install `pybind11`, e.g.:

```bash
pip3 install pybind11
```

### Setting needed environment variables

The compiler needs three things to work correctly:

1. the relevant PyBind files (headers);
2. the relevant Python files (headers);
3. the appropriate extension for the compiled file.

The CMakeLists.txt file loads these variables from the environment. If you know what you are doing, you can set these yourself. If not, run the following commands in the terminal in which you have activated your relevant Python environment:

```bash
export PYBIND_INCLUDES=`python3 -c'import pybind11;print(pybind11.get_include())'`
export PYTHON_INCLUDES=`python3 -c"from sysconfig import get_paths as gp; print(gp()[\"include\"])"`
export SUFFIX=`python3-config --extension-suffix`
```

### Compiling the Python linked C++ code

Compiling the Python interface is done by running CMake:

```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make psvWave
```

### Using the Python interface

This interface can be used by having the resulting `psvWave.*.so` file in your working directory, and importing it, e.g.:

```python
import psvWave

model = psvWave.fdModel(
    "../tests/test_configurations/forward_configuration.ini")

model.forward_shot(0, verbose=True, store_fields=True)

```
