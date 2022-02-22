# psvWave

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/psvWave)
[![Main GitHub Language](https://img.shields.io/github/languages/top/larsgeb/forward-virieux)](https://github.com/larsgeb/psvWave)
[![GitHub](https://img.shields.io/github/license/larsgeb/psvWave?color=4dc71f)](https://github.com/larsgeb/psvWave/blob/master/LICENSE)
[![PyPI version](https://badge.fury.io/py/psvWave.svg)](https://pypi.org/project/psvWave/)
[![](https://img.shields.io/badge/docs-latest-brightgreen)](https://larsgeb.github.io/psvWave/)

https://user-images.githubusercontent.com/21038893/155033663-5e22b5ee-1b2b-43eb-9aab-74bfd85f2a73.mp4

![Gradients](https://user-images.githubusercontent.com/21038893/155048974-63ce3dd3-5c07-4650-a7f2-9c2105151642.png)

[Check out the notebook here!](https://github.com/larsgeb/psvWave/blob/master/notebooks/L-BFGS%20FWI.ipynb)

A Python/C++ package for 2D P-SV wave propagation using finite differences and OpenMP.
This package was written to facilitate high-throughput numerical wave simulations for
Monte Carlo simulation in Seismology.
It uses the velocity-stress formulation on a staggered grid from [Virieux's
classical 1986 paper](https://doi.org/10.1190/1.1442147).
For compilation we require only OpenMP and the git subrepos (header-only):
[Eigen](http://eigen.tuxfamily.org) and [inih](https://github.com/jtilly/inih),
however installation can be easily done through pip.
Used as a PDE-simulation code for
[this publication](https://doi.org/10.1029/2019JB018428).


## Extremely quick start using Docker

The best way to run the code if you are not on Linux. Pull the docker image and start
the notebook server on port 7999. Feel free to change the port 7999 to something of your
preference.

```bash
docker run -it -p 7999:8888 larsgebraad/psvwave
```

You can then navigate in your browser to `localhost:7999`. This starts you right off 
with some fun notebooks! 

Note that the notebooks assume you have 6 cores available for your docker. You can
specify how many cores are available by starting the Docker the following way:

```bash
docker run -it --cpus=2 -p 7999:8888 larsgebraad/psvwave
```

## Installing the package using PyPi (pip)

There are many ways to install this package.
Installing directly from the PyPi archives is arguably the easiest:

```bash
pip install psvWave
```

To check if everything worked correctly, you can run the following in an interactive python shell or notebook:

```python
>>> import psvWave
>>> print(psvWave.__version__)
```

If this raises an `ImportError`, the C++ packages have not correctly compiled and you
are either on an unsupported system (Windows/MacOS) or I have made a terrible mistake.
Please contact me in any case!

## Getting started

## Working with the configuration files

**This section is a must read for anyone wishing to use this package.**

An example configuration file is given below.
The simulations performed make a few basic assumptions about the medium, wavefield and
sources:

### 1: Physics

All sources propagate waves through the same medium / domain, and are recorded by
the same network.
The physics is defined in a **right-handed** coordinate system.
However, you are allowed to interpret the simulations in any unit and orientation
you like.
Just make sure you keep track of the units, and don't use numbers outside the range
of either `float` or `double` (the package is by default compiled with doubles).
And, the physics is for **in-plane** shear waves.

### 2: Sources

All sources are normal/reverse faults (with strike parallel to the y-axis) with a
Ricker wavelet of all the same frequency as source time function.
Every source can have a different dip angle.
This source time function can be altered in both the Python and C++ API, the focal
mechanism /source type not (yet).

### 3: Simulations

Simulations are divided in 'shots', i.e. a single time length in which data is
recorded and some 'sources' fire.
It is thus allowed to have 2 sources in a single shot.
This allows for source stacking.
The delay_cycles_per_shot variable allows for time staggering, delaying the source
time function per source by that many cycles.
An example relevant to the given configuration file:

```
peak_frequency = 50.0
```

Means all source time functions (STF) are a Ricker wavelet with peak (central)
frequency of 50Hz.

```
delay_cycles_per_shot = 24
```

Means that if 2 sources are present in a shot, the STF of the second shot is delayed
by 24 cycles. For a peak frequency of 50 Hz, this turns out to be
`24cycl / 50cycl/s = 0.48s`. Every subsequent shot is delayed after the previous by
the same amount.

```
which_source_to_fire_in_which_shot = {{0, 1}}
```

Means that both source 0 and source 1 (zero-based indexing) are fired in shot 1.

In the below given configuration, total simulation time is 1 second.
This means that the second shot is 'fired' at almost half the simulation time.
The idea behind source stacking is that without strong reflections, we can take
advantage of the position of the wavefields to simulate multiple shots at the same
time, with minimal 'cross-talk'.

### 4: Domain and boundary

The domain is truncated on all 4 sides by absorbing boundary conditions.
It's width is variable, but as of yet, the same on all sides.
This does not directly allow for free boundary conditions, but this is planned to
change.
When measuring distance or counting gridpoints, the zero-point is the first points
not inside the boundary layer but in the actual simulation medium.
When updating medium properties within the domain, the boundary copies the medium
properties closest to it, to avoid creating reflectors.

### 5: Indexing

The location of the sources and receivers is not expressed in distance, but in
gridpoint numbering.
Because the actual indexing starts within the medium, and not the absorbing
boundary, sources and receivers can only be placed inside the medium.
However, the nx_inner_boundary and nz_inner_boundary variables determine how many
gridpoints are not considered free parameters.
The idea behind this is that this allows us to place sources/receivers in regions
of the domain that are not inverted for, and are also not inside the boundary.
This to avoid near-source and near-receiver effects.

### Example file

```
[domain]
nt = 4000
nx_inner = 200
nz_inner = 100
nx_inner_boundary = 10
nz_inner_boundary = 20
dx = 1.249
dz = 1.249
dt = 0.00025

[boundary]
np_boundary = 25
np_factor = 0.015

[medium]
scalar_rho = 1500.0
scalar_vp = 2000.0
scalar_vs = 800.0

[sources]
peak_frequency = 50.0
n_sources = 2
n_shots = 1
source_timeshift = 0.005
delay_cycles_per_shot = 24
moment_angles = {90, 180}
ix_sources = {25, 175}
iz_sources = {10, 10}
which_source_to_fire_in_which_shot = {{0, 1}}

[receivers]
nr = 19
ix_receivers = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190}
iz_receivers = {90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90}

[inversion]
snapshot_interval = 10
```

## Installing from GitHub

You might also be tempted to install it from tags or another GitHub hash.
This has the problem however that submodules are not automatically downloaded.
If you still wish to install from the repo, you have to clone it to your machine first,
and then also pull all the submodules:

```bash
git clone --recursive https://github.com/larsgeb/psvWave.git
cd psvWave
```

Afterwards you can ...

1. directly install from this directory:

   ```bash
   pip install -e .
   ```

2. create a source distribution (uncompiled) and install it on _any_ machine:

   ```bash
   python setup.py sdist
   cd dist
   pip install psvWave-*.tar.gz # this will compile the C++ modules
   ```

3. create a binary wheel in which the compiled code is present and install it on
   _similar_ machines:

   ```bash
   python setup.py bdist_wheel # this will compile the C++ modules
   cd dist
   pip install psvWave-*.whl
   ```

The main difference between 2 and 3 is that 2 doesn't compile the C++ code yet at the
distribution stage.
Option 3 does compile in this stage, and therefore might not work on machines with
wildly different architectures.

If you are really at the end of your rope, we can also send a precompiled wheel for the
platform you're using.

## Compiling the C++ interface

Compiling the C++ API into your C++ application is fairly straightforward. One needs
an OpenMP enabled compiler, `cmake` installed and C++11 support. The different
targets are defined in the CMakeLists.txt. Assuming your in a local clone of this repo:

```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make forward_test # or other targets
```

## Compiling the Python interface yourself

Compiling the CMake target `psvWave` creates a Python module for the current
environments' Python version. However, to find the right requirements, we first need to
do two things:

1. Install PyBind, to interface Python to C++, using your favourite package manager;
1. Set the relevant environment variables.

Before you continue, make sure you are in your desired Python environment, e.g. Conda or
PyEnv. **Python 3.6** or higher is recommended. Also, the `python3-dev` or equivalent
package is needed for compilation.

### Installing development dependencies.

Using your relevant package manager, you need to install all required development
dependencies. To at minimum compile the interface, `cmake` and `pybind11` are required:

```bash
pip install pybind11
pip install cmake
```

However, you might want to perform code-formatting and run tests. To install all the
dependencies for this, it might be easier to install them using the `setup.py` file.
Make sure you are in the cloned repo folder and run the following:

```bash
# On Bash / SH
pip install -e .[dev]
# On ZSH
pip install -e .\[dev\]
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

Compiling the Python interface is done by running CMake in the cloned repo:

```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make psvWave
```

### Using the Python interface

This interface can be used by having the resulting `__psvWave_cpp.*.so` file in your
working directory or PATH variable, and importing it, e.g.:

```python
import __psvWave_cpp

model = __psvWave_cpp.fdModel(
    "../tests/test_configurations/forward_configuration.ini")

model.forward_shot(0, verbose=True, store_fields=True)

```

However, the files in the `./psvWave/` Python module provides an interface that is a
little neater with additional functions. Place the .so file in this folder and have this
folder in your path.

## Compiling the documentation

The documentation for the Python and C++ API requires one extra thing after running
`pip install -e .[dev]`; A locally installed **doxygen**, to parse the C++ API into a
Sphinx readable structure (a bunch of XML files, really.
Installing this is a little platform dependent, with a quick
`install doxygen <platform>` typically being enough.
On e.g. Ubuntu the command to run would be:

```bash
$ sudo apt-get install doxygen
```

For compiling the total documentation the following needs to be run out of the local git clone:

```bash
$ cd docs-source
$ rm build/ -rf
$ make html
$ touch build/html/.nojekyll
```

The entire content of the `docs-source/build/html` directory, together with an empty file `.nojekyll`, is used as the gh-pages branch.
