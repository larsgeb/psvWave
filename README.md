# psvWave M1 GPU

This is the experimental GPU version of psvWave. Not suited for production, only used as 
a research code. It accompanies "Seamless GPU acceleration for C++ based physics using
the M1's unified processing units, a case study for elastic wave propagation and
full-waveform inversion".

Setup instruction when working with VSCode on your M1 Mac:

1. Install Xcode and Xcode developer tools, per [these instructions](https://developer.apple.com/metal/cpp/).

2. Install homebrew CLang and OMP:

```console
brew install llvm
brew install libomp
```

3. Clone this repo (this specific branch!).

4. Open the new repo in VSCode.

5. Install Conda and a Python 3.8 environment, activate it.

6. Run the appropriate build `(SHIFT+COMMAND+B)`, choose from:

   a. Build shared library with Python (for Python module).
   
   b. Build GPU benchmark (for C++ benchmark).

   These tasks automatically build the appropriate Metal code.

6. Run the tests. 

For C++:

```console
cd build
export OMP_NUM_THREADS=8
./benchmark.x
```

For Python:

a. Install Jupyter notebook, NumPy, Matplotlib (using Conda)

b. Start a server in the build directory

c. Run either notebook


[![GitHub](https://img.shields.io/github/license/larsgeb/psvWave?color=4dc71f)](https://github.com/larsgeb/psvWave/blob/master/LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6216312.svg)](https://doi.org/10.5281/zenodo.6216312)

