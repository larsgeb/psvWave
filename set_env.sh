export PYBIND_INCLUDES=`python3 -c'import pybind11;print(pybind11.get_include())'`
export PYTHON_INCLUDES=`python3 -c"from sysconfig import get_paths as gp; print(gp()[\"include\"])"`
export SUFFIX=`python3-config --extension-suffix`
