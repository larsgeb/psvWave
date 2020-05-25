cmake_minimum_required(VERSION 3.10)
project(forward-virieux)

add_executable(forward_test tests/forward_test.cpp src/fdWaveModel.cpp src/fdWaveModel.h)
add_library(pyWave SHARED src/pyWave.cpp src/fdWaveModel.cpp src/fdWaveModel.h)
add_executable(adjoint_test tests/adjoint_test.cpp src/fdWaveModel.cpp src/fdWaveModel.h)
add_executable(xcor_test tests/xcor_test.cpp src/fdWaveModel.cpp src/fdWaveModel.h)
add_executable(kernel_test tests/kernel_on_background_test.cpp src/fdWaveModel.cpp src/fdWaveModel.h)

set_property(TARGET pyWave PROPERTY POSITION_INDEPENDENT_CODE ON)

target_include_directories(forward_test PRIVATE ext/eigen)
target_include_directories(forward_test PRIVATE ext/inih)

target_include_directories(pyWave PRIVATE ext/eigen)
target_include_directories(pyWave PRIVATE ext/inih)
target_include_directories(pyWave PRIVATE /usr/include/python3.8)
target_include_directories(pyWave PRIVATE /home/larsgebraad/.local/lib/python3.8/site-packages/pybind11/include)

set_target_properties(pyWave PROPERTIES PREFIX "")
set_target_properties(pyWave PROPERTIES OUTPUT_NAME "pyWave")
set_target_properties(pyWave PROPERTIES SUFFIX ".cpython-38-x86_64-linux-gnu.so")


find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

