cmake_minimum_required(VERSION 3.10)
project(forward-virieux)

add_executable(forward_test tests/forward_test.cpp src/fdWaveModel.cpp src/contiguous_arrays.h src/fdWaveModel.h)
add_library(pyWave SHARED src/pyWave.cpp src/fdWaveModel.cpp src/contiguous_arrays.h src/fdWaveModel.h)

set_property(TARGET pyWave PROPERTY POSITION_INDEPENDENT_CODE ON)

target_include_directories(forward_test PRIVATE ext/eigen)
target_include_directories(forward_test PRIVATE ext/inih)

target_include_directories(pyWave PRIVATE ext/eigen)
target_include_directories(pyWave PRIVATE ext/inih)


find_package(PythonLibs REQUIRED)
target_include_directories(pyWave PRIVATE ${PYTHON_INCLUDE_DIRS})

MESSAGE(STATUS "Suffix: $ENV{SUFFIX}")

IF(DEFINED ENV{PYBIND_INCLUDES})
    MESSAGE(STATUS "Loaded PyBind includes from environment variable: $ENV{PYBIND_INCLUDES}")
    target_include_directories(pyWave PRIVATE $ENV{PYBIND_INCLUDES})
ELSE()
    MESSAGE(STATUS "Using default PyBind includes")
    target_include_directories(pyWave PRIVATE /home/larsgebraad/.local/lib/python3.8/site-packages/pybind11/include)
ENDIF()

set_target_properties(pyWave PROPERTIES PREFIX "")
set_target_properties(pyWave PROPERTIES OUTPUT_NAME "pyWave")
set_target_properties(pyWave PROPERTIES SUFFIX $ENV{SUFFIX}")


find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

