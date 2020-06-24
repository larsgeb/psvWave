cmake_minimum_required(VERSION 3.10)
project(forward-virieux)

message("PYBIND_INCLUDES=${PYBIND_INCLUDES}")
message("PYTHON_INCLUDES=${PYTHON_INCLUDES}")
message("EXTENSION=${EXTENSION}")

set(CMAKE_CXX_FLAGS "")
set(CMAKE_CXX_FLAGS_DEBUG "-g -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

add_executable(forward_test tests/forward_test.cpp src/fdWaveModel.cpp src/contiguous_arrays.h src/fdWaveModel.h)
add_library(pyWave_cpp SHARED src/pyWave.cpp src/fdWaveModel.cpp src/contiguous_arrays.h src/fdWaveModel.h)

set_property(TARGET pyWave_cpp PROPERTY POSITION_INDEPENDENT_CODE ON)
set_target_properties(pyWave_cpp PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

target_include_directories(forward_test PRIVATE ext/eigen)
target_include_directories(forward_test PRIVATE ext/inih)

target_include_directories(pyWave_cpp PRIVATE ext/eigen)
target_include_directories(pyWave_cpp PRIVATE ext/inih)

IF(DEFINED ENV{PYTHON_INCLUDES})
    MESSAGE(STATUS "Loaded PYTHON_INCLUDES from environment variable: $ENV{PYTHON_INCLUDES}")
    target_include_directories(pyWave_cpp PRIVATE $ENV{PYTHON_INCLUDES})
ELSE()
    MESSAGE(STATUS "Using passed PYTHON_INCLUDES")
    target_include_directories(pyWave_cpp PRIVATE ${PYTHON_INCLUDES})
ENDIF()

IF(DEFINED ENV{PYBIND_INCLUDES})
    MESSAGE(STATUS "Loaded PYBIND_INCLUDES from environment variable: $ENV{PYBIND_INCLUDES}")
    target_include_directories(pyWave_cpp PRIVATE $ENV{PYBIND_INCLUDES})
ELSE()
    MESSAGE(STATUS "Using passed PYBIND_INCLUDES")
    target_include_directories(pyWave_cpp PRIVATE ${PYBIND_INCLUDES})
ENDIF()

IF(DEFINED ENV{SUFFIX})
    MESSAGE(STATUS "Loaded Suffix from environment variable: $ENV{SUFFIX}")
    set_target_properties(pyWave_cpp PROPERTIES SUFFIX $ENV{SUFFIX})
ELSE()
    MESSAGE(STATUS "Using passed SUFFIX")
    set_target_properties(pyWave_cpp PROPERTIES SUFFIX ${EXTENSION})
ENDIF()

set_target_properties(pyWave_cpp PROPERTIES PREFIX "")
set_target_properties(pyWave_cpp PROPERTIES OUTPUT_NAME "__pyWave")

set_target_properties( pyWave_cpp
  PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/pyWave"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/pyWave"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/pyWave"
)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}") 