cmake_minimum_required(VERSION 3.10)
project(forward-virieux)

message("PYBIND_INCLUDES=${PYBIND_INCLUDES}")
message("PYTHON_INCLUDES=${PYTHON_INCLUDES}")
message("SUFFIX=${SUFFIX}")

set(CMAKE_CXX_FLAGS "")
set(CMAKE_CXX_FLAGS_DEBUG "-g -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# Create test executables
add_executable(test_file_constructor tests/test_file_constructor.cpp src/fdModel.cpp src/contiguous_arrays.h src/fdModel.h)
add_executable(test_variable_constructor tests/test_variable_constructor.cpp src/fdModel.cpp src/contiguous_arrays.h src/fdModel.h)
add_executable(test_constructor_comparison tests/test_constructor_comparison.cpp src/fdModel.cpp src/contiguous_arrays.h src/fdModel.h)
add_executable(test_copy_constructor tests/test_copy_constructor.cpp src/fdModel.cpp src/contiguous_arrays.h src/fdModel.h)

# Create the python extension
add_library(psvWave_cpp SHARED src/psvWave.cpp src/fdModel.cpp src/contiguous_arrays.h src/fdModel.h)

# Include the appropriate compile time dependencies
include_directories(ext/eigen)
include_directories(ext/inih)

# Make the extension position independent
set_property(TARGET psvWave_cpp PROPERTY POSITION_INDEPENDENT_CODE ON)
set_target_properties(psvWave_cpp PROPERTIES POSITION_INDEPENDENT_CODE TRUE)


IF(DEFINED PYTHON_INCLUDES)
    MESSAGE(STATUS "Using passed PYTHON_INCLUDES: ${PYTHON_INCLUDES}")
    target_include_directories(psvWave_cpp PRIVATE ${PYTHON_INCLUDES})
ELSE()
    MESSAGE(STATUS "Loaded PYTHON_INCLUDES from environment variable: $ENV{PYTHON_INCLUDES}")
    target_include_directories(psvWave_cpp PRIVATE $ENV{PYTHON_INCLUDES})
ENDIF()

IF(DEFINED PYBIND_INCLUDES)
    MESSAGE(STATUS "Using passed PYBIND_INCLUDES: ${PYBIND_INCLUDES}")
    target_include_directories(psvWave_cpp PRIVATE ${PYBIND_INCLUDES})
ELSE()
    MESSAGE(STATUS "Loaded PYBIND_INCLUDES from environment variable: $ENV{PYBIND_INCLUDES}")
    target_include_directories(psvWave_cpp PRIVATE $ENV{PYBIND_INCLUDES})
ENDIF()

IF(DEFINED SUFFIX)
    MESSAGE(STATUS "Using passed SUFFIX: ${SUFFIX}")
    set_target_properties(psvWave_cpp PROPERTIES SUFFIX "${SUFFIX}")
ELSE()
    MESSAGE(STATUS "Loaded SUFFIX from environment variable: $ENV{SUFFIX}")
    set_target_properties(psvWave_cpp PROPERTIES SUFFIX "$ENV{SUFFIX}")
ENDIF()

set_target_properties(psvWave_cpp PROPERTIES PREFIX "")
set_target_properties(psvWave_cpp PROPERTIES OUTPUT_NAME "__psvWave_cpp")

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}") 