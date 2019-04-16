cmake_minimum_required(VERSION 3.10)
project(forward-virieux)

add_executable(forward_test tests/forward_test.cpp src/fdWaveModel.cpp src/fdWaveModel.h)

add_executable(cpuWaveOOP src/deterministicTest.cpp src/fdWaveModel.cpp src/fdWaveModel.h)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

