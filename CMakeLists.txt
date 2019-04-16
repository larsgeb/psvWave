cmake_minimum_required(VERSION 3.10)
project(forward-virieux)

add_executable(forward_test tests/forward_test.cpp src/fdWaveModel.cpp src/fdWaveModel.h)

add_executable(cpuWaveOOP src/deterministicTest.cpp src/fdWaveModel.cpp src/fdWaveModel.h)
add_executable(cpuWaveOOPMPI src/deterministicTestMPI.cpp src/fdWaveModel.cpp src/fdWaveModel.h)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

find_package(MPI REQUIRED)
# Link against MPI
target_link_libraries(cpuWaveOOPMPI ${MPI_LIBRARIES})

# Include MPI includes
include_directories(${MPI_INCLUDE_PATH})
if (MPI_COMPILE_FLAGS)
    set_target_properties(cpuWaveOOPMPI PROPERTIES
            COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif ()
if (MPI_LINK_FLAGS)
    set_target_properties(cpuWaveOOPMPI PROPERTIES
            LINK_FLAGS "${MPI_LINK_FLAGS}")
endif ()
