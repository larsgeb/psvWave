cmake_minimum_required(VERSION 3.10)
project(gpuWave)

add_executable(cpuWaveOOP src/deterministicTest.cpp src/fdWaveModel.cpp src/fdWaveModel.h)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

find_package(MPI REQUIRED)
# Link against MPI
target_link_libraries(cpuWaveOOP ${MPI_LIBRARIES})

# Include MPI includes
include_directories(${MPI_INCLUDE_PATH})
if(MPI_COMPILE_FLAGS)
    set_target_properties(cpuWaveOOP PROPERTIES
            COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()
if(MPI_LINK_FLAGS)
    set_target_properties(cpuWaveOOP PROPERTIES
            LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()
