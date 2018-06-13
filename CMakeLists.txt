cmake_minimum_required(VERSION 3.5)
project(hmc-forward-fd C CXX)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lopenblas -fopenmp")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -Wextra -Wshadow -Wnon-virtual-dtor -pedantic") # Wan to debug?

include_directories(../armadillo-code/include)

add_executable(waveSimulation
        src/waveSimulation.cpp
        src/forward-psv/fwiPropagator.h
        src/forward-psv/fwiPropagator.cpp
        src/forward-psv/fwiModel.h
        src/forward-psv/fwiModel.cpp
        src/forward-psv/fwiExperiment.h
        src/forward-psv/fwiExperiment.cpp
        src/forward-psv/fwiShot.h
        src/forward-psv/fwiShot.cpp
        src/misc/functions.cpp
        src/misc/functions.h)

# - Config file for the Armadillo package
# It defines the following variables
#  ARMADILLO_INCLUDE_DIRS - include directories for Armadillo
#  ARMADILLO_LIBRARY_DIRS - library directories for Armadillo (normally not used!)
#  ARMADILLO_LIBRARIES    - libraries to link against

set(ARMADILLO_INSTALL_FOLDER "~/armadillo-install")

# Tell the user project where to find our headers and libraries
set(ARMADILLO_INCLUDE_DIRS "${ARMADILLO_INSTALL_FOLDER}/include")
set(ARMADILLO_LIBRARY_DIRS "${ARMADILLO_INSTALL_FOLDER}/lib")

# Our library dependencies (contains definitions for IMPORTED targets)
include("${ARMADILLO_INSTALL_FOLDER}/share/Armadillo/CMake/ArmadilloLibraryDepends.cmake")

# These are IMPORTED targets created by ArmadilloLibraryDepends.cmake
set(ARMADILLO_LIBRARIES armadillo)

target_link_libraries(waveSimulation armadillo)