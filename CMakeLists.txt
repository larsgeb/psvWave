cmake_minimum_required(VERSION 3.5)
project(hmc-forward-fd)
set (CMAKE_CXX_FLAGS "-O2")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_CXX_STANDARD 11)

#set(CMAKE_BUILD_TYPE release)
set(CMAKE_BUILD_TYPE debug)
#set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_C_FLAGS "fopenmp")

find_package(LAPACK)
find_package(Armadillo REQUIRED)
#include_directories(${ARMADILLO_INCLUDE_DIRS})
#add_executable(forward src/main.cpp)
add_executable(forwardAc src/forward-acoustic.cpp)
add_executable(forwardEl src/forward-elastic.cpp)
target_link_libraries(forwardAc ${lapackblas_libraries})
target_link_libraries(forwardAc ${ARMADILLO_LIBRARIES})

