cmake_minimum_required(VERSION 3.5)
project(hmc-forward-fd)
set (CMAKE_CXX_FLAGS "-O3")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_CXX_STANDARD 11)

#set(CMAKE_BUILD_TYPE release)
#set(CMAKE_BUILD_TYPE debug)

find_path(BLAS_INCLUDE_DIRS cblas.h
        /usr/include/openblas/)
set(BLAS_DIR /usr/lib/openblas-base/)
set(LAPACK_DIR /usr/lib/lapack/)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(Armadillo REQUIRED)

add_executable(forwardAc src/forward-acoustic.cpp)
add_executable(forwardEl src/forward-elastic.cpp)

target_link_libraries(forwardEl ${ARMADILLO_LIBRARIES})

