cmake_minimum_required(VERSION 3.5)
project(hmc-forward-fd C CXX)

#list(APPEND CMAKE_CXX_SOURCE_FILE_EXTENSIONS c)
SET(CMAKE_CXX_COMPILER /cluster/apps/openmpi/1.6.5/x86_64/gcc_4.8.2/bin/mpic++)

set(CMAKE_PREFIX_PATH /cluster/home/lgebraad/Arma-8.4-install/usr)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
#set(CMAKE_CXX_STANDARD 11)

#set(CMAKE_C_FLAGS " ${CMAKE_C_FLAGS} -fopenmp -lopenblas -llapack -std=c++0x")
#set(CMAKE_CXX_FLAGS "-std=c++11 -g -O3 ${CMAKE_CXX_FLAGS}")

#set(CMAKE_BUILD_TYPE release)
#set(CMAKE_BUILD_TYPE debug)

#find_package(OpenMP)
#find_package(BLAS REQUIRED)
#find_package(LAPACK REQUIRED)
find_package(Armadillo REQUIRED)

include_directories(${ARMADILLO_INCLUDE_DIRS})
add_executable(forwardAc src/forward-acoustic.cpp)
add_executable(forwardEl src/forward-elastic.cpp)

target_link_libraries(forwardEl ${ARMADILLO_LIBRARIES})

