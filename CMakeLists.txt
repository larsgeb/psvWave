cmake_minimum_required(VERSION 3.5)
project(hmc-forward-fd C CXX)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_CXX_STANDARD 11)
#set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lopenblas -fopenmp")
set(CMAKE_CXX_FLAGS_RELEASE  "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} -Wall -Wextra -Wshadow -Wnon-virtual-dtor -pedantic") # Wan to debug?
#set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} -Werror") # Wan to debug really bad?

include_directories(../armadillo-code/include)

add_executable(forwardObs src/forwardObservedData.cpp src/forward-psv/propagator.cpp src/forward-psv/propagator.h src/forward-psv/model.h
        src/forward-psv/model.cpp src/forward-psv/experiment.h src/forward-psv/experiment.cpp src/forward-psv/shot.cpp
        src/forward-psv/shot.h)
add_executable(calculateKernel src/calculateKernel.cpp src/forward-psv/propagator.cpp src/forward-psv/propagator.h src/forward-psv/model.h
        src/forward-psv/model.cpp src/forward-psv/experiment.h src/forward-psv/experiment.cpp src/forward-psv/shot.cpp
        src/forward-psv/shot.h)
add_executable(fdKernelTest src/tests/fdKernel.cpp src/forward-psv/propagator.cpp src/forward-psv/propagator.h src/forward-psv/model.h
        src/forward-psv/model.cpp src/forward-psv/experiment.h src/forward-psv/experiment.cpp src/forward-psv/shot.cpp
        src/forward-psv/shot.h)

target_link_libraries(forwardObs openblas)
target_link_libraries(calculateKernel openblas)
target_link_libraries(fdKernelTest openblas)
