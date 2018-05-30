cmake_minimum_required(VERSION 3.5)
project(hmc-forward-fd C CXX)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_CXX_STANDARD 11)
#set(CMAKE_BUILD_TYPE Debug)
#set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lopenblas -fopenmp")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -Wextra -Wshadow -Wnon-virtual-dtor -pedantic") # Wan to debug?
#set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} -Werror") # Wan to debug really bad?

include_directories(../armadillo-code/include)

add_executable(calculateKernel src/calculateKernel.cpp src/forward-psv/fwiPropagator.cpp src/forward-psv/fwiPropagator.h src/forward-psv/fwiModel.h
        src/forward-psv/fwiModel.cpp src/forward-psv/fwiExperiment.h src/forward-psv/fwiExperiment.cpp src/forward-psv/fwiShot.cpp
        src/forward-psv/fwiShot.h src/misc/functions.cpp src/misc/functions.h)
add_executable(fdTest src/tests/fdTest.cpp src/forward-psv/fwiPropagator.cpp src/forward-psv/fwiPropagator.h src/forward-psv/fwiModel.h
        src/forward-psv/fwiModel.cpp src/forward-psv/fwiExperiment.h src/forward-psv/fwiExperiment.cpp src/forward-psv/fwiShot.cpp
        src/forward-psv/fwiShot.h src/misc/functions.cpp src/misc/functions.h)

add_executable(waveSimulation src/waveSimulation.cpp src/forward-psv/fwiPropagator.cpp src/forward-psv/fwiPropagator.h src/forward-psv/fwiModel.h
        src/forward-psv/fwiModel.cpp src/forward-psv/fwiExperiment.h src/forward-psv/fwiExperiment.cpp src/forward-psv/fwiShot.cpp
        src/forward-psv/fwiShot.h src/misc/functions.cpp src/misc/functions.h)

target_link_libraries(calculateKernel openblas)
target_link_libraries(fdTest openblas)
