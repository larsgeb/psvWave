cmake_minimum_required(VERSION 3.5)
project(hmc-forward-fd C CXX)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lopenblas -fopenmp")

include_directories(../armadillo-code/include)

add_executable(main src/main.cpp src/forward-psv/propagator.cpp src/forward-psv/propagator.h src/forward-psv/model.h
        src/forward-psv/model.cpp src/forward-psv/experiment.h src/forward-psv/experiment.cpp src/forward-psv/shot.cpp
        src/forward-psv/shot.h)
add_executable(generateSynthetics src/generateSynthetics.cpp src/forward-psv/propagator.cpp src/forward-psv/propagator.h src/forward-psv/model.h
        src/forward-psv/model.cpp src/forward-psv/experiment.h src/forward-psv/experiment.cpp src/forward-psv/shot.cpp
        src/forward-psv/shot.h)

target_link_libraries(main openblas)
target_link_libraries(generateSynthetics openblas)
