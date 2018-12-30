cmake_minimum_required(VERSION 3.10)
project(gpuWave)

SET(CMAKE_C_COMPILER /home/lars/Applications/pgi/linux86-64/2018/bin/pgcc)
SET(CMAKE_CXX_COMPILER /home/lars/Applications/pgi/linux86-64/2018/bin/pgc++)

add_executable(gpuWave main.cpp)
add_executable(cpuWave main.cpp)

# 1 for double, 2 for float
add_definitions(-DREALTYPE=2)
set_target_properties(gpuWave PROPERTIES COMPILE_DEFINITIONS "OPENACCCOMPILE=1")

set(CMAKE_CXX_FLAGS_RELEASE "-fast -O3")

set_target_properties(gpuWave PROPERTIES COMPILE_OPTIONS "-Minfo=accel;-acc;-ta=tesla:cc60")
# Counter-intuitive, target link libraries is also able to pass flags. Fixed in CMake 3.13 using target_link_options
target_link_libraries(gpuWave -acc)

set_target_properties(cpuWave PROPERTIES COMPILE_OPTIONS "-Minfo=mp;-mp")

