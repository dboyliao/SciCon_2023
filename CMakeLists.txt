project(uSuperPoint CXX)
cmake_minimum_required(VERSION 3.10)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_STANDARD 11)

# Hack, handling no-RTTI is a pain but needed for embedded systems
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fno-rtti")

add_subdirectory(uTensor)
add_executable(usuperpoint main.cpp superpoint.cpp)
target_link_libraries(usuperpoint PUBLIC utensor)
target_compile_features(usuperpoint PUBLIC cxx_std_11)
