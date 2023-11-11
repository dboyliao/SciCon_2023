project(uSuperPoint CXX)
cmake_minimum_required(VERSION 3.10)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_STANDARD 11)

add_subdirectory(uTensor)
add_executable(usuperpoint main.cpp superpoint.cpp)
target_link_libraries(usuperpoint PUBLIC utensor)
