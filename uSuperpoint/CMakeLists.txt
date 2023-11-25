add_executable(usuperpoint main.cpp superpoint.cpp)
target_link_libraries(usuperpoint PRIVATE utensor PRIVATE ${OpenCV_LIBS})
target_include_directories(usuperpoint PRIVATE ${OpenCV_INCLUDE_DIRS})