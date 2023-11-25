add_executable(cifar10_cnn main.cpp cifar10_cnn.cpp)
target_link_libraries(cifar10_cnn PRIVATE utensor PRIVATE ${OpenCV_LIBS})
target_include_directories(cifar10_cnn PRIVATE ${OpenCV_INCLUDE_DIRS})
