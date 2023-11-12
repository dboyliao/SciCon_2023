pybind11_add_module(_pyuSuperpoint _pyuSuperPoint.cpp ${PROJECT_SOURCE_DIR}/superpoint.cpp)
target_include_directories(_pyuSuperpoint PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(_pyuSuperpoint PRIVATE utensor)
