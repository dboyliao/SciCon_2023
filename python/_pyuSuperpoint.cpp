#include <iostream>

#include "cstddef"
#include "pybind11/pybind11.h"
#include "superpoint.hpp"

namespace py = pybind11;

void talk(void) {
  std::cout << "Hello plugin with pybind11!" << std::endl;
  return;
}

PYBIND11_MODULE(_pyuSuperpoint, m) {
  m.doc() = "pybind11 superpoint plugin";  // optional module docstring
  m.def("talk", &talk, "A function returning hello");
}
