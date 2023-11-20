#include <iostream>

#include "opencv2/opencv.hpp"
#include "superpoint.hpp"
#include "uTensor.h"

using namespace uTensor;
static localCircularArenaAllocator<512> meta_alloc;
static localCircularArenaAllocator<100000, uint32_t> ram_alloc;

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " <img>" << std::endl;
    return 1;
  }
  Context::get_default_context()->set_metadata_allocator(&meta_alloc);
  Context::get_default_context()->set_ram_data_allocator(&ram_alloc);

  cv::Mat cv_img = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
  uint16_t rows = cv_img.rows, cols = cv_img.cols;
  Tensor t_img = new RamTensor({rows, cols, 1}, flt);
  Tensor out_encode = new RamTensor({1, 15, 11, 256}, flt);
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      t_img(r, c) = static_cast<float>(cv_img.at<uint8_t>(0, 0));
    }
  }
  Superpoint sp;
  sp.set_inputs({{Superpoint::input_0, t_img}})
      .set_outputs({{Superpoint::output_0, out_encode}})
      .eval();
  return 0;
}
