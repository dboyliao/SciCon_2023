#include <iostream>

#include "cifar10_cnn.hpp"
#include "opencv2/opencv.hpp"
#include "uTensor.h"

using namespace uTensor;
static localCircularArenaAllocator<512> meta_alloc;
static localCircularArenaAllocator<5000> ram_alloc;

int main(int argc, char const *argv[]) {
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " <img>" << std::endl;
    return 1;
  }
  Context::get_default_context()->set_metadata_allocator(&meta_alloc);
  Context::get_default_context()->set_ram_data_allocator(&ram_alloc);

  cv::Mat cv_img = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
  uint16_t rows = cv_img.rows, cols = cv_img.cols;
  Tensor t_img = new RamTensor({1, rows, cols, 1}, flt);
  Tensor logits = new RamTensor({1, 10}, flt);
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      t_img(1, r, c) = static_cast<float>(cv_img.at<uint8_t>(0, 0));
    }
  }
  Cifar10Cnn cifar10_cnn;
  cifar10_cnn.set_inputs({{Cifar10Cnn::input_0, t_img}})
      .set_outputs({{Cifar10Cnn::output_0, logits}})
      .eval();
  for (int i = 0; i < 10; ++i) {
    float val = static_cast<float>(logits(i));
    std::cout << val << ", ";
  }
  std::cout << std::endl;
  Context::get_default_context()->set_metadata_allocator(&meta_alloc);
  Context::get_default_context()->set_ram_data_allocator(&ram_alloc);
  return 0;
}
