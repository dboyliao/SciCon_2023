#include <iostream>

#include "cifar10_cnn.hpp"
#include "opencv2/opencv.hpp"
#include "uTensor.h"

using namespace uTensor;
static localCircularArenaAllocator<512> meta_alloc;
static localCircularArenaAllocator<50000, uint32_t> ram_alloc;

int main(int argc, char const *argv[]) {
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " <img>" << std::endl;
    return 1;
  }
  Context::get_default_context()->set_metadata_allocator(&meta_alloc);
  Context::get_default_context()->set_ram_data_allocator(&ram_alloc);

  cv::Mat cv_img = cv::imread(argv[1], cv::IMREAD_COLOR);
  uint16_t rows = cv_img.rows, cols = cv_img.cols;
  Tensor t_img = new RamTensor({1, rows, cols, 3}, flt);

  Tensor logits = new RamTensor({1, 10}, flt);
  // float *data_ptr = (float *)static_cast<RamTensor
  // *>(*logits)->get_address(); float arr[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9,
  // 10}; std::memcpy(data_ptr, &arr, sizeof(float) * 10);
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      cv::Vec3b bgr_value = cv_img.at<cv::Vec3b>(r, c);
      t_img(0, r, c, 0) = static_cast<float>(bgr_value(0)) / 255.0f;
      t_img(0, r, c, 1) = static_cast<float>(bgr_value(1)) / 255.0f;
      t_img(0, r, c, 2) = static_cast<float>(bgr_value(2)) / 255.0f;
    }
  }
  Cifar10Cnn cifar10_cnn;
  cifar10_cnn.set_inputs({{Cifar10Cnn::input_0, t_img}})
      .set_outputs({{Cifar10Cnn::output_0, logits}})
      .eval();
  int max_idx = 0;
  float max_logit = static_cast<float>(logits(0, 0));
  for (int i = 0; i < 10; ++i) {
    float val = static_cast<float>(logits(0, i));
    if (val >= max_logit) {
      max_logit = val;
      max_idx = i;
    }
  }
  std::cout << "Predicted class is " << max_idx << " for " << argv[1]
            << std::endl;
  Context::get_default_context()->set_metadata_allocator(&meta_alloc);
  Context::get_default_context()->set_ram_data_allocator(&ram_alloc);
  return 0;
}
