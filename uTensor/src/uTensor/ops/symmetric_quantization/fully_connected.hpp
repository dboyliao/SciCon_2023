#ifndef UTENSOR_S_QUANTIZED_FC_OPS_H
#define UTENSOR_S_QUANTIZED_FC_OPS_H
#include "uTensor/core/context.hpp"
#include "uTensor/core/operatorBase.hpp"
#include "uTensor/ops/Matrix.hpp"
#include "fully_connected_kernel.hpp"
#include "symmetric_quantization_utils.hpp"

namespace uTensor {
namespace TflmSymQuantOps {

template <typename T>
class QuantizedMatrixMultiplyOperator : public OperatorInterface<3, 1> {};

template <>
class QuantizedMatrixMultiplyOperator<int8_t> : public OperatorInterface<3, 1> {
 public:
  enum names_in : uint8_t { input, filter, bias };
  enum names_out : uint8_t { output };

  QuantizedMatrixMultiplyOperator(TFLM::TfLiteFusedActivation activation)
      : _activation(activation) {}

 private:
  TFLM::TfLiteFusedActivation _activation;

 protected:
  virtual void compute() {
    bool have_bias = inputs.has(bias);
    // Decide on c shape
    TensorShape& a_shape = inputs[input].tensor()->get_shape();
    TensorShape& b_shape = inputs[filter].tensor()->get_shape();
    TensorShape& c_shape = outputs[output].tensor()->get_shape();
    if (a_shape.num_dims() > 2 || b_shape.num_dims() > 2 ||
        c_shape.num_dims() > 2 || a_shape[1] != b_shape[0] ||
        a_shape[0] != c_shape[0] || b_shape[1] != c_shape[1]) {
      uTensor_printf("[Error] Invalid matrix multiple shape mismatch\n");
      Context::get_default_context()->throwError(
          new InvalidMatrixMultIndicesError);
    }
    // TODO check that all the Tensor types are int8

    // Taken from TFLu to maintain operator compatibility.
    // The scaling factor from input to output (aka the 'real multiplier') can
    // be represented as a fixed point multiplier plus a left shift.
    int32_t output_multiplier;
    int output_shift;
    // The range of the fused activation layer. For example for kNone and
    // uint8_t these would be 0 and 255.
    int32_t output_activation_min;
    int32_t output_activation_max;
    double real_multiplier = 0.0;
    int exponent;
    TFLM::GetQuantizedConvolutionMultipler(
        inputs[input].tensor(), inputs[filter].tensor(), inputs[bias].tensor(),
        outputs[output].tensor(), &real_multiplier);
    TFLM::QuantizeMultiplier(real_multiplier, &output_multiplier, &exponent);
    TFLM::CalculateActivationRangeQuantized(
        _activation, outputs[output].tensor(), &output_activation_min,
        &output_activation_max);

    output_shift = -exponent;
    // gets rid of IF case in mult loop
    if (have_bias) {
      const Tensor& b = inputs[bias].tensor();
      TFLM::quantized_matrix_mult_kernel(
          outputs[output].tensor(), inputs[input].tensor(),
          inputs[filter].tensor(), [&b](int32_t i){ return static_cast<int32_t>(b(i)); }, output_multiplier,
          output_shift, output_activation_min, output_activation_max);
    } else {
      TFLM::quantized_matrix_mult_kernel(
          outputs[output].tensor(), inputs[input].tensor(),
          inputs[filter].tensor(), [](int32_t i){ return 0; }, output_multiplier, output_shift,
          output_activation_min, output_activation_max);
    }
  }
};

template <typename Tout>
using FullyConnectedOperator = QuantizedMatrixMultiplyOperator<Tout>;

}
}  // namespace uTensor
#endif
