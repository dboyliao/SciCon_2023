#ifndef UTENSOR_FFT_HPP
#define UTENSOR_FFT_HPP

#include "uTensor/core/uPAL.hpp"
#if UT_ARCH(UT_ARCH_X86)

#include "fftw3.h"
 #include "uTensor/core/operatorBase.hpp"

namespace uTensor {
// fftw_plan fftw_plan_dft_r2c_1d(int n, double *in, fftw_complex *out, unsigned
// flags);

class PowerSpectrum : public OperatorInterface<1, 1>, FastOperator {
 public:
  enum names : uint8_t { input, power };
  PowerSpectrum(int N);

 protected:
  virtual void compute();
  void power_spectrum_kernel(Tensor& power, const Tensor& input, int N);

 private:
  int N;
};
}  // namespace uTensor
#endif //platform guard
#endif
