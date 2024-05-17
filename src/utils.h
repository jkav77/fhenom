#include <utility>
#include <fhenom/ckks_vector.h>
#include <fhenom/tensor.h>
#include <fhenom/common.h>

namespace fhenom {
std::pair<bool, std::string> validate_conv2d_input(const fhenom::Tensor& kernel, const fhenom::Tensor& bias,
                                                   const fhenom::shape_t& shape);
std::vector<fhenom::CkksVector> rotate_images(const fhenom::CkksVector& data, fhenom::shape_t kernel_shape);
}  // namespace fhenom
