#pragma once

#include <openfhe.h>

#include <vector>

namespace fhenom {
using shape_t              = std::vector<unsigned>;
using Ctxt                 = lbcrypto::Ciphertext<lbcrypto::DCRTPoly>;
using Ptxt                 = lbcrypto::Plaintext;
using PrecomputedRotations = std::shared_ptr<std::vector<lbcrypto::DCRTPoly>>;

static const std::vector<double> f3_coeffs{0.0, 2.1875, 0.0, -2.1875, 0.0, 1.3125, 0.0, -0.3125};
static const std::vector<double> g3_coeffs{0.0, 4.477073170731708, 0.0, -16.1884765625,
                                           0.0, 25.013671875,      0.0, -12.55859375};
}  // namespace fhenom
