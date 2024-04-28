#pragma once

#include <openfhe.h>

#include <vector>

namespace fhenom {
using shape_t              = std::vector<unsigned>;
using Ctxt                 = lbcrypto::Ciphertext<lbcrypto::DCRTPoly>;
using Ptxt                 = lbcrypto::Plaintext;
using PrecomputedRotations = std::shared_ptr<std::vector<lbcrypto::DCRTPoly>>;
}  // namespace fhenom
