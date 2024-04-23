#pragma once

#include <vector>

#include "openfhe.h"

namespace fhenom {
using shape_t = std::vector<size_t>;
using Ctxt = lbcrypto::Ciphertext<lbcrypto::DCRTPoly>;
using Ptxt = lbcrypto::Plaintext;
using PrecomputedRotations = std::shared_ptr<std::vector<lbcrypto::DCRTPoly>>;
}  // namespace fhenom
