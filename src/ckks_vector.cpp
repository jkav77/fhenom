#include "fhenom/ckks_vector.h"

#include <openfhe.h>

#include <filesystem>
#include <vector>

// Serialization stuff
#include <ciphertext-ser.h>
#include <cryptocontext-ser.h>
#include <key/key-ser.h>
#include <scheme/ckksrns/ckksrns-ser.h>

#include <cereal/archives/binary.hpp>

using fhenom::CkksVector;
using lbcrypto::Ciphertext;
using lbcrypto::DCRTPoly;

//////////////////////////////////////////////////////////////////////////////
// Homomorphic Operations

CkksVector CkksVector::SignUsingChebyshev(const double lower_bound,
                                          const double upper_bound,
                                          uint32_t degree) const {
  auto cryptoContext = context_.getCryptoContext();

  if (size() == 0) {
    spdlog::warn("Data is empty. Comparing nothing.");
  }

  std::vector<Ctxt> result;
  for (const auto& ctxt : data_) {
    result.emplace_back(cryptoContext->EvalChebyshevFunction(
        [](double ctxt_val) -> double {
          if (ctxt_val < 0) {
            return -1;
          }
          return 1;
        },
        ctxt, lower_bound, upper_bound, degree));
  }

  return CkksVector{result, numElements_, context_};
}

CkksVector CkksVector::IsEqual(const double value) const {
  auto vec = *this;
  auto diff = vec - value;
  auto sign = diff.Sign();
  return 1 - (sign * sign);
}

CkksVector CkksVector::Sum() const {
  auto cryptoContext = context_.getCryptoContext();
  auto batch_size = cryptoContext->GetEncodingParams()->GetBatchSize();
  CkksVector result(context_);
  result.setNumElements(1);
  result.data_.push_back(cryptoContext->EvalSum(data_[0], batch_size));
  for (unsigned i = 1; i < data_.size(); ++i) {
    result.data_[0] += cryptoContext->EvalSum(data_[i], batch_size);
  }
  return result;
}

CkksVector CkksVector::rotate(int rows_to_rotate) const {
  auto cryptoContext = context_.getCryptoContext();

  if (context_.getCryptoContext() == nullptr) {
    spdlog::error("Crypto context is not set. Cannot rotate.");
    throw std::invalid_argument("Crypto context is not set. Cannot rotate.");
  }

  if (size() == 0) {
    spdlog::error("Data is empty. Cannot rotate.");
    throw std::invalid_argument("Data is empty. Cannot rotate.");
  }

  std::vector<Ctxt> result;

  for (unsigned i = 0; i < data_.size(); ++i) {
    if (precomputedRotations_.empty()) {
      result.emplace_back(cryptoContext->EvalRotate(data_[i], rows_to_rotate));
    } else {
      result.emplace_back(cryptoContext->EvalFastRotation(
          data_[i], rows_to_rotate, cryptoContext->GetCyclotomicOrder(),
          precomputedRotations_[i]));
    }
  }

  return CkksVector{result, numElements_, context_};
}

CkksVector& CkksVector::operator*=(const std::vector<double>& rhs) {
  auto cryptoContext = context_.getCryptoContext();
  auto batch_size = cryptoContext->GetEncodingParams()->GetBatchSize();
  precomputedRotations_.clear();

  if (size() == 0) {
    spdlog::error("Data is empty. Cannot multiply.");
    throw std::invalid_argument("Data is empty. Cannot multiply.");
  }

  if (rhs.size() != size()) {
    spdlog::error("Cannot multiply vectors of different sizes.");
    throw std::invalid_argument("Cannot multiply vectors of different sizes.");
  }

  for (unsigned i = 0; i < data_.size(); ++i) {
    auto start = rhs.begin() + i * batch_size;
    auto end = rhs.begin() + std::min((i + 1) * batch_size,
                                      static_cast<unsigned>(rhs.size()));
    auto values_slice = std::vector<double>(start, end);
    data_[i] = cryptoContext->EvalMult(
        data_[i], cryptoContext->MakeCKKSPackedPlaintext(values_slice));
  }

  return *this;
}

CkksVector& CkksVector::operator*=(const CkksVector& rhs) {
  auto cryptoContext = context_.getCryptoContext();
  precomputedRotations_.clear();

  if (size() == 0) {
    spdlog::warn("Data is empty. Multiplying nothing.");
  }

  if (rhs.size() != size()) {
    spdlog::error("Cannot multiply vectors of different sizes.");
    throw std::invalid_argument("Cannot multiply vectors of different sizes.");
  }

  for (unsigned i = 0; i < data_.size(); ++i) {
    data_[i] = cryptoContext->EvalMult(data_[i], rhs.data_[i]);
  }

  return *this;
}

CkksVector& CkksVector::operator+=(const CkksVector& rhs) {
  if (rhs.size() != size()) {
    spdlog::error("Cannot add vectors of different sizes.");
    throw std::invalid_argument("Cannot add vectors of different sizes.");
  }

  precomputedRotations_.clear();

  for (unsigned i = 0; i < data_.size(); ++i) {
    data_[i] = context_.getCryptoContext()->EvalAdd(data_[i], rhs.data_[i]);
  }

  return *this;
}

CkksVector& CkksVector::operator-=(const double& rhs) {
  precomputedRotations_.clear();

  for (auto& ctxt : data_) {
    ctxt = context_.getCryptoContext()->EvalSub(ctxt, rhs);
  }

  return *this;
}

namespace fhenom {
CkksVector operator-(const double& lhs, CkksVector rhs) {
  rhs.precomputedRotations_.clear();
  for (auto& ctxt : rhs.data_) {
    ctxt = rhs.context_.getCryptoContext()->EvalSub(lhs, ctxt);
  }

  return rhs;
}
}  // namespace fhenom

void CkksVector::precomputeRotations() {
  auto cryptoContext = context_.getCryptoContext();

  precomputedRotations_.clear();
  for (auto& ctxt : data_) {
    precomputedRotations_.emplace_back(
        cryptoContext->EvalFastRotationPrecompute(ctxt));
  }
}

//////////////////////////////////////////////////////////////////////////////
// Modifiers

void CkksVector::Concat(const CkksVector& rhs) {
  auto cryptoContext = context_.getCryptoContext();
  auto batch_size = cryptoContext->GetEncodingParams()->GetBatchSize();

  numElements_ = data_.size() * batch_size + rhs.size();

  data_.insert(data_.end(), rhs.data_.begin(), rhs.data_.end());

  if (rhs.precomputedRotations_.empty()) {
    precomputedRotations_.clear();
  } else {
    precomputedRotations_.insert(precomputedRotations_.end(),
                                 rhs.precomputedRotations_.begin(),
                                 rhs.precomputedRotations_.end());
  }
}

//////////////////////////////////////////////////////////////////////////////
// Encryption and Decryption

void CkksVector::encrypt(const std::vector<double>& data) {
  precomputedRotations_.clear();

  if (data.empty()) {
    spdlog::error("Data is empty. Cannot encrypt.");
    throw std::invalid_argument("Data is empty. Cannot encrypt.");
  }

  if (context_.getCryptoContext() == nullptr) {
    spdlog::error("Crypto context is not set. Cannot encrypt.");
    throw std::invalid_argument("Crypto context is not set. Cannot encrypt.");
  }

  if (context_.getKeyPair().publicKey == nullptr) {
    spdlog::error("Public key is not set. Cannot encrypt.");
    throw std::invalid_argument("Public key is not set. Cannot encrypt.");
  }

  this->data_.clear();

  for (size_t i = 0; i < data.size();
       i += context_.getCryptoContext()->GetEncodingParams()->GetBatchSize()) {
    auto end = std::min(
        i + context_.getCryptoContext()->GetEncodingParams()->GetBatchSize(),
        data.size());
    const auto tmp = std::vector<double>(data.begin() + i, data.begin() + end);
    auto ptxt = context_.getCryptoContext()->MakeCKKSPackedPlaintext(tmp);
    auto ctxt = context_.getCryptoContext()->Encrypt(
        context_.getKeyPair().publicKey, ptxt);
    this->data_.push_back(ctxt);
  }

  numElements_ = data.size();
}

std::vector<double> CkksVector::decrypt() const {
  if (context_.getCryptoContext() == nullptr) {
    spdlog::error("Crypto context is not set. Cannot decrypt.");
    throw std::invalid_argument("Crypto context is not set. Cannot decrypt.");
  }

  if (context_.getKeyPair().secretKey == nullptr) {
    spdlog::error("Secret key is not set. Cannot decrypt.");
    throw std::invalid_argument("Secret key is not set. Cannot decrypt.");
  }

  if (size() == 0) {
    spdlog::error("Data is empty. Cannot decrypt.");
    throw std::invalid_argument("Data is empty. Cannot decrypt.");
  }

  std::vector<double> result;
  for (const auto& ctxt : data_) {
    Ptxt ptxt;
    context_.getCryptoContext()->Decrypt(context_.getKeyPair().secretKey, ctxt,
                                         &ptxt);

    auto remaining = numElements_ - result.size();
    if (remaining <
        context_.getCryptoContext()->GetEncodingParams()->GetBatchSize()) {
      ptxt->SetLength(remaining);
    }

    auto tmp = ptxt->GetRealPackedValue();
    result.insert(result.end(), tmp.begin(), tmp.end());
  }

  return result;
}

//////////////////////////////////////////////////////////////////////////////
// File I/O

template <class Archive>
void CkksVector::serialize(Archive& archive) {
  archive(numElements_, data_);
  spdlog::debug("Serializing EncVector with {} elements", numElements_);
  spdlog::debug("Data size: {}", data_.size());
}

void CkksVector::load(const std::filesystem::path& path) {
  std::ifstream if_stream{path};
  if (!if_stream.is_open()) {
    spdlog::error("Could not open file {}", path.string());
    throw std::invalid_argument("Could not open file");
  }

  cereal::BinaryInputArchive iarchive{if_stream};
  iarchive(*this);
}

void CkksVector::save(const std::filesystem::path& path) const {
  std::ofstream os{path};
  cereal::BinaryOutputArchive oarchive{os};
  oarchive(*this);
}
