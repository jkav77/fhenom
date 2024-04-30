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
using std::size_t;

//////////////////////////////////////////////////////////////////////////////
// Homomorphic Operations

CkksVector CkksVector::GetSignUsingChebyshev(const double lower_bound, const double upper_bound,
                                             uint32_t degree) const {
    auto crypto_context = context_.GetCryptoContext();

    if (size() == 0) {
        spdlog::warn("Data is empty. Comparing nothing.");
    }

    std::vector<Ctxt> result(data_.size());
    for (unsigned i = 0; i < data_.size(); ++i) {
        result[i] = crypto_context->EvalChebyshevFunction(
            [](double ctxt_val) -> double {
                if (ctxt_val < 0) {
                    return -1;
                }
                return 1;
            },
            data_[i], lower_bound, upper_bound, degree);
    }

    return CkksVector{result, numElements_, context_};
}

CkksVector CkksVector::IsEqual(const double value) const {
    auto vec  = *this;
    auto diff = vec - value;
    auto sign = diff.GetSign();
    return 1 - (sign * sign);
}

CkksVector CkksVector::GetSum() const {
    auto crypto_context = context_.GetCryptoContext();
    auto batch_size     = crypto_context->GetEncodingParams()->GetBatchSize();
    CkksVector result(context_);
    result.SetNumElements(1);
    result.data_.push_back(crypto_context->EvalSum(data_[0], batch_size));
    for (unsigned i = 1; i < data_.size(); ++i) {
        result.data_[0] += crypto_context->EvalSum(data_[i], batch_size);
    }
    return result;
}

CkksVector CkksVector::GetCount(const double value) const {
    auto is_equal = IsEqual(value);
    auto sum      = is_equal.GetSum();
    return sum;
}

CkksVector CkksVector::Rotate(int rotation_index) const {
    if (rotation_index == 0) {
        return *this;
    }

    auto crypto_context = context_.GetCryptoContext();

    if (crypto_context == nullptr) {
        spdlog::error("Crypto context is not set. Cannot rotate.");
        throw std::invalid_argument("Crypto context is not set. Cannot rotate.");
    }

    if (size() == 0) {
        spdlog::error("Data is empty. Cannot rotate.");
        throw std::invalid_argument("Data is empty. Cannot rotate.");
    }

    std::vector<Ctxt> result   = data_;
    auto precomputed_rotations = precomputedRotations_;
    auto am_key_map            = crypto_context->GetEvalAutomorphismKeyMap(data_[0]->GetKeyTag());
    int rotations_remaining    = rotation_index;

    while (rotations_remaining != 0) {
        auto am_idx = crypto_context->FindAutomorphismIndex(rotations_remaining);

        int next_rotation_amount = 0;
        if (am_key_map.contains(am_idx)) {
            next_rotation_amount = rotations_remaining;
        }
        else {
            next_rotation_amount = std::pow(2, std::floor(std::log2(std::abs(rotations_remaining))));
            if (rotations_remaining < 0) {
                next_rotation_amount = -next_rotation_amount;
            }
        }

        rotations_remaining -= next_rotation_amount;

        for (unsigned i = 0; i < result.size(); ++i) {
            if (precomputed_rotations.empty()) {
                result[i] = crypto_context->EvalRotate(result[i], next_rotation_amount);
            }
            else {
                result[i] = crypto_context->EvalFastRotation(
                    result[i], next_rotation_amount, crypto_context->GetCyclotomicOrder(), precomputed_rotations[i]);
                precomputed_rotations.clear();
            }
        }
    }

    return CkksVector{result, numElements_, context_};
}

CkksVector& CkksVector::operator*=(const std::vector<double>& rhs) {
    auto crypto_context = context_.GetCryptoContext();
    auto batch_size     = crypto_context->GetEncodingParams()->GetBatchSize();
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
        auto start        = rhs.begin() + i * batch_size;
        auto end          = rhs.begin() + std::min((i + 1) * batch_size, static_cast<unsigned>(rhs.size()));
        auto values_slice = std::vector<double>(start, end);
        data_[i]          = crypto_context->EvalMult(data_[i], crypto_context->MakeCKKSPackedPlaintext(values_slice));
    }

    return *this;
}

CkksVector& CkksVector::operator*=(const CkksVector& rhs) {
    auto crypto_context = context_.GetCryptoContext();
    precomputedRotations_.clear();

    if (size() == 0) {
        spdlog::warn("Data is empty. Multiplying nothing.");
    }

    if (rhs.size() != size()) {
        spdlog::error("Cannot multiply vectors of different sizes.");
        throw std::invalid_argument("Cannot multiply vectors of different sizes.");
    }

    for (unsigned i = 0; i < data_.size(); ++i) {
        data_[i] = crypto_context->EvalMult(data_[i], rhs.data_[i]);
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
        data_[i] = context_.GetCryptoContext()->EvalAdd(data_[i], rhs.data_[i]);
    }

    return *this;
}

CkksVector& CkksVector::operator-=(const double& rhs) {
    precomputedRotations_.clear();

    for (auto& ctxt : data_) {
        ctxt = context_.GetCryptoContext()->EvalSub(ctxt, rhs);
    }

    return *this;
}

namespace fhenom {
CkksVector operator-(const double& lhs, CkksVector rhs) {
    rhs.precomputedRotations_.clear();
    for (auto& ctxt : rhs.data_) {
        ctxt = rhs.context_.GetCryptoContext()->EvalSub(lhs, ctxt);
    }

    return rhs;
}
}  // namespace fhenom

void CkksVector::PrecomputeRotations() {
    auto crypto_context = context_.GetCryptoContext();

    precomputedRotations_.clear();
    for (auto& ctxt : data_) {
        precomputedRotations_.emplace_back(crypto_context->EvalFastRotationPrecompute(ctxt));
    }
}

//////////////////////////////////////////////////////////////////////////////
// Modifiers

void CkksVector::Concat(const CkksVector& rhs) {
    auto crypto_context = context_.GetCryptoContext();
    auto batch_size     = crypto_context->GetEncodingParams()->GetBatchSize();

    numElements_ = data_.size() * batch_size + rhs.size();

    data_.insert(data_.end(), rhs.data_.begin(), rhs.data_.end());

    if (rhs.precomputedRotations_.empty()) {
        precomputedRotations_.clear();
    }
    else {
        precomputedRotations_.insert(precomputedRotations_.end(), rhs.precomputedRotations_.begin(),
                                     rhs.precomputedRotations_.end());
    }
}

//////////////////////////////////////////////////////////////////////////////
// Encryption and Decryption

void CkksVector::Encrypt(const std::vector<double>& data) {
    precomputedRotations_.clear();

    if (data.empty()) {
        spdlog::error("Data is empty. Cannot encrypt.");
        throw std::invalid_argument("Data is empty. Cannot encrypt.");
    }

    if (context_.GetCryptoContext() == nullptr) {
        spdlog::error("Crypto context is not set. Cannot encrypt.");
        throw std::invalid_argument("Crypto context is not set. Cannot encrypt.");
    }

    if (context_.GetKeyPair().publicKey == nullptr) {
        spdlog::error("Public key is not set. Cannot encrypt.");
        throw std::invalid_argument("Public key is not set. Cannot encrypt.");
    }

    this->data_.clear();

    for (size_t i = 0; i < data.size(); i += context_.GetCryptoContext()->GetEncodingParams()->GetBatchSize()) {
        auto end         = std::min(i + context_.GetCryptoContext()->GetEncodingParams()->GetBatchSize(), data.size());
        const auto k_tmp = std::vector<double>(data.begin() + i, data.begin() + end);
        auto ptxt        = context_.GetCryptoContext()->MakeCKKSPackedPlaintext(k_tmp);
        auto ctxt        = context_.GetCryptoContext()->Encrypt(context_.GetKeyPair().publicKey, ptxt);
        this->data_.push_back(ctxt);
    }

    numElements_ = data.size();
}

std::vector<double> CkksVector::Decrypt() const {
    if (context_.GetCryptoContext() == nullptr) {
        spdlog::error("Crypto context is not set. Cannot decrypt.");
        throw std::invalid_argument("Crypto context is not set. Cannot decrypt.");
    }

    if (context_.GetKeyPair().secretKey == nullptr) {
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
        context_.GetCryptoContext()->Decrypt(context_.GetKeyPair().secretKey, ctxt, &ptxt);

        auto remaining = numElements_ - result.size();
        if (remaining < context_.GetCryptoContext()->GetEncodingParams()->GetBatchSize()) {
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
}

void CkksVector::Load(const std::filesystem::path& path) {
    std::ifstream if_stream{path};
    if (!if_stream.is_open()) {
        spdlog::error("Could not open file {}", path.string());
        throw std::invalid_argument("Could not open file");
    }

    cereal::BinaryInputArchive iarchive{if_stream};
    iarchive(*this);
}

void CkksVector::Save(const std::filesystem::path& path) const {
    std::ofstream os{path};
    cereal::BinaryOutputArchive oarchive{os};
    oarchive(*this);
}
