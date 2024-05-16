#include <fhenom/ckks_vector.h>
#include "coefficients.h"

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

void CkksVector::Bootstrap() {
    if (size() == 0) {
        spdlog::warn("Data is empty. Nothing to bootstrap.");
        return;
    }

    auto crypto_context = context_.GetCryptoContext();
    for (auto& ctxt : data_) {
        ctxt = crypto_context->EvalBootstrap(ctxt);
    }
}

CkksVector CkksVector::ReLU(unsigned depth) const {
    CkksVector result;
    switch (depth) {
        case 3:
            return EvalChebyshev(
                [](double x) -> double {
                    if (x < 0) {
                        return 0;
                    }
                    return x;
                },
                -1, 1, 7);
            break;
        case 4:
            return EvalChebyshev(
                [](double x) -> double {
                    if (x < 0) {
                        return 0;
                    }
                    return x;
                },
                -1, 1, 13);
            break;
        case 10:
            return EvalChebyshev(
                [](double x) -> double {
                    if (x < 0) {
                        return 0;
                    }
                    return x;
                },
                -1, 1, 1023);
            break;
        case 11:
            return EvalChebyshev(
                [](double x) -> double {
                    if (x < 0) {
                        return 0;
                    }
                    return x;
                },
                -1, 1, 2047);
            break;
        case 12:
            return EvalChebyshev(
                [](double x) -> double {
                    if (x < 0) {
                        return 0;
                    }
                    return x;
                },
                -1, 1, 4095);
            break;
        default:
            spdlog::error("ReLU of depth {} not implemented.", depth);
            throw std::invalid_argument("ReLU of depth not implemented.");
    }

    // (1/2) * (x + x * sign(x))
    result = (*this + *this * (result)) * std::vector<double>(this->size(), 0.5);
    return result;
}

CkksVector CkksVector::EvalPoly(const std::vector<double>& coefficients) const {
    auto crypto_context = context_.GetCryptoContext();

    if (size() == 0) {
        spdlog::warn("Data is empty. Nothing to compare.");
        throw std::invalid_argument("Data is empty. Nothing to compare.");
    }

    std::vector<Ctxt> result(data_.size());
    for (unsigned i = 0; i < data_.size(); ++i) {
        result[i] = crypto_context->EvalPoly(data_[i], coefficients);
    }

    return CkksVector{result, numElements_, context_};
}

CkksVector CkksVector::EvalChebyshev(const std::function<double(double)>& func, const double lower_bound,
                                     const double upper_bound, uint32_t degree) const {
    auto crypto_context = context_.GetCryptoContext();

    if (size() == 0) {
        spdlog::warn("Data is empty. Cannot evaluate Chebyshev on empty data.");
        throw std::invalid_argument("Data is empty. Cannot evaluate Chebyshev on empty data.");
    }

    std::vector<Ctxt> result(data_.size());
    for (unsigned i = 0; i < data_.size(); ++i) {
        result[i] = crypto_context->EvalChebyshevFunction(func, data_[i], lower_bound, upper_bound, degree);
    }

    return CkksVector{result, numElements_, context_};
}

CkksVector CkksVector::GetSignUsingPolyComp() const {
    auto crypto_context = context_.GetCryptoContext();

    if (size() == 0) {
        spdlog::warn("Data is empty. Nothing to compare.");
        throw std::invalid_argument("Data is empty. Nothing to compare.");
    }

    CkksVector result = EvalPoly(fhenom::kG3Coeffs);
    result            = result.EvalPoly(fhenom::kG3Coeffs);
    result            = result.EvalPoly(fhenom::kG3Coeffs);
    result            = result.EvalPoly(fhenom::kF3Coeffs);
    result            = result.EvalPoly(fhenom::kF3Coeffs);

    return result;
}

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

CkksVector CkksVector::IsEqual(const double value, const double max_value) const {
    auto diff = *this - value;
    diff *= std::vector<double>(size(), 1.0 / max_value);
    auto sign = diff.GetSign();
    return 1 - (sign * sign);
}

CkksVector CkksVector::GetSum() const {
    if (size() == 0) {
        spdlog::error("Data is empty. Cannot sum.");
        throw std::invalid_argument("Data is empty. Cannot sum.");
    }

    auto crypto_context = context_.GetCryptoContext();
    size_t batch_size   = crypto_context->GetEncodingParams()->GetBatchSize();

    Ctxt result = crypto_context->EvalSum(data_[0], batch_size);
    for (unsigned i = 1; i < data_.size(); ++i) {
        result += crypto_context->EvalSum(data_[i], batch_size);
    }

    return CkksVector(std::vector<Ctxt>{result}, 1, context_);
}

CkksVector CkksVector::GetCount(const double value, const double max_value) const {
    auto is_equal = IsEqual(value, max_value);

    // Remove trailing zeros that IsEqual counted as equal so EvalSum doesn't count them
    auto crypto_context = context_.GetCryptoContext();
    auto batch_size     = crypto_context->GetEncodingParams()->GetBatchSize();
    auto remainder      = numElements_ % batch_size;
    if (remainder) {
        std::vector<double> mask(batch_size, 0);
        std::fill(mask.begin() + remainder, mask.end(), -1);
        auto ptxt             = crypto_context->MakeCKKSPackedPlaintext(mask);
        is_equal.data_.back() = crypto_context->EvalAdd(is_equal.data_.back(), ptxt);
    }

    auto sum = is_equal.GetSum();
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
            try {
                if (precomputed_rotations.empty()) {
                    result[i] = crypto_context->EvalRotate(result[i], next_rotation_amount);
                }
                else {
                    result[i] = crypto_context->EvalFastRotation(result[i], next_rotation_amount,
                                                                 crypto_context->GetCyclotomicOrder(),
                                                                 precomputed_rotations[i]);
                    precomputed_rotations.clear();
                }
            }
            catch (const std::exception& e) {
                spdlog::error("Error rotating ciphertext by {}. No key for rotating {}.", rotation_index,
                              next_rotation_amount);
                throw e;
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
        context_.GetCryptoContext()->EvalAddInPlace(data_[i], rhs.data_[i]);
    }

    return *this;
}

CkksVector& CkksVector::operator+=(const double& rhs) {
    precomputedRotations_.clear();

    for (auto& ctxt : data_) {
        ctxt = context_.GetCryptoContext()->EvalAdd(ctxt, rhs);
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
    precomputedRotations_.clear();

    if (crypto_context != rhs.context_.GetCryptoContext()) {
        spdlog::error("Cannot concatenate vectors with different contexts.");
        throw std::invalid_argument("Cannot concatenate vectors with different contexts.");
    }

    if (size() == 0) {
        data_        = rhs.data_;
        numElements_ = rhs.size();
        return;
    }

    if (rhs.size() == 0) {
        return;
    }

    auto space_remaining = data_.size() * batch_size - numElements_;

    if (space_remaining == 0) {
        std::copy(rhs.data_.begin(), rhs.data_.end(), std::back_inserter(data_));
        numElements_ = data_.size() * batch_size + rhs.size();
        return;
    }

    if (space_remaining >= rhs.size()) {
        data_.back() += rhs.Rotate(space_remaining).data_.front();
        numElements_ += rhs.size();
        return;
    }

    std::vector<double> mask_first_part(rhs.size(), 1);
    std::fill(mask_first_part.begin(), mask_first_part.begin() + space_remaining, 0);
    auto mask_first_part_ptxt = crypto_context->MakeCKKSPackedPlaintext(mask_first_part);

    std::vector<double> mask_second_part(rhs.size(), 0);
    std::fill(mask_second_part.begin(), mask_second_part.begin() + space_remaining, 1);
    auto mask_second_part_ptxt = crypto_context->MakeCKKSPackedPlaintext(mask_second_part);

    auto tmp = rhs.Rotate(space_remaining);
    for (const auto& ctxt : tmp.data_) {
        crypto_context->EvalAddInPlace(data_.back(), crypto_context->EvalMult(ctxt, mask_first_part_ptxt));
        data_.push_back(crypto_context->EvalMult(ctxt, mask_second_part_ptxt));
    }
    numElements_ = size() + rhs.size();
}

CkksVector CkksVector::Merge(const std::vector<CkksVector>& vectors) {
    if (vectors.empty()) {
        spdlog::error("Cannot merge empty vectors.");
        throw std::invalid_argument("Cannot merge empty vectors.");
    }

    auto crypto_context = vectors[0].context_.GetCryptoContext();

    std::vector<double> mask(vectors[0].size(), 0);
    mask[0]           = 1;
    CkksVector result = vectors[0] * mask;
    result.SetNumElements(1);

    for (auto i = 1; i < vectors.size(); ++i) {
        if (vectors[i].context_.GetCryptoContext() != crypto_context) {
            spdlog::error("Cannot merge vectors with different contexts.");
            throw std::invalid_argument("Cannot merge vectors with different contexts.");
        }

        mask    = std::vector<double>(vectors[i].size(), 0);
        mask[0] = 1;
        result  = result.Rotate(-1);

        auto new_size = result.size() + 1;
        result.SetNumElements(vectors[i].size());
        result += vectors[i] * mask;
        result.SetNumElements(new_size);
    }

    return result;
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
