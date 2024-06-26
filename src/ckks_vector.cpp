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
#pragma omp parallel for
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

CkksVector CkksVector::AddMany(const std::vector<CkksVector>& vectors) {
    if (vectors.empty()) {
        spdlog::error("Cannot add empty vectors.");
        throw std::invalid_argument("Cannot add empty vectors.");
    }

    if (vectors.size() == 1) {
        return vectors[0];
    }

    auto crypto_context = vectors[0].GetContext().GetCryptoContext();

    std::vector<Ctxt> result_data(vectors[0].GetData().size());

#pragma omp parallel for
    for (unsigned i = 0; i < result_data.size(); ++i) {
        std::vector<Ctxt> ctxts;
        for (const auto& vec : vectors) {
            ctxts.push_back(vec.data_[i]);
        }
        result_data[i] = crypto_context->EvalAddMany(ctxts);
    }

    return CkksVector(result_data, vectors[0].size(), vectors[0].GetContext());
}

CkksVector CkksVector::GetSum() const {
    if (size() == 0) {
        spdlog::error("Data is empty. Cannot sum.");
        throw std::invalid_argument("Data is empty. Cannot sum.");
    }

    auto crypto_context = context_.GetCryptoContext();
    size_t batch_size   = crypto_context->GetEncodingParams()->GetBatchSize();

    Ctxt sum_ctxt;
    if (data_.size() == 1) {
        sum_ctxt = data_[0];
    }
    else {
        sum_ctxt = crypto_context->EvalAddMany(data_);
    }

    for (unsigned i = 0; i < log2(batch_size); ++i) {
        crypto_context->EvalAddInPlace(sum_ctxt, crypto_context->EvalRotate(sum_ctxt, 1 << i));
    }
    return CkksVector(std::vector<Ctxt>{sum_ctxt}, 1, context_);
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
    auto batch_size     = crypto_context->GetEncodingParams()->GetBatchSize();

    if (crypto_context == nullptr) {
        spdlog::error("Crypto context is not set. Cannot rotate.");
        throw std::invalid_argument("Crypto context is not set. Cannot rotate.");
    }

    if (size() == 0) {
        spdlog::error("Data is empty. Cannot rotate.");
        throw std::invalid_argument("Data is empty. Cannot rotate.");
    }

    if (std::abs(rotation_index) >= batch_size) {
        spdlog::error("Cannot rotate by more than the number of slots.");
        throw std::invalid_argument("Cannot rotate by more than the number of slots.");
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
                    spdlog::warn("Slow rotation of {} to get to {}", next_rotation_amount, rotation_index);
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

    if (rhs.size() != size() && rhs.size() != capacity()) {
        spdlog::error(
            "Cannot multiply vectors of different sizes, this is {}/{} and trying to multiply by vector size {}.",
            size(), data_.size() * batch_size, rhs.size());
        throw std::invalid_argument("Cannot multiply vectors of different sizes.");
    }

    for (unsigned i = 0; i < data_.size(); ++i) {
        auto start        = rhs.begin() + i * batch_size;
        auto end          = rhs.begin() + std::min(static_cast<size_t>((i + 1) * batch_size), rhs.size());
        auto values_slice = std::vector<double>(start, end);
        data_[i]          = crypto_context->EvalMult(data_[i], crypto_context->MakeCKKSPackedPlaintext(values_slice));
    }

    return *this;
}

CkksVector& CkksVector::operator*=(const double& rhs) {
    return this->operator*=(std::vector<double>(size(), rhs));
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

CkksVector& CkksVector::operator+=(const double& rhs) {
    precomputedRotations_.clear();

    for (auto& ctxt : data_) {
        ctxt = context_.GetCryptoContext()->EvalAdd(ctxt, rhs);
    }

    return *this;
}

CkksVector& CkksVector::operator+=(const std::vector<double>& rhs) {
    auto crypto_context = context_.GetCryptoContext();
    size_t batch_size   = crypto_context->GetEncodingParams()->GetBatchSize();
    if (rhs.size() != size() && rhs.size() != data_.size() * batch_size) {
        spdlog::error("Cannot add vectors of different sizes.");
        throw std::invalid_argument("Cannot add vectors of different sizes.");
    }

    precomputedRotations_.clear();

    for (unsigned i = 0; i < data_.size(); ++i) {
        auto batch_rhs =
            std::vector<double>(rhs.begin() + i * batch_size, rhs.begin() + std::min((i + 1) * batch_size, rhs.size()));
        data_[i] = context_.GetCryptoContext()->EvalAdd(
            data_[i], context_.GetCryptoContext()->MakeCKKSPackedPlaintext(batch_rhs));
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
        numElements_ += rhs.size();
        return;
    }

    if (space_remaining >= rhs.size()) {
        data_.back() += rhs.Rotate(-(size() % batch_size)).data_.front();
        numElements_ += rhs.size();
        return;
    }

    std::vector<double> mask_out_first_part(batch_size, 0);
    std::fill(mask_out_first_part.begin() + space_remaining, mask_out_first_part.begin() + batch_size, 1);

    std::vector<double> mask_out_second_part(batch_size, 0);
    std::fill(mask_out_second_part.begin(), mask_out_second_part.begin() + space_remaining, 1);

    auto first_part  = rhs * mask_out_second_part;
    first_part       = first_part.Rotate(space_remaining - batch_size);
    auto second_part = rhs * mask_out_first_part;
    second_part      = second_part.Rotate(batch_size - space_remaining);
    for (unsigned i = 0; i < rhs.GetData().size(); ++i) {
        crypto_context->EvalAddInPlace(data_.back(), first_part.GetData()[i]);
        data_.push_back(second_part.GetData()[i]);
    }
    numElements_ = size() + rhs.size();
}

//////////////////////////////////////////////////////////////////////////////
// Encryption and Decryption

void CkksVector::Encrypt(const std::vector<double>& data) {
    precomputedRotations_.clear();
    auto crypto_context = context_.GetCryptoContext();
    auto batch_size     = crypto_context->GetEncodingParams()->GetBatchSize();

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

    for (size_t i = 0; i < data.size(); i += batch_size) {
        auto end         = std::min(i + batch_size, data.size());
        const auto k_tmp = std::vector<double>(data.begin() + i, data.begin() + end);
        auto ptxt        = context_.GetCryptoContext()->MakeCKKSPackedPlaintext(k_tmp);
        auto ctxt        = context_.GetCryptoContext()->Encrypt(context_.GetKeyPair().publicKey, ptxt);
        this->data_.push_back(ctxt);
    }

    numElements_ = data.size();
}

std::vector<double> CkksVector::Decrypt() const {
    auto crypto_context = context_.GetCryptoContext();

    if (crypto_context == nullptr) {
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

    auto batch_size = crypto_context->GetEncodingParams()->GetBatchSize();

    std::vector<double> result;
    for (const auto& ctxt : data_) {
        Ptxt ptxt;
        spdlog::debug("Decrypting...");
        crypto_context->Decrypt(context_.GetKeyPair().secretKey, ctxt, &ptxt);

        auto remaining = numElements_ - result.size();
        if (remaining < batch_size) {
            ptxt->SetLength(remaining);
        }
        else {
            ptxt->SetLength(batch_size);
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
