#pragma once

#include <fhenom/common.h>
#include <fhenom/context.h>
#include <openfhe.h>
#include <spdlog/spdlog.h>

#include <cereal/archives/binary.hpp>
#include <filesystem>
#include <vector>
#include "tensor.h"

namespace fhenom {

/**
 * @brief This class represents a vector of CKKS encrypted data of arbitrary
 * length and provides some homomorphic vector operations.
 */
class CkksVector {
private:
    std::vector<Ctxt> data_;
    std::size_t numElements_;
    fhenom::Context context_;
    std::vector<PrecomputedRotations> precomputedRotations_;

public:
    CkksVector() : data_{}, numElements_{0} {}
    CkksVector(const fhenom::Context& context) : data_{}, numElements_{0}, context_{context} {}
    CkksVector(const std::vector<Ctxt>& data, std::size_t numElements, const fhenom::Context& context)
        : data_{data}, numElements_{numElements}, context_{context} {}

    ////////////////////////////////////////////////////////////////////////////
    // Encryption and Decryption

    /**
     * @brief Encrypts a vector of data
     *
     * @param data The unencrypted data
     */
    void Encrypt(const std::vector<double>& data);

    /**
     * @brief Decrypts the data in this vector
     *
     * @return vector<double> Returns the decrypted data vector
     */
    std::vector<double> Decrypt() const;

    ////////////////////////////////////////////////////////////////////////////
    // Homomorphic Operations

    /**
     * @brief Bootstrap each of the underlying ciphertexts
     * 
     * @note This method assumes that all necessary automorphism keys are loaded.
     * 
     */
    void Bootstrap();

    /**
     * @brief Evaluate a polynomial on the vector elements
     * 
     * @param coeffs The coefficients in decreasing degree
     * @return CkksVector the result of the polynomial evaluation
     */
    CkksVector EvalPoly(const std::vector<double>& coefficients) const;

    /**
     * @brief Evaluate a Chebyshev approximation of a function
     * 
     * @param func The function to approximate
     * @param lower_bound the lower limit of the domain
     * @param upper_bound the upper limit of the domain
     * @param degree the degree of the approximation
     * @return CkksVector The result of applying the approximation
     */
    CkksVector EvalChebyshev(const std::function<double(double)>& func, const double lower_bound = -1,
                             const double upper_bound = 1, unsigned degree = 4095) const;

    /**
     * @brief Evaluates the sign in each slot of the vector
     *
     * @return CkksVector A vector with -1 in negative slots, 0 in 0 slots, and 1
     * in positive slots
     * 
     * @note calls GetSignUsingPolyComp under the hood, so values must be scaled to [-1, 1]
     */
    inline CkksVector GetSign() const {
        return GetSignUsingPolyComp();
    };

    /**
     * @brief Evaluates the sign in each slot of the vector using the Chebyshev
     * function
     *
     * @return CkksVector A vector with -1 in negative slots, 0 in 0 slots, and 1
     * in positive slots
     */
    CkksVector GetSignUsingChebyshev(const double lower_bound = -100, const double upper_bound = 100,
                                     const uint32_t degree = 2031) const;

    /**
     * @brief Get the sign using polynomial composition of each slot in the vector
     * 
     * @return CkksVector A vector with -1 in negative slots, 0 in 0 slots, and 1 in positive slots
     * 
     * @note Values in the vector must be scaled to [-1, 1] for this method to work. Results will 
     * explode outside this range.
     */
    CkksVector GetSignUsingPolyComp() const;

    /**
     * @brief Checks if the data in this vector is equal to a plaintext value
     *
     * @param value The value to compare against
     * @param max_value The absolute value of the maximum value in the vector
     * @return CkksVector A vector of `1`s in the place of equal slots; `0`s
     * elsewhere
     */
    CkksVector IsEqual(const double value, const double max_value) const;

    /**
     * @brief Sum all elements in a vector
     *
     * @return CkksVector A vector where slot 1 contains the sum of all values in
     * the vector
     */
    CkksVector GetSum() const;

    /**
     * @brief Get the number of occurrences of value in the vector
     * 
     * @param value The value to count
     * @param max_value The absolute value of the maximum value in the vector
     * @return CkksVector A vector where the first slot contains the number of occurrences of value
     * 
     * @note Uses polynomial composition coefficients that are accurate for domains up to [-100,100]
     */
    CkksVector GetCount(const double value, const double max_value = 100) const;

    /**
     * @brief Rotates the data in this vector (+ is left, - is right)
     *
     * @param rotation_index
     * @return CkksVector
     *
     * @note The rotations occur within the each ciphertext, length `N/2`. For
     * N=8192, `Rotate(1)` would send index 0 to index 4095, and index 4096 to
     * index 8191.
     */
    CkksVector Rotate(int rotation_index) const;

    /**
     * @brief Elementwise multiplication with a plaintext of equal size
     *
     * @param rhs The plaintext to multiply
     * @return CkksVector& A reference to this object
     */
    CkksVector& operator*=(const std::vector<double>& rhs);

    /**
     * @brief Multiply every element in the vector by a scalar
     * 
     * @param rhs the scalar to multiply
     * @return CkksVector& The modified vector
     */
    CkksVector& operator*=(const double& rhs);

    /**
     * @brief Elementwise multiplication with a plaintext of equal size
     *
     * @param rhs The vector to multiply
     * @return CkksVector& A reference to this object
     */
    CkksVector& operator*=(const CkksVector& rhs);

    /**
     * @brief Elementwise addition of another encrypted vector
     *
     * @param rhs the vector to add
     * @return CkksVector& A reference to this object
     */
    CkksVector& operator+=(const CkksVector& rhs);

    /**
     * @brief Add a plaintext value to each element in the vector
     * 
     * @param rhs the plaintext value to add
     * @return CkksVector& a reference to this object
     */
    CkksVector& operator+=(const double& rhs);

    /**
     * @brief Add a plaintext vector to this vector
     * 
     * @param rhs the plaintext vector to add
     * @return the addition result
     */
    CkksVector& operator+=(const std::vector<double>& rhs);

    /**
     * @brief Add a plaintext tensor to this vector
     * 
     * @param rhs the tensor to add
     * @return CkksVector& the addition result
     */
    inline CkksVector& operator+=(const fhenom::Tensor& rhs) {
        return *this += rhs.GetData();
    }

    /**
     * @brief Elementwise subtraction
     *
     * @param rhs
     * @return CkksVector&
     */
    CkksVector& operator-=(const double& rhs);

    friend inline CkksVector operator-(CkksVector lhs, const double rhs) {
        return lhs -= rhs;
    }

    friend CkksVector operator-(const double& lhs, CkksVector rhs);

    /**
     * @brief Add together a list of vectors
     * 
     * @param vectors the list of vectors to add
     * @return the result of element-wise addition
     */
    static CkksVector AddMany(const std::vector<CkksVector>& vectors);

    /**
     * @brief Precompute rotations to use fast rotations
     */
    void PrecomputeRotations();

    //////////////////////////////////////////////////////////////////////////////
    // Modifiers

    /**
     * @brief Concatenate a vector onto this vector
     *
     * @param rhs The vector to concatenate on the end
     * 
     * @note This assumes that all unused slots are zero
     */
    void Concat(const CkksVector& rhs);

    // /**
    //  * @brief Condense the ciphertexts in the vector to the first `num_elements`
    //  *
    //  * @param num_elements The number of elements to keep from each ciphertext
    //  * @note This method assumes all ciphertexts are already masked with zeroes outside the elements to keep.
    //  */
    // void Condense(unsigned num_elements);

    // /**
    //  * @brief Create a vector consisting of the first `num_elements` slots of each of the parameter vectors
    //  *
    //  * @param vectors A list of vectors to merge
    //  * @param num_elements the number of elements to merge from each vector
    //  * @return CkksVector A vector containing the first `num_elements` slots of each input vector in the slot
    //  *
    //  * @note This method assumes all vectors are already masked with zeroes outside the elements to merge.
    //  */
    // static CkksVector Merge(const std::vector<CkksVector>& vectors, unsigned num_elements = 1);
    //
    // /**
    //  * @brief Create a vector consisting of the first `num_elements` slots of each of the parameter ciphertexts
    //  *
    //  * @param ctxts a list of ciphertexts to merge
    //  * @param num_elements the number of elements to merge from each ciphertext
    //  * @return CkksVector A vector containing the first `num_elements` slots of each input ciphertext
    //  *
    //  * @note This method assumes all ciphertexts are already masked with zeroes outside the elements to merge.
    //  */
    // static CkksVector Merge(const std::vector<Ctxt>& ctxts, unsigned num_elements = 1);

    ////////////////////////////////////////////////////////////////////////////
    // File I/O

    // friend class cereal::access;

    /**
     * @brief Tells cereal how to serialize this class
     *
     * @tparam Archive
     * @param archive
     */
    template <class Archive>
    void serialize(Archive& archive);

    /**
     * @brief Loads data from a file
     *
     * @param path The path to the file
     */
    void Load(const std::filesystem::path& path);

    /**
     * @brief Saves data to a file
     *
     * @param path The path to the file
     */
    void Save(const std::filesystem::path& path) const;

    ////////////////////////////////////////////////////////////////////////////
    // Getters and Setters

    /**
     * @brief Get the data
     *
     * @return vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>> The encrypted
     * data
     */
    inline std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> GetData() const {
        return data_;
    }

    /**
     * @brief Set the data
     *
     * @param data The encrypted data
     */
    inline void SetData(const std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>& data, std::size_t numElements) {
        this->data_        = std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(data.begin(), data.end());
        this->numElements_ = numElements;
    }

    /**
     * @brief Set the Context object
     *
     * @param context
     */
    inline void SetContext(const fhenom::Context& context) {
        this->context_ = context;
    }

    /**
     * @brief Get the Context object
     *
     * @return fhenom::Context
     */
    inline fhenom::Context GetContext() const {
        return context_;
    }

    /**
     * @brief Get the number of elements
     *
     * @return unsigned The number of elements
     */
    inline std::size_t size() const {
        return numElements_;
    }

    inline std::size_t capacity() const {
        return data_.size() * context_.GetCryptoContext()->GetEncodingParams()->GetBatchSize();
    }

    inline void SetNumElements(std::size_t numElements) {
        this->numElements_ = numElements;
    }
};

}  // namespace fhenom

inline fhenom::CkksVector operator*(fhenom::CkksVector lhs, const fhenom::CkksVector& rhs) {
    return lhs *= rhs;
}

inline fhenom::CkksVector operator*(fhenom::CkksVector lhs, const std::vector<double>& rhs) {
    return lhs *= rhs;
}

inline fhenom::CkksVector operator*(fhenom::CkksVector lhs, const double& rhs) {
    return lhs *= rhs;
}

inline fhenom::CkksVector operator+(fhenom::CkksVector lhs, const fhenom::CkksVector& rhs) {
    return lhs += rhs;
}

inline fhenom::CkksVector operator+(fhenom::CkksVector lhs, const fhenom::Tensor& rhs) {
    return lhs += rhs;
}

inline fhenom::CkksVector operator+(fhenom::CkksVector lhs, const double rhs) {
    return lhs += rhs;
}

inline fhenom::CkksVector operator+(fhenom::CkksVector lhs, const std::vector<double>& rhs) {
    return lhs += rhs;
}
