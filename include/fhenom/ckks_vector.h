#pragma once

#include <fhenom/common.h>
#include <fhenom/context.h>
#include <openfhe.h>
#include <spdlog/spdlog.h>

#include <cereal/archives/binary.hpp>
#include <filesystem>
#include <vector>

namespace fhenom {

/**
 * @brief This class represents a vector of CKKS encrypted data of arbitrary
 * length and provides some homomorphic vector operations.
 */
class CkksVector {
protected:
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
     * @brief Rectified linear unit (ReLU) activation function
     * 
     * @param degree The degree of the approximation (4 or 11)
     * @return CkksVector the ReLU of the vector
     */
    CkksVector ReLU(unsigned degree = 4) const;

    /**
     * @brief Evaluate a polynomial on the vector elements
     * 
     * @param coeffs The coefficients in decreasing degree
     * @return CkksVector the result of the polynomial evaluation
     */
    CkksVector EvalPoly(const std::vector<double>& coefficients) const;

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

    /**
     * @brief Create a vector consisting of the first slot of each of the parameter vectors
     * 
     * @param vectors A list of vectors to merge
     * @return CkksVector A vector containing the first slot of each input vector in the slot that
     * corresponds to the input vector index
     */
    static CkksVector Merge(const std::vector<CkksVector>& vectors);

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
        this->data_ = std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(data.begin(), data.end());
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

inline fhenom::CkksVector operator+(fhenom::CkksVector lhs, const fhenom::CkksVector& rhs) {
    return lhs += rhs;
}
