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
  size_t numElements_;
  fhenom::Context context_;
  std::vector<PrecomputedRotations> precomputedRotations_;

 public:
  CkksVector() : data_{}, numElements_{0} {}
  CkksVector(const fhenom::Context &context)
      : data_{}, numElements_{0}, context_{context} {}
  CkksVector(const std::vector<Ctxt> &data, size_t numElements,
             const fhenom::Context &context)
      : data_{data}, numElements_{numElements}, context_{context} {}

  ////////////////////////////////////////////////////////////////////////////
  // Encryption and Decryption

  /**
   * @brief Encrypts a vector of data
   *
   * @param data The unencrypted data
   */
  void encrypt(const std::vector<double> &data);

  /**
   * @brief Decrypts the data in this vector
   *
   * @return vector<double> Returns the decrypted data vector
   */
  std::vector<double> decrypt() const;

  ////////////////////////////////////////////////////////////////////////////
  // Homomorphic Operations

  /**
   * @brief Evaluates the sign in each slot of the vector
   *
   * @return CkksVector A vector with -1 in negative slots, 0 in 0 slots, and 1
   * in positive slots
   */
  inline CkksVector Sign() const { return SignUsingChebyshev(); };

  /**
   * @brief Evaluates the sign in each slot of the vector using the Chebyshev
   * function
   *
   * @return CkksVector A vector with -1 in negative slots, 0 in 0 slots, and 1
   * in positive slots
   */
  CkksVector SignUsingChebyshev(const double lower_bound = -100,
                                const double upper_bound = 100,
                                const uint32_t degree = 2031) const;

  /**
   * @brief Checks if the data in this vector is equal to a plaintext value
   *
   * @param value The value to compare against
   * @return CkksVector A vector of `1`s in the place of equal slots; `0`s
   * elsewhere
   */
  CkksVector IsEqual(const double value) const;

  /**
   * @brief Sum all elements in a vector
   *
   * @return CkksVector A vector where slot 1 contains the sum of all values in
   * the vector
   */
  CkksVector Sum() const;

  /**
   * @brief Rotates the data in this vector (+ is left, - is right)
   *
   * @param r
   * @return CkksVector
   *
   * @note The rotations occur within the each ciphertext, length `N/2`. For
   * N=8192, `rotate(1)` would send index 0 to index 4095, and index 4096 to
   * index 8191.
   */
  CkksVector rotate(int r) const;

  /**
   * @brief Elementwise multiplication with a plaintext of equal size
   *
   * @param rhs The plaintext to multiply
   * @return CkksVector& A reference to this object
   */
  CkksVector &operator*=(const std::vector<double> &rhs);

  /**
   * @brief Elementwise multiplication with a plaintext of equal size
   *
   * @param rhs The vector to multiply
   * @return CkksVector& A reference to this object
   */
  CkksVector &operator*=(const CkksVector &rhs);

  /**
   * @brief Elementwise addition of another encrypted vector
   *
   * @param rhs the vector to add
   * @return CkksVector& A reference to this object
   */
  CkksVector &operator+=(const CkksVector &rhs);

  /**
   * @brief Elementwise subtraction
   *
   * @param rhs
   * @return CkksVector&
   */
  CkksVector &operator-=(const double &rhs);

  friend CkksVector operator-(const double &lhs, CkksVector rhs);

  /**
   * @brief Precompute rotations to use fast rotations
   */
  void precomputeRotations();

  //////////////////////////////////////////////////////////////////////////////
  // Modifiers

  /**
   * @brief Concatenate a vector onto this vector
   *
   * @param rhs The vector to concatenate on the end
   *
   * @note Unused slots will be preserved and may create a gap in the data. This
   * method works best when the vector has used all of the slots.
   */
  void Concat(const CkksVector &rhs);

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
  void serialize(Archive &archive);

  /**
   * @brief Loads data from a file
   *
   * @param path The path to the file
   */
  void load(const std::filesystem::path &path);

  /**
   * @brief Saves data to a file
   *
   * @param path The path to the file
   */
  void save(const std::filesystem::path &path) const;

  ////////////////////////////////////////////////////////////////////////////
  // Getters and Setters

  /**
   * @brief Get the data
   *
   * @return vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>> The encrypted
   * data
   */
  inline std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> getData() const {
    return data_;
  }

  /**
   * @brief Set the data
   *
   * @param data The encrypted data
   */
  inline void setData(
      const std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> &data,
      size_t numElements) {
    this->data_ = std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(
        data.begin(), data.end());
  }

  /**
   * @brief Set the Context object
   *
   * @param context
   */
  inline void setContext(const fhenom::Context &context) {
    this->context_ = context;
  }

  /**
   * @brief Get the Context object
   *
   * @return fhenom::Context
   */
  inline fhenom::Context getContext() const { return context_; }

  /**
   * @brief Get the number of elements
   *
   * @return size_t The number of elements
   */
  inline size_t size() const { return numElements_; }

  inline void setNumElements(size_t numElements) {
    this->numElements_ = numElements;
  }
};

}  // namespace fhenom

inline fhenom::CkksVector operator*(fhenom::CkksVector lhs,
                                    const fhenom::CkksVector &rhs) {
  return lhs *= rhs;
}

inline fhenom::CkksVector operator*(fhenom::CkksVector lhs,
                                    const std::vector<double> &rhs) {
  return lhs *= rhs;
}

inline fhenom::CkksVector operator+(fhenom::CkksVector lhs,
                                    const fhenom::CkksVector &rhs) {
  return lhs += rhs;
}

inline fhenom::CkksVector operator-(fhenom::CkksVector lhs, const double &rhs) {
  return lhs -= rhs;
}
