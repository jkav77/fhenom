#pragma once

#include <fhenom/ckks_vector.h>
#include <fhenom/common.h>
#include <fhenom/context.h>
#include <fhenom/tensor.h>

namespace fhenom {

class CkksTensor {
    fhenom::CkksVector data_;
    fhenom::shape_t shape_;

public:
    CkksTensor() : data_{}, shape_{0} {}

    CkksTensor(fhenom::CkksVector data, fhenom::shape_t shape);

    //////////////////////////////////////////////////////////////////////////////
    // Homomorphic Operations

    /**
   * @brief Apply the kernel to the CkksTensor
   *
   * @param kernel A tensor representing the kernel to apply
   * @return fhenom::CkksTensor The result of the convolution
   *
   * @note The kernel must have the same number of dimensions as the CkksTensor
   */
    fhenom::CkksTensor Conv2D(fhenom::Tensor kernel);

    //////////////////////////////////////////////////////////////////////////////
    // Convolution Helper Functions

    /**
   * @brief Create a Masked Convolution vectors from the provided kernel
   *
   * @param kernel The kernel to be applied to the CkksTensor
   * @return std::vector<fhenom::Tensor>
   */
    std::vector<std::vector<double>> createMaskedConvVectors(const Tensor& kernel, size_t filter_number) const;

    /**
   * @brief Mask the provided vector with 0s on the left `numCols` elements of
   * each row
   *
   * @param vec The vector to modify
   * @param numCols The number of columns to mask (default 1)
   */
    void maskLeft(std::vector<double>& vec, size_t numCols = 1) const;

    /**
   * @brief Mask the provided vector with 0s on the top `numRows` rows
   *
   * @param vec The vector to modify
   * @param rowSize The row size
   * @param numRows The number of rows to mask (default 1)
   */
    void maskTop(std::vector<double>& vec, size_t numRows = 1) const;

    /**
   * @brief Mask the provided vector with 0s on the bottom `numRows` rows
   *
   * @param vec The vector to modify
   * @param rowSize The row size
   * @param numRows The number of rows to mask (default 1)
   */
    void maskBottom(std::vector<double>& vec, size_t numRows = 1) const;

    /**
   * @brief Mask the provided vector with 0s on the right `numCols` elements of
   * each row
   *
   * @param vec The vector to modify
   * @param rowSize The row size
   * @param numCols The number of columns to mask (default 1)
   */
    void maskRight(std::vector<double>& vec, size_t numCols = 1) const;

    //////////////////////////////////////////////////////////////////////////////
    // Getters and Setters

    fhenom::CkksVector GetData() const {
        return data_;
    }
    fhenom::shape_t GetShape() const {
        return shape_;
    }

    void SetData(fhenom::CkksVector data, shape_t shape);
};
}  // namespace fhenom
