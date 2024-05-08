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
    fhenom::CkksTensor Conv2D(const fhenom::Tensor& kernel);

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
