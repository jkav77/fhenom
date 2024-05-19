#pragma once

#include <fhenom/ckks_vector.h>
#include <fhenom/common.h>
#include <fhenom/context.h>
#include <fhenom/tensor.h>

namespace fhenom {

class CkksTensor {
    fhenom::CkksVector data_;
    fhenom::shape_t shape_;
    unsigned stripe_{0};

public:
    CkksTensor() : data_{}, shape_{0} {}

    CkksTensor(fhenom::CkksVector data, fhenom::shape_t shape, bool sparse = false);

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
    fhenom::CkksTensor Conv2D(const fhenom::Tensor& kernel, const fhenom::Tensor& bias);

    /**
     * @brief Apply average pooling to the CkksTensor
     * 
     * @return fhenom::CkksTensor the transformed tensor
     * 
     * @note This method only does average pooling with a 2x2 window
     */
    fhenom::CkksTensor AvgPool2D();

    //////////////////////////////////////////////////////////////////////////////
    // Getters and Setters

    void SetData(fhenom::CkksVector data, shape_t shape, bool sparse = false);

    fhenom::CkksVector GetData() const {
        return data_;
    }
    fhenom::shape_t GetShape() const {
        return shape_;
    }

    void SetStripe(unsigned stripe) {
        stripe_ = stripe;
    }

    unsigned GetStripe() const {
        return stripe_;
    }

    unsigned GetIndex(fhenom::shape_t position) const;

    //////////////////////////////////////////////////////////////////////////////
    // Utility Functions

    std::vector<CkksVector> rotate_images(const shape_t& kernel_shape);
};
}  // namespace fhenom
