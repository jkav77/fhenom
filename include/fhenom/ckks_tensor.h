#pragma once

#include <fhenom/ckks_vector.h>
#include <fhenom/common.h>
#include <fhenom/context.h>
#include <fhenom/tensor.h>

namespace fhenom {

class CkksTensor {
    fhenom::CkksVector data_;
    fhenom::shape_t shape_;
    bool sparse_;

public:
    CkksTensor() : data_{}, shape_{0} {}

    CkksTensor(const fhenom::CkksVector& data, const fhenom::shape_t& shape, bool sparse = false);

    //////////////////////////////////////////////////////////////////////////////
    // Homomorphic Operations

    inline CkksTensor& Bootstrap() {
        data_.Bootstrap();
        return *this;
    }

    /**
     * @brief Apply a fully connected layer to the CkksTensor
     * 
     * @param weights a tensor representing the weights, should have shape [input_size, output_size]
     * @param bias a tensor of the biases
     * @return fhenom::CkksTensor the transformed tensor
     */
    fhenom::CkksTensor Dense(const fhenom::Tensor& weights, const fhenom::Tensor& bias) const;

    /**
   * @brief Apply the kernel to the CkksTensor
   *
   * @param kernel A tensor representing the kernel to apply
   * @param bias A tensor of the biases
   * @return fhenom::CkksTensor The result of the convolution
   *
   * @note The kernel must have the same number of dimensions as the CkksTensor
   */
    fhenom::CkksTensor Conv2D(const fhenom::Tensor& kernel, const fhenom::Tensor& bias) const;

    /**
     * @brief Apply average pooling to the CkksTensor
     * 
     * @return fhenom::CkksTensor the transformed tensor
     * 
     * @note This method only does average pooling with a 2x2 window
     */
    fhenom::CkksTensor AvgPool2D() const;

    /**
     * @brief Rectified linear unit (ReLU) activation function
     * 
     * @param depth The depth of the approximation (4 or 11)
     * @param scale Scaling factor to bring x into [-1,1]
     * @return CkksTensor the ReLU of the tensor
     * 
     * @note This returns (x + x * sign(x/scale)) / 2
     */
    CkksTensor ReLU(unsigned depth = 4, double scale = 1) const;

    //////////////////////////////////////////////////////////////////////////////
    // Getters and Setters

    void SetData(const fhenom::CkksVector& data, const shape_t& shape, bool sparse = false);

    fhenom::CkksVector GetData() const {
        return data_;
    }
    fhenom::shape_t GetShape() const {
        return shape_;
    }

    unsigned GetIndex(fhenom::shape_t position) const;

    //////////////////////////////////////////////////////////////////////////////
    // Utility Functions

    std::vector<CkksVector> rotate_images(const shape_t& kernel_shape) const;
};
}  // namespace fhenom
