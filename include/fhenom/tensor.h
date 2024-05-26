#pragma once

#include <vector>

#include "fhenom/common.h"

namespace fhenom {

/**
 * @brief A tensor for unencrypted data
 * 
 */
class Tensor {
    std::vector<double> data_;
    fhenom::shape_t shape_;
    std::vector<int> offsets_;

public:
    Tensor() = default;

    /**
     * @brief Construct a new Tensor object
     * 
     * @param data the data to store in the tensor
     * @param shape the shape of the tensor must be consistent with data size
     * 
     * @note The shape is in the form [filters, channels, rows, cols]
     */
    Tensor(const std::vector<double>& data, const fhenom::shape_t& shape);

    //////////////////////////////////////////////////////////////////////////////
    // Getters and Setters

    std::vector<double> GetData() const {
        return data_;
    };

    fhenom::shape_t GetShape() const {
        return shape_;
    };

    /**
     * @brief Changes the shape of the tensor without altering the underlying data
     * 
     * @param shape The new shape, which must match the size of the data
     */
    void Reshape(const shape_t& shape);

    /**
     * @brief Set the Data object
     * 
     * @param data the data to store in the tensor
     * @param shape the shape of the tensor must be consistent with data size
     */
    void SetData(const std::vector<double>& data, const fhenom::shape_t& shape);

    /**
     * @brief Retrieve a specific element from the tensor
     * 
     * @param coordinates the indices of the element to retrieve, which must match the tensor shape
     * @return double the stored value
     */
    double Get(const shape_t& coordinates) const;

    /**
     * @brief Tells cereal how to serialize this class
     *
     * @tparam Archive
     * @param archive
     */
    template <class Archive>
    void serialize(Archive& archive);
};

}  // namespace fhenom
