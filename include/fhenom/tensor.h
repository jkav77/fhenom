#pragma once

#include <vector>

#include "fhenom/common.h"

namespace fhenom {
class Tensor {
    std::vector<double> data_;
    fhenom::shape_t shape_;

public:
    Tensor() = default;
    Tensor(const std::vector<double>& data, const fhenom::shape_t& shape);

    //////////////////////////////////////////////////////////////////////////////
    // Getters and Setters

    std::vector<double> getData() const {
        return data_;
    };

    fhenom::shape_t getShape() const {
        return shape_;
    };

    void reshape(const shape_t& shape);

    void setData(const std::vector<double>& data, const fhenom::shape_t& shape);

    double get(const shape_t& coordinates) const;
};
}  // namespace fhenom
