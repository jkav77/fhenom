#include <fhenom/tensor.h>

#include <utility>
#include <vector>

using fhenom::shape_t;
using fhenom::Tensor;

Tensor::Tensor(const std::vector<double>& data, const shape_t& shape) {
    setData(std::move(data), std::move(shape));
}

void Tensor::setData(const std::vector<double>& data, const fhenom::shape_t& shape) {
    size_t len = 1;
    for (auto i : shape) {
        len *= i;
    }

    if (len != data.size()) {
        throw std::invalid_argument("Data size does not match shape");
    }

    this->data_  = std::move(data);
    this->shape_ = shape;
}

double Tensor::get(const shape_t& coordinates) const {
    if (coordinates.size() != shape_.size()) {
        throw std::invalid_argument("Coordinates do not match tensor shape");
    }

    int index = 0;
    for (unsigned i = 0; i < shape_.size() - 1; ++i) {
        if (coordinates[i] > shape_[i] || coordinates[i] < 0) {
            throw std::invalid_argument("Index out of bounds");
        }

        index += coordinates[i] * shape_[i];
    }
    index += coordinates[shape_.size() - 1];

    return data_[index];
}

void Tensor::reshape(const shape_t& shape) {
    size_t len = 1;
    for (auto i : shape) {
        len *= i;
    }

    if (len != data_.size()) {
        throw std::invalid_argument("Data size does not match shape");
    }

    shape_ = shape;
}
