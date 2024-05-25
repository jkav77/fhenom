#include <fhenom/tensor.h>
#include <spdlog/spdlog.h>

#include <utility>
#include <vector>

using fhenom::shape_t;
using fhenom::Tensor;

Tensor::Tensor(const std::vector<double>& data, const shape_t& shape) {
    SetData(data, shape);
}

void Tensor::SetData(const std::vector<double>& data, const fhenom::shape_t& shape) {
    size_t len = 1;
    for (auto i : shape) {
        len *= i;
    }

    if (len != data.size()) {
        spdlog::error("Data size {} does not match shape {}", data.size(), len);
        throw std::invalid_argument("Data size does not match shape");
    }

    this->data_ = data;
    Reshape(shape);
}

double Tensor::Get(const shape_t& coordinates) const {
    if (coordinates.size() != shape_.size()) {
        throw std::invalid_argument("Coordinates do not match tensor shape");
    }

    int index = 0;
    for (unsigned i = 0; i < shape_.size(); ++i) {
        index += coordinates[i] * offsets_[i];
    }

    return data_[index];
}

void Tensor::Reshape(const shape_t& shape) {
    size_t len = 1;
    for (auto i : shape) {
        len *= i;
    }

    if (len != data_.size()) {
        throw std::invalid_argument("Data size does not match shape");
    }

    this->shape_ = shape;
    offsets_.clear();
    for (auto it = shape_.begin(); it != shape_.end(); ++it) {
        offsets_.push_back(std::reduce(it + 1, shape_.end(), 1, std::multiplies<int>{}));
    }
}
