#include <fhenom/tensor.h>
#include <spdlog/spdlog.h>
#include <cereal/archives/binary.hpp>

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

void Tensor::CalculateOffsets() {
    offsets_.clear();
    for (auto it = shape_.begin(); it != shape_.end(); ++it) {
        offsets_.push_back(std::reduce(it + 1, shape_.end(), 1, std::multiplies<int>{}));
    }
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
    CalculateOffsets();
}

std::pair<Tensor, Tensor> Tensor::FuseConvBN(const std::pair<Tensor, Tensor>& conv,
                                             const std::tuple<Tensor, Tensor, Tensor, Tensor>& bn,
                                             const double epsilon) {
    auto [conv_weights, conv_bias]                   = conv;
    auto [bn_weights, bn_bias, bn_mean, bn_variance] = bn;

    auto conv_shape          = conv_weights.GetShape();
    auto num_output_channels = conv_shape[0];
    auto output_channel_size = conv_shape[1] * conv_shape[2] * conv_shape[3];

    std::vector<double> expanded_bn_multiplier(conv_weights.GetData().size(), 0.0);
    std::vector<double> expanded_bn_bias(conv_weights.GetData().size(), 0.0);
    std::vector<double> expanded_bn_mean(conv_weights.GetData().size(), 0.0);

    for (unsigned i = 0; i < num_output_channels; i++) {
        auto start = expanded_bn_multiplier.begin() + i * output_channel_size;
        auto end   = start + output_channel_size;
        auto tmp   = bn_weights.GetData()[i] / (std::sqrt(bn_variance.GetData()[i] + epsilon));
        std::fill(start, end, tmp);

        start = expanded_bn_bias.begin() + i * output_channel_size;
        end   = start + output_channel_size;
        std::fill(start, end, bn_bias.GetData()[i]);

        start = expanded_bn_mean.begin() + i * output_channel_size;
        end   = start + output_channel_size;
        std::fill(start, end, bn_mean.GetData()[i]);
    }

    std::vector<double> new_weights_data(conv_weights.GetData().size(), 0);
    auto conv_weights_data = conv_weights.GetData();
    std::transform(conv_weights_data.begin(), conv_weights_data.end(), expanded_bn_multiplier.begin(),
                   new_weights_data.begin(), std::multiplies<>());

    std::vector<double> new_bias_data(conv_bias.GetData().size());
    auto conv_bias_data = conv_bias.GetData();
    std::transform(conv_bias_data.begin(), conv_bias_data.end(), expanded_bn_mean.begin(), new_bias_data.begin(),
                   std::plus<>());
    std::transform(new_bias_data.begin(), new_bias_data.end(), expanded_bn_multiplier.begin(), new_bias_data.begin(),
                   std::multiplies<>());
    std::transform(new_bias_data.begin(), new_bias_data.end(), expanded_bn_bias.begin(), new_bias_data.begin(),
                   std::minus<>());

    return {Tensor{new_weights_data, conv_weights.GetShape()}, Tensor{new_bias_data, conv_bias.GetShape()}};
}