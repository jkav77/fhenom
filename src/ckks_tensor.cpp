#include <fhenom/ckks_tensor.h>
#include <fhenom/ckks_vector.h>
#include <fhenom/tensor.h>
#include <spdlog/spdlog.h>

#include <utility>
#include "coefficients.h"

using fhenom::CkksTensor;
using fhenom::Tensor;

void report_time(const std::string& message, const std::chrono::time_point<std::chrono::high_resolution_clock>& start,
                 int64_t& total_time);

CkksTensor::CkksTensor(const CkksVector& data, const shape_t& shape, bool sparse) {
    SetData(data, shape, sparse);
}

void CkksTensor::SetData(const CkksVector& data, const shape_t& shape, bool sparse) {
    unsigned len = 1;
    for (auto dim : shape) {
        len *= dim;
    }
    if (sparse) {
        if (data.size() < len) {
            spdlog::error("Data vector size ({}) is less than shape ({})", data.size(), len);
            throw std::invalid_argument("Data vector size is less than shape");
        }
    }
    else if (data.size() != len) {
        spdlog::error("Data vector size ({}) does not match shape ({})", data.size(), len);
        throw std::invalid_argument("Data vector size does not match shape");
    }

    data_   = data;
    shape_  = shape;
    sparse_ = sparse;
}

std::vector<fhenom::CkksVector> CkksTensor::rotate_images(const fhenom::shape_t& kernel_shape) const {
    auto kernel_num_rows = kernel_shape[2];
    auto kernel_num_cols = kernel_shape[3];
    auto kernel_size     = kernel_num_rows * kernel_num_cols;
    auto data_num_cols   = shape_[2];
    auto padding         = (kernel_num_rows - 1) / 2;

    std::vector<fhenom::CkksVector> rotated_images(kernel_size);
    auto new_data = data_;
    new_data.PrecomputeRotations();

#pragma omp parallel for
    for (unsigned idx = 0; idx < kernel_size; ++idx) {
        const auto row = idx / kernel_num_cols;
        const auto col = idx % kernel_num_cols;
        rotated_images[row * kernel_num_cols + col] =
            new_data.Rotate((row - padding) * data_num_cols + (col - padding));
    }

    return rotated_images;
}

CkksTensor CkksTensor::Dense(const Tensor& weights, const Tensor& bias) const {
    auto weights_shape = weights.GetShape();
    auto num_inputs    = weights_shape[1];
    auto num_outputs   = weights_shape[0];

    if (bias.GetShape().size() != 1) {
        spdlog::error("Bias should have one dimension (has {})", bias.GetShape().size());
        throw std::invalid_argument("Bias does not have one dimension");
    }

    if (bias.GetShape()[0] != num_outputs) {
        spdlog::error("Bias shape ({}) does not match number of outputs ({})", bias.GetShape()[0], num_outputs);
        throw std::invalid_argument("Bias shape does not match number of outputs");
    }

    unsigned flattened_size = data_.size();

    if (flattened_size != num_inputs) {
        spdlog::error("Input size ({}) does not match number of weights ({})", flattened_size, num_inputs);
        throw std::invalid_argument("Input size does not match number of weights");
    }

    int64_t total_inner_product_time = 0;
    auto start                       = std::chrono::high_resolution_clock::now();
    std::vector<CkksVector> channel_outputs(num_outputs);
    auto weights_data = weights.GetData();
#pragma omp parallel for
    for (unsigned output_index = 0; output_index < num_outputs; ++output_index) {
        int64_t inner_product_time = 0;
        auto start                 = std::chrono::high_resolution_clock::now();
        // Multiply by weights for first part of inner product
        auto weights_start = weights_data.begin() + flattened_size * output_index;
        auto weights_end   = weights_start + flattened_size;
        std::vector<double> ctxt_weights(weights_start, weights_end);
        spdlog::debug("Output {}: Multiply by weights", output_index);
        CkksVector output_channel = data_ * ctxt_weights;

        // Sum up the results to complete inner product
        spdlog::debug("Output {}: Sum", output_index);
        output_channel = output_channel.GetSum();
        output_channel.SetNumElements(10);

        // Mask the correct index (all indices have the same value)
        spdlog::debug("Output {}: Mask", output_index);
        std::vector<double> mask(num_outputs);
        mask[output_index] = 1;
        output_channel *= mask;
        report_time("Inner product for channel " + std::to_string(output_index), start, inner_product_time);

        // Add to vector of results so we can use AddMany later
        channel_outputs[output_index] = output_channel;
    }
    report_time("Total inner product time", start, total_inner_product_time);

    // EvalAddMany the results
    spdlog::debug("Output: Add Many");
    auto output_data = CkksVector::AddMany(channel_outputs);

    // Add the bias
    spdlog::debug("Output: Add Bias");
    output_data += bias;

    output_data.SetNumElements(num_outputs);
    return CkksTensor(output_data, {num_outputs});
}

std::pair<bool, std::string> validate_conv2d_input(const fhenom::Tensor& kernel, const fhenom::Tensor& bias,
                                                   const fhenom::shape_t& shape) {
    auto kernel_shape = kernel.GetShape();
    auto num_channels = kernel_shape[1];
    auto num_filters  = kernel_shape[0];

    if (bias.GetShape()[0] != num_filters) {
        spdlog::error("Bias shape ({}) does not match number of filters ({})", bias.GetShape()[0], num_filters);
        return {false, "Bias shape does not match number of filters"};
    }

    if (shape.size() != 3) {
        spdlog::error("Image should have three dimensions (has {}): [channels, rows, cols]", shape.size());
        return {false, "Image does not have three dimensions"};
    }

    if (kernel_shape.size() != 4) {
        spdlog::error(
            "Kernel should have four dimensions (has {}): "
            "[filters, channels, rows, cols]",
            kernel_shape.size());
        return {false, "Kernel does not have four dimensions"};
    }

    if (num_channels != shape[0]) {
        spdlog::error(
            "Kernel channel size ({}) does not match image channel size "
            "({})",
            kernel_shape[2], shape[2]);
        return {false, "Kernel channel size does not match image"};
    }

    if (kernel_shape[2] != kernel_shape[3]) {
        return {false, "Kernel is not square"};
    }

    if (kernel_shape[2] % 2 == 0) {
        return {false, "Kernel size must be odd (e.g. 3x3 or 5x5)"};
    }

    return {true, ""};
}

void report_time(const std::string& message, const std::chrono::time_point<std::chrono::high_resolution_clock>& start,
                 int64_t& total_time) {
    auto end      = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    total_time += duration;
    spdlog::debug("{}: {}ms, {:.1f}s total", message, duration, total_time / 1000.0);
}

CkksTensor CkksTensor::Conv2D(const fhenom::Tensor& kernel, const fhenom::Tensor& bias) const {
    int64_t rotate_images_time           = 0;
    int64_t multiply_rotated_images_time = 0;
    int64_t add_filter_outputs_time      = 0;
    int64_t rotate_to_channel_time       = 0;

    {
        auto [validation_result, message] = validate_conv2d_input(kernel, bias, shape_);
        if (!validation_result) {
            throw std::invalid_argument(message);
        }
    }

    auto kernel_shape             = kernel.GetShape();
    auto kernel_size              = kernel_shape[2] * kernel_shape[3];
    auto num_input_channels       = kernel_shape[1];
    auto num_output_channels      = kernel_shape[0];
    auto kernel_num_rows          = kernel_shape[2];
    auto kernel_num_cols          = kernel_shape[3];
    auto channel_size             = shape_[1] * shape_[2];
    auto data_num_cols            = shape_[2];
    auto padding                  = (kernel_num_rows - 1) / 2;
    auto batch_size               = data_.GetContext().GetCryptoContext()->GetEncodingParams()->GetBatchSize();
    unsigned int num_output_ctxts = ceil(static_cast<double>(num_output_channels) * channel_size / batch_size);

    // Create rotated images, which will be reused for every filter
    auto start          = std::chrono::high_resolution_clock::now();
    auto rotated_images = rotate_images(kernel_shape);
    report_time("Rotate images time", start, rotate_images_time);

    std::vector<CkksVector> channel_outputs;

    // Generate an output channel for every filter in the kernel tensor
    for (unsigned filter_index = 0; filter_index < num_output_channels; ++filter_index) {
        std::vector<std::vector<double>> filter(kernel_size, std::vector<double>(num_input_channels * channel_size));

        // Generate kernel vectors for every (row, column) in the kernel
        for (unsigned row_index = 0; row_index < kernel_num_rows; ++row_index) {
            for (unsigned col_index = 0; col_index < kernel_num_cols; ++col_index) {
                // Add weights for each channel to the filter vector
                for (unsigned channel_index = 0; channel_index < num_input_channels; ++channel_index) {
                    auto channel_offset = channel_index * channel_size;
                    auto kernel_index   = row_index * kernel_num_cols + col_index;
                    auto start          = filter[kernel_index].begin() + channel_offset;
                    auto end            = start + channel_size;

                    std::fill(start, end, kernel.Get({filter_index, channel_index, row_index, col_index}));

                    if (row_index < padding) {  // Zero out the top row(s)
                        auto num_rows_to_pad = padding - row_index;
                        std::fill(start, start + num_rows_to_pad * data_num_cols, 0);
                    }
                    else if (kernel_num_rows - row_index <= padding) {  // Zero out the bottom row(s)
                        auto num_rows_to_pad = padding - (kernel_num_rows - row_index) + 1;
                        std::fill(end - num_rows_to_pad * data_num_cols, end, 0);
                    }

                    if (col_index < padding) {  // Zero out the left column(s)
                        auto num_cols_to_pad = padding - col_index;
                        for (auto mask_it = start; mask_it != end; mask_it += data_num_cols) {
                            std::fill(mask_it, mask_it + num_cols_to_pad, 0);
                        }
                    }
                    else if (kernel_num_cols - col_index <= padding) {  // Zero out the right column(s)
                        auto num_cols_to_pad = padding - (kernel_num_cols - col_index) + 1;
                        for (auto mask_it = start; mask_it != end; mask_it += data_num_cols) {
                            std::fill(mask_it + data_num_cols - num_cols_to_pad, mask_it + data_num_cols, 0);
                        }
                    }
                }
            }
        }

        // Multiply the rotated images by the kernel vectors and add up the results
        start = std::chrono::high_resolution_clock::now();
        std::vector<CkksVector> filter_outputs(kernel_size);
#pragma omp parallel for
        for (unsigned kernel_index = 0; kernel_index < kernel_size; ++kernel_index) {
            filter_outputs[kernel_index] = rotated_images[kernel_index] * filter[kernel_index];
        }
        report_time("Multiply rotated images", start, multiply_rotated_images_time);

        start              = std::chrono::high_resolution_clock::now();
        auto filter_output = CkksVector::AddMany(filter_outputs);
        report_time("Add filter outputs", start, add_filter_outputs_time);

        start = std::chrono::high_resolution_clock::now();
        filter_output.PrecomputeRotations();
        // Consolidate the output of every filter channel in the correct channel
        std::vector<CkksVector> channel_vectors(num_input_channels);
#pragma omp parallel for
        for (unsigned channel_index = 0; channel_index < num_input_channels; ++channel_index) {
            auto ctxt_position = ((channel_index - filter_index) * channel_size) % batch_size;
            spdlog::debug("Moving channel {} in filter {} to position {}", channel_index, filter_index, ctxt_position);
            channel_vectors[channel_index] = filter_output.Rotate(ctxt_position);
        }

        filter_output = CkksVector::AddMany(channel_vectors);
        report_time("Rotate to channel position", start, rotate_to_channel_time);

        spdlog::debug("Add bias");
        filter_output += bias.Get({filter_index});

        // Clone ciphertext to make room for all channels
        {
            auto tmp = filter_output.GetData();
            spdlog::debug("Cloning to create {} ciphertexts and support {} channels size {}", num_output_ctxts,
                          num_output_channels, channel_size);
            std::vector<Ctxt> new_data(num_output_ctxts, tmp[0]);
            filter_output.SetData(new_data, channel_size);
        }

        // Zero out the other channels
        spdlog::debug("Create filter mask for filter output size {} and capacity {}", filter_output.size(),
                      filter_output.capacity());
        std::vector<double> filter_mask(filter_output.capacity(), 0);
        auto start_offset = filter_index * channel_size;
        auto end_offset   = channel_size;
        spdlog::debug("Creating filter with size {}, Start offset: {}, End offset: {}", filter_output.capacity(),
                      start_offset, end_offset);
        auto start = filter_mask.begin() + filter_index * channel_size;
        auto end   = start + channel_size;
        std::fill(start, end, 1);

        spdlog::debug("Apply filter mask");
        filter_output *= filter_mask;
        filter_output.SetNumElements(channel_size);

        channel_outputs.push_back(filter_output);
    }

    // The vector to hold the results of applying all filter convolutions
    CkksVector conv_output = CkksVector::AddMany(channel_outputs);

    spdlog::info("Rotate images time: {:.1f}s", rotate_images_time / 1000.0);
    spdlog::info("Multiply rotated images time: {:.1f}s", multiply_rotated_images_time / 1000.0);
    spdlog::info("Add filter outputs time: {:.1f}s", add_filter_outputs_time / 1000.0);
    spdlog::info("Rotate to channel position time: {:.1f}s", rotate_to_channel_time / 1000.0);

    // Output number of channels is the number of filters we applied
    fhenom::shape_t conv_shape = {num_output_channels, shape_[1], shape_[2]};
    conv_output.SetNumElements(num_output_channels * channel_size);
    return CkksTensor(conv_output, conv_shape);
}

CkksTensor CkksTensor::ReLU(unsigned depth, double scale) const {
    auto relu_function = [](double x) -> double {
        if (x < 0.0) {
            return 0.0;
        }

        return x;
    };

    CkksVector relu;

    switch (depth) {
        case 4:
            relu = data_.EvalChebyshev(relu_function, -scale, scale, 7);
            break;

        case 5:
            relu = data_.EvalChebyshev(relu_function, -scale, scale, 13);
            break;

        case 6:
            relu = data_.EvalChebyshev(relu_function, -scale, scale, 27);
            break;

        case 11:
            relu = data_.EvalChebyshev(relu_function, -scale, scale, 511);
            break;

        case 12:
//             std::vector<double> new_g3_coeffs(kG3Coeffs.size());
//             for (int i = 0; i < kG3Coeffs.size(); ++i) {
// new_g3_coeffs[i] = kG3Coeffs[i]
//             }
//             relu = data_.EvalPoly(kG3Coeffs);
//             relu = relu.EvalPoly(kG2Coeffs);
//             relu = data_ + data_ * relu;
            relu = data_.EvalChebyshev(relu_function, -scale, scale, 2047);
            break;

        default:
            relu = data_.EvalChebyshev(relu_function, -scale, scale, 7);
            break;
    }

    return CkksTensor(relu, shape_, sparse_);
}
