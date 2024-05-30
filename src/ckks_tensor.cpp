#include <fhenom/ckks_tensor.h>
#include <fhenom/ckks_vector.h>
#include <fhenom/tensor.h>
#include <spdlog/spdlog.h>

#include <utility>
#include "coefficients.h"

using fhenom::CkksTensor;
using fhenom::Tensor;

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

    std::vector<CkksVector> channel_outputs(num_outputs);
    auto weights_data = weights.GetData();
#pragma omp parallel for
    for (unsigned output_index = 0; output_index < num_outputs; ++output_index) {
        // Multiply by weights
        auto weights_start = weights_data.begin() + flattened_size * output_index;
        auto weights_end   = weights_start + flattened_size;
        std::vector<double> ctxt_weights(weights_start, weights_end);
        CkksVector output_channel = data_ * ctxt_weights;

        // Sum up the results
        output_channel = output_channel.GetSum();
        output_channel.SetNumElements(10);

        // Mask the correct index
        std::vector<double> mask(num_outputs);
        mask[output_index] = 1;
        output_channel *= mask;

        // Add to vector of results
        channel_outputs[output_index] = output_channel;
    }

    // EvalAddMany the results
    auto output_data = CkksVector::AddMany(channel_outputs);

    // Add the bias
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
    static int64_t rotate_images_time           = 0;
    static int64_t gen_filter_vectors_time      = 0;
    static int64_t multiply_rotated_images_time = 0;
    static int64_t move_filter_outputs_time     = 0;
    static int64_t add_filter_outputs_time      = 0;
    static int64_t concat_filter_outputs_time   = 0;

    {
        auto [validation_result, message] = validate_conv2d_input(kernel, bias, shape_);
        if (!validation_result) {
            throw std::invalid_argument(message);
        }
    }

    auto kernel_shape    = kernel.GetShape();
    auto kernel_size     = kernel_shape[2] * kernel_shape[3];
    auto num_channels    = kernel_shape[1];
    auto num_filters     = kernel_shape[0];
    auto kernel_num_rows = kernel_shape[2];
    auto kernel_num_cols = kernel_shape[3];
    auto channel_size    = shape_[1] * shape_[2];
    auto data_num_cols   = shape_[2];
    auto padding         = (kernel_num_rows - 1) / 2;

    // Create rotated images, which will be reused for every filter
    auto start          = std::chrono::high_resolution_clock::now();
    auto rotated_images = rotate_images(kernel_shape);
    report_time("Rotate images time", start, rotate_images_time);

    // The vector to hold the results of applying all filter convolutions
    CkksVector conv_output(data_.GetContext());

    // Generate an output channel for every filter in the kernel tensor
    for (unsigned filter_index = 0; filter_index < num_filters; ++filter_index) {
        std::vector<std::vector<double>> filter(kernel_size, std::vector<double>(num_channels * channel_size));

        start = std::chrono::high_resolution_clock::now();
        // Generate a filter vector for every (row, column) in the kernel
        for (unsigned row_index = 0; row_index < kernel_num_rows; ++row_index) {
            for (unsigned col_index = 0; col_index < kernel_num_cols; ++col_index) {
                // Add weights for each channel to the filter vector
                for (unsigned channel_index = 0; channel_index < num_channels; ++channel_index) {
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
        report_time("Generate filter vectors", start, gen_filter_vectors_time);

        // Multiply the rotated images by the kernel vectors and add up the results
        start = std::chrono::high_resolution_clock::now();
        std::vector<CkksVector> filter_outputs(kernel_size);
#pragma omp parallel for
        for (unsigned kernel_index = 0; kernel_index < kernel_size; ++kernel_index) {
            filter_outputs[kernel_index] = rotated_images[kernel_index] * filter[kernel_index];
        }
        report_time("Multiply rotated images", start, multiply_rotated_images_time);

        start = std::chrono::high_resolution_clock::now();
        std::vector<Ctxt> ctxts;
        for (auto& vec : filter_outputs) {
            auto data = vec.GetData();
            std::move(data.begin(), data.end(), std::back_inserter(ctxts));
        }
        report_time("Move filter outputs", start, move_filter_outputs_time);

        start               = std::chrono::high_resolution_clock::now();
        auto crypto_context = data_.GetContext().GetCryptoContext();
        auto filter_ctxt    = crypto_context->EvalAddMany(ctxts);
        auto filter_output  = CkksVector({filter_ctxt}, channel_size, data_.GetContext()) + bias.Get({filter_index});
        report_time("Add filter outputs", start, add_filter_outputs_time);
        //         filter_output.PrecomputeRotations();

        //         // Consolidate the output of every filter channel in the first channel
        //         std::vector<CkksVector> channel_vectors(num_channels);
        // #pragma omp parallel for
        //         for (unsigned channel_index = 0; channel_index < num_channels; ++channel_index) {
        //             channel_vectors[channel_index] = filter_output.Rotate(channel_index * channel_size);
        //         }

        //         filter_output = CkksVector::AddMany(channel_vectors);

        //         // Zero out the other channels
        //         std::vector<double> filter_mask(num_channels * channel_size, 0);
        //         std::fill(filter_mask.begin(), filter_mask.begin() + channel_size, 1);
        //         filter_output += bias.Get({filter_index});
        //         filter_output *= filter_mask;
        //         filter_output.SetNumElements(channel_size);

        // conv_output.Concat(filter_output);
        start = std::chrono::high_resolution_clock::now();
        conv_output.Concat(filter_output);
        report_time("Concat filter outputs", start, concat_filter_outputs_time);
    }

    spdlog::info("Rotate images time: {:.1f}s", rotate_images_time);
    spdlog::info("Generate filter vectors time: {:.1f}s", gen_filter_vectors_time);
    spdlog::info("Multiply rotated images time: {:.1f}s", multiply_rotated_images_time);
    spdlog::info("Move filter outputs time: {:.1f}s", move_filter_outputs_time);
    spdlog::info("Add filter outputs time: {:.1f}s", add_filter_outputs_time);
    spdlog::info("Concat filter outputs time: {:.1f}s", concat_filter_outputs_time);

    // Output number of channels is the number of filters we applied
    fhenom::shape_t conv_shape = {num_filters, shape_[1], shape_[2]};
    conv_output.SetNumElements(num_filters * channel_size);
    return CkksTensor(conv_output, conv_shape);
}

// CkksTensor CkksTensor::AvgPool2D() const {
//     const auto crypto_context = data_.GetContext().GetCryptoContext();
//     const auto batch_size     = crypto_context->GetEncodingParams()->GetBatchSize();
//     const shape_t new_shape{shape_[0], shape_[1] / 2, shape_[2] / 2};
//     const auto new_channel_size = new_shape[1] * new_shape[2];

//     CkksVector output_data = data_;
//     std::vector<double> mask(output_data.GetData().size() * batch_size);

//     // Average
//     output_data += output_data.Rotate(1) + output_data.Rotate(shape_[2]) + output_data.Rotate(shape_[2] + 1);
//     for (unsigned i = 0; i < mask.size(); ++i) {
//         mask[i] = (i / shape_[2]) % 2 == 0 && i % 2 == 0 ? 0.25 : 0;  // Every other column in every other row
//     }
//     output_data *= mask;

//     // Condense columns
//     for (int i = 0; i < log2(shape_[2]) - 1; ++i) {
//         output_data += output_data.Rotate(1 << i);
//         for (unsigned j = 0; j < mask.size(); ++j) {
//             mask[j] = (j / shape_[2]) % 2 == 0 && j / (1 << (i + 1)) % 2 == 0 ? 1 : 0;
//         }
//         output_data *= mask;
//     }

//     // Condense rows
//     for (int i = 0; i < log2(new_shape[1]) - 1; i += 2) {
//         auto num_rotations = std::min(static_cast<unsigned>(4), new_shape[1] / (1 << i));

//         if (num_rotations == 4) {
//             output_data.PrecomputeRotations();
//         }
//         std::vector<CkksVector> rotations(num_rotations - 1);
// #pragma omp parallel for
//         for (unsigned j = 1; j < num_rotations; ++j) {
//             auto rotation_amount = 3 * j * (new_shape[2] * (1 << i));
//             rotations[j - 1]     = output_data.Rotate(rotation_amount);
//         }
//         for (auto& ctxt : rotations) {
//             output_data += ctxt;
//         }

// #pragma omp parallel for
//         for (unsigned j = 0; j < mask.size(); ++j) {
//             mask[j] = (j / (4 * new_shape[2] * (1 << i))) % 4 == 0 ? 1 : 0;
//         }
//         output_data *= mask;
//     }

//     // Condense Channels
//     for (int i = 0; i < log2(new_shape[0]) - 1; ++i) {
//         auto num_rotations  = std::min(static_cast<unsigned>(4), new_shape[0] / (1 << i));
//         unsigned block_size = new_channel_size * pow(4, i);
//         if (16 * block_size > batch_size) {
//             break;
//         }

//         // Move channels into position
//         if (num_rotations == 4) {
//             output_data.PrecomputeRotations();
//         }
//         std::vector<CkksVector> rotations(num_rotations - 1);
// #pragma omp parallel for
//         for (unsigned j = 1; j < num_rotations; ++j) {
//             auto rotation_amount = 3 * j * block_size;
//             rotations[j - 1]     = output_data.Rotate(rotation_amount);
//         }

//         for (auto& ctxt : rotations) {
//             output_data += ctxt;
//         }
//         // Mask in the first and mask out the next three (channels / block of 4 channels / block of 16 channels / etc.)
//         std::fill(mask.begin(), mask.end(), 0);
// #pragma omp parallel for
//         for (unsigned j = 0; j < mask.size(); ++j) {
//             auto block_number = j / (block_size * 4);
//             mask[j]           = block_number % 4 == 0 ? 1 : 0;
//         }
//         output_data *= mask;
//     }

//     // Condense Ciphertexts
//     output_data.Condense(batch_size / 4);

//     output_data.SetNumElements(new_shape[0] * new_shape[1] * new_shape[2]);
//     return CkksTensor(output_data, new_shape);
// }

CkksTensor CkksTensor::ReLU(unsigned depth, double scale) const {
    auto relu_function = [](double x) -> double {
        if (x < 0.0) {
            return 0.0;
        }

        return x;
    };

    CkksVector relu;

    if (scale != 1.0) {
        depth--;
    }

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
            relu = data_.EvalChebyshev(relu_function, -scale, scale, 2047);
            break;

        default:
            relu = data_.EvalChebyshev(relu_function, -scale, scale, 7);
            break;
    }

    return CkksTensor(relu, shape_, sparse_);
}
