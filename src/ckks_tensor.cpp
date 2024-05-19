#include <fhenom/ckks_tensor.h>
#include <fhenom/ckks_vector.h>
#include <spdlog/spdlog.h>

#include <utility>

using fhenom::CkksTensor;

CkksTensor::CkksTensor(CkksVector data, shape_t shape, bool sparse) {
    SetData(std::move(data), std::move(shape), sparse);
}

void CkksTensor::SetData(CkksVector data, shape_t shape, bool sparse) {
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

    data_  = std::move(data);
    shape_ = std::move(shape);
}

std::vector<fhenom::CkksVector> CkksTensor::rotate_images(const fhenom::shape_t& kernel_shape) {
    auto kernel_num_rows = kernel_shape[2];
    auto kernel_num_cols = kernel_shape[3];
    auto kernel_size     = kernel_num_rows * kernel_num_cols;
    auto data_num_cols   = shape_[2];
    auto padding         = (kernel_num_rows - 1) / 2;

    std::vector<fhenom::CkksVector> rotated_images(kernel_size);

    for (int row = 0; row < kernel_num_rows; ++row) {
        for (int col = 0; col < kernel_num_cols; ++col) {
            rotated_images[row * kernel_num_cols + col] =
                data_.Rotate((row - padding) * data_num_cols + (col - padding));
        }
    }

    return rotated_images;
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

CkksTensor CkksTensor::Conv2D(const fhenom::Tensor& kernel, const fhenom::Tensor& bias) {
    {
        auto validation_result = validate_conv2d_input(kernel, bias, shape_);
        if (!validation_result.first) {
            throw std::invalid_argument(validation_result.second);
        }
    }

    auto kernel_shape    = kernel.GetShape();
    auto kernel_size     = kernel_shape[2] * kernel_shape[3];
    auto num_channels    = kernel_shape[1];
    auto num_filters     = kernel_shape[0];
    auto kernel_num_rows = kernel_shape[2];
    auto kernel_num_cols = kernel_shape[3];
    auto crypto_context  = data_.GetContext().GetCryptoContext();
    auto channel_size    = shape_[1] * shape_[2];
    auto data_num_cols   = shape_[2];
    auto padding         = (kernel_num_rows - 1) / 2;

    // Create rotated images, which will be reused for every filter
    auto rotated_images = rotate_images(kernel_shape);

    // The vector to hold the results of applying all filter convolutions
    CkksVector conv_output(data_.GetContext());

    // Generate an output channel for every filter in the kernel tensor
    for (unsigned filter_index = 0; filter_index < num_filters; ++filter_index) {
        std::vector<std::vector<double>> filter(kernel_size, std::vector<double>(num_channels * channel_size));

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

        // Multiply the rotated images by the kernel vectors and add up the results
        CkksVector filter_output(data_.GetContext());
        for (unsigned kernel_index = 0; kernel_index < kernel_size; ++kernel_index) {
            if (filter_output.size() == 0) {
                filter_output = rotated_images[kernel_index] * filter[kernel_index];
            }
            else {
                filter_output += rotated_images[kernel_index] * filter[kernel_index];
            }
        }

        // Consolidate the output of every filter channel in the first channel
        for (auto channel_index = 1; channel_index < num_channels; ++channel_index) {
            filter_output += filter_output.Rotate(channel_index * channel_size);
        }

        // Zero out the other channels
        std::vector<double> filter_mask(num_channels * channel_size, 0);
        std::fill(filter_mask.begin(), filter_mask.begin() + channel_size, 1);
        filter_output += bias.Get({filter_index});
        filter_output *= filter_mask;
        filter_output.SetNumElements(channel_size);

        conv_output.Concat(filter_output);
    }

    // Output number of channels is the number of filters we applied
    fhenom::shape_t conv_shape = {num_filters, shape_[1], shape_[2]};
    conv_output.SetNumElements(num_filters * channel_size);
    return CkksTensor(conv_output, conv_shape);
}

CkksTensor CkksTensor::AvgPool2D() {
    auto crypto_context     = data_.GetContext().GetCryptoContext();
    const auto channel_size = shape_[1] * shape_[2];

    int row_shift = shape_[2];
    int col_shift = 1;
    if (stripe_ == 1) {
        row_shift = 1;  // With two stripes, the rows to average are already adjacent
        col_shift = 2;
    }

    CkksVector output_data =
        data_ + data_.Rotate(col_shift) + data_.Rotate(row_shift) + data_.Rotate(row_shift + col_shift);
    output_data *= 0.25;

    auto decrypted = output_data.Decrypt();
    spdlog::debug("Decrypted: {}", decrypted[0]);

    // Mask every other column in every other row
    std::vector<double> mask(data_.size(), 0);
    for (unsigned channel = 0; channel < shape_[0]; ++channel) {
        for (unsigned row = 0; row < shape_[1]; ++row) {
            for (unsigned col = 0; col < shape_[2]; ++col) {
                if (row % 2 == 0) {
                    mask[channel * channel_size + row * shape_[2] + col] = col % 2 == 0 ? 1 : 0;
                }
            }
        }
    }
    output_data *= mask;

    // Stripe the rows
    output_data += output_data.Rotate(2 * shape_[2] - 1);
    for (unsigned index = 0; index < data_.size(); ++index) {
        mask[index] = (index / (shape_[2])) % 4 == 0 ? 1 : 0;
    }
    output_data *= mask;

    // Condense ciphertexts
    auto ctxts = output_data.GetData();
    std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> new_ctxts(ceil(static_cast<double>(ctxts.size()) / 4));
    for (int index = 0; index < ctxts.size(); ++index) {
        if (index % 4 == 0) {
            new_ctxts[index / 4] = std::move(ctxts[index]);
        }
        else {
            new_ctxts[index / 4] += crypto_context->EvalRotate(ctxts[index], -(index % 4) * shape_[2]);
        }
    }

    CkksTensor output_tensor(output_data, {shape_[0], shape_[1] / 2, shape_[2] / 2}, true);
    output_tensor.SetStripe(1);

    return output_tensor;
}

unsigned CkksTensor::GetIndex(fhenom::shape_t position) const {
    const auto channel      = position[0];
    const auto row          = position[1];
    const auto col          = position[2];
    const auto channel_size = shape_[1] * shape_[2];

    if (channel >= shape_[0] || row >= shape_[1] || col >= shape_[2]) {
        spdlog::error("Position ({}, {}, {}) is out of bounds for shape ({}, {}, {})", channel, row, col, shape_[0],
                      shape_[1], shape_[2]);
        throw std::invalid_argument("Position is out of bounds");
    }

    if (stripe_ == 0) {
        return channel * channel_size + row * shape_[2] + col;
    }

    const auto crypto_context          = data_.GetContext().GetCryptoContext();
    const auto batch_size              = crypto_context->GetEncodingParams()->GetBatchSize();
    const auto channels_per_ciphertext = batch_size / channel_size;
    const auto ctxt_offset             = (channel / channels_per_ciphertext) * batch_size;
    const auto ch_block_offset         = (channel % channels_per_ciphertext) * 4 * channel_size;
    const auto ch_offset               = ((channel / channels_per_ciphertext) % 4) * 2 * shape_[2];
    const auto row_offset              = (row / 2) * 8 * shape_[2] + (row % 2);
    const auto col_offset              = 2 * col;

    const auto index = ctxt_offset + ch_block_offset + ch_offset + row_offset + col_offset;

    return index;
}
