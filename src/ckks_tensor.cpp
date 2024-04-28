#include <fhenom/ckks_tensor.h>
#include <fhenom/ckks_vector.h>
#include <spdlog/spdlog.h>

#include <utility>

using fhenom::CkksTensor;

CkksTensor::CkksTensor(CkksVector data, shape_t shape) {
    SetData(std::move(data), std::move(shape));
}

void CkksTensor::SetData(CkksVector data, shape_t shape) {
    unsigned len = 1;
    for (auto dim : shape) {
        len *= dim;
    }
    if (data.size() != len) {
        spdlog::error("Data vector size ({}) does not match shape ({})", data.size(), len);
        throw std::invalid_argument("Data vector size does not match shape");
    }

    data_  = std::move(data);
    shape_ = std::move(shape);
}

std::vector<fhenom::CkksVector> RotateImages(const fhenom::CkksVector& data, unsigned kernel_size) {
    auto rotation_range = (kernel_size - 1) / 2;
    std::vector<fhenom::CkksVector> rotated_images(kernel_size);
    for (int i = 0; i < kernel_size; ++i) {
        rotated_images[i] = data.Rotate(i - rotation_range);
    }
    return rotated_images;
}

std::vector<double> CreateKernelElementVector(const std::vector<double>& element, const unsigned channel_size,
                                              const int rotation) {
    auto num_channels = element.size();
    std::vector<double> kernel_vector(channel_size * num_channels, 0);

    for (unsigned channel = 0; channel < num_channels; ++channel) {
        unsigned start = std::max(0, static_cast<int>(channel * channel_size + rotation));
        unsigned end   = std::min(static_cast<unsigned>(kernel_vector.size()), (channel + 1) * channel_size + rotation);
        for (unsigned idx = start; idx < end; ++idx) {
            kernel_vector[idx] = element[channel];
        }
    }
    return kernel_vector;
}

void MaskRows(std::vector<double>& vec, int channel_size, int num_channels, int num_cols, int start_row, int rotation,
              int num_rows = 1) {
    for (int channel = 0; channel < num_channels; ++channel) {
        int channel_begin = channel * channel_size;
        for (int row = 0; row < num_rows; ++row) {
            int row_begin = channel_begin + start_row * num_cols + row * num_cols;
            for (int col = 0; col < num_cols; ++col) {
                int idx  = std::max(0, row_begin + col + rotation);
                idx      = std::min(static_cast<int>(vec.size() - 1), idx);
                vec[idx] = 0;
            }
        }
    }

    for (int i = 0; i < rotation; ++i) {
        vec[i] = 0;
    }

    for (int i = -1; i >= rotation; --i) {
        vec[vec.size() + i] = 0;
    }
}

void MaskCols(std::vector<double>& vec, int row_size, int start_col, int rotation, int num_cols = 1) {
    for (int row = 0; row < vec.size() / row_size; ++row) {
        int row_begin = row * row_size;
        for (int col = start_col; col < num_cols + start_col; col++) {
            int idx  = std::max(0, row_begin + col + rotation);
            idx      = std::min(static_cast<int>(vec.size() - 1), idx);
            vec[idx] = 0;
        }
    }
}

std::vector<std::vector<double>> CreateKernelVectors(const fhenom::Tensor& kernel, const fhenom::shape_t& image_shape,
                                                     const unsigned filter_number) {
    auto kernel_shape   = kernel.GetShape();
    auto num_cols       = kernel_shape[3];
    auto num_rows       = kernel_shape[2];
    auto kernel_size    = kernel_shape[2] * kernel_shape[3];
    auto num_channels   = kernel_shape[1];
    auto channel_size   = image_shape[2] * image_shape[3];
    auto rotation_range = (kernel_size - 1) / 2;
    // auto middle_row     = (num_rows - 1) / 2;
    // auto num_filters    = kernel_shape[0];
    auto padding = (kernel_shape[2] - 1) / 2;

    if (kernel_shape.size() != 4) {
        spdlog::debug(
            "Kernel tensor is {} dimensional; should be 4: [num_filters, "
            "num_channels, num_rows, num_cols]");
        throw std::invalid_argument(
            "Kernel tensor should be 4 dimensions [num_filters,"
            "num_channels, num_rows, num_cols]");
    }
    std::vector<std::vector<double>> kernel_vectors(kernel_size);

    for (unsigned row = 0; row < num_rows; ++row) {
        for (unsigned col = 0; col < num_cols; ++col) {
            std::vector<double> element_weights(num_channels);
            const int row_idx = row * num_cols + col;
            for (unsigned channel = 0; channel < num_channels; ++channel) {
                element_weights[channel] = kernel.Get({filter_number, channel, row, col});
            }
            kernel_vectors[row_idx] =
                CreateKernelElementVector(element_weights, channel_size, row_idx - rotation_range);

            if (row < padding) {
                MaskRows(kernel_vectors[row_idx], channel_size, num_channels, num_cols, padding - row,
                         row_idx - rotation_range);
            }
            else if (row >= num_rows - padding) {
                MaskRows(kernel_vectors[row_idx], channel_size, num_channels, num_cols, num_rows - padding,
                         row_idx - rotation_range, padding);
            }
        }
    }

    return kernel_vectors;
}

CkksTensor CkksTensor::Conv2D(Tensor kernel) {
    auto kernel_shape   = kernel.GetShape();
    auto kernel_size    = kernel_shape[0] * kernel_shape[1];
    auto num_channels   = kernel_shape[2];
    auto num_filters    = kernel_shape[3];
    auto crypto_context = data_.GetContext().GetCryptoContext();
    // auto channel_size   = shape_[0] * shape_[1];
    // auto channels_per_ctxt = crypto_context->GetEncodingParams()->GetBatchSize() / channel_size;
    // auto num_ctxts         = num_filters / channels_per_ctxt;
    // int rotation_range     = static_cast<int>((kernel_size - 1) / 2);

    if (kernel_shape.size() != 4) {
        if (kernel_shape.size() == 3) {
            kernel.Reshape({kernel_shape[0], kernel_shape[1], kernel_shape[2], 1});
        }
        else {
            spdlog::error(
                "Kernel should have four dimensions (has {}): "
                "[rows,cols,ch,num_filters]",
                kernel_shape.size());
            throw std::invalid_argument("Kernel does not have four dimensions");
        }
    }

    if (num_channels != shape_[2]) {
        spdlog::error(
            "Kernel channel size ({}) does not match data channel size "
            "({})",
            kernel_shape[2], shape_[2]);
        throw std::invalid_argument("Kernel channel size does not match data");
    }

    if (kernel_shape[0] != kernel_shape[1]) {
        throw std::invalid_argument("Kernel is not square");
    }

    if (kernel_shape[0] % 2 == 0) {
        throw std::invalid_argument("Kernel size must be odd (e.g. 3x3 or 5x5)");
    }

    auto rotated_images = RotateImages(data_, kernel_size);

    std::vector<CkksVector> conv_results(kernel_size);
    CkksVector output_data;

    for (int filter = 0; filter < num_filters; ++filter) {
        auto kernel_vectors = CreateKernelVectors(kernel, shape_, filter);
        for (int weights_idx = 0; weights_idx < kernel_size; ++weights_idx) {
            conv_results[weights_idx] = rotated_images[weights_idx] * kernel_vectors[weights_idx];
            output_data += conv_results[weights_idx];
        }
    }

    CkksTensor output_tensor(output_data, {num_filters, shape_[1], shape_[2]});

    return output_tensor;
}

// static std::vector<std::vector<double>> CkksTensor::CreateMaskedConvVectors(const Tensor& kernel,
//                                                                             const size_t filter_number) {
//     auto kernel_row_sz  = kernel.GetShape()[0];
//     auto kernel_col_sz  = kernel.GetShape()[1];
//     unsigned padding    = (kernel_row_sz - 1) / 2;
//     auto num_channels   = kernel.GetShape()[2];
//     auto channel_sz     = shape_[0] * shape_[1];
//     auto rotation_range = (kernel_row_sz * kernel_col_sz - 1) / 2;

//     std::vector<std::vector<double>> masked_vectors;

//     // Iterate over rows and columns of the kernel
//     for (size_t row = 0; row < kernel_row_sz; ++row) {
//         for (size_t col = 0; col < kernel_col_sz; ++col) {
//             // Vector of kernel weights for the current row and column
//             std::vector<double> vec_data;

//             // Iterate over channels
//             for (size_t ch = 0; ch < num_channels; ++ch) {
//                 // Fill the whole channel
//                 for (size_t i = 0; i < channel_sz; ++i) {
//                     vec_data.push_back(kernel.Get({filter_number, ch, row, col}));
//                 }
//             }

//             // Mask the upper and lower edges
//             if (row < padding) {
//                 maskTop(vec_data, padding);
//             }
//             else if (row >= kernel_row_sz - padding) {
//                 maskBottom(vec_data, padding);
//             }

//             // Mask the left and right edges
//             if (col < padding) {
//                 maskLeft(vec_data, padding);
//             }
//             else if (col >= kernel_col_sz - padding) {
//                 maskRight(vec_data, padding);
//             }

//             // Account for rotations
//             for (unsigned i = -rotation_range; i <= rotation_range; ++i) {
//                 auto vec = masked_vectors[i + rotation_range];
//                 if (i < 0) {
//                     vec.insert(vec.begin(), -i, 0);
//                     vec.erase(vec.end() + i, vec.end());
//                 }
//                 else if (i > 0) {
//                     vec.erase(vec.begin(), vec.begin() + i);
//                     for (unsigned j = 0; j < i; ++j) {
//                         vec.push_back(0);
//                     }
//                 }
//             }

//             masked_vectors.push_back(vec_data);
//         }
//     }

//     return masked_vectors;
// }

// static void CkksTensor::MaskRight(std::vector<double>& vec, size_t numCols) {
//     for (size_t i = 0; i < shape_[0]; i++) {
//         for (size_t j = shape_[1] - numCols; j < shape_[1]; j++) {
//             for (size_t ch = 0; ch < shape_[2]; ch++) {
//                 vec[i * shape_[1] + j + ch * shape_[0] * shape_[1]] = 0;
//             }
//         }
//     }
// }

// static void CkksTensor::MaskLeft(std::vector<double>& vec, size_t numCols) {
//     for (size_t i = 0; i < shape_[0]; i++) {
//         for (size_t j = 0; j < numCols; j++) {
//             for (size_t ch = 0; ch < shape_[2]; ch++) {
//                 vec[i * shape_[1] + j + ch * shape_[0] * shape_[1]] = 0;
//             }
//         }
//     }
// }

// static void CkksTensor::MaskTop(std::vector<double>& vec, size_t numRows) {
//     for (size_t i = 0; i < numRows; i++) {
//         for (size_t j = 0; j < shape_[1]; j++) {
//             for (size_t ch = 0; ch < shape_[2]; ch++) {
//                 vec[i * shape_[1] + j + ch * shape_[0] * shape_[1]] = 0;
//             }
//         }
//     }
// }

// static void CkksTensor::MaskBottom(std::vector<double>& vec, size_t numRows) {
//     for (size_t i = shape_[0] - numRows; i < shape_[0]; i++) {
//         for (size_t j = 0; j < shape_[1]; j++) {
//             for (size_t ch = 0; ch < shape_[2]; ch++) {
//                 vec[i * shape_[1] + j + ch * shape_[0] * shape_[1]] = 0;
//             }
//         }
//     }
// }
