#include <fhenom/ckks_tensor.h>
#include <fhenom/ckks_vector.h>

#include <utility>

using fhenom::CkksTensor;
using fhenom::Tensor;

CkksTensor::CkksTensor(CkksVector data, shape_t shape) {
  setData(std::move(data), std::move(shape));
}

void CkksTensor::setData(CkksVector data, shape_t shape) {
  size_t len = 1;
  for (auto dim : shape) {
    len *= dim;
  }
  if (data.size() != len) {
    spdlog::error("Data vector size ({}) does not match shape ({})",
                  data.size(), len);
    throw std::invalid_argument("Data vector size does not match shape");
  }

  data_ = std::move(data);
  shape_ = std::move(shape);
}

CkksTensor CkksTensor::conv2D(Tensor kernel) {
  auto kernel_shape = kernel.getShape();
  auto kernel_size = kernel_shape[0] * kernel_shape[1];
  auto channel_size = shape_[0] * shape_[1];
  // auto num_channels = kernel_shape[2];
  auto num_filters = kernel_shape[3];
  int rotation_range = (kernel_size - 1) / 2;
  auto cryptoContext = data_.getContext().getCryptoContext();
  auto channels_per_ctxt =
      cryptoContext->GetEncodingParams()->GetBatchSize() / channel_size;
  auto num_ctxts = num_filters / channels_per_ctxt;

  if (kernel_shape.size() != 4) {
    if (kernel_shape.size() == 3) {
      kernel.reshape({kernel_shape[0], kernel_shape[1], kernel_shape[2], 1});
    } else {
      spdlog::error(
          "Kernel should have four dimensions (has {}): "
          "[rows,cols,ch,num_filters]",
          kernel_shape.size());
      throw std::invalid_argument("Kernel does not have four dimensions");
    }
  }

  if (kernel_shape[2] != shape_[2]) {
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
    throw std::invalid_argument("Kernel size must be odd");
  }

  std::vector<CkksVector> rotated_ctxts;
  data_.precomputeRotations();

  for (int range = -rotation_range; range <= rotation_range; ++range) {
    rotated_ctxts.emplace_back(data_.rotate(range));
  }

  std::vector<CkksVector> output_channels(num_filters);

  std::vector<CkksVector> intermediate_results(kernel_size);
#pragma omp parallel for
  for (unsigned ctxt_index = 0; ctxt_index < num_ctxts; ++ctxt_index) {
    for (unsigned filter_index = 0; filter_index < channels_per_ctxt;
         ++filter_index) {
      auto kernel_vectors = createMaskedConvVectors(kernel, filter_index);

      for (unsigned i = 0; i < kernel_size; ++i) {
        intermediate_results[i] = rotated_ctxts[i] * kernel_vectors[i];
      }

      for (size_t idx = kernel_size - 1; idx > 0; --idx) {
        intermediate_results[0] += intermediate_results[idx];
        intermediate_results.pop_back();
      }

      intermediate_results[filter_index] +=
          intermediate_results[0].rotate(
              static_cast<int>(shape_[0] * shape_[1])) +
          intermediate_results[0].rotate(
              static_cast<int>(2 * shape_[0] * shape_[1]));
      intermediate_results[0].setNumElements(shape_[0] * shape_[1]);
    }
  }

  return CkksTensor{intermediate_results[0], shape_t{shape_[0], shape_[1], 1}};
}

std::vector<std::vector<double>> CkksTensor::createMaskedConvVectors(
    const Tensor &kernel, const size_t filter_number) const {
  auto kernel_row_sz = kernel.getShape()[0];
  auto kernel_col_sz = kernel.getShape()[1];
  unsigned padding = (kernel_row_sz - 1) / 2;
  auto num_channels = kernel.getShape()[2];
  auto channel_sz = shape_[0] * shape_[1];
  auto rotation_range = (kernel_row_sz * kernel_col_sz - 1) / 2;

  std::vector<std::vector<double>> masked_vectors;

  // Iterate over rows and columns of the kernel
  for (size_t row = 0; row < kernel_row_sz; ++row) {
    for (size_t col = 0; col < kernel_col_sz; ++col) {
      // Vector of kernel weights for the current row and column
      std::vector<double> vec_data;

      // Iterate over channels
      for (size_t ch = 0; ch < num_channels; ++ch) {
        // Fill the whole channel
        for (size_t i = 0; i < channel_sz; ++i) {
          vec_data.push_back(kernel.get({filter_number, ch, row, col}));
        }
      }

      // Mask the upper and lower edges
      if (row < padding) {
        maskTop(vec_data, padding);
      } else if (row >= kernel_row_sz - padding) {
        maskBottom(vec_data, padding);
      }

      // Mask the left and right edges
      if (col < padding) {
        maskLeft(vec_data, padding);
      } else if (col >= kernel_col_sz - padding) {
        maskRight(vec_data, padding);
      }

      // Account for rotations
      for (unsigned i = -rotation_range; i <= rotation_range; ++i) {
        auto vec = masked_vectors[i + rotation_range];
        if (i < 0) {
          vec.insert(vec.begin(), -i, 0);
          vec.erase(vec.end() + i, vec.end());
        } else if (i > 0) {
          vec.erase(vec.begin(), vec.begin() + i);
          for (unsigned j = 0; j < i; ++j) {
            vec.push_back(0);
          }
        }
      }

      masked_vectors.push_back(vec_data);
    }
  }

  return masked_vectors;
}

void CkksTensor::maskRight(std::vector<double> &vec, size_t numCols) const {
  for (size_t i = 0; i < shape_[0]; i++) {
    for (size_t j = shape_[1] - numCols; j < shape_[1]; j++) {
      for (size_t ch = 0; ch < shape_[2]; ch++) {
        vec[i * shape_[1] + j + ch * shape_[0] * shape_[1]] = 0;
      }
    }
  }
}

void CkksTensor::maskLeft(std::vector<double> &vec, size_t numCols) const {
  for (size_t i = 0; i < shape_[0]; i++) {
    for (size_t j = 0; j < numCols; j++) {
      for (size_t ch = 0; ch < shape_[2]; ch++) {
        vec[i * shape_[1] + j + ch * shape_[0] * shape_[1]] = 0;
      }
    }
  }
}

void CkksTensor::maskTop(std::vector<double> &vec, size_t numRows) const {
  for (size_t i = 0; i < numRows; i++) {
    for (size_t j = 0; j < shape_[1]; j++) {
      for (size_t ch = 0; ch < shape_[2]; ch++) {
        vec[i * shape_[1] + j + ch * shape_[0] * shape_[1]] = 0;
      }
    }
  }
}

void CkksTensor::maskBottom(std::vector<double> &vec, size_t numRows) const {
  for (size_t i = shape_[0] - numRows; i < shape_[0]; i++) {
    for (size_t j = 0; j < shape_[1]; j++) {
      for (size_t ch = 0; ch < shape_[2]; ch++) {
        vec[i * shape_[1] + j + ch * shape_[0] * shape_[1]] = 0;
      }
    }
  }
}
