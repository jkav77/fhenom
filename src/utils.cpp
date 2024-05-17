#include <utility>
#include <spdlog/spdlog.h>
#include <fhenom/ckks_vector.h>
#include <fhenom/tensor.h>
#include <fhenom/common.h>

namespace fhenom {
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

}  // namespace fhenom
