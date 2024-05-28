#include <fhenom/ckks_tensor.h>
#include <fhenom/tensor.h>
#include "nlohmann/json.hpp"

#include <cereal/archives/binary.hpp>
#include "gtest/gtest.h"

using fhenom::CkksTensor;
using fhenom::CkksVector;
using fhenom::Context;
using fhenom::shape_t;
using fhenom::Tensor;
using nlohmann::json;

std::vector<double> CreateKernelElementVector(const std::vector<double>& element, unsigned channel_size, int rotation);
void MaskRows(std::vector<double>& vec, int channel_size, int num_channels, int num_cols, int start_row, int rotation,
              int num_rows = 1);
void MaskCols(std::vector<double>& vec, int row_size, int start_col, int rotation, int num_cols = 1);

class CkksTensorTest : public ::testing::Test {
protected:
    Context context_;
    CkksVector ckks_vector_;
    CkksTensor ckks_tensor_;
    const std::filesystem::path test_data_dir_{"testData/ckks_tensor"};
    const Tensor kernel_{{0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                          0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1},
                         {1, 3, 3, 3}};
    const Tensor kernel2_{{0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                           0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
                           0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2},
                          {2, 3, 3, 3}};

    // Data for a 3x5x5 tensor representing a 3-channel 5x5 image
    const std::vector<double> test_data_{0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                                         0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                                         0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                                         0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                                         0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1};

    const double epsilon_ = 0.001;

    CkksTensorTest() {
        spdlog::set_level(spdlog::level::debug);

        context_ = Context{GetParameters()};
        context_.SetSlotsPerCtxt(1024);
        context_.GenerateKeys();
        context_.GenerateRotateKeys({-1,    -2,    -4,    -8,    -16,    -32,  -64,  -128, -256, -512,
                                     -1024, -2048, -4096, -8192, -16384, 1,    2,    4,    8,    16,
                                     32,    64,    128,   256,   512,    1024, 2048, 4096, 8192, 16384});
        ckks_vector_.SetContext(context_);
        ckks_vector_.Encrypt(test_data_);
        ckks_tensor_ = CkksTensor{ckks_vector_, {3, 5, 5}};
    }

    static lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS> GetParameters() {
        lbcrypto::ScalingTechnique sc_tech = lbcrypto::FLEXIBLEAUTO;
        uint32_t mult_depth                = 12;
        uint32_t scale_mod_size            = 39;
        uint32_t first_mod_size            = 40;
        uint32_t ring_dim                  = 32768;
        lbcrypto::SecurityLevel sl         = lbcrypto::HEStd_128_classic;

        lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS> ckks_parameters;
        ckks_parameters.SetMultiplicativeDepth(mult_depth);
        ckks_parameters.SetScalingModSize(scale_mod_size);
        ckks_parameters.SetFirstModSize(first_mod_size);
        ckks_parameters.SetScalingTechnique(sc_tech);
        ckks_parameters.SetSecurityLevel(sl);
        ckks_parameters.SetRingDim(ring_dim);
        ckks_parameters.SetSecretKeyDist(lbcrypto::UNIFORM_TERNARY);
        ckks_parameters.SetKeySwitchTechnique(lbcrypto::HYBRID);
        ckks_parameters.SetNumLargeDigits(3);
        return ckks_parameters;
    }
};

TEST_F(CkksTensorTest, DefaultConstructor) {
    CkksTensor ckks_tensor;
    EXPECT_EQ(ckks_tensor.GetData().size(), 0);
    EXPECT_EQ(ckks_tensor.GetShape(), shape_t{0});
}

TEST_F(CkksTensorTest, Conv2D) {
    context_.SetSlotsPerCtxt(25);
    ckks_vector_.SetContext(context_);
    ckks_vector_.Encrypt(test_data_);
    ckks_tensor_      = CkksTensor{ckks_vector_, {3, 5, 5}};
    CkksTensor tensor = ckks_tensor_.Conv2D(kernel_, fhenom::Tensor{{0}, {1}});
    ASSERT_EQ(tensor.GetShape(), (shape_t{1, 5, 5}));
    CkksVector vec = tensor.GetData();
    ASSERT_EQ(vec.size(), 25);
    std::vector<double> result = vec.Decrypt();
    ASSERT_NEAR(result[0], 0.12, epsilon_);
    ASSERT_NEAR(result[1], 0.18, epsilon_);
    ASSERT_NEAR(result[5], 0.18, epsilon_);
    ASSERT_NEAR(result[6], 0.27, epsilon_);
    ASSERT_NEAR(result[4], 0.12, epsilon_);
    ASSERT_NEAR(result[3], 0.18, epsilon_);
    ASSERT_NEAR(result[9], 0.18, epsilon_);
    ASSERT_NEAR(result[8], 0.27, epsilon_);
    ASSERT_NEAR(result[24], 0.12, epsilon_);
    ASSERT_NEAR(result[23], 0.18, epsilon_);
    ASSERT_NEAR(result[19], 0.18, epsilon_);
    ASSERT_NEAR(result[18], 0.27, epsilon_);

    tensor = ckks_tensor_.Conv2D(kernel2_, fhenom::Tensor{{0, 0}, {2}});
    vec    = tensor.GetData();
    ASSERT_EQ(vec.size(), 50);
    result = vec.Decrypt();
    // first filter
    ASSERT_NEAR(result[0], 0.12, epsilon_);
    ASSERT_NEAR(result[1], 0.18, epsilon_);
    ASSERT_NEAR(result[5], 0.18, epsilon_);
    ASSERT_NEAR(result[6], 0.27, epsilon_);
    ASSERT_NEAR(result[4], 0.12, epsilon_);
    ASSERT_NEAR(result[3], 0.18, epsilon_);
    ASSERT_NEAR(result[9], 0.18, epsilon_);
    ASSERT_NEAR(result[8], 0.27, epsilon_);
    ASSERT_NEAR(result[24], 0.12, epsilon_);
    ASSERT_NEAR(result[23], 0.18, epsilon_);
    ASSERT_NEAR(result[19], 0.18, epsilon_);
    ASSERT_NEAR(result[18], 0.27, epsilon_);

    // second filter
    ASSERT_NEAR(result[25], 0.24, epsilon_);
    ASSERT_NEAR(result[26], 0.36, epsilon_);
    ASSERT_NEAR(result[30], 0.36, epsilon_);
    ASSERT_NEAR(result[31], 0.54, epsilon_);
    ASSERT_NEAR(result[29], 0.24, epsilon_);
    ASSERT_NEAR(result[28], 0.36, epsilon_);
    ASSERT_NEAR(result[34], 0.36, epsilon_);
    ASSERT_NEAR(result[33], 0.54, epsilon_);
    ASSERT_NEAR(result[49], 0.24, epsilon_);
    ASSERT_NEAR(result[48], 0.36, epsilon_);
    ASSERT_NEAR(result[44], 0.36, epsilon_);
    ASSERT_NEAR(result[43], 0.54, epsilon_);

    // Conv layer with 4 filters
    std::vector<double> weights(108);
    for (unsigned i = 0; i < 4; ++i) {
        std::fill(weights.begin() + i * 27, weights.begin() + (i + 1) * 27, (i + 1) / 10.0);
    }
    Tensor filter(weights, {4, 3, 3, 3});
    tensor = ckks_tensor_.Conv2D(filter, fhenom::Tensor{{0, 0, 0, 0}, {4}});
    vec    = tensor.GetData();
    ASSERT_EQ(vec.size(), 100);
    result = vec.Decrypt();
    // first filter
    ASSERT_NEAR(result[0], 0.12, epsilon_);
    ASSERT_NEAR(result[1], 0.18, epsilon_);
    ASSERT_NEAR(result[5], 0.18, epsilon_);
    ASSERT_NEAR(result[6], 0.27, epsilon_);
    ASSERT_NEAR(result[4], 0.12, epsilon_);
    ASSERT_NEAR(result[3], 0.18, epsilon_);
    ASSERT_NEAR(result[9], 0.18, epsilon_);
    ASSERT_NEAR(result[8], 0.27, epsilon_);
    ASSERT_NEAR(result[24], 0.12, epsilon_);
    ASSERT_NEAR(result[23], 0.18, epsilon_);
    ASSERT_NEAR(result[19], 0.18, epsilon_);
    ASSERT_NEAR(result[18], 0.27, epsilon_);

    // second filter
    ASSERT_NEAR(result[25], 0.24, epsilon_);
    ASSERT_NEAR(result[26], 0.36, epsilon_);
    ASSERT_NEAR(result[30], 0.36, epsilon_);
    ASSERT_NEAR(result[31], 0.54, epsilon_);
    ASSERT_NEAR(result[29], 0.24, epsilon_);
    ASSERT_NEAR(result[28], 0.36, epsilon_);
    ASSERT_NEAR(result[34], 0.36, epsilon_);
    ASSERT_NEAR(result[33], 0.54, epsilon_);
    ASSERT_NEAR(result[49], 0.24, epsilon_);
    ASSERT_NEAR(result[48], 0.36, epsilon_);
    ASSERT_NEAR(result[44], 0.36, epsilon_);
    ASSERT_NEAR(result[43], 0.54, epsilon_);
}

TEST_F(CkksTensorTest, Conv2DSpanCiphertexts) {
    const std::vector<double> image_data(3072, 0.1);
    CkksVector image_vec(context_);
    image_vec.Encrypt(image_data);
    CkksTensor image(image_vec, {3, 32, 32});

    std::vector<double> weights_vec(27 * 5);
    for (unsigned i = 0; i < 5; ++i) {
        std::fill(weights_vec.begin() + i * 27, weights_vec.begin() + (i + 1) * 27, (i + 1) / 10.0);
    }
    Tensor weights(weights_vec, {5, 3, 3, 3});

    auto result = image.Conv2D(weights, Tensor{{0, 0, 0, 0, 0}, {5}});
    ASSERT_EQ(result.GetShape(), (shape_t{5, 32, 32}));
    auto result_vec = result.GetData();
    ASSERT_EQ(result_vec.size(), 5 * 32 * 32);

    auto decrypted_values = result_vec.Decrypt();
    // first filter
    ASSERT_NEAR(decrypted_values[0], 0.12, epsilon_);
    ASSERT_NEAR(decrypted_values[1], 0.18, epsilon_);
    ASSERT_NEAR(decrypted_values[31], 0.12, epsilon_);
    ASSERT_NEAR(decrypted_values[32], 0.18, epsilon_);
    ASSERT_NEAR(decrypted_values[33], 0.27, epsilon_);
    ASSERT_NEAR(decrypted_values[1022], 0.18, epsilon_);
    ASSERT_NEAR(decrypted_values[1023], 0.12, epsilon_);

    // first filter
    ASSERT_NEAR(decrypted_values[0 + 1024], 0.12 * 2, epsilon_);
    ASSERT_NEAR(decrypted_values[1 + 1024], 0.18 * 2, epsilon_);
    ASSERT_NEAR(decrypted_values[31 + 1024], 0.12 * 2, epsilon_);
    ASSERT_NEAR(decrypted_values[32 + 1024], 0.18 * 2, epsilon_);
    ASSERT_NEAR(decrypted_values[33 + 1024], 0.27 * 2, epsilon_);
    ASSERT_NEAR(decrypted_values[1022 + 1024], 0.18 * 2, epsilon_);
    ASSERT_NEAR(decrypted_values[1023 + 1024], 0.12 * 2, epsilon_);

    // fifth filter
    ASSERT_NEAR(decrypted_values[0 + 1024 * 4], 0.12 * 5, epsilon_);
    ASSERT_NEAR(decrypted_values[1 + 1024 * 4], 0.18 * 5, epsilon_);
    ASSERT_NEAR(decrypted_values[31 + 1024 * 4], 0.12 * 5, epsilon_);
    ASSERT_NEAR(decrypted_values[32 + 1024 * 4], 0.18 * 5, epsilon_);
    ASSERT_NEAR(decrypted_values[33 + 1024 * 4], 0.27 * 5, epsilon_);
    ASSERT_NEAR(decrypted_values[1022 + 1024 * 4], 0.18 * 5, epsilon_);
    ASSERT_NEAR(decrypted_values[1023 + 1024 * 4], 0.12 * 5, epsilon_);
}

std::vector<double> loadImage() {
    std::filesystem::path filename = "./test_batch.bin";
    std::ifstream file(filename, std::ios::binary);

    if (!file.is_open()) {
        spdlog::error("Failed to open file: {}", filename.string());
        throw std::ifstream::failure("Failed to open image file.");
    }

    unsigned char label;
    file >> label;

    std::vector<unsigned char> buffer(3072);
    file.read(reinterpret_cast<char*>(buffer.data()), 3072);
    std::vector<double> image(3072);
    std::transform(buffer.begin(), buffer.end(), image.begin(), [](unsigned char pixel) { return pixel / 127.5 - 1; });
    return image;
}

std::pair<Tensor, Tensor> loadWeights() {
    std::filesystem::path filename = "./SmallCNN_weights.json";
    std::ifstream file(filename);
    auto model   = json::parse(file);
    auto weights = Tensor(model["conv1"]["weights"], {16, 3, 3, 3});
    auto bias    = Tensor(model["conv1"]["bias"], {16});
    return {weights, bias};
}

TEST_F(CkksTensorTest, RotateImages) {
    auto image = loadImage();
    ckks_vector_.Encrypt(image);
    CkksTensor ckks_tensor(ckks_vector_, {3, 32, 32});

    fhenom::shape_t kernel_shape{16, 3, 3, 3};
    auto rotated_images = ckks_tensor.rotate_images(kernel_shape);

    std::vector<double> decrypted = rotated_images[0].Decrypt();
    for (int i = 0; i < 6; ++i) {
        ASSERT_NEAR(decrypted[i], 0, 1e-4);
    }
}

TEST_F(CkksTensorTest, Conv2DCifar10) {
    auto image           = loadImage();
    auto [weights, bias] = loadWeights();
    context_.SetSlotsPerCtxt(1024);
    ckks_vector_.SetContext(context_);
    ckks_vector_.Encrypt(image);
    ckks_tensor_.SetData(ckks_vector_, {3, 32, 32});
    auto result = ckks_tensor_.Conv2D(weights, bias);
    ASSERT_EQ(result.GetShape(), (shape_t{16, 32, 32}));

    auto decrypted_result = result.GetData().Decrypt();
    ASSERT_NEAR(decrypted_result[0], -0.6053400635719299, 0.001);
}

// TEST_F(CkksTensorTest, AvgPool2D) {
//     auto image = loadImage();
//     ckks_vector_.Encrypt(image);
//     CkksTensor ckks_tensor(ckks_vector_, {3, 32, 32});
//     auto result = ckks_tensor.AvgPool2D();
//     ASSERT_EQ(result.GetShape(), (shape_t{3, 16, 16}));

//     auto decrypted = result.GetData().Decrypt();

//     for (int row = 0; row < 32; row += 2) {
//         for (int col = 0; col < 32; col += 2) {
//             auto avg = (image[row * 32 + col] + image[row * 32 + col + 1] + image[(row + 1) * 32 + col] +
//                         image[(row + 1) * 32 + col + 1]) /
//                        4;
//             ASSERT_NEAR(decrypted[row / 2 * 16 + col / 2], avg, 0.001);
//         }
//     }
// }

// TEST_F(CkksTensorTest, AvgPool2DCifarOutput) {
//     lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS> ckks_parameters;
//     ckks_parameters.SetMultiplicativeDepth(20);
//     ckks_parameters.SetScalingModSize(59);
//     ckks_parameters.SetFirstModSize(60);
//     ckks_parameters.SetScalingTechnique(lbcrypto::FLEXIBLEAUTO);
//     ckks_parameters.SetSecurityLevel(lbcrypto::HEStd_NotSet);
//     ckks_parameters.SetRingDim(8192);
//     ckks_parameters.SetSecretKeyDist(lbcrypto::UNIFORM_TERNARY);
//     ckks_parameters.SetKeySwitchTechnique(lbcrypto::HYBRID);

//     context_ = Context{ckks_parameters};
//     spdlog::debug("Generating keys...");
//     context_.GenerateKeys();
//     spdlog::debug("Generating rotate keys...");
//     context_.GenerateRotateKeys({1,     2,     4,     8,     16,    32,    64,    128,   256,   512,
//                                  1024,  2048,  4096,  8192,  16384, 32768, -1,    -2,    -4,    -8,
//                                  -16,   -32,   3072,  4096,  5120,  6144,  7168,  9216,  10240, 11264,
//                                  12288, 13312, 14336, 15360, 16384, 17408, 18432, 19456, 20480, 21504,
//                                  22528, 23552, 24576, 25600, 26624, 27648, 28672, 29696, 30720, 31744});

//     auto file        = std::ifstream("testData/relu_output.json");
//     auto relu_output = json::parse(file).get<std::vector<double>>();

//     CkksVector relu_vec(context_);
//     relu_vec.Encrypt(relu_output);
//     CkksTensor relu_tensor(relu_vec, {16, 32, 32});

//     auto avg_pool_output    = relu_tensor.AvgPool2D();
//     auto avg_pool_decrypted = avg_pool_output.GetData().Decrypt();
//     ASSERT_NEAR(avg_pool_decrypted[0], (relu_output[0] + relu_output[1] + relu_output[32] + relu_output[33]) / 4,
//                 0.001);
// }

TEST_F(CkksTensorTest, ReLU4) {
    lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS> params;
    params.SetMultiplicativeDepth(4);
    params.SetScalingModSize(59);
    params.SetRingDim(16384);
    params.SetSecurityLevel(lbcrypto::HEStd_128_classic);

    fhenom::Context context(params);
    context.GenerateKeys();

    std::vector<double> relu_data(201);
    std::iota(relu_data.begin(), relu_data.end(), -100.0);
    std::transform(relu_data.begin(), relu_data.end(), relu_data.begin(), [](double x) { return x * 0.01; });

    CkksVector vec(context);
    vec.Encrypt(relu_data);
    CkksTensor tensor(vec, {201});

    double epsilon     = 0.01;
    auto compute_error = [](std::vector<double> result, std::vector<double> data) -> std::vector<double> {
        std::vector<double> error(data.size());
        for (unsigned i = 0; i < data.size(); ++i) {
            error[i] = std::abs(result[i] - std::max(0.0, data[i]));
        }

        return error;
    };

    spdlog::debug("ReLU(4)");
    auto result         = tensor.ReLU(4).GetData().Decrypt();
    auto error          = compute_error(result, relu_data);
    auto number_correct = std::count_if(error.begin(), error.end(), [epsilon](double x) { return x < epsilon; });
    spdlog::debug("Number correct: {}", number_correct);
    spdlog::debug("Percent correct: {0:.0f}%", static_cast<double>(number_correct) / relu_data.size() * 100);
    spdlog::debug("Average error: {}", std::accumulate(error.begin(), error.end(), 0.0) / error.size());
    spdlog::debug("Max error: {}", *std::max_element(error.begin(), error.end()));
}

TEST_F(CkksTensorTest, ReLU12) {
    std::vector<double> relu_data(201);
    std::iota(relu_data.begin(), relu_data.end(), -100.0);
    std::transform(relu_data.begin(), relu_data.end(), relu_data.begin(), [](double x) { return x * 0.01; });

    lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS> params;
    params.SetMultiplicativeDepth(12);
    params.SetScalingModSize(59);
    params.SetRingDim(65536);
    params.SetSecurityLevel(lbcrypto::HEStd_128_classic);

    fhenom::Context context(params);
    context.GenerateKeys();

    fhenom::CkksVector vec(context);
    vec.Encrypt(relu_data);
    CkksTensor tensor(vec, {201});

    auto epsilon       = 0.01;
    auto compute_error = [](std::vector<double> result, std::vector<double> data) -> std::vector<double> {
        std::vector<double> error(data.size());
        for (unsigned i = 0; i < data.size(); ++i) {
            error[i] = std::abs(result[i] - std::max(0.0, data[i]));
        }

        return error;
    };

    ASSERT_EQ(vec.GetData()[0]->GetLevel(), 0);
    auto result = tensor.ReLU(12);
    ASSERT_EQ(result.GetData().GetData()[0]->GetLevel(), 12);
    auto decrypted = result.GetData().Decrypt();

    auto error          = compute_error(decrypted, relu_data);
    auto number_correct = std::count_if(error.begin(), error.end(), [epsilon](double x) { return x < epsilon; });
    spdlog::debug("Number correct: {}", number_correct);
    spdlog::debug("Percent correct: {0:.2f}%", static_cast<double>(number_correct) / relu_data.size() * 100);
    spdlog::debug("Average error: {}", std::accumulate(error.begin(), error.end(), 0.0) / error.size());
    spdlog::debug("Max error: {}", *std::max_element(error.begin(), error.end()));
}

TEST_F(CkksTensorTest, ScaledReLU12) {
    std::vector<double> relu_data(201);
    std::iota(relu_data.begin(), relu_data.end(), -100.0);
    std::transform(relu_data.begin(), relu_data.end(), relu_data.begin(), [](double x) { return x * 0.02; });

    lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS> params;
    params.SetMultiplicativeDepth(12);
    params.SetScalingModSize(59);
    params.SetRingDim(65536);
    params.SetSecurityLevel(lbcrypto::HEStd_128_classic);

    fhenom::Context context(params);
    context.GenerateKeys();

    fhenom::CkksVector vec(context);
    vec.Encrypt(relu_data);
    CkksTensor tensor(vec, {201});

    auto epsilon       = 0.01;
    auto compute_error = [](std::vector<double> result, std::vector<double> data) -> std::vector<double> {
        std::vector<double> error(data.size());
        for (unsigned i = 0; i < data.size(); ++i) {
            error[i] = std::abs(result[i] - std::max(0.0, data[i]));
        }

        return error;
    };

    ASSERT_EQ(vec.GetData()[0]->GetLevel(), 0);
    auto result = tensor.ReLU(12, 2);
    ASSERT_LE(result.GetData().GetData()[0]->GetLevel(), 12);
    auto decrypted = result.GetData().Decrypt();

    auto error          = compute_error(decrypted, relu_data);
    auto number_correct = std::count_if(error.begin(), error.end(), [epsilon](double x) { return x < epsilon; });
    spdlog::debug("Number correct: {}", number_correct);
    spdlog::debug("Percent correct: {0:.2f}%", static_cast<double>(number_correct) / relu_data.size() * 100);
    spdlog::debug("Average error: {}", std::accumulate(error.begin(), error.end(), 0.0) / error.size());
    spdlog::debug("Max error: {}", *std::max_element(error.begin(), error.end()));
}

TEST_F(CkksTensorTest, ScaledReLU6) {
    lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS> params;

    params.SetMultiplicativeDepth(6);
    params.SetScalingModSize(59);
    params.SetRingDim(32768);

    fhenom::Context context(params);
    context.GenerateKeys();

    std::vector<double> relu_data(201);
    std::iota(relu_data.begin(), relu_data.end(), -100.0);
    std::transform(relu_data.begin(), relu_data.end(), relu_data.begin(), [](double x) { return x * 0.02; });

    CkksVector vec(context);
    vec.Encrypt(relu_data);
    CkksTensor tensor(vec, {201});

    double epsilon     = 0.003;
    auto compute_error = [](std::vector<double> result, std::vector<double> data) -> std::vector<double> {
        std::vector<double> error(data.size());
        for (unsigned i = 0; i < data.size(); ++i) {
            error[i] = std::abs(result[i] - std::max(0.0, data[i]));
        }

        return error;
    };

    auto result = tensor.ReLU(6, 2);
    ASSERT_LE(result.GetData().GetData()[0]->GetLevel(), 6);

    auto decrypted      = result.GetData().Decrypt();
    auto error          = compute_error(decrypted, relu_data);
    auto number_correct = std::count_if(error.begin(), error.end(), [epsilon](double x) { return x < epsilon; });
    spdlog::debug("Number correct: {}", number_correct);
    spdlog::debug("Percent correct: {0:.2f}%", static_cast<double>(number_correct) / relu_data.size() * 100);
    spdlog::debug("Average error: {}", std::accumulate(error.begin(), error.end(), 0.0) / error.size());
    spdlog::debug("Max error: {}", *std::max_element(error.begin(), error.end()));
}

TEST(CkksTensor, Dense) {
    spdlog::set_level(spdlog::level::debug);

    Context context;
    std::filesystem::path context_path{"testData/dense/context"};
    if (std::filesystem::exists("context/cryptocontext.txt")) {
        spdlog::debug("Loading context...");
        context.Load(context_path);
        context.LoadPublicKey(context_path / "public-key.txt");
        context.LoadSecretKey(context_path / "secret-key.txt");
    }
    else {
        spdlog::debug("Generating context...");
        lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS> ckks_parameters;
        ckks_parameters.SetMultiplicativeDepth(32);
        ckks_parameters.SetScalingModSize(59);
        ckks_parameters.SetFirstModSize(60);
        ckks_parameters.SetSecurityLevel(lbcrypto::HEStd_128_classic);
        ckks_parameters.SetRingDim(131072);
        context = Context{ckks_parameters};

        spdlog::debug("Generating keys...");
        context.GenerateKeys();
        spdlog::debug("Generating rotate keys...");
        context.GenerateRotateKeys(
            {1,         2,         4,         8,         16,        31,        32,        33,        64,
             128,       256,       512,       1024,      2048,      3072,      4096,      8192,      16384,
             32768,     -1,        -31,       -32,       -33,       5120,      6144,      7168,      9216,
             10240,     11264,     12288,     13312,     14336,     15360,     17 * 1024, 18 * 1024, 19 * 1024,
             20 * 1024, 21 * 1024, 22 * 1024, 23 * 1024, 24 * 1024, 25 * 1024, 26 * 1024, 27 * 1024, 28 * 1024,
             29 * 1024, 30 * 1024, 31 * 1024, 32 * 1024, 33 * 1024, 34 * 1024, 35 * 1024, 36 * 1024, 37 * 1024,
             38 * 1024, 39 * 1024, 40 * 1024, 41 * 1024, 42 * 1024, 43 * 1024, 44 * 1024, 45 * 1024, 46 * 1024,
             47 * 1024, 48 * 1024, 49 * 1024, 50 * 1024, 51 * 1024, 52 * 1024, 53 * 1024, 54 * 1024, 55 * 1024,
             56 * 1024, 57 * 1024, 58 * 1024, 59 * 1024, 60 * 1024, 61 * 1024, 62 * 1024, 63 * 1024});

        context.Save(context_path);
        context.SavePublicKey(context_path / "public-key.txt");
        context.SaveSecretKey(context_path / "secret-key.txt");
    }

    std::filesystem::path relu3_path{"testData/dense/relu3_output"};
    spdlog::debug("Loading relu3");
    CkksVector relu3_data;
    relu3_data.SetContext(context);
    relu3_data.Load(relu3_path);

    CkksTensor dense_input{relu3_data, {64, 32, 32}};

    Tensor dense_weights;
    Tensor dense_bias;
    {
        std::ifstream file{"testData/dense/dense_weights.bin"};
        cereal::BinaryInputArchive archive{file};
        archive(dense_weights, dense_bias);
    }

    auto output    = dense_input.Dense(dense_weights, dense_bias);
    auto decrypted = output.GetData().Decrypt();
    spdlog::debug("Dense input decrypted");
}
