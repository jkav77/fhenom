#include <fhenom/ckks_tensor.h>
#include <fhenom/tensor.h>
#include "nlohmann/json.hpp"

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

        if (!std::filesystem::exists(test_data_dir_)) {
            context_ = Context{GetParameters()};
            context_.GenerateKeys();
            context_.GenerateRotateKeys({-1, -2, -4, -8, -16, -32, -64, -128, -256, -512, -1024, -2048, -4096,
                                         1,  2,  4,  8,  16,  32,  64,  128,  256,  512,  1024,  2048,  4096});
            context_.Save(test_data_dir_);
            context_.SaveRotationKeys(test_data_dir_ / "key-rotate.txt");
            context_.SavePublicKey(test_data_dir_ / "key-public.txt");
            context_.SaveSecretKey(test_data_dir_ / "key-secret.txt");
        }
        else {
            context_.Load(test_data_dir_);
            context_.LoadRotationKeys(test_data_dir_ / "key-rotate.txt");
            context_.LoadPublicKey(test_data_dir_ / "key-public.txt");
            context_.LoadSecretKey(test_data_dir_ / "key-secret.txt");
        }

        ckks_vector_.SetContext(context_);
        ckks_vector_.Encrypt(test_data_);
        ckks_tensor_ = CkksTensor{ckks_vector_, {3, 5, 5}};
    }

    static lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS> GetParameters() {
        lbcrypto::ScalingTechnique sc_tech = lbcrypto::FLEXIBLEAUTOEXT;
        uint32_t mult_depth                = 2;
        if (sc_tech == lbcrypto::FLEXIBLEAUTOEXT)
            mult_depth += 1;
        uint32_t scale_mod_size    = 24;
        uint32_t first_mod_size    = 30;
        uint32_t ring_dim          = 8192;
        lbcrypto::SecurityLevel sl = lbcrypto::HEStd_128_classic;
        // uint32_t slots = 16;  // sparsely-packed
        // uint32_t batchSize = slots;

        lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS> ckks_parameters;
        ckks_parameters.SetMultiplicativeDepth(mult_depth);
        ckks_parameters.SetScalingModSize(scale_mod_size);
        ckks_parameters.SetFirstModSize(first_mod_size);
        ckks_parameters.SetScalingTechnique(sc_tech);
        ckks_parameters.SetSecurityLevel(sl);
        ckks_parameters.SetRingDim(ring_dim);
        // ckksParameters.SetBatchSize(batchSize);
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
        std::fill(weights.begin() + i * 27, weights.begin() + (i + 1) * 27, (i + 1) / 100.0);
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
    auto batch_size = context_.GetCryptoContext()->GetEncodingParams()->GetBatchSize();

    const std::vector<double> image_data(batch_size * 0.75, 0.1);
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
    std::transform(buffer.begin(), buffer.end(), image.begin(), [](unsigned char pixel) { return pixel / 255.0; });
    return image;
}

std::pair<Tensor, Tensor> loadWeights() {
    std::filesystem::path filename = "./SmallCNN_weights.json";
    std::ifstream file(filename);
    auto model   = json::parse(file);
    auto weights = Tensor(model["cnn.0"]["weights"], {16, 3, 3, 3});
    auto bias    = Tensor(model["cnn.0"]["bias"], {16});
    return {weights, bias};
}

TEST_F(CkksTensorTest, ConvVaryingValues) {
    auto image = loadImage();
    ckks_vector_.Encrypt(image);
    ckks_tensor_.SetData(ckks_vector_, {3, 32, 32});
    std::vector<double> kernel_data(27);
    for (int idx = 0; idx < 27; ++idx) {
        kernel_data[idx] = (idx + 1) / 100.0;
    }
    Tensor kernel{kernel_data, {1, 3, 3, 3}};
    auto result = ckks_tensor_.Conv2D(kernel, fhenom::Tensor{{0}, {1}});
    ASSERT_EQ(result.GetShape(), (shape_t{1, 32, 32}));

    auto decrypted_result = result.GetData().Decrypt();
    ASSERT_NEAR(decrypted_result[0], 0.6312549, epsilon_);
}

TEST_F(CkksTensorTest, Conv2DCifar10) {
    auto image           = loadImage();
    auto [weights, bias] = loadWeights();
    ckks_vector_.Encrypt(image);
    ckks_tensor_.SetData(ckks_vector_, {3, 32, 32});
    auto result = ckks_tensor_.Conv2D(weights, bias);
    ASSERT_EQ(result.GetShape(), (shape_t{16, 32, 32}));

    auto decrypted_result = result.GetData().Decrypt();
    ASSERT_NEAR(decrypted_result[0], -0.0804625, 0.001);
}
