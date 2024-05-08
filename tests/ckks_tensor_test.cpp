#include <fhenom/ckks_tensor.h>
#include <fhenom/tensor.h>

#include "gtest/gtest.h"

using fhenom::CkksTensor;
using fhenom::CkksVector;
using fhenom::Context;
using fhenom::shape_t;
using fhenom::Tensor;

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
    const Tensor kernel_{{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {1, 3, 3, 3}};
    const Tensor kernel2_{{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                           2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2},
                          {2, 3, 3, 3}};

    // Data for a 3x5x5 tensor representing a 3-channel 5x5 image
    const std::vector<double> test_data_{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    const double epsilon_ = 0.01;

    CkksTensorTest() {
        spdlog::set_level(spdlog::level::debug);

        if (!std::filesystem::exists(test_data_dir_)) {
            context_ = Context{GetParameters()};
            context_.GenerateKeys();
            context_.GenerateRotateKeys({-1, -2, -4, -8, -16, -32, -64, -128, -256, 1, 2, 4, 8, 16, 32, 64, 128, 256});
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
    CkksTensor tensor = ckks_tensor_.Conv2D(kernel_);
    ASSERT_EQ(tensor.GetShape(), (shape_t{1, 5, 5}));
    CkksVector vec = tensor.GetData();
    ASSERT_EQ(vec.size(), 25);
    std::vector<double> result = vec.Decrypt();
    ASSERT_NEAR(result[0], 12, epsilon_);
    ASSERT_NEAR(result[1], 18, epsilon_);
    ASSERT_NEAR(result[5], 18, epsilon_);
    ASSERT_NEAR(result[6], 27, epsilon_);
    ASSERT_NEAR(result[4], 12, epsilon_);
    ASSERT_NEAR(result[3], 18, epsilon_);
    ASSERT_NEAR(result[9], 18, epsilon_);
    ASSERT_NEAR(result[8], 27, epsilon_);
    ASSERT_NEAR(result[24], 12, epsilon_);
    ASSERT_NEAR(result[23], 18, epsilon_);
    ASSERT_NEAR(result[19], 18, epsilon_);
    ASSERT_NEAR(result[18], 27, epsilon_);

    tensor = ckks_tensor_.Conv2D(kernel2_);
    vec    = tensor.GetData();
    ASSERT_EQ(vec.size(), 50);
    result = vec.Decrypt();
    // first filter
    ASSERT_NEAR(result[0], 12, epsilon_);
    ASSERT_NEAR(result[1], 18, epsilon_);
    ASSERT_NEAR(result[5], 18, epsilon_);
    ASSERT_NEAR(result[6], 27, epsilon_);
    ASSERT_NEAR(result[4], 12, epsilon_);
    ASSERT_NEAR(result[3], 18, epsilon_);
    ASSERT_NEAR(result[9], 18, epsilon_);
    ASSERT_NEAR(result[8], 27, epsilon_);
    ASSERT_NEAR(result[24], 12, epsilon_);
    ASSERT_NEAR(result[23], 18, epsilon_);
    ASSERT_NEAR(result[19], 18, epsilon_);
    ASSERT_NEAR(result[18], 27, epsilon_);

    // second filter
    ASSERT_NEAR(result[25], 24, epsilon_);
    ASSERT_NEAR(result[26], 36, epsilon_);
    ASSERT_NEAR(result[5], 18, epsilon_);
    ASSERT_NEAR(result[6], 27, epsilon_);
    ASSERT_NEAR(result[4], 12, epsilon_);
    ASSERT_NEAR(result[3], 18, epsilon_);
    ASSERT_NEAR(result[9], 18, epsilon_);
    ASSERT_NEAR(result[8], 27, epsilon_);
    ASSERT_NEAR(result[24], 12, epsilon_);
    ASSERT_NEAR(result[23], 18, epsilon_);
    ASSERT_NEAR(result[19], 18, epsilon_);
    ASSERT_NEAR(result[18], 27, epsilon_);
}
