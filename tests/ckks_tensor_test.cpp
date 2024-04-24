#include <fhenom/ckks_tensor.h>
#include <fhenom/tensor.h>

#include "gtest/gtest.h"

using fhenom::CkksTensor;
using fhenom::CkksVector;
using fhenom::Context;
using fhenom::shape_t;
using fhenom::Tensor;

class CkksTensorTest : public ::testing::Test {
protected:
    Context context_;
    CkksVector ckks_vector_;
    CkksTensor ckks_tensor_;
    const std::filesystem::path test_data_dir_{"testData/ckks_tensor"};
    const Tensor kernel_{{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         {3, 3, 3, 1}};
    const Tensor kernel2_{{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                           2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2},
                          {3, 3, 3, 2}};

    // Data for a 5x5x3 tensor representing a 3-channel 5x5 image
    const std::vector<double> test_data_{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    const double epsilon_ = 0.01;

    CkksTensorTest() {
        spdlog::set_level(spdlog::level::debug);

        if (!std::filesystem::exists(test_data_dir_)) {
            context_ = Context{getParameters()};
            context_.generateKeys();
            context_.generateRotateKeys({-1, -2, -3, -4, 1, 2, 3, 4, 25, 50});
            context_.save(test_data_dir_);
            context_.saveRotationKeys(test_data_dir_ / "key-rotate.txt");
            context_.savePublicKey(test_data_dir_ / "key-public.txt");
            context_.saveSecretKey(test_data_dir_ / "key-secret.txt");
        }
        else {
            context_.load(test_data_dir_);
            context_.loadRotationKeys(test_data_dir_ / "key-rotate.txt");
            context_.loadPublicKey(test_data_dir_ / "key-public.txt");
            context_.loadSecretKey(test_data_dir_ / "key-secret.txt");
        }

        ckks_vector_.setContext(context_);
        ckks_vector_.encrypt(test_data_);
        spdlog::debug("Creating tensor with {} element vector and shape {} {} {}", ckks_vector_.size(), 5, 5, 3);
        ckks_tensor_ = CkksTensor{ckks_vector_, {5, 5, 3}};
    }

    static lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS> getParameters() {
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
    EXPECT_EQ(ckks_tensor.getData().size(), 0);
    EXPECT_EQ(ckks_tensor.getShape(), shape_t{0});
}

TEST_F(CkksTensorTest, conv2D) {
    CkksTensor tensor          = ckks_tensor_.conv2D(kernel_);
    CkksVector vec             = tensor.getData();
    std::vector<double> result = vec.decrypt();
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
}

TEST_F(CkksTensorTest, masking) {
    auto masked_vectors = ckks_tensor_.createMaskedConvVectors(kernel_, 0);
    EXPECT_EQ(masked_vectors.size(), 9);
    EXPECT_EQ(masked_vectors[0].size(), 75);
    EXPECT_EQ(masked_vectors[0][0], 0);
    EXPECT_EQ(masked_vectors[0][4], 0);
    EXPECT_EQ(masked_vectors[0][5], 0);
    EXPECT_EQ(masked_vectors[0][6], 1);
    EXPECT_EQ(masked_vectors[0][7], 1);
    EXPECT_EQ(masked_vectors[0][9], 1);
    EXPECT_EQ(masked_vectors[0][25], 0);
    EXPECT_EQ(masked_vectors[0][29], 0);
    EXPECT_EQ(masked_vectors[0][70], 0);
    EXPECT_EQ(masked_vectors[0][74], 1);
    EXPECT_EQ(masked_vectors[0][4 * 5], 0);
    EXPECT_EQ(masked_vectors[0][4 * 5 + 1], 1);
    auto total = std::reduce(masked_vectors[0].begin(), masked_vectors[0].end());
    EXPECT_EQ(total, 48);

    EXPECT_EQ(masked_vectors[4].size(), 75);
    EXPECT_EQ(masked_vectors[4][0], 1);
    EXPECT_EQ(masked_vectors[4][4], 1);
    EXPECT_EQ(masked_vectors[4][5], 1);
    EXPECT_EQ(masked_vectors[4][6], 1);
    EXPECT_EQ(masked_vectors[4][9], 1);
    EXPECT_EQ(masked_vectors[4][74], 1);
    EXPECT_EQ(masked_vectors[4][4 * 5], 1);
    EXPECT_EQ(masked_vectors[4][4 * 5 + 1], 1);
    total = std::reduce(masked_vectors[4].begin(), masked_vectors[4].end());
    EXPECT_EQ(total, 75);

    EXPECT_EQ(masked_vectors[8].size(), 75);
    EXPECT_EQ(masked_vectors[8][4], 0);
    EXPECT_EQ(masked_vectors[8][24], 0);
    EXPECT_EQ(masked_vectors[8][20], 0);
    EXPECT_EQ(masked_vectors[8][6], 1);
    EXPECT_EQ(masked_vectors[8][9], 0);
    EXPECT_EQ(masked_vectors[0][69], 1);
    EXPECT_EQ(masked_vectors[8][74], 0);
    EXPECT_EQ(masked_vectors[8][0], 1);
    total = std::reduce(masked_vectors[8].begin(), masked_vectors[8].end());
    EXPECT_EQ(total, 48);

    ckks_tensor_.createMaskedConvVectors(kernel2_, 0);
    EXPECT_EQ(masked_vectors.size(), 9);
    EXPECT_EQ(masked_vectors[0].size(), 75);
    EXPECT_EQ(masked_vectors[0][0], 0);
    EXPECT_EQ(masked_vectors[0][4], 0);
    EXPECT_EQ(masked_vectors[0][5], 0);
    EXPECT_EQ(masked_vectors[0][6], 1);
    EXPECT_EQ(masked_vectors[0][7], 1);
    EXPECT_EQ(masked_vectors[0][9], 1);
    EXPECT_EQ(masked_vectors[0][25], 0);
    EXPECT_EQ(masked_vectors[0][29], 0);
    EXPECT_EQ(masked_vectors[0][70], 0);
    EXPECT_EQ(masked_vectors[0][74], 1);
    EXPECT_EQ(masked_vectors[0][4 * 5], 0);
    EXPECT_EQ(masked_vectors[0][4 * 5 + 1], 1);
    total = std::reduce(masked_vectors[0].begin(), masked_vectors[0].end());
    EXPECT_EQ(total, 48);

    EXPECT_EQ(masked_vectors[4].size(), 75);
    EXPECT_EQ(masked_vectors[4][0], 1);
    EXPECT_EQ(masked_vectors[4][4], 1);
    EXPECT_EQ(masked_vectors[4][5], 1);
    EXPECT_EQ(masked_vectors[4][6], 1);
    EXPECT_EQ(masked_vectors[4][9], 1);
    EXPECT_EQ(masked_vectors[4][74], 1);
    EXPECT_EQ(masked_vectors[4][4 * 5], 1);
    EXPECT_EQ(masked_vectors[4][4 * 5 + 1], 1);
    total = std::reduce(masked_vectors[4].begin(), masked_vectors[4].end());
    EXPECT_EQ(total, 75);

    EXPECT_EQ(masked_vectors[8].size(), 75);
    EXPECT_EQ(masked_vectors[8][4], 0);
    EXPECT_EQ(masked_vectors[8][24], 0);
    EXPECT_EQ(masked_vectors[8][20], 0);
    EXPECT_EQ(masked_vectors[8][6], 1);
    EXPECT_EQ(masked_vectors[8][9], 0);
    EXPECT_EQ(masked_vectors[0][69], 1);
    EXPECT_EQ(masked_vectors[8][74], 0);
    EXPECT_EQ(masked_vectors[8][0], 1);
    total = std::reduce(masked_vectors[8].begin(), masked_vectors[8].end());
    EXPECT_EQ(total, 48);
}
