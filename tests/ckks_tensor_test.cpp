#include "gtest/gtest.h"
#include "fhenom/ckks_tensor.h"
#include "fhenom/tensor.h"

using fhenom::CkksTensor;
using fhenom::CkksVector;
using fhenom::Context;
using fhenom::shape_t;
using fhenom::Tensor;

class CkksTensorTest : public ::testing::Test {
 protected:
  Context context;
  CkksVector ckksVector;
  CkksTensor ckksTensor;
  std::filesystem::path testDataDir{"testData/ckks_tensor"};
  Tensor kernel{{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                {3, 3, 3, 1}};
  Tensor kernel2{{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                  2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2},
                 {3, 3, 3, 2}};

  // Data for a 5x5x3 tensor representing a 3-channel 5x5 image
  std::vector<double> testData{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                               1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                               1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                               1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  double epsilon = 0.01;

  CkksTensorTest() {
    spdlog::set_level(spdlog::level::debug);

    if (!std::filesystem::exists(testDataDir)) {
      context = Context{getParameters()};
      context.generateKeys();
      context.generateRotateKeys({-1, -2, -3, -4, 1, 2, 3, 4, 25, 50});
      context.save(testDataDir);
      context.saveRotationKeys(testDataDir / "key-rotate.txt");
      context.savePublicKey(testDataDir / "key-public.txt");
      context.saveSecretKey(testDataDir / "key-secret.txt");
    } else {
      context.load(testDataDir);
      context.loadRotationKeys(testDataDir / "key-rotate.txt");
      context.loadPublicKey(testDataDir / "key-public.txt");
      context.loadSecretKey(testDataDir / "key-secret.txt");
    }

    ckksVector.setContext(context);
    ckksVector.encrypt(testData);
    spdlog::debug("Creating tensor with {} element vector and shape {} {} {}", ckksVector.size(), 5,
                  5, 3);
    ckksTensor = CkksTensor{ckksVector, {5, 5, 3}};
  }

  lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS> getParameters() {
    lbcrypto::ScalingTechnique scTech = lbcrypto::FLEXIBLEAUTOEXT;
    uint32_t multDepth = 2;
    if (scTech == lbcrypto::FLEXIBLEAUTOEXT) multDepth += 1;
    uint32_t scaleModSize = 24;
    uint32_t firstModSize = 30;
    uint32_t ringDim = 8192;
    lbcrypto::SecurityLevel sl = lbcrypto::HEStd_128_classic;
    // uint32_t slots = 16;  // sparsely-packed
    // uint32_t batchSize = slots;

    lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS> ckksParameters;
    ckksParameters.SetMultiplicativeDepth(multDepth);
    ckksParameters.SetScalingModSize(scaleModSize);
    ckksParameters.SetFirstModSize(firstModSize);
    ckksParameters.SetScalingTechnique(scTech);
    ckksParameters.SetSecurityLevel(sl);
    ckksParameters.SetRingDim(ringDim);
    // ckksParameters.SetBatchSize(batchSize);
    ckksParameters.SetSecretKeyDist(lbcrypto::UNIFORM_TERNARY);
    ckksParameters.SetKeySwitchTechnique(lbcrypto::HYBRID);
    ckksParameters.SetNumLargeDigits(3);
    return ckksParameters;
  }
};

TEST_F(CkksTensorTest, DefaultConstructor) {
  CkksTensor ckksTensor;
  EXPECT_EQ(ckksTensor.getData().size(), 0);
  EXPECT_EQ(ckksTensor.getShape(), shape_t{0});
}

TEST_F(CkksTensorTest, conv2D) {
  CkksTensor tensor = ckksTensor.conv2D(kernel);
  CkksVector vec = tensor.getData();
  std::vector<double> result = vec.decrypt();
  ASSERT_NEAR(result[0], 12, epsilon);
  ASSERT_NEAR(result[1], 18, epsilon);
  ASSERT_NEAR(result[5], 18, epsilon);
  ASSERT_NEAR(result[6], 27, epsilon);
  ASSERT_NEAR(result[4], 12, epsilon);
  ASSERT_NEAR(result[3], 18, epsilon);
  ASSERT_NEAR(result[9], 18, epsilon);
  ASSERT_NEAR(result[8], 27, epsilon);
  ASSERT_NEAR(result[24], 12, epsilon);
  ASSERT_NEAR(result[23], 18, epsilon);
  ASSERT_NEAR(result[19], 18, epsilon);
  ASSERT_NEAR(result[18], 27, epsilon);
}

TEST_F(CkksTensorTest, masking) {
  auto masked_vectors = ckksTensor.createMaskedConvVectors(kernel, 0);
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

  ckksTensor.createMaskedConvVectors(kernel2, 0);
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
