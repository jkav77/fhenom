#include <filesystem>
#include <vector>

#include "gtest/gtest.h"
#include "fhenom/ckks_vector.h"
#include <spdlog/spdlog.h>

using fhenom::CkksVector;

class CkksVectorTest : public ::testing::Test {
 protected:
  const std::vector<double> testData{0, 1, -1, 16, -16, 5, -100, 50, 100, 2, 10, 1, 2, 3, 4, 5, 17};
  const std::vector<double> testDomain{0, 1, 2, 4, 8, 16, 32, 64, 96, 100};
  const std::filesystem::path testDataDir{"testData/ckks_vector"};

  // lbcrypto::ScalingTechnique scTech = lbcrypto::FLEXIBLEAUTOEXT;
  const uint32_t multDepth = 2;
  const uint32_t scaleModSize = 24;
  const uint32_t firstModSize = 30;
  const uint32_t ringDim = 8192;
  const lbcrypto::SecurityLevel sl = lbcrypto::HEStd_128_classic;
  // uint32_t scaleModSize = 50;
  // uint32_t firstModSize = 60;
  // uint32_t batchSize = slots;

  double epsilon = 0.01;

  CkksVector ckksVector, precise_vector;
  CkksVectorTest() {
    spdlog::set_level(spdlog::level::debug);
    fhenom::Context context;
    fhenom::Context precise_context;

    if (std::filesystem::exists(testDataDir)) {
      context.load("testData/ckks_vector");
      context.loadRotationKeys(testDataDir / "key-rotate.txt");
      context.loadPublicKey("testData/ckks_vector/key-public.txt");
      context.loadSecretKey("testData/ckks_vector/key-secret.txt");
      ckksVector.setContext(context);
      ckksVector.load(testDataDir / "data.txt");

      precise_context.load(testDataDir / "precise");
      precise_context.loadRotationKeys(testDataDir / "precise" / "key-rotate.txt");
      precise_context.loadPublicKey(testDataDir / "precise" / "key-public.txt");
      precise_context.loadSecretKey(testDataDir / "precise" / "key-secret.txt");
      precise_vector.setContext(precise_context);
      precise_vector.load(testDataDir / "precise" / "data.txt");
    } else {
      lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS> ckksParameters;
      ckksParameters.SetMultiplicativeDepth(multDepth);
      ckksParameters.SetScalingModSize(scaleModSize);
      ckksParameters.SetFirstModSize(firstModSize);
      // ckksParameters.SetScalingTechnique(scTech);
      ckksParameters.SetSecurityLevel(sl);
      ckksParameters.SetRingDim(ringDim);
      // ckksParameters.SetBatchSize(batchSize);
      ckksParameters.SetSecretKeyDist(lbcrypto::UNIFORM_TERNARY);
      ckksParameters.SetKeySwitchTechnique(lbcrypto::HYBRID);
      ckksParameters.SetNumLargeDigits(2);
      context = fhenom::Context{ckksParameters};

      context.generateKeys();
      context.generateRotateKeys({-1, 1, 8, -8});

      lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS> precise_params;
      precise_params.SetMultiplicativeDepth(24);
      precise_params.SetScalingModSize(40);
      precise_params.SetFirstModSize(50);
      precise_params.SetSecurityLevel(sl);
      precise_params.SetRingDim(65536);
      precise_params.SetSecretKeyDist(lbcrypto::UNIFORM_TERNARY);
      precise_params.SetKeySwitchTechnique(lbcrypto::HYBRID);
      precise_params.SetNumLargeDigits(3);
      precise_context = fhenom::Context{precise_params};

      precise_context.generateKeys();
      precise_context.generateSumKey();
      precise_context.generateRotateKeys({-1, 1, 8, -8});

      std::filesystem::create_directories(testDataDir);
      context.save(testDataDir);
      context.saveRotationKeys(testDataDir / "key-rotate.txt");
      context.savePublicKey(testDataDir / "key-public.txt");
      context.saveSecretKey(testDataDir / "key-secret.txt");

      ckksVector.setContext(context);
      ckksVector.encrypt(testData);
      ckksVector.save(testDataDir / "data.txt");

      std::filesystem::create_directories(testDataDir / "precise");
      precise_context.save(testDataDir / "precise");
      precise_context.saveRotationKeys(testDataDir / "precise" / "key-rotate.txt");
      precise_context.savePublicKey(testDataDir / "precise" / "key-public.txt");
      precise_context.saveSecretKey(testDataDir / "precise" / "key-secret.txt");

      precise_vector.setContext(precise_context);
      precise_vector.encrypt(testData);
      precise_vector.save(testDataDir / "precise" / "data.txt");
    }
  }
};

//////////////////////////////////////////////////////////////////////////////
// Encryption and Decryption

TEST_F(CkksVectorTest, encrypt) {
  CkksVector col1{};
  col1.setContext(ckksVector.getContext());
  col1.encrypt(testData);

  ASSERT_GT(ckksVector.size(), 0);
  ASSERT_GT(ckksVector.getData().size(), 0);

  std::vector<double> largeData(48842, 1);
  CkksVector col2{};
  col2.setContext(ckksVector.getContext());
  col2.encrypt(largeData);
  ASSERT_EQ(col2.size(), 48842);
  ASSERT_EQ(col2.getData().size(), ceil(48842.0 / (ringDim / 2)));
}

TEST_F(CkksVectorTest, decrypt) {
  ckksVector.load(testDataDir / "data.txt");
  auto decrypted = ckksVector.decrypt();

  ASSERT_EQ(decrypted.size(), testData.size());
  for (int i = 0; i < testData.size(); ++i) {
    ASSERT_NEAR(decrypted[i], testData[i], epsilon);
  }
}

//////////////////////////////////////////////////////////////////////////////
// Homomorphic Operations

TEST_F(CkksVectorTest, Sign) {
  auto result = precise_vector.Sign();
  auto decrypted = result.decrypt();

  ASSERT_EQ(decrypted.size(), testData.size());
  for (int i = 0; i < testData.size(); ++i) {
    if (testData[i] < 0) {
      ASSERT_NEAR(decrypted[i], -1, 0.05);
    } else if (testData[i] > 0) {
      ASSERT_NEAR(decrypted[i], 1, 0.05);
    } else {
      ASSERT_NEAR(decrypted[i], 0, 0.05);
    }
  }
}

TEST_F(CkksVectorTest, IsEqual) {
  auto epsilon = 0.05;
  precise_vector.encrypt(testDomain);
  auto result = precise_vector.IsEqual(1);
  auto decrypted = result.decrypt();
  ASSERT_EQ(decrypted.size(), testDomain.size());
  for (int i = 0; i < testDomain.size(); ++i) {
    ASSERT_NEAR(decrypted[i], testDomain[i] == 1, epsilon) << "Index: " << i;
  }

  result = precise_vector.IsEqual(4);
  decrypted = result.decrypt();
  ASSERT_EQ(decrypted.size(), testDomain.size());
  for (int i = 0; i < testDomain.size(); ++i) {
    ASSERT_NEAR(decrypted[i], testDomain[i] == 4, epsilon);
  }

  result = precise_vector.IsEqual(17);
  decrypted = result.decrypt();
  ASSERT_EQ(decrypted.size(), testDomain.size());
  for (int i = 0; i < testDomain.size(); ++i) {
    ASSERT_NEAR(decrypted[i], testDomain[i] == 17, epsilon);
  }

  result = precise_vector.IsEqual(0);
  decrypted = result.decrypt();
  ASSERT_EQ(decrypted.size(), testDomain.size());
  for (int i = 0; i < testDomain.size(); ++i) {
    ASSERT_NEAR(decrypted[i], testDomain[i] == 0, epsilon);
  }
}

TEST_F(CkksVectorTest, Sum) {
  auto result = precise_vector.Sum();
  auto decrypted = result.decrypt();
  ASSERT_EQ(decrypted.size(), 1);
  ASSERT_NEAR(decrypted[0], std::reduce(testData.begin(), testData.end()), epsilon);
}

TEST_F(CkksVectorTest, rotate) {
  auto rotated = ckksVector.rotate(1);
  auto decrypted = rotated.decrypt();

  ASSERT_EQ(decrypted.size(), testData.size());
  for (int i = 0; i < testData.size(); ++i) {
    ASSERT_NEAR(decrypted[i], testData[(i + 1) % 17], epsilon);
  }

  rotated = ckksVector.rotate(-1);
  decrypted = rotated.decrypt();

  ASSERT_EQ(decrypted.size(), testData.size());
  for (int i = 0; i < testData.size(); ++i) {
    ASSERT_NEAR(decrypted[i], testData[(i - 1) % 17], epsilon);
  }

  rotated = ckksVector.rotate(8);
  decrypted = rotated.decrypt();
  ASSERT_EQ(decrypted.size(), testData.size());
  for (int i = 0; i < testData.size() - 8; ++i) {
    ASSERT_NEAR(decrypted[i], testData[(i + 8) % 17], epsilon);
  }

  std::vector<double> large_data(ringDim / 2);
  for (int i = 0; i < ringDim / 2; ++i) {
    large_data[i] = i;
  }
  CkksVector large_vec;
  large_vec.setContext(ckksVector.getContext());
  large_vec.encrypt(large_data);
  rotated = large_vec.rotate(-8);
  decrypted = rotated.decrypt();
  ASSERT_EQ(decrypted.size(), large_data.size());
  for (int i = 0; i < 8; ++i) {
    ASSERT_NEAR(decrypted[i], large_data[large_data.size() - 8 + i], epsilon);
  }
  ASSERT_NEAR(decrypted[8], large_data[0], epsilon);
  ASSERT_NEAR(decrypted[9], large_data[1], epsilon);

  large_data.clear();
  for (int i = 0; i < ringDim; ++i) {
    large_data.push_back(i);
  }
  large_vec.encrypt(large_data);
  rotated = large_vec.rotate(8);
  decrypted = rotated.decrypt();
  ASSERT_EQ(decrypted.size(), large_data.size());
  for (int i = 0; i < 8; ++i) {
    ASSERT_NEAR(decrypted[i], large_data[i + 8], epsilon);
    ASSERT_NEAR(decrypted[ringDim / 2 + i], large_data[ringDim / 2 + i + 8], epsilon);
  }
}

TEST_F(CkksVectorTest, fastRotate) {
  auto cryptoContext = ckksVector.getContext().getCryptoContext();

  auto keyMap = cryptoContext->GetEvalAutomorphismKeyMap(ckksVector.getData()[0]->GetKeyTag());
  for (const auto &rot_idx : {-1, 1, 8, -8}) {
    auto am_idx = lbcrypto::FindAutomorphismIndex2n(rot_idx, cryptoContext->GetCyclotomicOrder());
    ASSERT_EQ(keyMap.count(am_idx), 1);
  }

  auto idx = lbcrypto::FindAutomorphismIndex2n(7, cryptoContext->GetCyclotomicOrder());
  ASSERT_EQ(keyMap.count(idx), 0);
}

TEST_F(CkksVectorTest, multiply) {
  auto cryptoContext = ckksVector.getContext().getCryptoContext();

  CkksVector vector_1(ckksVector.getContext());
  CkksVector vector_2(ckksVector.getContext());
  auto test_size = ringDim - 1024;

  std::vector<double> values_1(test_size, 1);
  vector_1.encrypt(values_1);

  std::vector<double> values_2(test_size, 2);
  vector_2.encrypt(values_2);

  CkksVector vector_3 = vector_1 * values_2;

  ASSERT_EQ(vector_3.size(), test_size);
  auto values_3 = vector_3.decrypt();
  ASSERT_EQ(values_3.size(), test_size);
  for (int i = 0; i < test_size; ++i) {
    ASSERT_NEAR(values_3[i], 2, epsilon);
  }
}

TEST_F(CkksVectorTest, Concat) {
  auto cryptoContext = ckksVector.getContext().getCryptoContext();

  CkksVector vector_1(ckksVector.getContext());
  CkksVector vector_2(ckksVector.getContext());
  auto test_size = ringDim;

  std::vector<double> values_1(test_size, 1);
  vector_1.encrypt(values_1);

  std::vector<double> values_2(test_size, 2);
  vector_2.encrypt(values_2);

  vector_1.Concat(vector_2);

  ASSERT_EQ(vector_1.size(), test_size * 2);
  auto decrypted_values = vector_1.decrypt();
  ASSERT_EQ(decrypted_values.size(), test_size * 2);
  for (int i = 0; i < test_size; ++i) {
    ASSERT_NEAR(decrypted_values[i], 1, epsilon);
    ASSERT_NEAR(decrypted_values[i + test_size], 2, epsilon);
  }

  CkksVector vector_3(ckksVector.getContext());
  std::vector<double> values_3(values_1.begin(), values_1.end() - 1024);
  vector_3.encrypt(values_3);
  ASSERT_EQ(vector_3.size(), test_size - 1024);

  vector_3.Concat(vector_2);
  ASSERT_EQ(vector_3.size(), test_size * 2);
  decrypted_values = vector_3.decrypt();
  ASSERT_EQ(decrypted_values.size(), test_size * 2);
  for (int i = 0; i < test_size - 1024; ++i) {
    ASSERT_NEAR(decrypted_values[i], 1, epsilon);
    ASSERT_NEAR(decrypted_values[i + test_size], 2, epsilon);
  }

  for (int i = test_size - 1024; i < test_size; ++i) {
    ASSERT_NEAR(decrypted_values[i], 0, epsilon);
    ASSERT_NEAR(decrypted_values[i + test_size], 2, epsilon);
  }
}

//////////////////////////////////////////////////////////////////////////////
// File I/O

TEST_F(CkksVectorTest, save) {
  ckksVector.load(testDataDir / "data.txt");
  ckksVector.save(testDataDir / "data_test.txt");

  ASSERT_TRUE(std::filesystem::exists(testDataDir / "data_test.txt"));
  ASSERT_GT(std::filesystem::file_size(testDataDir / "data_test.txt"), 0);
}

TEST_F(CkksVectorTest, load) {
  ckksVector.load(testDataDir / "data.txt");

  ASSERT_GT(ckksVector.getData().size(), 0);
  ASSERT_EQ(ckksVector.size(), testData.size());

  CkksVector tmp;
  tmp.load(testDataDir / "data.txt");
  auto keys = tmp.getData()[0]->GetCryptoContext()->GetAllEvalMultKeys();
  ASSERT_GT(keys.size(), 0);
}
