#include <binfhecontext.h>
#include <openfhe.h>

#include <filesystem>

#include "gtest/gtest.h"
#include "fhenom/context.h"
#include <spdlog/spdlog.h>

using fhenom::Context;

const uint MULT_DEPTH{2};
const uint RING_DIM{8192};
// const uint BATCH_SIZE{16};

class ContextTest : public ::testing::Test {
 protected:
  std::filesystem::path testDataDir{"testData/ckks_context"};
  Context context;
  ContextTest() {
    using namespace lbcrypto;
    spdlog::set_level(spdlog::level::debug);

    if (!std::filesystem::exists(testDataDir)) {
      ScalingTechnique scTech = FIXEDAUTO;
      uint32_t multDepth = MULT_DEPTH;
      if (scTech == FLEXIBLEAUTOEXT) multDepth += 1;
      uint32_t scaleModSize = 24;
      uint32_t firstModSize = 30;
      uint32_t ringDim = RING_DIM;
      SecurityLevel sl = HEStd_128_classic;
      // uint32_t slots = BATCH_SIZE;  // sparsely-packed
      // uint32_t batchSize = slots;

      lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS> ckksParameters;
      ckksParameters.SetMultiplicativeDepth(multDepth);
      ckksParameters.SetScalingModSize(scaleModSize);
      ckksParameters.SetFirstModSize(firstModSize);
      ckksParameters.SetScalingTechnique(scTech);
      ckksParameters.SetSecurityLevel(sl);
      ckksParameters.SetRingDim(ringDim);
      // ckksParameters.SetBatchSize(batchSize);
      ckksParameters.SetSecretKeyDist(UNIFORM_TERNARY);
      ckksParameters.SetKeySwitchTechnique(HYBRID);
      ckksParameters.SetNumLargeDigits(3);

      spdlog::debug("Testing CKKS parameters");
      context = Context{ckksParameters};
      context.generateKeys();
      context.generateSumKey();
      context.generateRotateKeys({-1, 1, -3, 3});
      context.save(testDataDir);
      context.savePublicKey(testDataDir / "key-public.txt");
      context.saveSecretKey(testDataDir / "key-secret.txt");
      context.saveEvalSumKeys(testDataDir / "key-eval-sum.txt");
      context.saveRotationKeys(testDataDir / "key-rotate.txt");
    } else {
      context = Context{};
      context.load(testDataDir);
      context.loadEvalSumKeys(testDataDir / "key-eval-sum.txt");
      context.loadRotationKeys(testDataDir / "key-rotate.txt");
      context.loadPublicKey(testDataDir / "key-public.txt");
    }
  }
};

TEST_F(ContextTest, constructor) {
  Context context;
  SUCCEED();
}

TEST_F(ContextTest, hasRotationIndex) {
  ASSERT_TRUE(context.hasRotationIdx(-1));
  ASSERT_TRUE(context.hasRotationIdx(1));
  ASSERT_TRUE(context.hasRotationIdx(-3));
  ASSERT_TRUE(context.hasRotationIdx(3));
  ASSERT_TRUE(context.hasRotationIdx(4));
  ASSERT_TRUE(context.hasRotationIdx(8));
  ASSERT_FALSE(context.hasRotationIdx(0));
  ASSERT_FALSE(context.hasRotationIdx(5));
}

TEST_F(ContextTest, saveCryptoContext) {
  std::filesystem::path path = testDataDir / "cryptocontext_test.txt";
  context.saveCryptoContext(path);

  ASSERT_TRUE(std::filesystem::exists(path));
  ASSERT_GT(std::filesystem::file_size(path), 0);

  std::filesystem::remove(path);
}

TEST_F(ContextTest, loadCryptoContext) {
  std::filesystem::path path = testDataDir / "cryptocontext.txt";
  context.loadCryptoContext(path);

  ASSERT_THROW(context.loadCryptoContext(testDataDir / "nonexistent.txt"),
               std::filesystem::filesystem_error);

  ASSERT_EQ(context.getCryptoContext()->GetRingDimension(), RING_DIM);
}

TEST_F(ContextTest, saveEvalMultKeys) {
  auto keyFile = testDataDir / "evalmultkeys_test.txt";
  context.loadEvalMultKeys(testDataDir / "key-eval-mult.txt");
  context.saveEvalMultKeys(keyFile);

  ASSERT_TRUE(std::filesystem::exists(keyFile));
  ASSERT_TRUE(std::filesystem::file_size(keyFile) > 0);

  std::filesystem::remove(keyFile);
}

// This test doesn't work because the eval keys are already loaded in the
// existing context
// TEST_F(ContextTest, loadEvalMultKeys) {
//   context.loadCryptoContext(testDataDir / "cryptocontext.txt");
//   ASSERT_EQ(context.getCryptoContext()->GetAllEvalMultKeys().size(), 0);
//
//   ASSERT_THROW(
//       context.loadEvalMultKeys(testDataDir / "nonexistent_evalmultkeys.txt"),
//       std::filesystem::filesystem_error);
//
//   context.loadEvalMultKeys(testDataDir / "key-eval-mult.txt");
//
//   ASSERT_GT(context.getCryptoContext()->GetAllEvalMultKeys().size(), 0);
// }

TEST_F(ContextTest, saveEvalSumKeys) {
  auto keyFile = testDataDir / "evalsumkeys_test.txt";
  context.loadEvalSumKeys(testDataDir / "key-eval-sum.txt");
  context.saveEvalSumKeys(keyFile);

  ASSERT_TRUE(std::filesystem::exists(keyFile));
  ASSERT_TRUE(std::filesystem::file_size(keyFile) > 0);

  std::filesystem::remove(keyFile);
}

// This test doesn't work because the eval keys are already loaded in the
// existing context
//  TEST_F(ContextTest, loadEvalSumKeys) {
//   context.loadCryptoContext(testDataDir / "cryptocontext.txt");
//   ASSERT_EQ(context.getCryptoContext()->GetAllEvalSumKeys().size(), 0);
//
//   ASSERT_THROW(
//       context.loadEvalSumKeys(testDataDir / "nonexistent_evalsumkeys.txt"),
//       std::filesystem::filesystem_error);
//
//   context.loadEvalSumKeys(testDataDir / "key-eval-sum.txt");
//
//   spdlog::info("Number of eval mult keys: {}",
//                context.getCryptoContext()->GetAllEvalSumKeys().size());
//   ASSERT_GT(context.getCryptoContext()->GetAllEvalSumKeys().size(), 0);
// }

// This test doesn't work because the eval keys are already loaded in the
// existing context
TEST_F(ContextTest, load) {
  // ASSERT_NE(context.getCryptoContext()
  //               ->GetCryptoParameters()
  //               ->GetPlaintextModulus(),
  //           PlaintextModulus{65537});
  // ASSERT_EQ(context.getCryptoContext()->GetAllEvalMultKeys().size(), 0);
  // ASSERT_EQ(context.getCryptoContext()->GetAllEvalSumKeys().size(), 0);

  context.load(testDataDir);

  ASSERT_GT(context.getCryptoContext()->GetAllEvalMultKeys().size(), 0);
  ASSERT_GT(context.getCryptoContext()->GetAllEvalSumKeys().size(), 0);
}

TEST_F(ContextTest, save) {
  auto testDir = testDataDir / "save_test";
  context.load(testDataDir);
  context.loadEvalMultKeys(testDataDir / "key-eval-mult.txt");
  context.loadEvalSumKeys(testDataDir / "key-eval-sum.txt");
  context.save(testDir);
  ASSERT_TRUE(std::filesystem::exists(testDir));
  ASSERT_TRUE(std::filesystem::exists(testDir / "cryptocontext.txt"));
  ASSERT_TRUE(std::filesystem::exists(testDir / "key-eval-mult.txt"));
  ASSERT_TRUE(std::filesystem::exists(testDir / "key-eval-sum.txt"));

  std::filesystem::remove_all(testDir);
}

// Doesn't work because the public key is already loaded in the existing context
// TEST_F(ContextTest, loadPublicKey) {
// ASSERT_EQ(context.getKeyPair().publicKey, nullptr);
//
// context.loadPublicKey(testDataDir / "key-public.txt");
//
// ASSERT_NE(context.getKeyPair().publicKey, nullptr);
// }

TEST_F(ContextTest, savePublicKey) {
  auto keyFile = testDataDir / "key-public_test.txt";
  context.loadPublicKey(testDataDir / "key-public.txt");
  context.savePublicKey(keyFile);

  ASSERT_TRUE(std::filesystem::exists(keyFile));
  ASSERT_TRUE(std::filesystem::file_size(keyFile) > 0);

  std::filesystem::remove(keyFile);
}

TEST_F(ContextTest, loadSecretKey) {
  ASSERT_EQ(context.getKeyPair().secretKey, nullptr);

  context.loadSecretKey(testDataDir / "key-secret.txt");

  ASSERT_NE(context.getKeyPair().secretKey, nullptr);
}

TEST_F(ContextTest, saveSecretKey) {
  auto keyFile = testDataDir / "key-secret_test.txt";
  context.loadSecretKey(testDataDir / "key-secret.txt");
  context.saveSecretKey(keyFile);

  ASSERT_TRUE(std::filesystem::exists(keyFile));
  ASSERT_TRUE(std::filesystem::file_size(keyFile) > 0);

  std::filesystem::remove(keyFile);
}

TEST_F(ContextTest, saveRotationKeys) {
  auto keyFile = testDataDir / "key-rotate_test.txt";
  context.saveRotationKeys(keyFile);

  ASSERT_TRUE(std::filesystem::exists(keyFile));
  ASSERT_TRUE(std::filesystem::file_size(keyFile) > 0);

  std::filesystem::remove(keyFile);
}

TEST_F(ContextTest, loadRotationKeys) {
  auto keyFile = testDataDir / "key-rotate.txt";
  context.loadRotationKeys(keyFile);

  ASSERT_GT(context.getCryptoContext()->GetAllEvalAutomorphismKeys().size(), 0);
}
