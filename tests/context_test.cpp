#include <binfhecontext.h>
#include <fhenom/context.h>
#include <openfhe.h>
#include <spdlog/spdlog.h>

#include <filesystem>

#include "gtest/gtest.h"

using fhenom::Context;

const uint kMultDepth{2};
const uint kRingDim{8192};
// const uint BATCH_SIZE{16};

class ContextTest : public ::testing::Test {
protected:
    std::filesystem::path test_data_dir_{"testData/ckks_context"};
    Context context_;
    ContextTest() {
        spdlog::set_level(spdlog::level::debug);
        using lbcrypto::FIXEDAUTO;
        using lbcrypto::FLEXIBLEAUTOEXT;
        using lbcrypto::HEStd_128_classic;
        using lbcrypto::HYBRID;
        using lbcrypto::ScalingTechnique;
        using lbcrypto::SecurityLevel;
        using lbcrypto::UNIFORM_TERNARY;

        if (!std::filesystem::exists(test_data_dir_)) {
            ScalingTechnique sc_tech = FIXEDAUTO;
            uint32_t mult_depth      = kMultDepth;
            if (sc_tech == FLEXIBLEAUTOEXT)
                mult_depth += 1;
            uint32_t scale_mod_size = 24;
            uint32_t first_mod_size = 30;
            uint32_t ring_dim       = kRingDim;
            SecurityLevel sl        = HEStd_128_classic;
            // uint32_t slots = BATCH_SIZE;  // sparsely-packed
            // uint32_t batchSize = slots;

            lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS> ckks_parameters;
            ckks_parameters.SetMultiplicativeDepth(mult_depth);
            ckks_parameters.SetScalingModSize(scale_mod_size);
            ckks_parameters.SetFirstModSize(first_mod_size);
            ckks_parameters.SetScalingTechnique(sc_tech);
            ckks_parameters.SetSecurityLevel(sl);
            ckks_parameters.SetRingDim(ring_dim);
            // ckksParameters.SetBatchSize(batchSize);
            ckks_parameters.SetSecretKeyDist(UNIFORM_TERNARY);
            ckks_parameters.SetKeySwitchTechnique(HYBRID);
            ckks_parameters.SetNumLargeDigits(3);

            spdlog::debug("Testing CKKS parameters");
            context_ = Context{ckks_parameters};
            context_.generateKeys();
            context_.generateSumKey();
            context_.generateRotateKeys({-1, 1, -3, 3});
            context_.save(test_data_dir_);
            context_.savePublicKey(test_data_dir_ / "key-public.txt");
            context_.saveSecretKey(test_data_dir_ / "key-secret.txt");
            context_.saveEvalSumKeys(test_data_dir_ / "key-eval-sum.txt");
            context_.saveRotationKeys(test_data_dir_ / "key-rotate.txt");
        }
        else {
            context_ = Context{};
            context_.load(test_data_dir_);
            context_.loadEvalSumKeys(test_data_dir_ / "key-eval-sum.txt");
            context_.loadRotationKeys(test_data_dir_ / "key-rotate.txt");
            context_.loadPublicKey(test_data_dir_ / "key-public.txt");
        }
    }
};

TEST_F(ContextTest, constructor) {
    Context context;
    SUCCEED();
}

TEST_F(ContextTest, hasRotationIndex) {
    ASSERT_TRUE(context_.hasRotationIdx(-1));
    ASSERT_TRUE(context_.hasRotationIdx(1));
    ASSERT_TRUE(context_.hasRotationIdx(-3));
    ASSERT_TRUE(context_.hasRotationIdx(3));
    ASSERT_TRUE(context_.hasRotationIdx(4));
    ASSERT_TRUE(context_.hasRotationIdx(8));
    ASSERT_FALSE(context_.hasRotationIdx(0));
    ASSERT_FALSE(context_.hasRotationIdx(5));
}

TEST_F(ContextTest, saveCryptoContext) {
    std::filesystem::path path = test_data_dir_ / "cryptocontext_test.txt";
    context_.saveCryptoContext(path);

    ASSERT_TRUE(std::filesystem::exists(path));
    ASSERT_GT(std::filesystem::file_size(path), 0);

    std::filesystem::remove(path);
}

TEST_F(ContextTest, loadCryptoContext) {
    std::filesystem::path path = test_data_dir_ / "cryptocontext.txt";
    context_.loadCryptoContext(path);

    ASSERT_THROW(context_.loadCryptoContext(test_data_dir_ / "nonexistent.txt"), std::filesystem::filesystem_error);

    ASSERT_EQ(context_.getCryptoContext()->GetRingDimension(), kRingDim);
}

TEST_F(ContextTest, saveEvalMultKeys) {
    auto key_file = test_data_dir_ / "evalmultkeys_test.txt";
    context_.loadEvalMultKeys(test_data_dir_ / "key-eval-mult.txt");
    context_.saveEvalMultKeys(key_file);

    ASSERT_TRUE(std::filesystem::exists(key_file));
    ASSERT_TRUE(std::filesystem::file_size(key_file) > 0);

    std::filesystem::remove(key_file);
}

// This test doesn't work because the eval keys are already loaded in the
// existing context
// TEST_F(ContextTest, loadEvalMultKeys) {
//   context.loadCryptoContext(test_data_dir / "cryptocontext.txt");
//   ASSERT_EQ(context.getCryptoContext()->GetAllEvalMultKeys().size(), 0);
//
//   ASSERT_THROW(
//       context.loadEvalMultKeys(test_data_dir / "nonexistent_evalmultkeys.txt"),
//       std::filesystem::filesystem_error);
//
//   context.loadEvalMultKeys(test_data_dir / "key-eval-mult.txt");
//
//   ASSERT_GT(context.getCryptoContext()->GetAllEvalMultKeys().size(), 0);
// }

TEST_F(ContextTest, saveEvalSumKeys) {
    auto key_file = test_data_dir_ / "evalsumkeys_test.txt";
    context_.loadEvalSumKeys(test_data_dir_ / "key-eval-sum.txt");
    context_.saveEvalSumKeys(key_file);

    ASSERT_TRUE(std::filesystem::exists(key_file));
    ASSERT_TRUE(std::filesystem::file_size(key_file) > 0);

    std::filesystem::remove(key_file);
}

// This test doesn't work because the eval keys are already loaded in the
// existing context
//  TEST_F(ContextTest, loadEvalSumKeys) {
//   context.loadCryptoContext(test_data_dir / "cryptocontext.txt");
//   ASSERT_EQ(context.getCryptoContext()->GetAllEvalSumKeys().size(), 0);
//
//   ASSERT_THROW(
//       context.loadEvalSumKeys(test_data_dir / "nonexistent_evalsumkeys.txt"),
//       std::filesystem::filesystem_error);
//
//   context.loadEvalSumKeys(test_data_dir / "key-eval-sum.txt");
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

    context_.load(test_data_dir_);

    ASSERT_GT(context_.getCryptoContext()->GetAllEvalMultKeys().size(), 0);
    ASSERT_GT(context_.getCryptoContext()->GetAllEvalSumKeys().size(), 0);
}

TEST_F(ContextTest, save) {
    auto test_dir = test_data_dir_ / "save_test";
    context_.load(test_data_dir_);
    context_.loadEvalMultKeys(test_data_dir_ / "key-eval-mult.txt");
    context_.loadEvalSumKeys(test_data_dir_ / "key-eval-sum.txt");
    context_.save(test_dir);
    ASSERT_TRUE(std::filesystem::exists(test_dir));
    ASSERT_TRUE(std::filesystem::exists(test_dir / "cryptocontext.txt"));
    ASSERT_TRUE(std::filesystem::exists(test_dir / "key-eval-mult.txt"));
    ASSERT_TRUE(std::filesystem::exists(test_dir / "key-eval-sum.txt"));

    std::filesystem::remove_all(test_dir);
}

// Doesn't work because the public key is already loaded in the existing context
// TEST_F(ContextTest, loadPublicKey) {
// ASSERT_EQ(context.getKeyPair().publicKey, nullptr);
//
// context.loadPublicKey(test_data_dir / "key-public.txt");
//
// ASSERT_NE(context.getKeyPair().publicKey, nullptr);
// }

TEST_F(ContextTest, savePublicKey) {
    auto key_file = test_data_dir_ / "key-public_test.txt";
    context_.loadPublicKey(test_data_dir_ / "key-public.txt");
    context_.savePublicKey(key_file);

    ASSERT_TRUE(std::filesystem::exists(key_file));
    ASSERT_TRUE(std::filesystem::file_size(key_file) > 0);

    std::filesystem::remove(key_file);
}

TEST_F(ContextTest, loadSecretKey) {
    ASSERT_EQ(context_.getKeyPair().secretKey, nullptr);

    context_.loadSecretKey(test_data_dir_ / "key-secret.txt");

    ASSERT_NE(context_.getKeyPair().secretKey, nullptr);
}

TEST_F(ContextTest, saveSecretKey) {
    auto key_file = test_data_dir_ / "key-secret_test.txt";
    context_.loadSecretKey(test_data_dir_ / "key-secret.txt");
    context_.saveSecretKey(key_file);

    ASSERT_TRUE(std::filesystem::exists(key_file));
    ASSERT_TRUE(std::filesystem::file_size(key_file) > 0);

    std::filesystem::remove(key_file);
}

TEST_F(ContextTest, saveRotationKeys) {
    auto key_file = test_data_dir_ / "key-rotate_test.txt";
    context_.saveRotationKeys(key_file);

    ASSERT_TRUE(std::filesystem::exists(key_file));
    ASSERT_TRUE(std::filesystem::file_size(key_file) > 0);

    std::filesystem::remove(key_file);
}

TEST_F(ContextTest, loadRotationKeys) {
    auto key_file = test_data_dir_ / "key-rotate.txt";
    context_.loadRotationKeys(key_file);

    ASSERT_GT(context_.getCryptoContext()->GetAllEvalAutomorphismKeys().size(), 0);
}
