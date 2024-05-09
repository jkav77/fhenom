#include <fhenom/context.h>
#include <openfhe.h>
#include <spdlog/spdlog.h>
#include "test_utils.h"

#include <filesystem>

#include "gtest/gtest.h"

using fhenom::Context;

class ContextTest : public ::testing::Test {
protected:
    std::filesystem::path test_data_dir_{"testData/ckks_context"};
    Context context_;
    Context fhe_context_;
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
            fhe_context_ = GetFheContext();
            fhe_context_.GenerateKeys();
            fhe_context_.GenerateBootstrapKeys();
            auto fhe_dir = test_data_dir_ / "fhe";
            fhe_context_.Save(fhe_dir);
            fhe_context_.SavePublicKey(fhe_dir / "key-public.txt");
            fhe_context_.SaveSecretKey(fhe_dir / "key-secret.txt");

            context_ = GetLeveledContext();
            context_.GenerateKeys();
            context_.GenerateSumKey();
            context_.GenerateRotateKeys({-1, 1, -3, 3});
            context_.Save(test_data_dir_);
            context_.SavePublicKey(test_data_dir_ / "key-public.txt");
            context_.SaveSecretKey(test_data_dir_ / "key-secret.txt");
            context_.SaveEvalSumKeys(test_data_dir_ / "key-eval-sum.txt");
            context_.SaveRotationKeys(test_data_dir_ / "key-rotate.txt");
        }
        else {
            context_ = Context(test_data_dir_);
            context_.LoadPublicKey(test_data_dir_ / "key-public.txt");

            fhe_context_ = Context(test_data_dir_ / "fhe");
        }
    }
};

TEST_F(ContextTest, constructor) {
    Context context;
    SUCCEED();
}

TEST_F(ContextTest, hasRotationIndex) {
    ASSERT_TRUE(context_.HasRotationIdx(-1));
    ASSERT_TRUE(context_.HasRotationIdx(1));
    ASSERT_TRUE(context_.HasRotationIdx(-3));
    ASSERT_TRUE(context_.HasRotationIdx(3));
    ASSERT_TRUE(context_.HasRotationIdx(4));
    ASSERT_TRUE(context_.HasRotationIdx(8));
    ASSERT_FALSE(context_.HasRotationIdx(0));
    ASSERT_FALSE(context_.HasRotationIdx(5));
}

TEST_F(ContextTest, SaveCryptoContext) {
    std::filesystem::path path = test_data_dir_ / "cryptocontext_test.txt";
    context_.SaveCryptoContext(path);

    ASSERT_TRUE(std::filesystem::exists(path));
    ASSERT_GT(std::filesystem::file_size(path), 0);

    std::filesystem::remove(path);
}

TEST_F(ContextTest, LoadCryptoContext) {
    std::filesystem::path path = test_data_dir_ / "cryptocontext.txt";
    context_.LoadCryptoContext(path);

    ASSERT_THROW(context_.LoadCryptoContext(test_data_dir_ / "nonexistent.txt"), std::filesystem::filesystem_error);

    ASSERT_EQ(context_.GetCryptoContext()->GetRingDimension(), 8192);
}

TEST_F(ContextTest, SaveEvalMultKeys) {
    auto key_file = test_data_dir_ / "evalmultkeys_test.txt";
    context_.LoadEvalMultKeys(test_data_dir_ / "key-eval-mult.txt");
    context_.SaveEvalMultKeys(key_file);

    ASSERT_TRUE(std::filesystem::exists(key_file));
    ASSERT_TRUE(std::filesystem::file_size(key_file) > 0);

    std::filesystem::remove(key_file);
}

TEST_F(ContextTest, GenerateBootstrapKeys) {
    auto fhe_context    = GetFheContext();
    auto crypto_context = fhe_context.GetCryptoContext();
    crypto_context->ClearEvalAutomorphismKeys();
    ASSERT_EQ(crypto_context->GetAllEvalAutomorphismKeys().size(), 0);

    fhe_context.GenerateKeys();
    fhe_context.GenerateBootstrapKeys();
    ASSERT_GT(crypto_context->GetAllEvalAutomorphismKeys().size(), 0);
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

TEST_F(ContextTest, SaveEvalSumKeys) {
    auto key_file = test_data_dir_ / "evalsumkeys_test.txt";
    context_.LoadEvalSumKeys(test_data_dir_ / "key-eval-sum.txt");
    context_.SaveEvalSumKeys(key_file);

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
TEST_F(ContextTest, Load) {
    // ASSERT_NE(context.getCryptoContext()
    //               ->GetCryptoParameters()
    //               ->GetPlaintextModulus(),
    //           PlaintextModulus{65537});
    // ASSERT_EQ(context.getCryptoContext()->GetAllEvalMultKeys().size(), 0);
    // ASSERT_EQ(context.getCryptoContext()->GetAllEvalSumKeys().size(), 0);

    context_.Load(test_data_dir_);

    ASSERT_GT(context_.GetCryptoContext()->GetAllEvalMultKeys().size(), 0);
    ASSERT_GT(context_.GetCryptoContext()->GetAllEvalSumKeys().size(), 0);
}

TEST_F(ContextTest, Save) {
    auto test_dir = test_data_dir_ / "save_test";
    context_.Load(test_data_dir_);
    context_.LoadEvalMultKeys(test_data_dir_ / "key-eval-mult.txt");
    context_.LoadEvalSumKeys(test_data_dir_ / "key-eval-sum.txt");
    context_.Save(test_dir);
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

TEST_F(ContextTest, SavePublicKey) {
    auto key_file = test_data_dir_ / "key-public_test.txt";
    context_.LoadPublicKey(test_data_dir_ / "key-public.txt");
    context_.SavePublicKey(key_file);

    ASSERT_TRUE(std::filesystem::exists(key_file));
    ASSERT_TRUE(std::filesystem::file_size(key_file) > 0);

    std::filesystem::remove(key_file);
}

TEST_F(ContextTest, loadSecretKey) {
    ASSERT_EQ(context_.GetKeyPair().secretKey, nullptr);

    context_.LoadSecretKey(test_data_dir_ / "key-secret.txt");

    ASSERT_NE(context_.GetKeyPair().secretKey, nullptr);
}

TEST_F(ContextTest, SaveSecretKey) {
    auto key_file = test_data_dir_ / "key-secret_test.txt";
    context_.LoadSecretKey(test_data_dir_ / "key-secret.txt");
    context_.SaveSecretKey(key_file);

    ASSERT_TRUE(std::filesystem::exists(key_file));
    ASSERT_TRUE(std::filesystem::file_size(key_file) > 0);

    std::filesystem::remove(key_file);
}

TEST_F(ContextTest, SaveRotationKeys) {
    auto key_file = test_data_dir_ / "key-rotate_test.txt";
    context_.SaveRotationKeys(key_file);

    ASSERT_TRUE(std::filesystem::exists(key_file));
    ASSERT_TRUE(std::filesystem::file_size(key_file) > 0);

    std::filesystem::remove(key_file);
}

TEST_F(ContextTest, LoadRotationKeys) {
    auto key_file = test_data_dir_ / "key-rotate.txt";
    context_.LoadRotationKeys(key_file);

    ASSERT_GT(context_.GetCryptoContext()->GetAllEvalAutomorphismKeys().size(), 0);
}
