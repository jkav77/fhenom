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
    ContextTest() {
        spdlog::set_level(spdlog::level::debug);

        context_ = get_leveled_context();
        context_.GenerateKeys();
        context_.GenerateSumKey();
        context_.GenerateRotateKeys({-1, 1, -3, 3});
        context_.Save(test_data_dir_);
        context_.SavePublicKey(test_data_dir_ / "key-public.txt");
        context_.SaveSecretKey(test_data_dir_ / "key-secret.txt");
        context_.SaveEvalSumKeys(test_data_dir_ / "key-eval-sum.txt");
        context_.SaveRotationKeys(test_data_dir_ / "key-rotate.txt");
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
