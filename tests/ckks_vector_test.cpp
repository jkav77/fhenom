#include <fhenom/ckks_vector.h>
#include <gtest/gtest.h>
#include <spdlog/spdlog.h>
#include <openfhe.h>
#include "test_utils.h"
#include "../src/coefficients.h"

#include <filesystem>
#include <vector>

using fhenom::CkksVector;

class CkksVectorTest : public ::testing::Test {
protected:
    fhenom::Context context_;
    fhenom::Context precise_context_;
    const std::vector<double> test_data_{0, 1, -1, 16, -16, 5, 50, 2, 10, 1, 2, 3, 4, 5, 17};
    const std::vector<double> testDomain_{0, 1, 2, 4, 8, 16, 32, 64, 96, 100};
    const std::filesystem::path test_data_dir_{"testData/ckks_vector"};
    const std::vector<int> rotation_indices_{1,    2,    4,    8,    16,    32,    64,    128,   256,   512,
                                             1024, 2048, 4096, 8192, 16384, -1,    -2,    -4,    -8,    -16,
                                             -32,  -64,  -128, -256, -512,  -1024, -2048, -4096, -8192, -16384};

    double epsilon_ = 0.01;

    CkksVector ckks_vector_, precise_vector_;
    CkksVectorTest() {
        spdlog::set_level(spdlog::level::debug);

        spdlog::debug("No saved test data found, generating new data...");
        context_ = get_leveled_context();
        spdlog::debug("Generating keys");
        context_.GenerateKeys();

        ckks_vector_.SetContext(context_);
        ckks_vector_.Encrypt(test_data_);

        precise_context_ = get_high_mult_depth_leveled_context();
        spdlog::debug("Generating keys for precise vector");
        precise_context_.GenerateKeys();

        precise_vector_.SetContext(precise_context_);
        precise_vector_.Encrypt(test_data_);
    }
};

//////////////////////////////////////////////////////////////////////////////
// Encryption and Decryption

TEST_F(CkksVectorTest, EncryptDecrypt) {
    CkksVector col1{};
    col1.SetContext(ckks_vector_.GetContext());
    col1.Encrypt(test_data_);

    ASSERT_GT(ckks_vector_.size(), 0);
    ASSERT_GT(ckks_vector_.GetData().size(), 0);

    std::vector<double> large_data(48842, 1);
    CkksVector col2{};
    col2.SetContext(ckks_vector_.GetContext());
    col2.Encrypt(large_data);
    ASSERT_EQ(col2.size(), 48842);

    auto decrypted = ckks_vector_.Decrypt();

    ASSERT_EQ(decrypted.size(), test_data_.size());
    for (unsigned i = 0; i < test_data_.size(); ++i) {
        ASSERT_NEAR(decrypted[i], test_data_[i], epsilon_);
    }
}

//////////////////////////////////////////////////////////////////////////////
// Homomorphic Operations

// Uncomment to run. Very slow.
// TEST_F(CkksVectorTest, Bootstrap) {
//     auto fhe_context = get_fhe_context();
//     fhe_context.GenerateKeys();
//     fhe_context.GenerateBootstrapKeys();

//     CkksVector fhe_vector(fhe_context);
//     fhe_vector.Encrypt(test_data_);

//     fhe_vector.GetSignUsingPolyComp();
//     spdlog::debug("Ciphertext level before bootstrap: {}", fhe_vector.GetData()[0]->GetLevel());
//     fhe_vector.Bootstrap();
//     spdlog::debug("Ciphertext level after bootstrap: {}", fhe_vector.GetData()[0]->GetLevel());
// }

TEST_F(CkksVectorTest, GetSign) {
    precise_vector_ *= std::vector<double>(test_data_.size(), 1.0 / 50.0);
    auto result = precise_vector_.GetSign().Decrypt();

    ASSERT_EQ(result.size(), test_data_.size());
    for (unsigned i = 0; i < test_data_.size(); ++i) {
        if (test_data_[i] < 0) {
            ASSERT_NEAR(result[i], -1, epsilon_);
        }
        else if (test_data_[i] > 0) {
            ASSERT_NEAR(result[i], 1, epsilon_);
        }
        else {
            ASSERT_NEAR(result[i], 0, epsilon_);
        }
    }
}

TEST_F(CkksVectorTest, GetSignUsingPolyComp) {
    precise_vector_ *= std::vector<double>(15, 1.0 / 100.0);
    auto decrypted = precise_vector_.Decrypt();
    auto result    = precise_vector_.GetSignUsingPolyComp().Decrypt();

    ASSERT_EQ(result.size(), test_data_.size());

    for (unsigned i = 0; i < test_data_.size(); ++i) {
        if (test_data_[i] < 0) {
            ASSERT_NEAR(result[i], -1, epsilon_);
        }
        else if (test_data_[i] > 0) {
            ASSERT_NEAR(result[i], 1, epsilon_);
        }
        else {
            ASSERT_NEAR(result[i], 0, epsilon_);
        }
    }
}

TEST_F(CkksVectorTest, IsEqual) {
    precise_vector_.Encrypt(testDomain_);
    auto result    = precise_vector_.IsEqual(1, 100);
    auto decrypted = result.Decrypt();
    ASSERT_EQ(decrypted.size(), testDomain_.size());
    for (unsigned i = 0; i < testDomain_.size(); ++i) {
        ASSERT_NEAR(decrypted[i], testDomain_[i] == 1, epsilon_) << "Index: " << i;
    }

    result    = precise_vector_.IsEqual(4, 100);
    decrypted = result.Decrypt();
    ASSERT_EQ(decrypted.size(), testDomain_.size());
    for (unsigned i = 0; i < testDomain_.size(); ++i) {
        ASSERT_NEAR(decrypted[i], testDomain_[i] == 4, epsilon_);
    }

    result    = precise_vector_.IsEqual(17, 100);
    decrypted = result.Decrypt();
    ASSERT_EQ(decrypted.size(), testDomain_.size());
    for (unsigned i = 0; i < testDomain_.size(); ++i) {
        ASSERT_NEAR(decrypted[i], testDomain_[i] == 17, epsilon_);
    }

    result    = precise_vector_.IsEqual(0, 100);
    decrypted = result.Decrypt();
    ASSERT_EQ(decrypted.size(), testDomain_.size());
    for (unsigned i = 0; i < testDomain_.size(); ++i) {
        ASSERT_NEAR(decrypted[i], testDomain_[i] == 0, epsilon_);
    }
}

TEST_F(CkksVectorTest, GetSum) {
    auto ring_dim = precise_vector_.GetContext().GetCryptoContext()->GetRingDimension();
    precise_context_.GenerateRotateKeys(rotation_indices_);

    precise_vector_.Encrypt(std::vector<double>(ring_dim * 1.25, 1));
    auto result    = precise_vector_.GetSum();
    auto decrypted = result.Decrypt();
    ASSERT_NEAR(decrypted[0], ring_dim * 1.25, epsilon_);
}

TEST_F(CkksVectorTest, GetCount) {
    precise_context_.GenerateSumKey();
    auto ring_dim = precise_vector_.GetContext().GetCryptoContext()->GetRingDimension();

    precise_vector_.Encrypt(std::vector<double>(40000, 1));
    auto result    = precise_vector_.GetCount(1);
    auto decrypted = result.Decrypt();
    ASSERT_NEAR(decrypted[0], 40000, epsilon_);

    std::vector<double> large_data(ring_dim * 1.25, 0);
    precise_vector_.Encrypt(large_data);
    result    = precise_vector_.GetCount(0);
    decrypted = result.Decrypt();
    ASSERT_NEAR(decrypted[0], ring_dim * 1.25, epsilon_);
}

TEST_F(CkksVectorTest, Rotate) {
    context_.GenerateRotateKeys(rotation_indices_);

    auto rotated   = ckks_vector_.Rotate(1);
    auto decrypted = rotated.Decrypt();
    ASSERT_EQ(decrypted.size(), test_data_.size());
    for (unsigned i = 0; i < test_data_.size(); ++i) {
        ASSERT_NEAR(decrypted[i], test_data_[(i + 1) % test_data_.size()], epsilon_);
    }

    rotated   = ckks_vector_.Rotate(-3);
    decrypted = rotated.Decrypt();
    ASSERT_EQ(decrypted.size(), test_data_.size());
    for (unsigned i = 0; i < test_data_.size() - 3; ++i) {
        ASSERT_NEAR(decrypted[i + 3], test_data_[i], epsilon_);
    }

    rotated   = ckks_vector_.Rotate(-1);
    decrypted = rotated.Decrypt();
    ASSERT_EQ(decrypted.size(), test_data_.size());
    for (unsigned i = 0; i < test_data_.size(); ++i) {
        ASSERT_NEAR(decrypted[i], test_data_[(i - 1) % test_data_.size()], epsilon_);
    }

    rotated   = ckks_vector_.Rotate(8);
    decrypted = rotated.Decrypt();
    ASSERT_EQ(decrypted.size(), test_data_.size());
    for (unsigned i = 0; i < test_data_.size() - 8; ++i) {
        ASSERT_NEAR(decrypted[i], test_data_[(i + 8) % test_data_.size()], epsilon_);
    }

    rotated   = ckks_vector_.Rotate(3);
    decrypted = rotated.Decrypt();
    ASSERT_EQ(decrypted.size(), test_data_.size());
    for (unsigned i = 0; i < test_data_.size() - 3; ++i) {
        ASSERT_NEAR(decrypted[i], test_data_[(i + 3) % test_data_.size()], epsilon_);
    }

    auto batch_size = ckks_vector_.GetContext().GetCryptoContext()->GetEncodingParams()->GetBatchSize();
    std::vector<double> large_data(2 * batch_size);
    for (unsigned i = 0; i < large_data.size(); ++i) {
        large_data[i] = i;
    }
    CkksVector large_vec;
    large_vec.SetContext(ckks_vector_.GetContext());
    large_vec.Encrypt(large_data);
    rotated   = large_vec.Rotate(-8);
    decrypted = rotated.Decrypt();
    ASSERT_EQ(decrypted.size(), large_data.size());
    for (int i = 0; i < 8; ++i) {
        ASSERT_NEAR(decrypted[i], large_data[batch_size - 8 + i], epsilon_);
    }
    ASSERT_NEAR(decrypted[8], large_data[0], epsilon_);
    ASSERT_NEAR(decrypted[9], large_data[1], epsilon_);

    large_data = std::vector<double>(2 * batch_size);
    for (unsigned i = 0; i < large_data.size(); ++i) {
        large_data[i] = i;
    }
    large_vec.Encrypt(large_data);
    rotated   = large_vec.Rotate(8);
    decrypted = rotated.Decrypt();
    ASSERT_EQ(decrypted.size(), large_data.size());
    for (int i = 0; i < 8; ++i) {
        ASSERT_NEAR(decrypted[i], large_data[i + 8], epsilon_);
        ASSERT_NEAR(decrypted[batch_size + i], large_data[batch_size + i + 8], epsilon_);
    }
}

TEST_F(CkksVectorTest, Multiply) {
    auto crypto_context = ckks_vector_.GetContext().GetCryptoContext();
    auto batch_size     = crypto_context->GetEncodingParams()->GetBatchSize();

    CkksVector vector_1(ckks_vector_.GetContext());
    CkksVector vector_2(ckks_vector_.GetContext());
    auto test_size = batch_size - 1024;

    std::vector<double> values_1(test_size, 1);
    vector_1.Encrypt(values_1);

    std::vector<double> values_2(test_size, 2);
    vector_2.Encrypt(values_2);

    CkksVector vector_3 = vector_1 * values_2;

    ASSERT_EQ(vector_3.size(), test_size);
    auto values_3 = vector_3.Decrypt();
    ASSERT_EQ(values_3.size(), test_size);
    for (unsigned i = 0; i < test_size; ++i) {
        ASSERT_NEAR(values_3[i], 2, epsilon_);
    }
}

TEST_F(CkksVectorTest, Subtract) {
    auto test_vector = ckks_vector_ - 1;
    SUCCEED();

    ckks_vector_.Encrypt(std::vector<double>(48842, 2));
    test_vector = ckks_vector_ - 1;
    SUCCEED();
}

TEST_F(CkksVectorTest, Addition) {
    auto test_vector = ckks_vector_ + ckks_vector_;
    auto result      = test_vector.Decrypt();
    ASSERT_NEAR(result[0], test_data_[0] * 2, epsilon_);

    test_vector += ckks_vector_;
    result = test_vector.Decrypt();
    ASSERT_NEAR(result[0], test_data_[0] * 3, epsilon_);

    result = ckks_vector_.Decrypt();
    ASSERT_NEAR(result[0], 2 * test_data_[0], epsilon_);
}

TEST_F(CkksVectorTest, AddMany) {
    CkksVector test_result = CkksVector::AddMany({ckks_vector_, ckks_vector_, ckks_vector_});
    auto decrypted         = test_result.Decrypt();
    ASSERT_NEAR(decrypted[0], test_data_[0] * 3, epsilon_);
    ASSERT_EQ(decrypted.size(), test_data_.size());
    ASSERT_EQ(ckks_vector_.size(), test_data_.size());
}

TEST_F(CkksVectorTest, Concat) {
    context_.GenerateRotateKeys(rotation_indices_);
    auto crypto_context = ckks_vector_.GetContext().GetCryptoContext();
    auto batch_size     = crypto_context->GetEncodingParams()->GetBatchSize();

    CkksVector test_vector(ckks_vector_.GetContext());
    CkksVector vector_1(ckks_vector_.GetContext());
    CkksVector vector_2(ckks_vector_.GetContext());

    vector_1.Encrypt(std::vector<double>(batch_size * 0.5, 1));
    vector_2.Encrypt(std::vector<double>(batch_size * 0.75, 2));

    test_vector.Concat(vector_1);
    auto decrypted_values = test_vector.Decrypt();
    ASSERT_EQ(test_vector.size(), batch_size * 0.5);
    ASSERT_NEAR(decrypted_values[0], 1, epsilon_);
    ASSERT_NEAR(decrypted_values[batch_size * 0.5 - 1], 1, epsilon_);
    ASSERT_NEAR(decrypted_values[batch_size * 0.5], 0, epsilon_);

    test_vector.Concat(vector_2);
    ASSERT_EQ(test_vector.GetData().size(), 2);

    ASSERT_EQ(test_vector.size(), batch_size * 1.25);
    decrypted_values = test_vector.Decrypt();
    ASSERT_EQ(decrypted_values.size(), test_vector.size());
    ASSERT_NEAR(decrypted_values[0], 1, epsilon_);
    ASSERT_NEAR(decrypted_values[batch_size * 0.5 - 1], 1, epsilon_);
    ASSERT_NEAR(decrypted_values[batch_size * 0.5], 2, epsilon_);
    ASSERT_NEAR(decrypted_values[batch_size], 2, epsilon_);
    ASSERT_NEAR(decrypted_values[batch_size * 1.25 - 1], 2, epsilon_);

    test_vector.Concat(vector_2);
    ASSERT_EQ(test_vector.size(), batch_size * 2.0);
    decrypted_values = test_vector.Decrypt();
    ASSERT_NEAR(decrypted_values[0], 1, epsilon_);
    ASSERT_NEAR(decrypted_values[batch_size * 0.5 - 1], 1, epsilon_);
    ASSERT_NEAR(decrypted_values[batch_size * 0.5], 2, epsilon_);
    ASSERT_NEAR(decrypted_values[batch_size], 2, epsilon_);
    ASSERT_NEAR(decrypted_values[batch_size * 1.25 - 1], 2, epsilon_);
    ASSERT_NEAR(decrypted_values[batch_size * 1.25], 2, epsilon_);
    ASSERT_NEAR(decrypted_values[batch_size * 2.0 - 1], 2, epsilon_);
}

TEST_F(CkksVectorTest, Condense) {
    //TODO (jkav77) Add test assertions
    ckks_vector_.Condense(test_data_.size());
}

//////////////////////////////////////////////////////////////////////////////
// File I/O

TEST_F(CkksVectorTest, Save) {
    ckks_vector_.Load(test_data_dir_ / "data.txt");
    ckks_vector_.Save(test_data_dir_ / "data_test.txt");

    ASSERT_TRUE(std::filesystem::exists(test_data_dir_ / "data_test.txt"));
    ASSERT_GT(std::filesystem::file_size(test_data_dir_ / "data_test.txt"), 0);
}

TEST_F(CkksVectorTest, Load) {
    ckks_vector_.Load(test_data_dir_ / "data.txt");

    ASSERT_GT(ckks_vector_.GetData().size(), 0);
    ASSERT_EQ(ckks_vector_.size(), test_data_.size());

    CkksVector tmp;
    tmp.Load(test_data_dir_ / "data.txt");
    auto keys = tmp.GetData()[0]->GetCryptoContext()->GetAllEvalMultKeys();
    ASSERT_GT(keys.size(), 0);
}
