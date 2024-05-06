#include <fhenom/ckks_vector.h>
#include <gtest/gtest.h>
#include <spdlog/spdlog.h>
#include <openfhe.h>

#include <filesystem>
#include <vector>

using fhenom::CkksVector;

class CkksVectorTest : public ::testing::Test {
protected:
    const std::vector<double> test_data_{0, 1, -1, 16, -16, 5, 50, 2, 10, 1, 2, 3, 4, 5, 17};
    const std::vector<double> testDomain_{0, 1, 2, 4, 8, 16, 32, 64, 96, 100};
    const std::filesystem::path test_data_dir_{"testData/ckks_vector"};
    const std::vector<int> rotation_indices_{1, 2, 4, 8, -1, -2, -4, -8};

    // lbcrypto::ScalingTechnique scTech = lbcrypto::FLEXIBLEAUTOEXT;
    const uint32_t multDepth_         = 2;
    const uint32_t scaleModSize_      = 26;
    const uint32_t firstModSize_      = 30;
    const uint32_t ringDim_           = 8192;
    const lbcrypto::SecurityLevel sl_ = lbcrypto::HEStd_128_classic;
    // uint32_t scaleModSize = 50;
    // uint32_t firstModSize = 60;
    // uint32_t batchSize = slots;

    double epsilon_ = 0.01;

    CkksVector ckks_vector_, precise_vector_;
    CkksVectorTest() {
        spdlog::set_level(spdlog::level::debug);
        fhenom::Context context;
        fhenom::Context precise_context;

        if (std::filesystem::exists(test_data_dir_)) {
            spdlog::debug("Saved test data found, loading...");
            context.Load("testData/ckks_vector");
            context.LoadRotationKeys(test_data_dir_ / "key-rotate.txt");
            context.LoadPublicKey("testData/ckks_vector/key-public.txt");
            context.LoadSecretKey("testData/ckks_vector/key-secret.txt");
            ckks_vector_.SetContext(context);
            ckks_vector_.Load(test_data_dir_ / "data.txt");

            precise_context.Load(test_data_dir_ / "precise");
            precise_context.LoadRotationKeys(test_data_dir_ / "precise" / "key-rotate.txt");
            precise_context.LoadPublicKey(test_data_dir_ / "precise" / "key-public.txt");
            precise_context.LoadSecretKey(test_data_dir_ / "precise" / "key-secret.txt");
            precise_vector_.SetContext(precise_context);
            precise_vector_.Load(test_data_dir_ / "precise" / "data.txt");
        }
        else {
            spdlog::debug("No saved test data found, generating new data...");
            lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS> ckks_parameters;
            ckks_parameters.SetMultiplicativeDepth(multDepth_);
            ckks_parameters.SetScalingModSize(scaleModSize_);
            ckks_parameters.SetFirstModSize(firstModSize_);
            // ckksParameters.SetScalingTechnique(scTech);
            ckks_parameters.SetSecurityLevel(sl_);
            ckks_parameters.SetRingDim(ringDim_);
            // ckksParameters.SetBatchSize(batchSize);
            ckks_parameters.SetSecretKeyDist(lbcrypto::UNIFORM_TERNARY);
            ckks_parameters.SetKeySwitchTechnique(lbcrypto::HYBRID);
            ckks_parameters.SetNumLargeDigits(2);
            context = fhenom::Context{ckks_parameters};

            context.GenerateKeys();
            context.GenerateRotateKeys(rotation_indices_);

            lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS> precise_params;
            precise_params.SetMultiplicativeDepth(23);
            precise_params.SetScalingModSize(50);
            precise_params.SetFirstModSize(60);
            precise_params.SetSecurityLevel(sl_);
            precise_params.SetRingDim(65536);
            precise_params.SetSecretKeyDist(lbcrypto::UNIFORM_TERNARY);
            precise_params.SetKeySwitchTechnique(lbcrypto::HYBRID);
            precise_params.SetNumLargeDigits(3);
            precise_context = fhenom::Context{precise_params};

            precise_context.GenerateKeys();
            precise_context.GenerateSumKey();
            precise_context.GenerateRotateKeys(rotation_indices_);

            std::filesystem::create_directories(test_data_dir_);
            context.Save(test_data_dir_);
            context.SaveRotationKeys(test_data_dir_ / "key-rotate.txt");
            context.SavePublicKey(test_data_dir_ / "key-public.txt");
            context.SaveSecretKey(test_data_dir_ / "key-secret.txt");

            ckks_vector_.SetContext(context);
            ckks_vector_.Encrypt(test_data_);
            ckks_vector_.Save(test_data_dir_ / "data.txt");

            std::filesystem::create_directories(test_data_dir_ / "precise");
            precise_context.Save(test_data_dir_ / "precise");
            precise_context.SaveRotationKeys(test_data_dir_ / "precise" / "key-rotate.txt");
            precise_context.SavePublicKey(test_data_dir_ / "precise" / "key-public.txt");
            precise_context.SaveSecretKey(test_data_dir_ / "precise" / "key-secret.txt");

            precise_vector_.SetContext(precise_context);
            precise_vector_.Encrypt(test_data_);
            precise_vector_.Save(test_data_dir_ / "precise" / "data.txt");
        }
    }
};

//////////////////////////////////////////////////////////////////////////////
// Encryption and Decryption

TEST_F(CkksVectorTest, Encrypt) {
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
    ASSERT_EQ(col2.GetData().size(), ceil(48842.0 / (static_cast<double>(ringDim_) / 2)));
}

TEST_F(CkksVectorTest, Decrypt) {
    ckks_vector_.Load(test_data_dir_ / "data.txt");
    auto decrypted = ckks_vector_.Decrypt();

    ASSERT_EQ(decrypted.size(), test_data_.size());
    for (unsigned i = 0; i < test_data_.size(); ++i) {
        ASSERT_NEAR(decrypted[i], test_data_[i], epsilon_);
    }
}

//////////////////////////////////////////////////////////////////////////////
// Homomorphic Operations

TEST_F(CkksVectorTest, GetSign) {
    precise_vector_ *= std::vector<double>(test_data_.size(), 1.0 / 50.0);
    auto result = precise_vector_.GetSign().Decrypt();

    ASSERT_EQ(result.size(), test_data_.size());
    for (unsigned i = 0; i < test_data_.size(); ++i) {
        if (test_data_[i] < 0) {
            ASSERT_NEAR(result[i], -1, 0.05);
        }
        else if (test_data_[i] > 0) {
            ASSERT_NEAR(result[i], 1, 0.05);
        }
        else {
            ASSERT_NEAR(result[i], 0, 0.05);
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

    auto result    = precise_vector_.GetSum();
    auto decrypted = result.Decrypt();
    ASSERT_EQ(decrypted.size(), 1);
    ASSERT_NEAR(decrypted[0], std::reduce(test_data_.begin(), test_data_.end()), epsilon_);

    precise_vector_.Encrypt(std::vector<double>(15, 1));
    result    = precise_vector_.GetSum();
    decrypted = result.Decrypt();
    ASSERT_NEAR(decrypted[0], 15, epsilon_);

    precise_vector_.Encrypt(std::vector<double>(ring_dim * 1.5, 0));
    result    = precise_vector_.GetSum();
    decrypted = result.Decrypt();
    ASSERT_NEAR(decrypted[0], 0, epsilon_);

    precise_vector_.Encrypt(std::vector<double>(ring_dim / 2, 1));
    result    = precise_vector_.GetSum();
    decrypted = result.Decrypt();
    ASSERT_NEAR(decrypted[0], ring_dim / 2, epsilon_);
}

TEST_F(CkksVectorTest, GetCount) {
    auto ring_dim = precise_vector_.GetContext().GetCryptoContext()->GetRingDimension();

    std::vector<double> test_data(ring_dim / 2, 1);
    precise_vector_.Encrypt(test_data);
    auto test = precise_vector_;
    test.Encrypt(test_data);
    auto result         = precise_vector_.IsEqual(1, 100);
    auto decrypted      = result.Decrypt();
    auto new_result     = result.GetSum();
    test                = test.GetSum();
    decrypted           = new_result.Decrypt();
    auto decrypted_test = test.Decrypt();
    ASSERT_NEAR(decrypted_test[0], ring_dim / 2.0, epsilon_);
    ASSERT_NEAR(decrypted[0], ring_dim / 2.0, epsilon_);

    test_data = std::vector<double>(ring_dim / 2, 1);
    precise_vector_.Encrypt(test_data);
    result    = precise_vector_.GetCount(1);
    decrypted = result.Decrypt();
    ASSERT_NEAR(decrypted[0], ring_dim / 2.0, epsilon_);

    test_data = std::vector<double>(ring_dim / 2, 0);
    precise_vector_.Encrypt(test_data);
    result    = precise_vector_.GetCount(1);
    decrypted = result.Decrypt();
    ASSERT_NEAR(decrypted[0], 0, epsilon_);

    std::vector<double> large_data(ring_dim * 1.5, 0);
    precise_vector_.Encrypt(large_data);
    result    = precise_vector_.GetCount(0);
    decrypted = result.Decrypt();
    ASSERT_NEAR(decrypted[0], ring_dim * 1.5, epsilon_);
}

TEST_F(CkksVectorTest, Rotate) {
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

    std::vector<double> large_data(ringDim_ / 2);
    for (unsigned i = 0; i < ringDim_ / 2; ++i) {
        large_data[i] = i;
    }
    CkksVector large_vec;
    large_vec.SetContext(ckks_vector_.GetContext());
    large_vec.Encrypt(large_data);
    rotated   = large_vec.Rotate(-8);
    decrypted = rotated.Decrypt();
    ASSERT_EQ(decrypted.size(), large_data.size());
    for (int i = 0; i < 8; ++i) {
        ASSERT_NEAR(decrypted[i], large_data[large_data.size() - 8 + i], epsilon_);
    }
    ASSERT_NEAR(decrypted[8], large_data[0], epsilon_);
    ASSERT_NEAR(decrypted[9], large_data[1], epsilon_);

    large_data.clear();
    for (unsigned i = 0; i < ringDim_; ++i) {
        large_data.push_back(i);
    }
    large_vec.Encrypt(large_data);
    rotated   = large_vec.Rotate(8);
    decrypted = rotated.Decrypt();
    ASSERT_EQ(decrypted.size(), large_data.size());
    for (int i = 0; i < 8; ++i) {
        ASSERT_NEAR(decrypted[i], large_data[i + 8], epsilon_);
        ASSERT_NEAR(decrypted[ringDim_ / 2 + i], large_data[ringDim_ / 2 + i + 8], epsilon_);
    }
}

TEST_F(CkksVectorTest, FastRotate) {
    // TODO(jkav77): This isn't testing anything...
    auto crypto_context = ckks_vector_.GetContext().GetCryptoContext();

    auto key_map = crypto_context->GetEvalAutomorphismKeyMap(ckks_vector_.GetData()[0]->GetKeyTag());
    for (const auto& rot_idx : {-1, 1, 8, -8}) {
        auto am_idx = lbcrypto::FindAutomorphismIndex2n(rot_idx, crypto_context->GetCyclotomicOrder());
        ASSERT_EQ(key_map.count(am_idx), 1);
    }

    auto idx = lbcrypto::FindAutomorphismIndex2n(7, crypto_context->GetCyclotomicOrder());
    ASSERT_EQ(key_map.count(idx), 0);
}

TEST_F(CkksVectorTest, Multiply) {
    auto crypto_context = ckks_vector_.GetContext().GetCryptoContext();

    CkksVector vector_1(ckks_vector_.GetContext());
    CkksVector vector_2(ckks_vector_.GetContext());
    auto test_size = ringDim_ - 1024;

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

TEST_F(CkksVectorTest, Concat) {
    auto crypto_context = ckks_vector_.GetContext().GetCryptoContext();

    CkksVector vector_1(ckks_vector_.GetContext());
    CkksVector vector_2(ckks_vector_.GetContext());
    auto test_size = ringDim_;

    std::vector<double> values_1(test_size, 1);
    vector_1.Encrypt(values_1);

    std::vector<double> values_2(test_size, 2);
    vector_2.Encrypt(values_2);

    vector_1.Concat(vector_2);

    ASSERT_EQ(vector_1.size(), test_size * 2);
    auto decrypted_values = vector_1.Decrypt();
    ASSERT_EQ(decrypted_values.size(), test_size * 2);
    for (unsigned i = 0; i < test_size; ++i) {
        ASSERT_NEAR(decrypted_values[i], 1, epsilon_);
        ASSERT_NEAR(decrypted_values[i + test_size], 2, epsilon_);
    }

    CkksVector vector_3(ckks_vector_.GetContext());
    std::vector<double> values_3(values_1.begin(), values_1.end() - 1024);
    vector_3.Encrypt(values_3);
    ASSERT_EQ(vector_3.size(), test_size - 1024);

    vector_3.Concat(vector_2);
    ASSERT_EQ(vector_3.size(), test_size * 2);
    decrypted_values = vector_3.Decrypt();
    ASSERT_EQ(decrypted_values.size(), test_size * 2);
    for (unsigned i = 0; i < test_size - 1024; ++i) {
        ASSERT_NEAR(decrypted_values[i], 1, epsilon_);
        ASSERT_NEAR(decrypted_values[i + test_size], 2, epsilon_);
    }

    for (unsigned i = test_size - 1024; i < test_size; ++i) {
        ASSERT_NEAR(decrypted_values[i], 0, epsilon_);
        ASSERT_NEAR(decrypted_values[i + test_size], 2, epsilon_);
    }
}

TEST_F(CkksVectorTest, Merge) {
    const size_t num_vectors = 10;
    std::vector<CkksVector> vectors(num_vectors);
    auto context = ckks_vector_.GetContext();
    for (auto i = 0; i < 10; ++i) {
        vectors[i] = CkksVector(context);
        vectors[i].Encrypt(test_data_);
    }

    auto result = CkksVector::Merge(vectors);

    ASSERT_EQ(result.size(), num_vectors);

    auto decrypted = result.Decrypt();
    ASSERT_EQ(decrypted.size(), num_vectors);

    for (const auto& element : decrypted) {
        ASSERT_NEAR(element, test_data_[0], epsilon_);
    }
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
