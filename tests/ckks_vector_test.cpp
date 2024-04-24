#include <fhenom/ckks_vector.h>
#include <gtest/gtest.h>
#include <spdlog/spdlog.h>

#include <filesystem>
#include <vector>

using fhenom::CkksVector;

class CkksVectorTest : public ::testing::Test {
protected:
    const std::vector<double> test_data_{0, 1, -1, 16, -16, 5, -100, 50, 100, 2, 10, 1, 2, 3, 4, 5, 17};
    const std::vector<double> testDomain_{0, 1, 2, 4, 8, 16, 32, 64, 96, 100};
    const std::filesystem::path test_data_dir_{"testData/ckks_vector"};

    // lbcrypto::ScalingTechnique scTech = lbcrypto::FLEXIBLEAUTOEXT;
    const uint32_t multDepth_         = 2;
    const uint32_t scaleModSize_      = 24;
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
            context.load("testData/ckks_vector");
            context.loadRotationKeys(test_data_dir_ / "key-rotate.txt");
            context.loadPublicKey("testData/ckks_vector/key-public.txt");
            context.loadSecretKey("testData/ckks_vector/key-secret.txt");
            ckks_vector_.setContext(context);
            ckks_vector_.load(test_data_dir_ / "data.txt");

            precise_context.load(test_data_dir_ / "precise");
            precise_context.loadRotationKeys(test_data_dir_ / "precise" / "key-rotate.txt");
            precise_context.loadPublicKey(test_data_dir_ / "precise" / "key-public.txt");
            precise_context.loadSecretKey(test_data_dir_ / "precise" / "key-secret.txt");
            precise_vector_.setContext(precise_context);
            precise_vector_.load(test_data_dir_ / "precise" / "data.txt");
        }
        else {
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

            context.generateKeys();
            context.generateRotateKeys({-1, 1, 8, -8});

            lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS> precise_params;
            precise_params.SetMultiplicativeDepth(24);
            precise_params.SetScalingModSize(40);
            precise_params.SetFirstModSize(50);
            precise_params.SetSecurityLevel(sl_);
            precise_params.SetRingDim(65536);
            precise_params.SetSecretKeyDist(lbcrypto::UNIFORM_TERNARY);
            precise_params.SetKeySwitchTechnique(lbcrypto::HYBRID);
            precise_params.SetNumLargeDigits(3);
            precise_context = fhenom::Context{precise_params};

            precise_context.generateKeys();
            precise_context.generateSumKey();
            precise_context.generateRotateKeys({-1, 1, 8, -8});

            std::filesystem::create_directories(test_data_dir_);
            context.save(test_data_dir_);
            context.saveRotationKeys(test_data_dir_ / "key-rotate.txt");
            context.savePublicKey(test_data_dir_ / "key-public.txt");
            context.saveSecretKey(test_data_dir_ / "key-secret.txt");

            ckks_vector_.setContext(context);
            ckks_vector_.encrypt(test_data_);
            ckks_vector_.save(test_data_dir_ / "data.txt");

            std::filesystem::create_directories(test_data_dir_ / "precise");
            precise_context.save(test_data_dir_ / "precise");
            precise_context.saveRotationKeys(test_data_dir_ / "precise" / "key-rotate.txt");
            precise_context.savePublicKey(test_data_dir_ / "precise" / "key-public.txt");
            precise_context.saveSecretKey(test_data_dir_ / "precise" / "key-secret.txt");

            precise_vector_.setContext(precise_context);
            precise_vector_.encrypt(test_data_);
            precise_vector_.save(test_data_dir_ / "precise" / "data.txt");
        }
    }
};

//////////////////////////////////////////////////////////////////////////////
// Encryption and Decryption

TEST_F(CkksVectorTest, encrypt) {
    CkksVector col1{};
    col1.setContext(ckks_vector_.getContext());
    col1.encrypt(test_data_);

    ASSERT_GT(ckks_vector_.size(), 0);
    ASSERT_GT(ckks_vector_.getData().size(), 0);

    std::vector<double> large_data(48842, 1);
    CkksVector col2{};
    col2.setContext(ckks_vector_.getContext());
    col2.encrypt(large_data);
    ASSERT_EQ(col2.size(), 48842);
    ASSERT_EQ(col2.getData().size(), ceil(48842.0 / (static_cast<double>(ringDim_) / 2)));
}

TEST_F(CkksVectorTest, decrypt) {
    ckks_vector_.load(test_data_dir_ / "data.txt");
    auto decrypted = ckks_vector_.decrypt();

    ASSERT_EQ(decrypted.size(), test_data_.size());
    for (unsigned i = 0; i < test_data_.size(); ++i) {
        ASSERT_NEAR(decrypted[i], test_data_[i], epsilon_);
    }
}

//////////////////////////////////////////////////////////////////////////////
// Homomorphic Operations

TEST_F(CkksVectorTest, Sign) {
    auto result    = precise_vector_.Sign();
    auto decrypted = result.decrypt();

    ASSERT_EQ(decrypted.size(), test_data_.size());
    for (unsigned i = 0; i < test_data_.size(); ++i) {
        if (test_data_[i] < 0) {
            ASSERT_NEAR(decrypted[i], -1, 0.05);
        }
        else if (test_data_[i] > 0) {
            ASSERT_NEAR(decrypted[i], 1, 0.05);
        }
        else {
            ASSERT_NEAR(decrypted[i], 0, 0.05);
        }
    }
}

TEST_F(CkksVectorTest, IsEqual) {
    auto epsilon = 0.05;
    precise_vector_.encrypt(testDomain_);
    auto result    = precise_vector_.IsEqual(1);
    auto decrypted = result.decrypt();
    ASSERT_EQ(decrypted.size(), testDomain_.size());
    for (unsigned i = 0; i < testDomain_.size(); ++i) {
        ASSERT_NEAR(decrypted[i], testDomain_[i] == 1, epsilon) << "Index: " << i;
    }

    result    = precise_vector_.IsEqual(4);
    decrypted = result.decrypt();
    ASSERT_EQ(decrypted.size(), testDomain_.size());
    for (unsigned i = 0; i < testDomain_.size(); ++i) {
        ASSERT_NEAR(decrypted[i], testDomain_[i] == 4, epsilon);
    }

    result    = precise_vector_.IsEqual(17);
    decrypted = result.decrypt();
    ASSERT_EQ(decrypted.size(), testDomain_.size());
    for (unsigned i = 0; i < testDomain_.size(); ++i) {
        ASSERT_NEAR(decrypted[i], testDomain_[i] == 17, epsilon);
    }

    result    = precise_vector_.IsEqual(0);
    decrypted = result.decrypt();
    ASSERT_EQ(decrypted.size(), testDomain_.size());
    for (unsigned i = 0; i < testDomain_.size(); ++i) {
        ASSERT_NEAR(decrypted[i], testDomain_[i] == 0, epsilon);
    }
}

TEST_F(CkksVectorTest, Sum) {
    auto result    = precise_vector_.Sum();
    auto decrypted = result.decrypt();
    ASSERT_EQ(decrypted.size(), 1);
    ASSERT_NEAR(decrypted[0], std::reduce(test_data_.begin(), test_data_.end()), epsilon_);
}

TEST_F(CkksVectorTest, rotate) {
    auto rotated   = ckks_vector_.rotate(1);
    auto decrypted = rotated.decrypt();

    ASSERT_EQ(decrypted.size(), test_data_.size());
    for (unsigned i = 0; i < test_data_.size(); ++i) {
        ASSERT_NEAR(decrypted[i], test_data_[(i + 1) % 17], epsilon_);
    }

    rotated   = ckks_vector_.rotate(-1);
    decrypted = rotated.decrypt();

    ASSERT_EQ(decrypted.size(), test_data_.size());
    for (unsigned i = 0; i < test_data_.size(); ++i) {
        ASSERT_NEAR(decrypted[i], test_data_[(i - 1) % 17], epsilon_);
    }

    rotated   = ckks_vector_.rotate(8);
    decrypted = rotated.decrypt();
    ASSERT_EQ(decrypted.size(), test_data_.size());
    for (unsigned i = 0; i < test_data_.size() - 8; ++i) {
        ASSERT_NEAR(decrypted[i], test_data_[(i + 8) % 17], epsilon_);
    }

    std::vector<double> large_data(ringDim_ / 2);
    for (unsigned i = 0; i < ringDim_ / 2; ++i) {
        large_data[i] = i;
    }
    CkksVector large_vec;
    large_vec.setContext(ckks_vector_.getContext());
    large_vec.encrypt(large_data);
    rotated   = large_vec.rotate(-8);
    decrypted = rotated.decrypt();
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
    large_vec.encrypt(large_data);
    rotated   = large_vec.rotate(8);
    decrypted = rotated.decrypt();
    ASSERT_EQ(decrypted.size(), large_data.size());
    for (int i = 0; i < 8; ++i) {
        ASSERT_NEAR(decrypted[i], large_data[i + 8], epsilon_);
        ASSERT_NEAR(decrypted[ringDim_ / 2 + i], large_data[ringDim_ / 2 + i + 8], epsilon_);
    }
}

TEST_F(CkksVectorTest, fastRotate) {
    auto crypto_context = ckks_vector_.getContext().getCryptoContext();

    auto key_map = crypto_context->GetEvalAutomorphismKeyMap(ckks_vector_.getData()[0]->GetKeyTag());
    for (const auto& rot_idx : {-1, 1, 8, -8}) {
        auto am_idx = lbcrypto::FindAutomorphismIndex2n(rot_idx, crypto_context->GetCyclotomicOrder());
        ASSERT_EQ(key_map.count(am_idx), 1);
    }

    auto idx = lbcrypto::FindAutomorphismIndex2n(7, crypto_context->GetCyclotomicOrder());
    ASSERT_EQ(key_map.count(idx), 0);
}

TEST_F(CkksVectorTest, multiply) {
    auto crypto_context = ckks_vector_.getContext().getCryptoContext();

    CkksVector vector_1(ckks_vector_.getContext());
    CkksVector vector_2(ckks_vector_.getContext());
    auto test_size = ringDim_ - 1024;

    std::vector<double> values_1(test_size, 1);
    vector_1.encrypt(values_1);

    std::vector<double> values_2(test_size, 2);
    vector_2.encrypt(values_2);

    CkksVector vector_3 = vector_1 * values_2;

    ASSERT_EQ(vector_3.size(), test_size);
    auto values_3 = vector_3.decrypt();
    ASSERT_EQ(values_3.size(), test_size);
    for (unsigned i = 0; i < test_size; ++i) {
        ASSERT_NEAR(values_3[i], 2, epsilon_);
    }
}

TEST_F(CkksVectorTest, Concat) {
    auto crypto_context = ckks_vector_.getContext().getCryptoContext();

    CkksVector vector_1(ckks_vector_.getContext());
    CkksVector vector_2(ckks_vector_.getContext());
    auto test_size = ringDim_;

    std::vector<double> values_1(test_size, 1);
    vector_1.encrypt(values_1);

    std::vector<double> values_2(test_size, 2);
    vector_2.encrypt(values_2);

    vector_1.Concat(vector_2);

    ASSERT_EQ(vector_1.size(), test_size * 2);
    auto decrypted_values = vector_1.decrypt();
    ASSERT_EQ(decrypted_values.size(), test_size * 2);
    for (unsigned i = 0; i < test_size; ++i) {
        ASSERT_NEAR(decrypted_values[i], 1, epsilon_);
        ASSERT_NEAR(decrypted_values[i + test_size], 2, epsilon_);
    }

    CkksVector vector_3(ckks_vector_.getContext());
    std::vector<double> values_3(values_1.begin(), values_1.end() - 1024);
    vector_3.encrypt(values_3);
    ASSERT_EQ(vector_3.size(), test_size - 1024);

    vector_3.Concat(vector_2);
    ASSERT_EQ(vector_3.size(), test_size * 2);
    decrypted_values = vector_3.decrypt();
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

//////////////////////////////////////////////////////////////////////////////
// File I/O

TEST_F(CkksVectorTest, save) {
    ckks_vector_.load(test_data_dir_ / "data.txt");
    ckks_vector_.save(test_data_dir_ / "data_test.txt");

    ASSERT_TRUE(std::filesystem::exists(test_data_dir_ / "data_test.txt"));
    ASSERT_GT(std::filesystem::file_size(test_data_dir_ / "data_test.txt"), 0);
}

TEST_F(CkksVectorTest, load) {
    ckks_vector_.load(test_data_dir_ / "data.txt");

    ASSERT_GT(ckks_vector_.getData().size(), 0);
    ASSERT_EQ(ckks_vector_.size(), test_data_.size());

    CkksVector tmp;
    tmp.load(test_data_dir_ / "data.txt");
    auto keys = tmp.getData()[0]->GetCryptoContext()->GetAllEvalMultKeys();
    ASSERT_GT(keys.size(), 0);
}
