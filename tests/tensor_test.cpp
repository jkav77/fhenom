#include <spdlog/spdlog.h>
#include <gtest/gtest.h>
#include <fhenom/tensor.h>

class TensorTest : public ::testing::Test {
protected:
    fhenom::Tensor tensor_;
    const std::vector<double> test_data_{1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                                         1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                                         1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    const fhenom::shape_t test_data_shape_{2, 3, 3, 3};

    TensorTest() {
        spdlog::set_level(spdlog::level::debug);
        tensor_ = fhenom::Tensor{test_data_, test_data_shape_};
    }
};

TEST_F(TensorTest, Get) {
    EXPECT_EQ(tensor_.Get({0, 0, 0, 0}), 1);
    EXPECT_EQ(tensor_.Get({1, 0, 0, 0}), 1);
    EXPECT_EQ(tensor_.Get({0, 1, 0, 0}), 1);
    EXPECT_EQ(tensor_.Get({1, 1, 0, 0}), 1);

    EXPECT_EQ(tensor_.Get({0, 0, 1, 0}), 4);
    EXPECT_EQ(tensor_.Get({0, 0, 0, 1}), 2);
    EXPECT_EQ(tensor_.Get({1, 1, 1, 1}), 5);
    EXPECT_EQ(tensor_.Get({1, 1, 0, 1}), 2);
}