//
// Created by Fabian Stiegler on 13.01.2025.
//

#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "spicenet_lib/Matrix.h"


TEST(MatrixTestSuite, Equal1_1) {
    // Arrange
    float_t data[] = {0.0, 1.0, 2.0, 3.0};
    Matrix<float_t> matrix(std::vector<uint16_t>{2, 2}, data);

    // Act
    auto result = matrix.getField({0, 0});

    // Assert
    EXPECT_EQ(0.0, result);
}

TEST(MatrixTestSuite, Equal2_2) {
    // Arrange
    float_t data[] = {0.0, 1.0, 2.0, 3.0};
    Matrix<float_t> matrix(std::vector<uint16_t>{2, 2}, data);

    // Act
    auto result = matrix.getField({1, 1});

    // Assert
    EXPECT_EQ(3.0, result);
}

TEST(MatrixTestSuite, Equal2_1) {
    // Arrange
    float_t data[] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
    Matrix<float_t> matrix(std::vector<uint16_t>{3, 2}, data);

    // Act
    auto result = matrix.getField({1, 0});

    // Assert
    EXPECT_EQ(2.0, result);
}

TEST(MatrixTestSuite, SetValue) {
    // Arrange
    float_t data[] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
    Matrix<float_t> matrix(std::vector<uint16_t>{3, 2}, data);

    // Act
    matrix.setField({0, 0}, 9.0);
    auto result = matrix.getField({0, 0});

    // Assert
    EXPECT_EQ(9.0, result);
}

TEST(MatrixTestSuite, TestToBigSize) {
    // Arrange
    // Act
    // Assert
    EXPECT_THAT([&]() {
        Matrix<float_t> matrix(
                std::vector<uint16_t>{USHRT_MAX, USHRT_MAX, USHRT_MAX, USHRT_MAX});
    },
                testing::Throws<std::invalid_argument>(testing::Property(&std::invalid_argument::what,
                                                                         testing::HasSubstr(
                                                                                 "Matrix size overflow, the shape is to large! Shape: (65535, 2)"))));
}

TEST(MatrixTestSuite, InitValue) {
    // Arrange

    // Act
    Matrix<float_t> matrix(std::vector<uint16_t>{3, 2}, 1, nullptr);

    // Assert
    EXPECT_EQ(matrix.getField({0, 0}), 1.0);
    EXPECT_EQ(matrix.getField({1, 0}), 1.0);
    EXPECT_EQ(matrix.getField({2, 0}), 1.0);
    EXPECT_EQ(matrix.getField({0, 1}), 1.0);
    EXPECT_EQ(matrix.getField({1, 1}), 1.0);
    EXPECT_EQ(matrix.getField({2, 1}), 1.0);
}