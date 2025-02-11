//
// Created by Fabian Stiegler on 13.01.2025.
//

#include "googlemock/include/gmock/gmock-matchers.h"
#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "spicenet_lib/MatrixIterator.h"
#include "spicenet_lib/Matrix.h"

TEST(MatrixIteratorTestSuite, TestWithSetOverMatrix) {
    // Arrange
    float_t data[] = {0.0, 1.0, 2.0, 3.0};
    Matrix<float_t> matrix({2, 2}, 0.0);
    matrix.setField({0, 0}, 0.0);
    matrix.setField({0, 1}, 1.0);
    matrix.setField({1, 0}, 2.0);
    matrix.setField({1, 1}, 3.0);
    MatrixIterator<float_t> iterator = matrix.getIterator();
    std::vector<uint16_t> index(2);
    float_t *pointer;
    bool result = false;

    // Act
    // Assert
    result = iterator.next(index, pointer);
    EXPECT_EQ(result, true);
    ASSERT_THAT(index, ::testing::ElementsAre(0, 0));
    EXPECT_EQ(0.0, *pointer);

    result = iterator.next(index, pointer);
    EXPECT_EQ(result, true);
    ASSERT_THAT(index, ::testing::ElementsAre(0, 1));
    EXPECT_EQ(1.0, *pointer);

    result = iterator.next(index, pointer);
    EXPECT_EQ(result, true);
    ASSERT_THAT(index, ::testing::ElementsAre(1, 0));
    EXPECT_EQ(2.0, *pointer);

    result = iterator.next(index, pointer);
    EXPECT_EQ(result, true);
    ASSERT_THAT(index, ::testing::ElementsAre(1, 1));
    EXPECT_EQ(3.0, *pointer);

    result = iterator.next(index, pointer);
    EXPECT_EQ(result, false);
}

TEST(MatrixIteratorTestSuite, Test2x2) {
    // Arrange
    float_t data[] = {0.0, 1.0, 2.0, 3.0};
    Matrix<float_t> matrix({2, 2}, data);
    MatrixIterator<float_t> iterator = matrix.getIterator();
    std::vector<uint16_t> index(2);
    float_t *pointer;
    bool result = false;

    // Act
    // Assert
    result = iterator.next(index, pointer);
    EXPECT_EQ(result, true);
    ASSERT_THAT(index, ::testing::ElementsAre(0, 0));
    EXPECT_EQ(0.0, *pointer);

    result = iterator.next(index, pointer);
    EXPECT_EQ(result, true);
    ASSERT_THAT(index, ::testing::ElementsAre(0, 1));
    EXPECT_EQ(1.0, *pointer);

    result = iterator.next(index, pointer);
    EXPECT_EQ(result, true);
    ASSERT_THAT(index, ::testing::ElementsAre(1, 0));
    EXPECT_EQ(2.0, *pointer);

    result = iterator.next(index, pointer);
    EXPECT_EQ(result, true);
    ASSERT_THAT(index, ::testing::ElementsAre(1, 1));
    EXPECT_EQ(3.0, *pointer);

    result = iterator.next(index, pointer);
    EXPECT_EQ(result, false);
}

TEST(MatrixIteratorTestSuite, Test3x2) {
    // Arrange
    float_t data[] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
    Matrix<float_t> matrix({3, 2}, data);
    MatrixIterator<float_t> iterator = matrix.getIterator();
    std::vector<uint16_t> index(2);
    float_t *pointer;
    bool result = false;

    // Act
    // Assert
    result = iterator.next(index, pointer);
    EXPECT_EQ(result, true);
    ASSERT_THAT(index, ::testing::ElementsAre(0, 0));
    EXPECT_EQ(0.0, *pointer);

    result = iterator.next(index, pointer);
    EXPECT_EQ(result, true);
    ASSERT_THAT(index, ::testing::ElementsAre(0, 1));
    EXPECT_EQ(1.0, *pointer);

    result = iterator.next(index, pointer);
    EXPECT_EQ(result, true);
    ASSERT_THAT(index, ::testing::ElementsAre(1, 0));
    EXPECT_EQ(2.0, *pointer);

    result = iterator.next(index, pointer);
    EXPECT_EQ(result, true);
    ASSERT_THAT(index, ::testing::ElementsAre(1, 1));
    EXPECT_EQ(3.0, *pointer);

    result = iterator.next(index, pointer);
    EXPECT_EQ(result, true);
    ASSERT_THAT(index, ::testing::ElementsAre(2, 0));
    EXPECT_EQ(4.0, *pointer);

    result = iterator.next(index, pointer);
    EXPECT_EQ(result, true);
    ASSERT_THAT(index, ::testing::ElementsAre(2, 1));
    EXPECT_EQ(5.0, *pointer);

    result = iterator.next(index, pointer);
    EXPECT_EQ(result, false);
}

TEST(MatrixIteratorTestSuite, Test3x3x3){
    // Arrange
    Matrix<float> matrix({3,3,3});
    // z = 0
    matrix.setField({0, 0, 0}, 1);
    matrix.setField({1, 0, 0}, 1);
    matrix.setField({2, 0, 0}, 1);

    matrix.setField({0, 1, 0}, 2);
    matrix.setField({1, 1, 0}, 5);
    matrix.setField({2, 1, 0}, 4);

    matrix.setField({0, 2, 0}, 3);
    matrix.setField({1, 2, 0}, 2);
    matrix.setField({2, 2, 0}, 1);

    // z = 1
    matrix.setField({0, 0, 1}, 6);
    matrix.setField({1, 0, 1}, 7);
    matrix.setField({2, 0, 1}, 8);

    matrix.setField({0, 1, 1}, 4);
    matrix.setField({1, 1, 1}, 8);
    matrix.setField({2, 1, 1}, 2);

    matrix.setField({0, 2, 1}, 9);
    matrix.setField({1, 2, 1}, 3);
    matrix.setField({2, 2, 1}, 1);

    // z = 2
    matrix.setField({0, 0, 2}, 1);
    matrix.setField({1, 0, 2}, 2);
    matrix.setField({2, 0, 2}, 2);

    matrix.setField({0, 1, 2}, 2);
    matrix.setField({1, 1, 2}, 1);
    matrix.setField({2, 1, 2}, 2);

    matrix.setField({0, 2, 2}, 2);
    matrix.setField({1, 2, 2}, 2);
    matrix.setField({2, 2, 2}, 1);

    auto iterator = matrix.getIterator();
    std::vector<uint16_t> index(2);
    float *pointer;
    bool result = false;

    // Act
    // Assert
    result = iterator.next(index, pointer);
    EXPECT_EQ(result, true);
    EXPECT_THAT(index, ::testing::ElementsAre(0, 0, 0));
    EXPECT_FLOAT_EQ(1.0, *pointer);

    result = iterator.next(index, pointer);
    EXPECT_EQ(result, true);
    EXPECT_THAT(index, ::testing::ElementsAre(0, 0, 1));
    EXPECT_FLOAT_EQ(6.0, *pointer);

    result = iterator.next(index, pointer);
    EXPECT_EQ(result, true);
    EXPECT_THAT(index, ::testing::ElementsAre(0, 0, 2));
    EXPECT_FLOAT_EQ(1.0, *pointer);

    result = iterator.next(index, pointer);
    EXPECT_EQ(result, true);
    EXPECT_THAT(index, ::testing::ElementsAre(0, 1, 0));
    EXPECT_FLOAT_EQ(2.0, *pointer);

    result = iterator.next(index, pointer);
    EXPECT_EQ(result, true);
    EXPECT_THAT(index, ::testing::ElementsAre(0, 1, 1));
    EXPECT_FLOAT_EQ(4.0, *pointer);

    result = iterator.next(index, pointer);
    EXPECT_EQ(result, true);
    EXPECT_THAT(index, ::testing::ElementsAre(0, 1, 2));
    EXPECT_FLOAT_EQ(2.0, *pointer);

    result = iterator.next(index, pointer);
    EXPECT_EQ(result, true);
    EXPECT_THAT(index, ::testing::ElementsAre(0, 2, 0));
    EXPECT_FLOAT_EQ(3.0, *pointer);

    result = iterator.next(index, pointer);
    EXPECT_EQ(result, true);
    EXPECT_THAT(index, ::testing::ElementsAre(0, 2, 1));
    EXPECT_FLOAT_EQ(9.0, *pointer);

    result = iterator.next(index, pointer);
    EXPECT_EQ(result, true);
    EXPECT_THAT(index, ::testing::ElementsAre(0, 2, 2));
    EXPECT_FLOAT_EQ(2.0, *pointer);

    result = iterator.next(index, pointer);
    EXPECT_EQ(result, true);
    EXPECT_THAT(index, ::testing::ElementsAre(1, 0, 0));
    EXPECT_FLOAT_EQ(1.0, *pointer);

    result = iterator.next(index, pointer);
    EXPECT_EQ(result, true);
    EXPECT_THAT(index, ::testing::ElementsAre(1, 0, 1));
    EXPECT_FLOAT_EQ(7.0, *pointer);

    result = iterator.next(index, pointer);
    EXPECT_EQ(result, true);
    EXPECT_THAT(index, ::testing::ElementsAre(1, 0, 2));
    EXPECT_FLOAT_EQ(2.0, *pointer);

    result = iterator.next(index, pointer);
    EXPECT_EQ(result, true);
    EXPECT_THAT(index, ::testing::ElementsAre(1, 1, 0));
    EXPECT_FLOAT_EQ(5.0, *pointer);

    result = iterator.next(index, pointer);
    EXPECT_EQ(result, true);
    EXPECT_THAT(index, ::testing::ElementsAre(1, 1, 1));
    EXPECT_FLOAT_EQ(8.0, *pointer);

    result = iterator.next(index, pointer);
    EXPECT_EQ(result, true);
    EXPECT_THAT(index, ::testing::ElementsAre(1, 1, 2));
    EXPECT_FLOAT_EQ(1.0, *pointer);

    result = iterator.next(index, pointer);
    EXPECT_EQ(result, true);
    EXPECT_THAT(index, ::testing::ElementsAre(1, 2, 0));
    EXPECT_FLOAT_EQ(2.0, *pointer);

    result = iterator.next(index, pointer);
    //EXPECT_EQ(result, false);
}