//
// Created by Fabian Stiegler on 02.01.2025.
//

#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "spicenet_lib/SpicenetHcm.h"

static uint64_t trainingIterationTestCounter = 0;

TEST(SpicenetHcmTestSuite, FitMallformedTrainingData_NotEnoughMappings) {
    // Arrange
    auto weightsLrf = [](uint64_t x) -> float {
        trainingIterationTestCounter = x;
        return 0.7;
    };
    auto trustOfNewLrf = [](uint64_t x) -> float { return 0.3; };

    std::list<std::list<std::vector<float_t>>> data;
    std::list<std::vector<float_t>> a1 = {{0.0}};
    data.push_back(a1);
    auto hcm = SpicenetHcm<float_t>({1, 1}, weightsLrf, trustOfNewLrf);

    // Act
    // Assert
    EXPECT_THAT([&]() { hcm.fit(data, 0); },
                testing::Throws<std::invalid_argument>(testing::Property(&std::invalid_argument::what,
                                                                         testing::HasSubstr(
                                                                                 "SpicenetHcm fit: the amount of activation lists is unequal to the amount of soms"))));
}

TEST(SpicenetHcmTestSuite, FitMallformedTrainingData_ToManyMappings) {
    // Arrange
    auto weightsLrf = [](uint64_t x) -> float {
        trainingIterationTestCounter = x;
        return 0.7;
    };
    auto trustOfNewLrf = [](uint64_t x) -> float { return 0.3; };

    std::list<std::list<std::vector<float_t>>> data;
    std::list<std::vector<float_t>> a1 = {{0.0}};
    std::list<std::vector<float_t>> a2 = {{1.0}};
    std::list<std::vector<float_t>> a3 = {{2.0}};
    data.push_back(a1);
    data.push_back(a2);
    data.push_back(a3);
    auto hcm = SpicenetHcm<float_t>({1, 1}, weightsLrf, trustOfNewLrf);

    // Act
    // Assert
    EXPECT_THAT([&]() { hcm.fit(data, 0); },
                testing::Throws<std::invalid_argument>(testing::Property(&std::invalid_argument::what,
                                                                         testing::HasSubstr(
                                                                                 "SpicenetHcm fit: the amount of activation lists is unequal to the amount of soms"))));
}

TEST(SpicenetHcmTestSuite, FitMallformedTrainingData_UnequalActivationAmount) {
    // Arrange
    auto weightsLrf = [](uint64_t x) -> float {
        trainingIterationTestCounter = x;
        return 0.7;
    };
    auto trustOfNewLrf = [](uint64_t x) -> float { return 0.3; };

    std::list<std::list<std::vector<float_t>>> data;
    std::list<std::vector<float_t>> a1 = {{0.0}};
    std::list<std::vector<float_t>> a2 = {{1.0},
                                          {2.0}};
    std::list<std::vector<float_t>> a3 = {{2.0}};
    data.push_back(a1);
    data.push_back(a2);
    data.push_back(a3);
    auto hcm = SpicenetHcm<float_t>({1, 1, 1}, weightsLrf, trustOfNewLrf);

    // Act
    // Assert
    EXPECT_THAT([&]() { hcm.fit(data, 0); },
                testing::Throws<std::invalid_argument>(testing::Property(&std::invalid_argument::what,
                                                                         testing::HasSubstr(
                                                                                 "SpicenetHcm fit: not all lists contain an equal amount of activations"))));
}

TEST(SpicenetHcmTestSuite, FitMallformedTrainingData_UnequalActivationAmount2) {
    // Arrange
    auto weightsLrf = [](uint64_t x) -> float {
        trainingIterationTestCounter = x;
        return 0.7;
    };
    auto trustOfNewLrf = [](uint64_t x) -> float { return 0.3; };

    std::list<std::list<std::vector<float_t>>> data;
    std::list<std::vector<float_t>> a1 = {{0.0}};
    std::list<std::vector<float_t>> a2 = {{1.0}};
    std::list<std::vector<float_t>> a3 = {{2.0},
                                          {2.0}};
    data.push_back(a1);
    data.push_back(a2);
    data.push_back(a3);
    auto hcm = SpicenetHcm<float_t>({1, 1, 1}, weightsLrf, trustOfNewLrf);

    // Act
    // Assert
    EXPECT_THAT([&]() { hcm.fit(data, 0); },
                testing::Throws<std::invalid_argument>(testing::Property(&std::invalid_argument::what,
                                                                         testing::HasSubstr(
                                                                                 "SpicenetHcm fit: not all lists contain an equal amount of activations"))));
}

TEST(SpicenetHcmTestSuite, FitMallformedTrainingData_UnequalActivationAmount3) {
    // Arrange
    auto weightsLrf = [](uint64_t x) -> float {
        trainingIterationTestCounter = x;
        return 0.7;
    };
    auto trustOfNewLrf = [](uint64_t x) -> float { return 0.3; };

    std::list<std::list<std::vector<float_t>>> data;
    std::list<std::vector<float_t>> a1 = {{0.0},
                                          {2.0}};
    std::list<std::vector<float_t>> a2 = {{1.0}};
    std::list<std::vector<float_t>> a3 = {{2.0}};
    data.push_back(a1);
    data.push_back(a2);
    data.push_back(a3);
    auto hcm = SpicenetHcm<float_t>({1, 1, 1}, weightsLrf, trustOfNewLrf);

    // Act
    // Assert
    EXPECT_THAT([&]() { hcm.fit(data, 0); },
                testing::Throws<std::invalid_argument>(testing::Property(&std::invalid_argument::what,
                                                                         testing::HasSubstr(
                                                                                 "SpicenetHcm fit: not all lists contain an equal amount of activations"))));
}

TEST(SpicenetHcmTestSuite, Fit2D) {
    // Arrange
    auto weightsLrf = [](uint64_t x) -> float {
        trainingIterationTestCounter = x;
        return 0.8;
    };
    auto trustOfNewLrf = [](uint64_t x) -> float { return 0.8; };


    std::list<std::list<std::vector<float_t>>> data;
    std::list<std::vector<float_t>> a1 = {{0.2},
                                          {0.4},
                                          {0.001}};
    std::list<std::vector<float_t>> a2 = {{0.9},
                                          {1.0},
                                          {1.3}};
    data.push_back(a1);
    data.push_back(a2);
    auto hcm = SpicenetHcm<float_t>({1, 1}, weightsLrf, trustOfNewLrf);

    // Act
    // Assert
    hcm.fit(data, 1);
    EXPECT_FLOAT_EQ(hcm.getMatrix()->getField({0, 0}), 1.003911808);
    EXPECT_EQ(trainingIterationTestCounter, 3 - 1);
}

TEST(SpicenetHcmTestSuite, Fit2DSteps) {
    // Arrange
    auto weightsLrf = [](uint64_t x) -> float {
        trainingIterationTestCounter = x;
        return 0.8;
    };
    auto trustOfNewLrf = [](uint64_t x) -> float { return 0.8; };

    auto hcm = SpicenetHcm<float_t>({1, 1}, weightsLrf, trustOfNewLrf);

    // Act
    // Assert
    std::list<std::list<std::vector<float_t>>> data1 = {{{0.2}},
                                                        {{0.9}}};
    hcm.fit(data1, 1);
    EXPECT_FLOAT_EQ(hcm.getMatrix()->getField({0, 0}), 1.00576);
    std::list<std::list<std::vector<float_t>>> data2 = {{{0.4}},
                                                        {{1.0}}};
    hcm.fit(data2, 1);
    EXPECT_FLOAT_EQ(hcm.getMatrix()->getField({0, 0}), 1.0079104);
    std::list<std::list<std::vector<float_t>>> data3 = {{{0.001}},
                                                        {{1.3}}};
    hcm.fit(data3, 1);
    EXPECT_FLOAT_EQ(hcm.getMatrix()->getField({0, 0}), 1.003911808);
    // Why -1? because we get the counter value with the last call see lambda at top
    EXPECT_EQ(trainingIterationTestCounter, 3 - 1);
}


TEST(SpicenetHcmTestSuite, Fit3D) {
    // Arrange
    auto weightsLrf = [](uint64_t x) -> float {
        trainingIterationTestCounter = x;
        return 0.7;
    };
    auto trustOfNewLrf = [](uint64_t x) -> float { return 0.3; };


    std::list<std::list<std::vector<float_t>>> data;
    std::list<std::vector<float_t>> a1 = {{0.5},
                                          {0.0}};
    std::list<std::vector<float_t>> a2 = {{2.0},
                                          {2.0}};
    std::list<std::vector<float_t>> a3 = {{2.0},
                                          {3.0}};
    data.push_back(a1);
    data.push_back(a2);
    data.push_back(a3);
    auto hcm = SpicenetHcm<float_t>({1, 1, 1}, weightsLrf, trustOfNewLrf);

    // Act
    hcm.fit(data, 1);

    // Assert
}

TEST(SpicenetHcmTestSuite, Fit2Epochs) {
    // Arrange
    auto weightsLrf = [](uint64_t x) -> float {
        trainingIterationTestCounter = x;
        return 0.8;
    };
    auto trustOfNewLrf = [](uint64_t x) -> float { return 0.8; };


    std::list<std::list<std::vector<float_t>>> data;
    std::list<std::vector<float_t>> a1 = {{0.2}};
    std::list<std::vector<float_t>> a2 = {{0.9}};
    data.push_back(a1);
    data.push_back(a2);
    auto hcm = SpicenetHcm<float_t>({1, 1}, weightsLrf, trustOfNewLrf);

    // Act
    hcm.fit(data, 2);

    // Assert
    EXPECT_FLOAT_EQ(hcm.getMatrix()->getField({0, 0}), 1.0059904);
    EXPECT_EQ(trainingIterationTestCounter, 2 - 1);
}


void fillMatrixWithTestData(Matrix<float_t> &matrix) {
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
}

TEST(SpicenetHcmTestSuite, ShouldCalculationRightValues) {
    // Arrange
    auto weightsLrf = [](uint64_t x) -> float {
        trainingIterationTestCounter = x;
        return 0.8;
    };
    auto trustOfNewLrf = [](uint64_t x) -> float { return 0.8; };

    Matrix<float_t> matrix({3, 3, 3});
    fillMatrixWithTestData(matrix);

    unsigned int size;
    auto hcm = SpicenetHcm<float_t>(matrix.rawData(size), {3, 3, 3}, weightsLrf, trustOfNewLrf);
    const std::map<uint8_t, std::vector<float_t>> activation = {
            {0, {0.8, 2, 0.1}},
            {1, {1,   9, 0.3}},
    };;

    // Act
    auto result = hcm.calculateShouldPattern(2, activation);

    // Assert
    ASSERT_THAT(result, ::testing::ElementsAre(112.85, 196.57, 39.29));
}

TEST(SpicenetHcmTestSuite, MissingSecondDim) {
    // Arrange
    auto weightsLrf = [](uint64_t x) -> float {
        trainingIterationTestCounter = x;
        return 0.8;
    };
    auto trustOfNewLrf = [](uint64_t x) -> float { return 0.8; };

    Matrix<float_t> matrix({3, 3, 3});
    fillMatrixWithTestData(matrix);

    unsigned int size;
    auto hcm = SpicenetHcm<float_t>(matrix.rawData(size), {3, 3, 3}, weightsLrf, trustOfNewLrf);
    const std::map<uint8_t, std::vector<float_t>> activation = {
            {0, {0.8, 2, 0.1}},
    };;

    // Act
    // Assert
    EXPECT_THAT([&]() { hcm.calculateShouldPattern(2, activation); },
                testing::Throws<std::invalid_argument>(testing::Property(&std::invalid_argument::what,
                                                                         testing::HasSubstr(
                                                                                 "SpicenetHcm calculateShouldPattern: for dimension 1 is no activation given"))));

}

