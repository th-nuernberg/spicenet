//
// Created by Fabian on 06.02.2025.
//

#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "spicenet_lib/RegressionMetrics.h"

TEST(RegressionMetricsTestSuite, MeanBiasTest) {
    // Arrange
    float_t predicted[] = {0.2, 0.9, 2.0, 3.5};
    float_t actual[] = {0.1, 1.0, 2.0, 3.0};

    // Act
    auto result = meanBias(std::cbegin(predicted),
                           std::cend(predicted),
                           std::cbegin(actual),
                           std::cend(actual));

    // Assert
    EXPECT_NEAR(0.125, result, 0.00001);
}

TEST(RegressionMetricsTestSuite, MedianBiasTest) {
    // Arrange
    float_t predicted[] = {0.2, 0.9, 2.0, 3.5};
    float_t actual[] = {0.1, 1.0, 2.0, 3.0};

    // Act
    auto result = medianBias(std::cbegin(predicted),
                             std::cend(predicted),
                             std::cbegin(actual),
                             std::cend(actual));

    // Assert
    EXPECT_NEAR(0.05, result, 0.00001);
}

TEST(RegressionMetricsTestSuite, standardDeviationBiasTest) {
    // Arrange
    float_t predicted[] = {0.2, 0.9, 2.0, 3.5};
    float_t actual[] = {0.1, 1.0, 2.0, 3.0};

    // Act
    auto result = standardDeviationBias(std::cbegin(predicted),
                                        std::cend(predicted),
                                        std::cbegin(actual),
                                        std::cend(actual));

    // Assert
    EXPECT_NEAR(0.22776083947860748, result, 0.00001);
}

TEST(RegressionMetricsTestSuite, meanAbsoluteGrossErrorTest) {
    // Arrange
    float_t predicted[] = {0.2, 0.9, 2.0, 3.5};
    float_t actual[] = {0.1, 1.0, 2.0, 3.0};

    // Act
    auto result = meanAbsoluteGrossError(std::cbegin(predicted),
                                         std::cend(predicted),
                                         std::cbegin(actual),
                                         std::cend(actual));

    // Assert
    EXPECT_NEAR(0.175, result, 0.00001);
}

TEST(RegressionMetricsTestSuite, meanSquaredErrorTest) {
    // Arrange
    float_t predicted[] = {0.2, 0.9, 2.0, 3.5};
    float_t actual[] = {0.1, 1.0, 2.0, 3.0};

    // Act
    auto result = meanSquaredError(std::cbegin(predicted),
                                   std::cend(predicted),
                                   std::cbegin(actual),
                                   std::cend(actual));

    // Assert
    EXPECT_NEAR(0.0675, result, 0.00001);
}

TEST(RegressionMetricsTestSuite, rootMeanSquaredErrorTest) {
    // Arrange
    float_t predicted[] = {0.2, 0.9, 2.0, 3.5};
    float_t actual[] = {0.1, 1.0, 2.0, 3.0};

    // Act
    auto result = rootMeanSquaredError(std::cbegin(predicted),
                                       std::cend(predicted),
                                       std::cbegin(actual),
                                       std::cend(actual));

    // Assert
    EXPECT_NEAR(0.2598076211353316, result, 0.00001);
}

TEST(RegressionMetricsTestSuite, centeredMeanSquareDifferenceTest) {
    // Arrange
    float_t predicted[] = {0.2, 0.9, 2.0, 3.5};
    float_t actual[] = {0.1, 1.0, 2.0, 3.0};

    // Act
    auto result = centeredMeanSquareDifference(std::cbegin(predicted),
                                               std::cend(predicted),
                                               std::cbegin(actual),
                                               std::cend(actual));

    // Assert
    EXPECT_NEAR(0.051875, result, 0.00001);
}

TEST(RegressionMetricsTestSuite, centeredRootMeanSquaredErrorTest) {
    // Arrange
    float_t predicted[] = {0.2, 0.9, 2.0, 3.5};
    float_t actual[] = {0.1, 1.0, 2.0, 3.0};

    // Act
    auto result = centeredRootMeanSquaredError(std::cbegin(predicted),
                                               std::cend(predicted),
                                               std::cbegin(actual),
                                               std::cend(actual));

    // Assert
    EXPECT_NEAR(0.22776083947860748, result, 0.00001);
}

TEST(RegressionMetricsTestSuite, meanNormalizedBiasTest) {
    // Arrange
    float_t predicted[] = {0.2, 0.9, 2.0, 3.5};
    float_t actual[] = {0.1, 1.0, 2.0, 3.0};

    // Act
    auto result = meanNormalizedBias(std::cbegin(predicted),
                                     std::cend(predicted),
                                     std::cbegin(actual),
                                     std::cend(actual));

    // Assert
    EXPECT_NEAR(0.2666666666666666, result, 0.00001);
}

TEST(RegressionMetricsTestSuite, meanNormalizedGrossErrorTest) {
    // Arrange
    float_t predicted[] = {0.2, 0.9, 2.0, 3.5};
    float_t actual[] = {0.1, 1.0, 2.0, 3.0};

    // Act
    auto result = meanNormalizedGrossError(std::cbegin(predicted),
                                           std::cend(predicted),
                                           std::cbegin(actual),
                                           std::cend(actual));

    // Assert
    EXPECT_NEAR(0.3166666666666667, result, 0.00001);
}

TEST(RegressionMetricsTestSuite, normalizedMeanBiasTest) {
    // Arrange
    float_t predicted[] = {0.2, 0.9, 2.0, 3.5};
    float_t actual[] = {0.1, 1.0, 2.0, 3.0};

    // Act
    auto result = normalizedMeanBias(std::cbegin(predicted),
                                     std::cend(predicted),
                                     std::cbegin(actual),
                                     std::cend(actual));

    // Assert
    EXPECT_NEAR(0.08196721311475419, result, 0.00001);
}

TEST(RegressionMetricsTestSuite, normalizedMeanErrorTest) {
    // Arrange
    float_t predicted[] = {0.2, 0.9, 2.0, 3.5};
    float_t actual[] = {0.1, 1.0, 2.0, 3.0};

    // Act
    auto result = normalizedMeanError(std::cbegin(predicted),
                                      std::cend(predicted),
                                      std::cbegin(actual),
                                      std::cend(actual));

    // Assert
    EXPECT_NEAR(0.11475409836065574, result, 0.00001);
}

TEST(RegressionMetricsTestSuite, fractionalBiasTest) {
    // Arrange
    float_t predicted[] = {0.2, 0.9, 2.0, 3.5};
    float_t actual[] = {0.1, 1.0, 2.0, 3.0};

    // Act
    auto result = fractionalBias(std::cbegin(predicted),
                                 std::cend(predicted),
                                 std::cbegin(actual),
                                 std::cend(actual));

    // Assert
    EXPECT_NEAR(0.1788124156545209, result, 0.00001);
}

TEST(RegressionMetricsTestSuite, fractionalGrossErrorTest) {
    // Arrange
    float_t predicted[] = {0.2, 0.9, 2.0, 3.5};
    float_t actual[] = {0.1, 1.0, 2.0, 3.0};

    // Act
    auto result = fractionalGrossError(std::cbegin(predicted),
                                       std::cend(predicted),
                                       std::cbegin(actual),
                                       std::cend(actual));

    // Assert
    EXPECT_NEAR(0.23144399460188933, result, 0.00001);
}

TEST(RegressionMetricsTestSuite, theilsUiTest) {
    // Arrange
    float_t predicted[] = {0.2, 0.9, 2.0, 3.5};
    float_t actual[] = {0.1, 1.0, 2.0, 3.0};

    // Act
    auto result = theilsUi(std::cbegin(predicted),
                           std::cend(predicted),
                           std::cbegin(actual),
                           std::cend(actual));

    // Assert
    EXPECT_NEAR(0.06595601905783499, result, 0.00001);
}

TEST(RegressionMetricsTestSuite, indexOfAgreementTest) {
    // Arrange
    float_t predicted[] = {0.2, 0.9, 2.0, 3.5};
    float_t actual[] = {0.1, 1.0, 2.0, 3.0};

    // Act
    auto result = indexOfAgreement(std::cbegin(predicted),
                                   std::cend(predicted),
                                   std::cbegin(actual),
                                   std::cend(actual));

    // Assert
    EXPECT_NEAR(0.9875518672199171, result, 0.00001);
}
