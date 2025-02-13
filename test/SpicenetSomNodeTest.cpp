//
// Created by Fabian Stiegler on 05.12.2024.
//

#include <cmath>
#include "gtest/gtest.h"
#include "spicenet_lib/SpicenetSomNode.h"


TEST(SpicenetSomNodeTestSuite, UpdateNode) {
    // Arrange
    auto node = SpicenetSomNode<float_t, 1>();
    node.tuningCurveWidth = 0.8;
    node.preferredValue[0] = 1;

    float_t inputValue[1] = {1.1};

    // Act
    node.updateNode(inputValue,
                    0.7,
                    0.3,
                    1);

    // Assert
    float_t newPreferredValue = 0.0002706144098 + 1.0;
    float_t newTuningCurveWidth = -0.001704870782 + 0.8;
    EXPECT_EQ(node.preferredValue[0], newPreferredValue);
    EXPECT_EQ(node.tuningCurveWidth, newTuningCurveWidth);
}

TEST(SpicenetSomNodeTestSuite, UpdateNode2D) {
    // Arrange
    auto node = SpicenetSomNode<float_t, 2>();
    node.tuningCurveWidth = 0.8;
    node.preferredValue[0] = 1;
    node.preferredValue[1] = 1;

    float_t inputValue[2] = {1.1, 1.1};

    // Act
    node.updateNode(inputValue,
                    0.7,
                    0.3,
                    1);

    // Assert
    float_t newPreferredValue = 0.0002706144098 + 1.0;
    float_t newTuningCurveWidth = -0.001704870782 + 0.8;
    EXPECT_EQ(node.preferredValue[0], newPreferredValue);
    EXPECT_EQ(node.preferredValue[1], newPreferredValue);
    EXPECT_EQ(node.tuningCurveWidth, newTuningCurveWidth);
}

TEST(SpicenetSomNodeTestSuite, ActivationOfNode) {
    // Arrange
    auto node = SpicenetSomNode<double_t, 1>();
    node.tuningCurveWidth = 0.8;
    node.preferredValue[0] = 1;

    double_t inputValue[1] = {1.1};

    // Act
    auto result = node.nodeActivation(inputValue);

    // Assert
    double_t activation = 0.49479710868093685;
    EXPECT_EQ(result, activation);
}


TEST(SpicenetSomNodeTestSuite, ToTableString2D) {
    // Arrange
    auto node = SpicenetSomNode<double_t, 2>();
    node.tuningCurveWidth = 0.8;
    node.preferredValue[0] = 0.2;
    node.preferredValue[1] = 0.3;

    // Act
    auto result = node.toTableString();

    // Assert
    EXPECT_EQ(result, "| 0.800000 | [0.200000, 0.300000] |");
}

TEST(SpicenetSomNodeTestSuite, ToTableString1D) {
    // Arrange
    auto node = SpicenetSomNode<double_t, 1>();
    node.tuningCurveWidth = 0.8;
    node.preferredValue[0] = 0.2;

    // Act
    auto result = node.toTableString();

    // Assert
    EXPECT_EQ(result, "| 0.800000 | [0.200000] |");
    // auto test = 'x_dlla';
}