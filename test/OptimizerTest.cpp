//
// Created by Fabian on 18.01.2025.
//


#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include <cmath>
#include "spicenet_lib/Optimizer.h"

TEST(OptimizerTestSuite, BasicTest){
    // Arrange
    double start = -4;
    double end = -1;
    //double tolerance = std::abs((1.0e-6) * (-4 + -1) / 2.0);
    auto fn = [](double x) -> double {return pow((x + 3.0), 2.0) + 1.0;};

    // Act
    double result = approximateLocalMin<double>(start, end, fn);

    // Assert
    EXPECT_DOUBLE_EQ(result, -2.999999523162842);
}

TEST(OptimizerTestSuite, BasicTest2){
    // Arrange
    double start = -7;
    double end = 10;
    //double tolerance = std::abs((1.0e-6) * (-7 + 10) / 2.0);
    auto fn = [](double x) -> double {return pow((x - 3.0), 2.0) + 1.0;};

    // Act
    double result = approximateLocalMin<double>(start, end, fn);

    // Assert
    EXPECT_DOUBLE_EQ(result, 3.0000004172325134);
}