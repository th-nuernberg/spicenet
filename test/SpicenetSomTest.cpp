//
// Created by Fabian Stiegler on 05.12.2024.
//

#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "spicenet_lib/SpicenetSom.h"

TEST(SpicenetSomTestSuite, InitSomPrefferedValue1D) {
    // Arrange
    auto spicenetSom = SpicenetSom<float_t, 1>(0.0,
                                               1.0,
                                               2,
                                               [](auto x) -> float_t { return 0.0; },
                                               [](auto x) -> float_t { return 0.0; });

    // Act
    size_t size = 99;
    auto nodes = spicenetSom.getNodes(size);

    // Assert
    EXPECT_EQ(size, 2);
    EXPECT_EQ(nodes[0].preferredValue[0], 0.25);
    EXPECT_EQ(nodes[1].preferredValue[0], 0.75);
}

TEST(SpicenetSomTestSuite, InitSomPrefferedValue2D) {
    // Arrange
    auto spicenetSom = SpicenetSom<float_t, 2>(0.0,
                                               1.0,
                                               3,
                                               [](auto x) -> float_t { return 0.0; },
                                               [](auto x) -> float_t { return 0.0; });

    // Act
    size_t size = 99;
    auto nodes = spicenetSom.getNodes(size);

    // Assert
    ASSERT_EQ(size, 3);
    EXPECT_FLOAT_EQ(nodes[0].preferredValue[0], 0.166666672);
    EXPECT_FLOAT_EQ(nodes[0].preferredValue[1], 0.166666672);
    EXPECT_FLOAT_EQ(nodes[1].preferredValue[0], 0.5);
    EXPECT_FLOAT_EQ(nodes[1].preferredValue[1], 0.5);
    EXPECT_FLOAT_EQ(nodes[2].preferredValue[0], 0.833333373);
    EXPECT_FLOAT_EQ(nodes[2].preferredValue[1], 0.833333373);
}

static uint64_t trainingIterationTestCounter = 0;

TEST(SpicenetSomTestSuite, FitTest) {
    // Arrange
    auto tuningCurveLrf = [](uint64_t x) -> float {
        trainingIterationTestCounter = x;
        return 0.7;
    };
    auto interactionKernelLrf = [](uint64_t x) -> float { return 0.3; };

    SpicenetSomNode<float_t, 1> nodes[2];
    nodes[0].tuningCurveWidth = 0.8;
    nodes[0].preferredValue[0] = 1;
    nodes[1].tuningCurveWidth = 0.8;
    nodes[1].preferredValue[0] = 2;

    SpicenetSomNode<float_t, 1> resultingNodes[2];
    memcpy(resultingNodes, nodes, sizeof(SpicenetSomNode<float_t, 1>) * 2);

    std::list<std::vector<float_t> > trainingData;
    trainingData.push_back(std::vector<float_t>{0.83928});
    trainingData.push_back(std::vector<float_t>{0.5});
    trainingData.push_back(std::vector<float_t>{2.1653});

    for (int i = 0; i < 2; ++i) {
        float_t data_1[1]{0.83928};
        resultingNodes[0].updateNode(data_1, 0.7, 0.3, 0);
        resultingNodes[1].updateNode(data_1, 0.7, 0.3, 1);

        float_t data_2[1]{0.5};
        resultingNodes[0].updateNode(data_2, 0.7, 0.3, 0);
        resultingNodes[1].updateNode(data_2, 0.7, 0.3, 1);

        float_t data_3[1]{2.1653};
        resultingNodes[0].updateNode(data_3, 0.7, 0.3, 1);
        resultingNodes[1].updateNode(data_3, 0.7, 0.3, 0);
    }

    auto spicenetSom = SpicenetSom<float_t, 1>(nodes,
                                               2,
                                               (Lrf) tuningCurveLrf,
                                               (Lrf) interactionKernelLrf);

    // Act
    bool result  = spicenetSom.fit(trainingData, 2);

    // Assert
    EXPECT_EQ(result, true);
    EXPECT_EQ(nodes[0].tuningCurveWidth, resultingNodes[0].tuningCurveWidth);
    EXPECT_EQ(nodes[1].tuningCurveWidth, resultingNodes[1].tuningCurveWidth);
    EXPECT_EQ(nodes[0].preferredValue[0], resultingNodes[0].preferredValue[0]);
    EXPECT_EQ(nodes[1].preferredValue[0], resultingNodes[1].preferredValue[0]);
    EXPECT_EQ(trainingIterationTestCounter, 5);
    EXPECT_EQ(spicenetSom.getTrainingIterations(), 6);
}

TEST(SpicenetSomTestSuite, FitMallformedDataTest) {
    // Arrange
    auto tuningCurveLrf = [](uint64_t x) -> float {
        trainingIterationTestCounter = x;
        return 0.7;
    };
    auto interactionKernelLrf = [](uint64_t x) -> float { return 0.3; };

    SpicenetSomNode<float_t, 1> nodes[2];
    nodes[0].tuningCurveWidth = 0.8;
    nodes[0].preferredValue[0] = 1;
    nodes[1].tuningCurveWidth = 0.8;
    nodes[1].preferredValue[0] = 2;


    std::list<std::vector<float_t> > trainingData;
    trainingData.push_back(std::vector<float_t>{0.83928, 0.83928});
    trainingData.push_back(std::vector<float_t>{0.5});
    trainingData.push_back(std::vector<float_t>{2.1653});


    auto spicenetSom = SpicenetSom<float_t, 1>(nodes,
                                               2,
                                               (Lrf) tuningCurveLrf,
                                               (Lrf) interactionKernelLrf);

    // Act
    bool result = spicenetSom.fit(trainingData, 2);

    // Assert
    EXPECT_EQ(false, result);
}

TEST(SpicenetSomTestSuite, FitMallformedDataTest_2) {
    // Arrange
    auto tuningCurveLrf = [](uint64_t x) -> float {
        trainingIterationTestCounter = x;
        return 0.7;
    };
    auto interactionKernelLrf = [](uint64_t x) -> float { return 0.3; };

    SpicenetSomNode<float_t, 1> nodes[2];
    nodes[0].tuningCurveWidth = 0.8;
    nodes[0].preferredValue[0] = 1;
    nodes[1].tuningCurveWidth = 0.8;
    nodes[1].preferredValue[0] = 2;


    std::list<std::vector<float_t> > trainingData;
    trainingData.push_back(std::vector<float_t>{});
    trainingData.push_back(std::vector<float_t>{0.5});
    trainingData.push_back(std::vector<float_t>{2.1653});


    auto spicenetSom = SpicenetSom<float_t, 1>(nodes,
                                               2,
                                               (Lrf) tuningCurveLrf,
                                               (Lrf) interactionKernelLrf);

    // Act
    bool result = spicenetSom.fit(trainingData, 2);

    // Assert
    EXPECT_EQ(false, result);
}

TEST(SpicenetSomTestSuite, ActivationTest) {
    // Arrange
    SpicenetSomNode<float_t, 1> nodes[2];
    nodes[0].tuningCurveWidth = 0.8;
    nodes[0].preferredValue[0] = 1;
    nodes[1].tuningCurveWidth = 0.8;
    nodes[1].preferredValue[0] = 2;

    float_t input[1] = {0.83928};


    auto spicenetSom = SpicenetSom<float_t, 1>(nodes,
                                               2,
                                               [](auto x) -> float_t { return 0.0; },
                                               [](auto x) -> float_t { return 0.0; });

    // Act
    auto result = spicenetSom.activation(input);

    // Assert
    EXPECT_THAT(result, ::testing::ElementsAre(
            nodes[0].nodeActivation(input),
            nodes[1].nodeActivation(input)));
}


TEST(SpicenetSomTestSuite, ActivationVectorEqualArray) {
    // Arrange
    SpicenetSomNode<float_t, 1> nodes[2];
    nodes[0].tuningCurveWidth = 0.8;
    nodes[0].preferredValue[0] = 1;
    nodes[1].tuningCurveWidth = 0.8;
    nodes[1].preferredValue[0] = 2;

    float_t input[1] = {0.83928};

    int n = sizeof(input) / sizeof(input[0]);
    std::vector<float_t> inputVector(input, input + n);

    auto spicenetSom = SpicenetSom<float_t, 1>(nodes,
                                               2,
                                               [](auto x) -> float_t { return 0.0; },
                                               [](auto x) -> float_t { return 0.0; });

    // Act
    auto result = spicenetSom.activation(input);
    auto result2 = spicenetSom.activation(inputVector);

    // Assert
    EXPECT_THAT(result2, ::testing::ElementsAreArray(result));
}
