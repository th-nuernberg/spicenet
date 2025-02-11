//
// Created by Fabian Stiegler on 25.01.2025.
//

#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "spicenet_lib/Spicenet.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>


std::list<std::list<std::vector<double_t>>> loadTrainingsDataDouble() {
    std::list<std::vector<double_t>> initData;
    std::list<std::vector<double_t>> resultData;
    std::ifstream file(R"(C:\Users\Fabian\source\repos\spicenet-cpp\Google_tests\data\test_data.csv)");

    std::string line;
    std::getline(file, line);
    while (std::getline(file, line)) {
        std::vector<std::string> row;
        std::stringstream lineStream(line);
        std::string cell;

        std::getline(lineStream, cell, ',');
        initData.push_back({std::stod(cell)});
        std::getline(lineStream, cell, ',');
        resultData.push_back({std::stod(cell)});
    }
    file.close();


    return {
            initData,
            resultData
    };
}

std::list<std::list<std::vector<float_t>>> loadTrainingsDataFloat() {
    std::list<std::vector<float_t>> initData;
    std::list<std::vector<float_t>> resultData;
    std::ifstream file(R"(C:\Users\Fabian\source\repos\spicenet-cpp\Google_tests\data\test_data.csv)");

    std::string line;
    std::getline(file, line);
    while (std::getline(file, line)) {
        std::vector<std::string> row;
        std::stringstream lineStream(line);
        std::string cell;

        std::getline(lineStream, cell, ',');
        initData.push_back({std::stof(cell)});
        std::getline(lineStream, cell, ',');
        resultData.push_back({std::stof(cell)});
    }
    file.close();


    return {
            initData,
            resultData
    };
}


TEST(SpicenetTestSuite, FitTest) {
    // Arrange
    std::list<std::list<std::vector<float_t>>> inputData = {
            {{0.2}, {0.3}, {0.4}},
            {{0.2}, {0.3}, {0.4}},
    };
    auto *spicenetSom1 = new SpicenetSom<float_t, 1>(0,
                                                     1,
                                                     10,
                                                     [](auto x) -> float_t { return 0.8; },
                                                     [](auto x) -> float_t { return 0.8; });

    auto *spicenetSom2 = new SpicenetSom<float_t, 1>(0,
                                                     1,
                                                     10,
                                                     [](auto x) -> float_t { return 0.8; },
                                                     [](auto x) -> float_t { return 0.8; });

    auto *hcm = new SpicenetHcm<float_t>({10, 10},
                                         [](auto x) -> float_t { return 0.8; },
                                         [](auto x) -> float_t { return 0.8; });

    std::array<SpicenetSomBase<float_t> *, 2> somArr = {
            spicenetSom1,
            spicenetSom2
    };
    Spicenet<float_t, 2> spicenet(hcm, somArr);

    // Act
    spicenet.fit(inputData, 1);

    // Assert
    EXPECT_EQ(spicenetSom1->getTrainingIterations(), 3);
    delete spicenetSom1;
    delete spicenetSom2;
    delete hcm;
}


TEST(SpicenetTestSuite, FitTestWithDataFromFile) {
    // Arrange
    std::list<std::list<std::vector<float_t>>> inputData = loadTrainingsDataFloat();

    auto *spicenetSom1 = new SpicenetSom<float_t, 1>(0,
                                                     1,
                                                     10,
                                                     [](auto x) -> float_t { return 0.8; },
                                                     [](auto x) -> float_t { return 0.8; });

    auto *spicenetSom2 = new SpicenetSom<float_t, 1>(0,
                                                     1,
                                                     10,
                                                     [](auto x) -> float_t { return 0.8; },
                                                     [](auto x) -> float_t { return 0.8; });

    auto *hcm = new SpicenetHcm<float_t>({10, 10},
                                         [](auto x) -> float_t { return 0.8; },
                                         [](auto x) -> float_t { return 0.8; });

    std::array<SpicenetSomBase<float_t> *, 2> somArr = {
            spicenetSom1,
            spicenetSom2
    };
    Spicenet<float_t, 2> spicenet(hcm, somArr);

    // Act
    spicenet.fit(inputData, 10);

    // Assert
    EXPECT_EQ(spicenetSom1->getTrainingIterations(), 30'000);
    delete spicenetSom1;
    delete spicenetSom2;
    delete hcm;

}


TEST(SpicenetTestSuite, FitTestWithDataFromFileWithOutput) {
    // Arrange
    std::list<std::list<std::vector<double_t>>> inputData = loadTrainingsDataDouble();

    auto *spicenetSom1 = new SpicenetSom<double_t, 1>(-1,
                                                      1,
                                                      100,
                                                      [](auto x) -> float_t { return 0.8; },
                                                      [](auto x) -> float_t { return 0.8; });

    auto *spicenetSom2 = new SpicenetSom<double_t, 1>(-1,
                                                      1,
                                                      100,
                                                      [](auto x) -> float_t { return 0.8; },
                                                      [](auto x) -> float_t { return 0.8; });

    auto *hcm = new SpicenetHcm<double_t>({100, 100},
                                          [](auto x) -> float_t { return 0.8; },
                                          [](auto x) -> float_t { return 0.8; });


    std::array<SpicenetSomBase<double_t> *, 2> somArr = {
            spicenetSom1,
            spicenetSom2
    };
    Spicenet<double_t, 2> spicenet(hcm, somArr);

    // Act
    spicenet.fit(inputData, 10);


    // Assert
    EXPECT_EQ(spicenetSom1->getTrainingIterations(), 30'000);

    std::ofstream outPutFileSom1(R"(C:\Users\Fabian\source\repos\spicenet-cpp\Google_tests\data\output_som_1.csv)");
    size_t nodes = 0;
    auto nodeArr = spicenetSom1->getNodes(nodes);
    outPutFileSom1 << "tuningCurveWidth,preferredValue\n";
    for (int i = 0; i < nodes; ++i) {
        outPutFileSom1 << nodeArr[i].tuningCurveWidth << ",";
        outPutFileSom1 << nodeArr[i].preferredValue[0] << "\n";
    }
    outPutFileSom1.close();

    std::ofstream outPutFileSom2(R"(C:\Users\Fabian\source\repos\spicenet-cpp\Google_tests\data\output_som_2.csv)");
    nodes = 0;
    nodeArr = spicenetSom2->getNodes(nodes);
    outPutFileSom2 << "tuningCurveWidth,preferredValue\n";
    for (int i = 0; i < nodes; ++i) {
        outPutFileSom2 << nodeArr[i].tuningCurveWidth << ",";
        outPutFileSom2 << nodeArr[i].preferredValue[0] << "\n";
    }
    outPutFileSom2.close();

    std::ofstream outPutFile(R"(C:\Users\Fabian\source\repos\spicenet-cpp\Google_tests\data\output_hcm.csv)");

    outPutFile << "x,y,value\n";
    auto matrixIterator = hcm->getMatrix()->getIterator();
    std::vector<uint16_t> index;
    double_t *weight;
    while (matrixIterator.next(index, weight)) {
        for (auto &i: index) {
            outPutFile << std::to_string(i) << ",";
        }
        outPutFile << *weight << "\n";
    }
    outPutFile.close();

    delete spicenetSom1;
    delete spicenetSom2;
    delete hcm;

}

TEST(SpicenetTestSuite, DecodeTest) {
    // Arrange
    std::list<std::list<std::vector<double_t>>> inputData = loadTrainingsDataDouble();

    auto *spicenetSom1 = new SpicenetSom<double_t, 1>(-1,
                                                      1,
                                                      100,
                                                      [](auto x) -> float_t { return 0.8; },
                                                      [](auto x) -> float_t { return 0.8; });

    auto *spicenetSom2 = new SpicenetSom<double_t, 1>(-1,
                                                      1,
                                                      100,
                                                      [](auto x) -> float_t { return 0.8; },
                                                      [](auto x) -> float_t { return 0.8; });

    auto *hcm = new SpicenetHcm<double_t>({100, 100},
                                          [](auto x) -> float_t { return 0.8; },
                                          [](auto x) -> float_t { return 0.8; });


    std::array<SpicenetSomBase<double_t> *, 2> somArr = {
            spicenetSom1,
            spicenetSom2
    };
    Spicenet<double_t, 2> spicenet(hcm, somArr);


    // Act
    spicenet.fit(inputData, 10);


    // Assert
    double sum = 0;
    std::srand(42);
    for (int i = 0; i < 100; ++i) {
        double_t value = (double_t) std::rand() / (double_t) RAND_MAX;
        std::map<uint8_t, std::vector<double_t>> inputValues{{0, {value}}};
        auto predicted = spicenet.decode(1, inputValues);
        EXPECT_EQ(predicted, pow(value, 3));
        sum += predicted - pow(value, 3);
    }
    EXPECT_EQ(sum / 100.0, 0.0);

    //init: -0.3807291804874664, result: -0.055188487517647725
    // 0.3809857727068724,0.05530014548706537
    /*
    std::map<uint8_t, std::vector<double_t>> inputValues{{0, {0.3809857727068724}}};
    EXPECT_EQ(spicenet.decode(1, inputValues), 0.0);
    std::map<uint8_t, std::vector<double_t>> inputValues2{{0, {-0.3807291804874664}}};
    EXPECT_EQ(spicenet.decode(1, inputValues2), 0.0);
     */

    delete spicenetSom1;
    delete spicenetSom2;
    delete hcm;
}