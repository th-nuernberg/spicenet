//
// Created by Fabian on 11.02.2025.
//

#ifndef SPICENET_EXAMPLE_H
#define SPICENET_EXAMPLE_H

#include <Arduino.h>
#include "spicenet_lib/Spicenet.h"
#include "3DTestData.h"
#include "spicenet_lib/RegressionMetrics.h"

#define SOM_SIZE 25

auto *spicenetSom1 = new SpicenetSom<double, 1>(-1,
                                               1,
                                               SOM_SIZE,
                                               [](auto x) -> float { return 0.8; },
                                               [](auto x) -> float { return 0.8; });

auto *spicenetSom2 = new SpicenetSom<double, 1>(-1,
                                               1,
                                               SOM_SIZE,
                                               [](auto x) -> float { return 0.8; },
                                               [](auto x) -> float { return 0.8; });

auto *spicenetSom3 = new SpicenetSom<double, 1>(-1,
                                               1,
                                               SOM_SIZE,
                                               [](auto x) -> float { return 0.8; },
                                               [](auto x) -> float { return 0.8; });

auto *hcm = new SpicenetHcm<double>({SOM_SIZE, SOM_SIZE, SOM_SIZE},
                                   [](auto x) -> float { return 0.8; },
                                   [](auto x) -> float { return 0.8; });


std::array<SpicenetSomBase<double> *, 3> somArr = {
        spicenetSom1,
        spicenetSom2,
        spicenetSom3
};
Spicenet<double, 3> spicenet(hcm, somArr);


void trainModel() {
    Serial.println("[Training]: Start loading test dataset");
    const auto startLoadData = millis();
    auto trainingsData = getTrainingsData3D();
    Serial.print("[Training]: finished loading test dataset in ");
    Serial.print(millis() - startLoadData);
    Serial.println(" millis");


    //display_freeram();


    Serial.println("[Training]: Start training");
    Serial.print("[Training]: Trainset size ");
    Serial.println(trainingsData.begin()->size());
    const auto startTraining = millis();
    spicenet.fit(trainingsData, 1);
    auto endTraining = millis();
    Serial.print("[Training]: Training finished ");
    Serial.print(endTraining - startTraining);
    Serial.println(" millis");
}

void evaluateModel() {
    Serial.println("[Evaluation]: Metrics");
    auto trainingsData = getTestData3D();

    auto it = trainingsData.begin();
    unsigned int size = it->size();
    auto xIter = it->begin();
    ++it;
    auto zIter = it->begin();
    Serial.println("[Evaluation]: Metrics");
    ++it;
    auto yData = *it;

    std::list<double> resultData;
    for (auto &dataVec: yData) {
        resultData.push_back(dataVec.at(0));
    }

    Serial.println("[Evaluation] Predicted values: ");
    unsigned int ind = 0;
    std::list<double> predictedData;

    for (unsigned int i = 0; i < size; ++i) {
        Serial.print("Calculated data: ");
        Serial.print(ind);
        Serial.print(" / ");
        Serial.print(resultData.size());
        double prediction;
        const auto start = millis();
        auto successful = spicenet.tryDecode(2, {{0, *xIter}, {1, *zIter}}, prediction);
        const auto end = millis();
        if (successful) {
            predictedData.push_back(prediction);
        }
        Serial.print('\r');
        Serial.print("                                                 ");
        Serial.print('\r');
        if (successful){
            Serial.print(prediction, 20);
        }else{
            Serial.print("error");
        }
        Serial.print(',');
        Serial.print(end - start);
        Serial.println();
        ++ind;

        ++xIter;
        ++zIter;
    }
    Serial.print("\r");
    Serial.println("[Evaluation] Stats: ");

    auto metrics = metricsMap(predictedData.begin(),
                              predictedData.end(),
                              resultData.begin(),
                              resultData.end());
    for (auto &metric: metrics) {
        Serial.print(metric.first.c_str());
        Serial.print(",");
        Serial.println(metric.second, 20);
    }
}

void runTest() {
    trainModel();
    evaluateModel();
}


#endif //SPICENET_EXAMPLE_H
