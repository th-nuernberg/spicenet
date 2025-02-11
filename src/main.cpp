#include <Arduino.h>
#include <mbed_stats.h>
#include "spicenet_lib/Spicenet.h"
#include "trainings-data.h"
#include "spicenet_lib/RegressionMetrics.h"

#define SOM_SIZE 100

const String title = "   _____ _____ _____ _____ ______            _   \n"
                     "  / ____|  __ \\_   _/ ____|  ____|          | |  \n"
                     " | (___ | |__) || || |    | |__   _ __   ___| |_ \n"
                     "  \\___ \\|  ___/ | || |    |  __| | '_ \\ / _ \\ __|\n"
                     "  ____) | |    _| || |____| |____| | | |  __/ |_ \n"
                     " |_____/|_|   |_____\\_____|______|_| |_|\\___|\\__|\n"
                     "                                                 \n";


auto *spicenetSom1 = new SpicenetSom<float, 1>(-1,
                                               1,
                                               SOM_SIZE,
                                               [](auto x) -> float { return 0.8; },
                                               [](auto x) -> float { return 0.8; });

auto *spicenetSom2 = new SpicenetSom<float, 1>(-1,
                                               1,
                                               SOM_SIZE,
                                               [](auto x) -> float { return 0.8; },
                                               [](auto x) -> float { return 0.8; });

auto *hcm = new SpicenetHcm<float>({SOM_SIZE, SOM_SIZE},
                                   [](auto x) -> float { return 0.8; },
                                   [](auto x) -> float { return 0.8; });


std::array<SpicenetSomBase<float> *, 2> somArr = {
        spicenetSom1,
        spicenetSom2
};
Spicenet<float, 2> spicenet(hcm, somArr);

extern "C" char *sbrk(int incr);

int freeRam() {
    char top;
    return &top - reinterpret_cast<char *>(sbrk(0));
}


void display_freeram() {
    Serial.print(F("- SRAM left: "));
    Serial.println(freeRam());

    mbed_stats_heap_t heap_stats;
    mbed_stats_heap_get(&heap_stats);
    Serial.println(heap_stats.current_size);
    Serial.println(heap_stats.reserved_size);
}

void trainModel() {
    Serial.println("[Training]: Start loading test dataset");
    const auto startLoadData = millis();
    auto trainingsData = getTrainingsData();
    Serial.print("[Training]: finished loading test dataset in ");
    Serial.print(millis() - startLoadData, 20);
    Serial.println(" millis");


    display_freeram();


    Serial.println("[Training]: Start training");
    Serial.print("[Training]: Trainset size ");
    Serial.println(trainingsData.begin()->size());
    const auto startTraining = millis();
    spicenet.fit(trainingsData, 10);
    auto endTraining = millis();
    Serial.print("[Training]: Training finished ");
    Serial.print((endTraining - startTraining) / 1000, 20);
    Serial.println(" seconds");
}

void evaluateModel() {
    Serial.println("[Evaluation]: Metrics");
    auto trainingsData = getTestData();
    auto initData = *trainingsData.begin();
    auto resultDataGeneric = *(++trainingsData.begin());
    std::list<float> resultData;
    for (auto &dataVec: resultDataGeneric) {
        resultData.push_back(dataVec.at(0));
    }

    Serial.println("[Evaluation] Predicted values: ");
    unsigned int ind = 0;
    std::list<float> predictedData;
    for (auto &init: initData) {
        Serial.print("Calculated data: ");
        Serial.print(ind);
        Serial.print(" / ");
        Serial.print(resultData.size());
        auto prediction = spicenet.decode(1, {{0, init}});
        predictedData.push_back(prediction);
        Serial.print('\r');
        Serial.print("                                                 ");
        Serial.print('\r');
        Serial.println(prediction, 20);
        ++ind;
    }
    Serial.print("\r");
    Serial.println("[Evaluation] Stats: ");

    auto metrics = metricsMap(predictedData.begin(),
                              predictedData.end(),
                              resultData.begin(),
                              resultData.end());
    for (auto &metric:metrics) {
        Serial.print(metric.first.c_str());
        Serial.print(",");
        Serial.println(metric.second, 20);
    }
}

void setup() {
    Serial.begin(9600);
    delay(5000);
    Serial.print(title);

    trainModel();
    evaluateModel();

    randomSeed(42);
}

void loop() {
    return;
    Serial.println("Doing stuff");
    float rnd = (float) random(-100000, 100000) / (float) 100000.0;
    Serial.println(rnd, 20);
    std::map<uint8_t, std::vector<float>> inputValues{{0, {rnd}}};
    auto start = millis();
    auto predicted = spicenet.decode(1, inputValues);
    auto end = millis();
    auto real = pow(rnd, 3);
    Serial.print("Decoding millis: ");
    Serial.println(end - start);
    Serial.print(predicted, 20);
    Serial.print(" real: ");
    Serial.println(real, 20);
    Serial.print("bias: ");
    Serial.println(predicted - real, 20);
    display_freeram();
    delay(1000);
}