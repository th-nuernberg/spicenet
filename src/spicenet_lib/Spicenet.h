//
// Created by Fabian on 19.01.2025.
//

#ifndef SPICENET_CPP_SPICENET_H
#define SPICENET_CPP_SPICENET_H

#include <vector>
#include <string>
#include <cmath>
#include <cstdint>
#include <array>
#include <list>
#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <map>

#include "SpicenetSomBase.h"
#include "SpicenetHcm.h"
#include "Optimizer.h"

enum Decoder {
    NAIVE,
    BERT_OPTIMIZER,
};

template<typename T, uint8_t D>
class Spicenet {
    static_assert(std::is_integral<T>::value || std::is_floating_point<T>::value,
                  "Spicenet: T must be a numeric type!");
private:
    SpicenetHcm<T> *hcm;
    std::array<SpicenetSomBase<T> *, D> soms;

    std::vector<T> normVector(const std::vector<T> &vector);

public:
    Spicenet(SpicenetHcm<T> *hcm, const std::array<SpicenetSomBase<T> *, D> &soms);

    // TODO:  uint32_t batchSize
    void fit(const std::list<std::list<std::vector<T>>> &inputData,
             uint16_t epochsOnBatch,
             void (*iterationCallback)(uint64_t iteration) = nullptr);

    T decode(uint8_t targetSom, const std::map<uint8_t, std::vector<T>> &inputValues, Decoder decoder = BERT_OPTIMIZER);
};

template<typename T, uint8_t D>
std::vector<T> Spicenet<T, D>::normVector(const std::vector<T> &vector) {
    T max = *std::max_element(vector.begin(), vector.end());
    std::vector<T> result = vector;
    for (auto &field: result) {
        field /= max;
    }
    return result;
}

template<typename T, uint8_t D>
Spicenet<T, D>::Spicenet(SpicenetHcm<T> *hcm, const std::array<SpicenetSomBase<T> *, D> &soms):hcm(hcm), soms(soms) {

}

template<typename T, uint8_t D>
void Spicenet<T, D>::fit(const std::list<std::list<std::vector<T>>> &inputData,
                         uint16_t epochsOnBatch,
                         void (*iterationCallback)(uint64_t)) {
    if (inputData.size() != D) {
        throw std::invalid_argument("Spicenet fit: the amount of data columns is unequal to the amount of soms");
    }
    auto it = inputData.begin();
    unsigned int dataCount = it->size();
    for (++it; it != inputData.end(); ++it) {
        if (it->size() != dataCount) {
            throw std::invalid_argument("Spicenet fit: not all lists contain an equal amount of data");
        }
    }
    if (dataCount <= 0) {
        return;
    }

    std::vector<typename std::list<std::vector<T>>::const_iterator> colIterators(inputData.size());
    int t = 0;
    for (const std::list<std::vector<T>> &col: inputData) {
        colIterators.at(t) = col.begin();
        ++t;
    }

    // Problematic durch das wir feste listen und kein Generator brauchen müssen wir viel Speicherplatz vorhalten
    // Schlechtes Designe für Embedded Systems mit batches

    int somNumber = 0;
    // it = inputData.begin();
    for (it = inputData.begin(); it != inputData.end(); ++it) {
        soms[somNumber]->fit(*it, epochsOnBatch);
        ++somNumber;
    }

    for (int epoch = 0; epoch < epochsOnBatch; ++epoch) {
        int colCounter = 0;
        for (const std::list<std::vector<T>> &col: inputData) {
            colIterators.at(colCounter) = col.begin();
            ++colCounter;
        }
        for (unsigned int row = 0; row < dataCount; ++row) {
            std::list<std::list<std::vector<T>>> activations;
            somNumber = 0;
            for (auto &iterator: colIterators) {
                std::list<std::vector<T>> stuff;
                stuff.push_back(soms.at(somNumber)->activation(*iterator));
                activations.push_back(stuff);
                std::advance(iterator, 1);
            }

            this->hcm->fit(activations, 1);
        }
    }
}

template<typename T, uint8_t D>
T Spicenet<T, D>::decode(uint8_t targetSom, const std::map<uint8_t, std::vector<T>> &inputValues, Decoder decoder) {
    // TODO: hier auch validieren?
    for (uint8_t i = 0; i < this->soms.size(); ++i) {
        if (i == targetSom) {
            continue;
        }
        if (inputValues.find(i) == inputValues.end()) {
            std::stringstream ss;
            ss << "Spicenet decode: for som " << std::to_string(i) << " is no activation given";
            throw std::invalid_argument(ss.str());
        }
    }

    std::map<uint8_t, std::vector<T> > inputActivations;
    for (uint8_t i = 0; i < this->soms.size(); ++i) {
        if (i == targetSom) {
            continue;
        }
        auto temp = this->soms[i]->activation(inputValues.at(i));
        inputActivations[i] = temp;
    }

    /*
    auto test = this->hcm->calculateShouldPattern(targetSom, inputActivations);
    for(auto temp: test){
        Serial.print(temp);
        Serial.print(" ");
    }
    Serial.println();
     */
    std::vector<T> shouldActivationNormed = normVector(this->hcm->calculateShouldPattern(targetSom, inputActivations));
    int winningIndex = std::distance(shouldActivationNormed.begin(),
                                     std::max_element(shouldActivationNormed.begin(), shouldActivationNormed.end()));

    auto targetSomInstance = this->soms.at(targetSom);

    auto fn = [&](T x) {
        T sum = 0;
        auto distanceVec = vectorSubtraction(shouldActivationNormed, normVector(targetSomInstance->activation({x})));
        for (unsigned int i = 0; i < distanceVec.size(); ++i) {
            sum += pow(distanceVec[i], 2);
        }
        return pow(sqrt(sum), 2);
    };

    std::tuple<T, T> searchRange;
    switch (decoder) {
        case Decoder::BERT_OPTIMIZER:
            searchRange = targetSomInstance->getDecodingBoundaries(winningIndex);
            return approximateLocalMin<T>(std::get<0>(searchRange), std::get<1>(searchRange), fn);
        case Decoder::NAIVE:
            break;
    }

}

#endif //SPICENET_CPP_SPICENET_H
