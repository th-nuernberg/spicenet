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
#include <map>

#include "SpicenetSomBase.h"
#include "SpicenetHcm.h"
#include "Optimizer.h"
#include "SpicenetLogging.h"

#ifdef SPICENET_LOGGING

#include <Arduino.h>

#endif


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
    bool fit(const std::list<std::list<std::vector<T>>> &inputData,
             uint16_t epochsOnBatch,
             void (*iterationCallback)(uint64_t iteration) = nullptr);

    bool tryDecode(uint8_t targetSom, const std::map<uint8_t, std::vector<T>> &inputValues, T &result,
                   Decoder decoder = BERT_OPTIMIZER);
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
bool Spicenet<T, D>::fit(const std::list<std::list<std::vector<T>>> &inputData,
                         uint16_t epochsOnBatch,
                         void (*iterationCallback)(uint64_t)) {
    if (inputData.size() != D) {
#ifdef SPICENET_LOGGING
        LOG_LN("Spicenet fit: the amount of data columns is unequal to the amount of soms");
#endif
        return false;
    }
    auto it = inputData.begin();
    unsigned int dataCount = it->size();
    for (++it; it != inputData.end(); ++it) {
        if (it->size() != dataCount) {
#ifdef SPICENET_LOGGING
            LOG_LN("Spicenet fit: not all lists contain an equal amount of data");
#endif
            return false;
        }
    }
    if (dataCount <= 0) {
        return true;
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
#ifdef SPICENET_LOGGING
        LOG_LN("");
        LOG_LN("Start HCM training");
        unsigned int iteration = 0;
#endif
        for (unsigned int row = 0; row < dataCount; ++row) {
            std::list<std::list<std::vector<T>>> activations;
            somNumber = 0;
            for (auto &iterator: colIterators) {
                std::list<std::vector<T>> stuff;
                stuff.push_back(soms.at(somNumber)->activation(*iterator));
                activations.push_back(stuff);
                std::advance(iterator, 1);
                ++somNumber;
            }

            this->hcm->fit(activations, 1);
#ifdef SPICENET_LOGGING
            LOG("SOM trainings iterations");
            LOG("HCM fit data: ");
            LOG(iteration);
            LOG('\r');
            ++iteration;
#endif
        }
    }
#ifdef SPICENET_LOGGING
    LOG_LN("");
#endif
    return true;
}

template<typename T, uint8_t D>
bool Spicenet<T, D>::tryDecode(uint8_t targetSom, const std::map<uint8_t, std::vector<T>> &inputValues, T &result,
                               Decoder decoder) {
    // TODO: hier auch validieren?
    for (uint8_t i = 0; i < this->soms.size(); ++i) {
        if (i == targetSom) {
            continue;
        }
        if (inputValues.find(i) == inputValues.end()) {
#ifdef SPICENET_LOGGING
            std::stringstream ss;
            ss << "Spicenet tryDecode: for som " << std::to_string(i) << " is no activation given";
            LOG_LN(ss.str().c_str());
#endif
            return false;
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

    std::vector<T> shouldActivationNormed = normVector(this->hcm->calculateShouldPattern(targetSom, inputActivations));
    int winningIndex = std::distance(shouldActivationNormed.begin(),
                                     std::max_element(shouldActivationNormed.begin(), shouldActivationNormed.end()));

    SpicenetSomBase<T> *targetSomInstance = this->soms.at(targetSom);

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
            T start, end;
            if (targetSomInstance->tryGetDecodingBoundaries(winningIndex, start, end)) {
                result = approximateLocalMin<T>(start, end, fn);
                return true;
            }
            return false;
        case Decoder::NAIVE:
            return false;
    }

}

#endif //SPICENET_CPP_SPICENET_H
