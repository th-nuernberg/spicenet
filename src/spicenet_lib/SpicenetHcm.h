//
// Created by Fabian Stiegler on 02.01.2025.
//

#ifndef SPICENETHCM_H
#define SPICENETHCM_H

#include "SpicenetLearningRateFunction.h"
#include "SpicenetSom.h"
#include "Matrix.h"
#include "VectorMath.h"
#include "SpicenetLogging.h"
#ifdef SPICENET_LOGGING
#include <Arduino.h>
#endif

template<typename T>
class SpicenetHcm {
    static_assert(std::is_integral<T>::value || std::is_floating_point<T>::value,
                  "SpicenetHcm: T must be a numeric type!");
private:
    Matrix<T> *weights;
    std::vector<std::vector<T> *> activationBars;
    Lrf weightsLrf;
    Lrf trustOfNewLrf;
    uint64_t trainingIteration = 0;
    uint8_t somCount;

    void createActivationBars(const std::vector<uint16_t> &shape);

public:
    SpicenetHcm(const std::vector<uint16_t> &shape, Lrf weightsLrf, Lrf trustOfNewLrf);

    SpicenetHcm(T *weights, const std::vector<uint16_t> &shape, Lrf weightsLrf, Lrf trustOfNewLrf);

    ~SpicenetHcm();

    bool fit(const std::list<std::list<std::vector<T>>> &inputActivations, uint16_t epochs);

    Matrix<T> *getMatrix();

    std::vector<T> calculateShouldPattern(uint8_t targetSom,
                                          const std::map<uint8_t, std::vector<T> > &inputActivations);
};

template<typename T>
SpicenetHcm<T>::SpicenetHcm(T *weights, const std::vector<uint16_t> &shape, Lrf weightsLrf, Lrf trustOfNewLrf):
        weightsLrf(weightsLrf),
        trustOfNewLrf(trustOfNewLrf),
        somCount(shape.size()) {
    this->weights = new Matrix<T>(shape, weights);
    createActivationBars(shape);
}

template<typename T>
SpicenetHcm<T>::SpicenetHcm(const std::vector<uint16_t> &shape, Lrf weightsLrf, Lrf trustOfNewLrf) :
        weightsLrf(weightsLrf),
        trustOfNewLrf(trustOfNewLrf),
        somCount(shape.size()) {
    this->weights = new Matrix<T>(shape, 1, nullptr);
    createActivationBars(shape);
}

template<typename T>
SpicenetHcm<T>::~SpicenetHcm() {
    delete this->weights;
    for (auto v: this->activationBars) {
        delete v;
    }
}

template<typename T>
void SpicenetHcm<T>::createActivationBars(const std::vector<uint16_t> &shape) {
    for (auto &d: shape) {
        this->activationBars.push_back(new std::vector<T>(d, 0));
    }
}

template<typename T>
Matrix<T> *SpicenetHcm<T>::getMatrix() {
    return this->weights;
}

template<typename T>
std::vector<T> SpicenetHcm<T>::calculateShouldPattern(uint8_t targetSom,
                                                      const std::map<uint8_t, std::vector<T>> &inputActivations) {
    for (uint8_t i = 0; i < somCount; ++i) {
        if (i == targetSom) {
            continue;
        }
        if (inputActivations.find(i) == inputActivations.end()) {
            std::stringstream ss;
            ss << "SpicenetHcm calculateShouldPattern: for dimension " << std::to_string(i)
               << " is no activation given";
#ifdef SPICENET_LOGGING
            LOG_LN(ss.str().c_str());
#endif
            return {};
        }
    }
    std::vector<T> result(this->weights->getShape().at(targetSom), 0);

    auto matrixIterator = this->weights->getIterator();
    std::vector<uint16_t> index;
    T *weight;
    while (matrixIterator.next(index, weight)) {
        T temp = 1;
        // auto activationTupel = inputActivations.at(index.at(targetSom));
        for (auto &activationTupel: inputActivations) {
            temp *= activationTupel.second.at(index.at(activationTupel.first));
        }
        result.at(index.at(targetSom)) += (*weight) * temp;
        // result.at(0) += (*weight) * temp;
    }

    return result;
}


template<typename T>
bool SpicenetHcm<T>::fit(const std::list<std::list<std::vector<T>>> &inputActivations, uint16_t epochs) {
    // TODO: switch to iterators instead of lists
    if (inputActivations.size() != this->somCount) {
#ifdef SPICENET_LOGGING
        LOG_LN("SpicenetHcm fit: the amount of activation lists is unequal to the amount of soms");
#endif
        return false;
    }
    auto it = inputActivations.begin();
    unsigned int dataCount = it->size();
    for (++it; it != inputActivations.end(); ++it) {
        if (it->size() != dataCount) {
#ifdef SPICENET_LOGGING
            LOG_LN("SpicenetHcm fit: not all lists contain an equal amount of activations");
#endif
            return false;
        }
    }
    if (dataCount <= 0) {
        return true;
    }

    std::vector<std::vector<T>> tempsDiffBarActivation(this->somCount);
    for (unsigned int epoch = 0; epoch < epochs; ++epoch) {
        std::vector<typename std::list<std::vector<T>>::const_iterator> colIterators(inputActivations.size());
        int t = 0;
        for (const std::list<std::vector<T>> &col: inputActivations) {
            colIterators.at(t) = col.begin();
            ++t;
        }
        for (unsigned int row = 0; row < dataCount; ++row) {
            auto colIt = inputActivations.begin();

            for (unsigned int col = 0; col < this->somCount; ++col) {
                std::vector<T> *activationBar = this->activationBars.at(col);
                std::vector<T> currentActivation(*(colIterators.at(col)));
                std::vector<T> aBarTemp = vectorAddition<T>(
                        vectorScalarMultiplication<T>((*activationBar),
                                                      (1.0 - this->trustOfNewLrf(this->trainingIteration))),
                        vectorScalarMultiplication<T>(currentActivation,
                                                      this->trustOfNewLrf(this->trainingIteration)));
                activationBar->assign(aBarTemp.begin(), aBarTemp.end());
                tempsDiffBarActivation.at(col) = vectorSubtraction<T>(currentActivation, *activationBar);
                ++colIt;
            }

            // Iterate over the matrix
            MatrixIterator<T> weightIt = this->weights->getIterator();
            std::vector<uint16_t> index;
            T *weight;
            while (weightIt.next(index, weight)) {
                T delta = 1;
                for (unsigned int i = 0; i < index.size(); ++i) {
                    delta *= tempsDiffBarActivation.at(i).at(index.at(i));
                }
                *weight += this->weightsLrf(this->trainingIteration) * delta;
            }

            for (auto &iterator: colIterators) {
                std::advance(iterator, 1);
            }
            ++(this->trainingIteration);
        }
    }
    return true;
}

#endif //SPICENETHCM_H
