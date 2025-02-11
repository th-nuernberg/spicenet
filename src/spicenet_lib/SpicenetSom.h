//
// Created by Fabian Stiegler on 23.11.2024.
//


#ifndef SPICENET_SPICENETSOM_H
#define SPICENET_SPICENETSOM_H

#include <vector>
#include <string>
#include <cmath>
#include <cstdint>
#include <array>
#include <list>
#include <algorithm>
#include <sstream>
#include <stdexcept>

#include "SpicenetSomNode.h"
#include "SpicenetLearningRateFunction.h"
#include "SpicenetSomShape.h"
#include "SpicenetSomBase.h"

template<typename T, size_t D>
class SpicenetSom : public SpicenetSomBase<T> {
    static_assert(std::is_integral<T>::value || std::is_floating_point<T>::value,
                  "SpicenetSom: T must be a numeric type!");
private:
    SpicenetSomNode<T, D> *nodes;
    size_t nodesCount;
    Lrf tuningCurveLrf;
    Lrf interactionLrf;
    bool isSelfManaged;
    uint64_t trainingIteration = 0;

public:
    // TODO: rename tuning curve lrf to ?
    SpicenetSom(T valueRangeStart,
                T valueRangeEnd,
                size_t neuronCount,
                Lrf tuningCurveLrf,
                Lrf interactionLrf);

    SpicenetSom(SpicenetSomNode<T, D> *nodes,
                size_t nodesCount,
                Lrf tuningCurveLrf,
                Lrf interactionLrf);

    ~SpicenetSom();

    SpicenetSomNode<T, D> *getNodes(size_t &size);

    SpicenetSomShape getShape();

    uint64_t getTrainingIterations();

    std::string toTableString();

    void fit(const std::list<std::vector<T> > &trainingData, uint16_t epochs) override;

    std::tuple<T, T> getDecodingBoundaries(unsigned int index) override;

    std::vector<T> activation(const T (&data)[D]);

    std::vector<T> activation(const std::vector<T> &data) override;
};

template<typename T, size_t D>
std::tuple<T, T> SpicenetSom<T, D>::getDecodingBoundaries(unsigned int index) {
    if (D != 1){
        throw std::invalid_argument("SpicenetSom getDecodingBoundaries: this som has to many dimensions for decoding");
    }
    unsigned int startIndex, endIndex;
    startIndex = index > 0 ? index - 1 : index;
    endIndex = index < this->nodesCount ? index + 1 : index;
    return std::tuple<T, T>(this->nodes[startIndex].preferredValue[0], this->nodes[endIndex].preferredValue[0]);
}

template<typename T, size_t D>
SpicenetSomShape SpicenetSom<T, D>::getShape() {
    return {.nodes = this->nodesCount, .dimensions = D};
}

template<typename T, size_t D>
uint64_t SpicenetSom<T, D>::getTrainingIterations() {
    return this->trainingIteration;
}

template<typename T, size_t D>
SpicenetSom<T, D>::~SpicenetSom() {
    if (this->isSelfManaged) {
        delete[] nodes;
    }
}

template<typename T, size_t D>
SpicenetSom<T, D>::SpicenetSom(T valueRangeStart,
                               T valueRangeEnd,
                               const size_t neuronCount,
                               const Lrf tuningCurveLrf,
                               const Lrf interactionLrf): nodes(new SpicenetSomNode<T, D>[neuronCount]),
                                                          nodesCount(neuronCount),
                                                          tuningCurveLrf(tuningCurveLrf),
                                                          interactionLrf(interactionLrf),
                                                          isSelfManaged(true) {
    T stepSize = (valueRangeEnd - valueRangeStart) / neuronCount;
    T pos = valueRangeStart + stepSize / 2.0;
    for (unsigned int n = 0; n < nodesCount; ++n) {
        for (unsigned int d = 0; d < D; ++d) {
            this->nodes[n].preferredValue[d] = pos;
        }
        pos += stepSize;
    }
}

template<typename T, size_t D>
SpicenetSom<T, D>::SpicenetSom(SpicenetSomNode<T, D> *nodes,
                               const size_t nodesCount,
                               const Lrf tuningCurveLrf,
                               const Lrf interactionLrf): nodes(nodes),
                                                          nodesCount(nodesCount),
                                                          tuningCurveLrf(tuningCurveLrf),
                                                          interactionLrf(interactionLrf),
                                                          isSelfManaged(false) {
}


template<typename T, size_t D>
std::string SpicenetSom<T, D>::toTableString() {
    for (int i = 0; i < nodesCount; ++i) {
    }
    return "";
}

template<typename T, size_t D>
void SpicenetSom<T, D>::fit(const std::list<std::vector<T> > &trainingData, const uint16_t epochs) {
    T currentData[D];
    for (auto &data: trainingData) {
        if (data.size() != D) {
            std::stringstream ss;
            ss << "SpicenetSom fit: data size mismatch, a vector has " << data.size()
               << " values instead of the expected " << D;
            throw std::invalid_argument(ss.str());
        }
    }

    for (uint16_t i = 0; i < epochs; ++i) {
        for (auto &data: trainingData) {
            memcpy(currentData, data.data(), sizeof(currentData));
            auto activation = this->activation(currentData);
            auto maxElement = std::max_element(activation.begin(), activation.end());
            const size_t indexMaxElement = std::distance(activation.begin(), maxElement);

            for (unsigned int j = 0; j < this->nodesCount; ++j) {
                this->nodes[j].updateNode(currentData,
                                          this->tuningCurveLrf(this->trainingIteration),
                                          this->interactionLrf(this->trainingIteration),
                                          abs(static_cast<int>(j) - static_cast<int>(indexMaxElement)));
            }
            ++this->trainingIteration;
        }
    }
}

template<typename T, size_t D>
std::vector<T> SpicenetSom<T, D>::activation(const std::vector<T> &data) {
    if (data.size() != D) {
        throw std::invalid_argument("SpicenetSom activation: input data size unequal to dimensions");
    }
    T inputArr[D];
    std::copy(data.begin(), data.end(), inputArr); // TODO: nötig?
    return activation(inputArr);
}

template<typename T, size_t D>
std::vector<T> SpicenetSom<T, D>::activation(const T (&data)[D]) {
    std::vector<T> result(nodesCount);
    for (unsigned int i = 0; i < this->nodesCount; ++i) {
        result.at(i) = nodes[i].nodeActivation(data);
    }
    return result;
}

template<typename T, size_t D>
SpicenetSomNode<T, D> *SpicenetSom<T, D>::getNodes(size_t &size) {
    size = this->nodesCount;
    return this->nodes;
}

#endif //SPICENET_SPICENETSOM_H
