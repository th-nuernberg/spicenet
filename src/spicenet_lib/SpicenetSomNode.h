//
// Created by Fabian Stiegler on 23.11.2024.
//

#ifndef SPICENET_SPICENETSOMNODE_H
#define SPICENET_SPICENETSOMNODE_H

#define M_PI		3.14159265358979323846
#include "string"

template<typename T, size_t D>
T distanceBetweenPoints(const T (&tip)[D], const T (&tail)[D]);

template<typename T, size_t D>
struct SpicenetSomNode {
    static_assert(std::is_integral<T>::value || std::is_floating_point<T>::value,
                  "SpicenetSomNode: T must be a numeric type!");
    T tuningCurveWidth = 0.001;
    T preferredValue[D];

    void updateNode(const T (&inputValue)[D],
                    T learnRate,
                    T interactionKernelLearningRate,
                    uint16_t distanceToWinner);

    T nodeActivation(const T (&data)[D]) const;

    std::string toTableString() const;
};

template<typename T, size_t D>
std::string SpicenetSomNode<T, D>::toTableString() const {
    auto prefStr = std::string();
    for (int i = 0; i < D; ++i) {
        prefStr += std::to_string(this->preferredValue[i]);
        if (i < D - 1) {
            prefStr += ", ";
        }
    }
    return "| " + std::to_string(this->tuningCurveWidth) + " | [" + prefStr + "] |";
}

template<typename T, size_t D>
void SpicenetSomNode<T, D>::updateNode(
        const T (&inputValue)[D],
        const T learnRate,
        const T interactionKernelLearningRate,
        const uint16_t distanceToWinner) {
    T interactionKernelValue = exp((-pow(distanceToWinner, 2)) / (2 * pow(interactionKernelLearningRate, 2)));
    // T oldPreferredValue[D];
    // memcpy(oldPreferredValue, this->preferredValue, sizeof(T[D]));
    for (unsigned int i = 0; i < D; i++) {
        this->preferredValue[i] +=
                learnRate * interactionKernelValue * (inputValue[i] - this->preferredValue[i]);
    }

    this->tuningCurveWidth += learnRate * interactionKernelValue * (
            pow(distanceBetweenPoints<T, D>(inputValue, this->preferredValue), 2)
            - pow(this->tuningCurveWidth, 2)
    );
}

template<typename T, size_t D>
T SpicenetSomNode<T, D>::nodeActivation(const T (&data)[D]) const {
    return (1.0 / (sqrt(2.0 * static_cast<T>(M_PI)) * this->tuningCurveWidth))
           *
           exp((-pow(distanceBetweenPoints<T, D>(data, this->preferredValue), 2)) /
               (2.0 * pow(this->tuningCurveWidth, 2)));
}

template<typename T, size_t D>
inline T distanceBetweenPoints(const T (&tip)[D], const T (&tail)[D]) {
    T sum = 0;
    for (unsigned int i = 0; i < D; ++i) {
        sum += pow(tip[i] - tail[i], 2);
    }
    return sqrt(sum);
}

#endif //SPICENET_SPICENETSOMNODE_H
