//
// Created by Fabian Stiegler on 16.01.2025.
//

#ifndef SPICENET_CPP_MATRIXITERATOR_H
#define SPICENET_CPP_MATRIXITERATOR_H

#include <vector>
#include <cstdint>

template<typename T>
class MatrixIterator {
    static_assert(std::is_integral<T>::value || std::is_floating_point<T>::value,
                  "MatrixIterator: T must be a numeric type!");
private:
    T *currentPosition;
    unsigned int remainingSteps;
    std::vector<uint16_t> currentIndex;
    const std::vector<uint16_t> shape;
public:
    MatrixIterator(std::vector<uint16_t> shape, T *start, unsigned int size);

    bool next(std::vector<uint16_t> &index, T *&position);
};

template<typename T>
MatrixIterator<T>::MatrixIterator(const std::vector<uint16_t> shape, T *start, unsigned int size):
        currentPosition(start),
        remainingSteps(size),
        currentIndex(shape.size(), 0),
        shape(shape) {
}

template<typename T>
bool MatrixIterator<T>::next(std::vector<uint16_t> &index, T *&position) {
    if (this->remainingSteps == 0) {
        return false;
    }

    index.assign(currentIndex.begin(), currentIndex.end());
    position = currentPosition;
    ++currentPosition;
    --this->remainingSteps;
    if (remainingSteps != 0) {
        for (int i = this->shape.size() - 1; i >= 0; --i) {
            // for (int i = 0; i < this->shape.size(); ++i) {
            currentIndex[i] = (currentIndex[i] + 1) % shape[i];
            if (currentIndex[i] != 0) {
                break;
            }
        }
    }
    return true;
}

#endif //SPICENET_CPP_MATRIXITERATOR_H
