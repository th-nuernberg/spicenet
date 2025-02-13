//
// Created by Fabian Stiegler on 13.01.2025.
//

#ifndef SPICENET_CPP_MATRIX_H
#define SPICENET_CPP_MATRIX_H

#include <cstdint>
#include <vector>
#include <stdexcept>
#include <sstream>
#include "MatrixIterator.h"

template<typename T>
class Matrix {
private:
    bool selfManaged = false;
    T *data;
    std::vector<uint16_t> shape;
    unsigned int size = 1;

    size_t indexToOffset(const std::vector<uint16_t> &indizes);

    bool checkShapeBounds(const std::vector<uint16_t> &indizes);

public:
    ~Matrix();

    Matrix(const std::vector<uint16_t> &shape, T *data = nullptr);

    Matrix(const std::vector<uint16_t> &shape, T initValue, T *data = nullptr);

    MatrixIterator<T> getIterator();

    std::vector<uint16_t> getShape();

    T *rawData(unsigned int &matrixSize);

    T getField(const std::vector<uint16_t> &indizes);

    bool setField(const std::vector<uint16_t> &indizes, T value);
};

template<typename T>
std::vector<uint16_t> Matrix<T>::getShape() {
    return this->shape;
}

template<typename T>
T *Matrix<T>::rawData(unsigned int &matrixSize) {
    matrixSize = this->size;
    return data;
}

template<typename T>
MatrixIterator<T> Matrix<T>::getIterator() {
    return MatrixIterator<T>(this->shape, this->data, this->size);
}

template<typename T>
Matrix<T>::Matrix(const std::vector<uint16_t> &shape, T *data) {
    // TODO: change data to vector for memory safety
    if (shape.empty()) {
#ifdef SPICENET_LOGGING
        LOG_LN("Warning: No values in matrix shape");
#endif
    }
    this->shape = std::vector<uint16_t>(shape);
    for (auto dimSize: shape) {
        this->size *= dimSize;
    }
    if (data == nullptr) {
        this->selfManaged = true;
        this->data = new T[this->size];
    } else {
        this->data = data;
    }
}

template<typename T>
Matrix<T>::Matrix(const std::vector<uint16_t> &shape, T initValue, T *data): Matrix(shape, data) {
    std::fill(this->data, this->data + size, initValue);
}

template<typename T>
Matrix<T>::~Matrix() {
    if (selfManaged) {
        delete this->data;
    }
}

template<typename T>
T Matrix<T>::getField(const std::vector<uint16_t> &indizes) {
    if (!checkShapeBounds(indizes)){
        return std::numeric_limits<T>::max();
    }
    return this->data[indexToOffset(indizes)];
}

template<typename T>
bool Matrix<T>::setField(const std::vector<uint16_t> &indizes, T value) {
    if (!checkShapeBounds(indizes)){
        return false;
    }
    this->data[indexToOffset(indizes)] = value;
    return true;
}

template<typename T>
size_t Matrix<T>::indexToOffset(const std::vector<uint16_t> &indizes) {
    size_t offset = 0;
    size_t multiplier = 1;

    for (int i = this->shape.size() - 1; i >= 0; --i) {
        offset += indizes[i] * multiplier;
        multiplier *= this->shape[i];
    }

    return offset;
}

template<typename T>
bool Matrix<T>::checkShapeBounds(const std::vector<uint16_t> &indizes) {
    if (indizes.size() != this->shape.size()) {
#ifdef SPICENET_LOGGING
        LOG_LN("The shape of the indizes not matching the matrix");
#endif
        return false;
    }
    for (int i = 0; i < this->shape.size(); ++i) {
        if (indizes.at(i) >= shape.at(i)) {
#ifdef SPICENET_LOGGING
            std::stringstream ss;
            ss << "Illegal index in dimension " << std::to_string(i);
            LOG_LN(ss.str().c_str());
#endif
            return false;
        }
    }
    return true;
}
/*
template<typename T>
std::ostream &operator<<(std::ostream &outs, const Matrix<T> &matrix){
    return outs << "[" << matrix. << "," << matrix.y << ")";
}
 */
#endif //SPICENET_CPP_MATRIX_H
