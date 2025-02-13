//
// Created by Fabian Stiegler on 15.01.2025.
//

#ifndef SPICENET_CPP_VECTORMATH_H
#define SPICENET_CPP_VECTORMATH_H

#include <vector>
#include <stdexcept>

template<typename T>
inline std::vector<T> vectorScalarMultiplication(const std::vector<T> &vector, T scalar) {
    static_assert(std::is_integral<T>::value || std::is_floating_point<T>::value,
                  "vectorScalarMultiplication: T must be a numeric type!");
    std::vector<T> result(vector);
    for (unsigned int i = 0; i < vector.size(); ++i) {
        result[i] *= scalar;
    }
    return result;
}

template<typename T>
inline std::vector<T> vectorAddition(const std::vector<T> &a, const std::vector<T> &b) {
    static_assert(std::is_integral<T>::value || std::is_floating_point<T>::value,
                  "vectorAddition: T must be a numeric type!");
    if (a.size() != b.size()) {
#ifdef SPICENET_LOGGING
        LOG_LN("vectorAddition: both vectors need to have the same size");
#endif
        return {};
    }
    std::vector<T> result(a);
    for (unsigned int i = 0; i < b.size(); ++i) {
        result[i] += b[i];
    }
    return result;
}

/**
 * Makes the vector subtraction.
 * @tparam T A numeric type.
 * @param a first vector
 * @param b second vector
 * @return a - b
 */
template<typename T>
inline std::vector<T> vectorSubtraction(const std::vector<T> &a, const std::vector<T> &b) {
    static_assert(std::is_integral<T>::value || std::is_floating_point<T>::value,
                  "vectorSubtraction: T must be a numeric type!");
    if (a.size() != b.size()) {
#ifdef SPICENET_LOGGING
        LOG_LN("vectorSubtraction: both vectors need to have the same size");
#endif
        return {};
    }
    std::vector<T> result(a);
    for (unsigned int i = 0; i < b.size(); ++i) {
        result[i] -= b[i];
    }
    return result;
}

#endif //SPICENET_CPP_VECTORMATH_H
