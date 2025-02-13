//
// Created by Fabian on 20.01.2025.
//

#ifndef SPICENET_CPP_SPICENETSOMBASE_H
#define SPICENET_CPP_SPICENETSOMBASE_H


#include <vector>
#include <cstdint>
#include <list>
#include <cstdio>

template<typename T>
class SpicenetSomBase {
    static_assert(std::is_integral<T>::value || std::is_floating_point<T>::value,
                  "SpicenetSomBase: T must be a numeric type!");
public:
    virtual bool fit(const std::list<std::vector<T> > &trainingData, uint16_t epochs) = 0;

    virtual std::vector<T> activation(const std::vector<T> &data) = 0;
    virtual bool tryGetDecodingBoundaries(unsigned int index, T &start, T &end) = 0;
    virtual ~SpicenetSomBase();
};


template<typename T>
SpicenetSomBase<T>::~SpicenetSomBase() {

}
/*
template<typename T>
std::vector<T> SpicenetSomBase<T>::activation(const std::vector<T> &data) {
    return std::vector<T>();
}

template<typename T>
bool SpicenetSomBase<T>::fit(const std::list<std::vector<T>> &trainingData, const uint16_t epochs) {

}
 */

#endif //SPICENET_CPP_SPICENETSOMBASE_H
