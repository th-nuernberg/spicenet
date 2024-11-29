//
// Created by Fabian on 23.11.2024.
//


#ifndef SPICENET_SPICENETSOM_H
#define SPICENET_SPICENETSOM_H

#include <vector>

#include "SpicenetSomNode.h"

class SpicenetSom {
private:
    SpicenetSomNode *data = nullptr;
    int dimensions = 0;

    float nodeActivation(const SpicenetSomNode &node);

public:
    std::string toString();

    void fit(const float *data);

    std::vector<float> activation(const float *data);
};


#endif //SPICENET_SPICENETSOM_H
