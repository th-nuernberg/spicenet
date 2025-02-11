//
// Created by Fabian Stiegler on 29.11.2024.
//

#ifndef SPICENET_SPICENETLEARNINGRATEFUNCTION_H
#define SPICENET_SPICENETLEARNINGRATEFUNCTION_H

/**
 * Learning-Rate-Function this is supposed to give the developer control over the learning rate while training.
 */
typedef float (*Lrf)(uint64_t d);

#endif //SPICENET_SPICENETLEARNINGRATEFUNCTION_H
