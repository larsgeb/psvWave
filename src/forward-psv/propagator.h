//
// Created by lars on 17.03.18.
//

#ifndef HMC_FORWARD_FD_PROPAGATOR_H
#define HMC_FORWARD_FD_PROPAGATOR_H


#include <armadillo>
#include "model.h"
#include "shot.h"

class propagator {
public:
    constexpr static double coeff1 = 9.0 / 8.0;
    constexpr static double coeff2 = 1.0 / 24.0;

    // static propagators
    static void propagateForward(model &_currentModel, shot &_shot, bool storeWavefield, int n);

};


#endif //HMC_FORWARD_FD_PROPAGATOR_H
