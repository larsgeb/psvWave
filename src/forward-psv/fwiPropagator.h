//
// Created by lars on 17.03.18.
//

#ifndef HMC_FORWARD_FD_PROPAGATOR_H
#define HMC_FORWARD_FD_PROPAGATOR_H


#include <armadillo>
#include "fwiModel.h"
#include "fwiShot.h"

class fwiPropagator {
public:
    constexpr static double coeff1 = 9.0 / 8.0;
    constexpr static double coeff2 = 1.0 / 24.0;

    // static propagators
    static void propagateForward(fwiModel &_currentModel, fwiShot &_shot);

    static void propagateAdjoint(fwiModel &_currentModel, fwiShot &_shot, arma::mat &_denistyKernel,
                                 arma::mat &_muKernel, arma::mat &_lambdaKernel);
};


#endif //HMC_FORWARD_FD_PROPAGATOR_H
