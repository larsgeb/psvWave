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
    static void propagate(model &_currentModel, bool accTraces, bool accFields, arma::imat &receivers,
                          arma::imat &sources, arma::mat &sourceFunctions, arma::mat &data_obj_vx,
                          arma::mat &data_obj_vz, int nt, double dt, bool sameSource);
};


#endif //HMC_FORWARD_FD_PROPAGATOR_H
