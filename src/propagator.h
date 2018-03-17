//
// Created by lars on 17.03.18.
//

#ifndef HMC_FORWARD_FD_PROPAGATOR_H
#define HMC_FORWARD_FD_PROPAGATOR_H


#include <armadillo>
#include "model.h"

class propagator {
public:
    const double dt = 0.00025;
    double coeff1 = 9.0 / 8.0;
    double coeff2 = 1.0 / 24.0;
    const int nt = 3000;

    static void propagate(model &_currentModel);

};


#endif //HMC_FORWARD_FD_PROPAGATOR_H
