//
// Created by lars on 17.03.18.
//

#ifndef HMC_FORWARD_FD_SETUP_H
#define HMC_FORWARD_FD_SETUP_H

#include <armadillo>
#include "shot.h"
#include "model.h"

class experiment {
public:
    // Fields
    arma::imat receivers;
    arma::imat sources;
    arma::vec sourceFunction;
    model currentModel;
    std::vector<shot> shots;
    double timestep = 0.00025;
    int experimentSteps = 3000;
    double experimentTime = (experimentSteps - 1) * timestep;

    // Constructors
    experiment(arma::imat _receivers, arma::imat _sources, arma::vec _sourceFunction, int nt, double dt);

    // Methods
    // TODO method to calculate forward field of current model and store in the shot subobjects.
    void writeShots();

    void forwardData();
};


#endif //HMC_FORWARD_FD_SETUP_H
