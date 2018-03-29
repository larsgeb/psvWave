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
    double dt;
    int nt;

    // FWI parameters
    int snapshotInterval = 10;
    double misfit;
    arma::vec misfitGradient;

    // Constructors
    experiment(arma::imat _receivers, arma::imat _sources, arma::vec _sourceFunction);

    // Methods
    void writeShots();

    void forwardData();

    void calculateAdjointSources();

    double calculateMisfit();

    arma::vec calculateMisfitGradient();

    void backwardData();

    void computeKernel();
};


#endif //HMC_FORWARD_FD_SETUP_H
