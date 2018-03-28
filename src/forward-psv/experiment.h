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
    double tTot;
    double misfit;
    arma::vec misfitGradient;

    // Constructors
    experiment(arma::imat _receivers, arma::imat _sources, arma::vec _sourceFunction, int nt, double dt);

    // Methods
    void writeShots();

    void forwardData();

    arma::vec getModelVector();

    arma::vec getDataVector();

    double getMisfit();

    double calculateMisfit();

    arma::vec getMisfitGradient();

    arma::vec calculateMisfitGradient();

    void backwardData();
};


#endif //HMC_FORWARD_FD_SETUP_H
