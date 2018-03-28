//
// Created by lars on 26/03/18.
//

#ifndef HMC_FORWARD_FD_SHOT_H
#define HMC_FORWARD_FD_SHOT_H


#include <armadillo>
#include "model.h"

class shot {
public:
    // Fields
    arma::irowvec source;
    arma::imat receivers;
    arma::vec sourceFunction;
    arma::mat seismogramSyn_ux;
    arma::mat seismogramSyn_uz;
    double dt;
    int nt;

    arma::mat seismogramObs_ux;
    arma::mat seismogramObs_uz;

    // Constructor
    shot(arma::irowvec source, arma::imat &_receivers, arma::vec &_sourceFunction, int nt, double dt, model &_model);

    // Methods
    void writeShot(const std::string& filename);
};


#endif //HMC_FORWARD_FD_SHOT_H
