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
    arma::mat forwardData_vx;
    arma::mat forwardData_vz;
    double dt;
    int nt;
    arma::vec adjointSource;

    arma::mat boundaryRecVxTop;
    arma::mat boundaryRecVzTop;
    arma::mat boundaryRecTxxTop;
    arma::mat boundaryRecTzzTop;
    arma::mat boundaryRecTxzTop;

    arma::mat boundaryRecVxBottom;
    arma::mat boundaryRecVzBottom;
    arma::mat boundaryRecTxxBottom;
    arma::mat boundaryRecTzzBottom;
    arma::mat boundaryRecTxzBottom;

    arma::mat boundaryRecVxLeft;
    arma::mat boundaryRecVzLeft;
    arma::mat boundaryRecTxxLeft;
    arma::mat boundaryRecTzzLeft;
    arma::mat boundaryRecTxzLeft;

    arma::mat boundaryRecVxRight;
    arma::mat boundaryRecVzRight;
    arma::mat boundaryRecTxxRight;
    arma::mat boundaryRecTzzRight;
    arma::mat boundaryRecTxzRight;

    // Constructor
    shot(arma::irowvec source, arma::imat &_receivers, arma::vec &_sourceFunction, int nt, double dt, model &_model);

    // Methods
    void writeShot(const std::string& filename);
};


#endif //HMC_FORWARD_FD_SHOT_H
