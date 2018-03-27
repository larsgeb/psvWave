//
// Created by lars on 26/03/18.
//

#ifndef HMC_FORWARD_FD_SHOT_H
#define HMC_FORWARD_FD_SHOT_H


#include <armadillo>

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
//    arma::vec adjointSource;

    // Constructor
    shot(arma::irowvec source, arma::imat &_receivers, arma::vec &_sourceFunction, int nt, double dt);

    // Methods
    void writeShot(const std::string& filename);
};


#endif //HMC_FORWARD_FD_SHOT_H
