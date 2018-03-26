//
// Created by lars on 26/03/18.
//

#ifndef HMC_FORWARD_FD_SHOT_H
#define HMC_FORWARD_FD_SHOT_H


#include <armadillo>

class shot {
public:
    // Fields
    arma::Row<arma::sword> source;
    arma::imat receivers;
    arma::vec sourceFunction;
    arma::mat forwardData_vx;
    arma::mat forwardData_vz;
//    arma::vec adjointSource;

    // Constructor
    shot(arma::Row<arma::sword> source, arma::imat &_receivers, arma::vec &_sourceFunction, int nt);

    // Methods
    void writeShot(const std::string& filename);
};


#endif //HMC_FORWARD_FD_SHOT_H
