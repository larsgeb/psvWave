//
// Created by lars on 17.03.18.
//

#ifndef HMC_FORWARD_FD_MODEL_H
#define HMC_FORWARD_FD_MODEL_H

#include <armadillo>

class model {
public:
    // Fields
    const double dx = 1.249;
    const double dz = 1.249;
    const arma::uword nx_domain = 400;
    const arma::uword nz_domain = 200;
    const arma::uword np_boundary = 50;
    const double np_factor = 0.0075;
    const arma::uword nx = nx_domain + 2 * np_boundary;
    const arma::uword nz = nz_domain + np_boundary;
    arma::mat la = 4000000000 * arma::ones(nx, nz);
    arma::mat mu = 1000000000 * arma::ones(nx, nz);
    arma::mat lm = la + 2.0 * mu;
    arma::mat b_vx = (1.0 / 1500.0) * arma::ones(nx, nz);
    arma::mat b_vz = b_vx;
    arma::vec model_vector;

    // Constructors
    model();

    // Methods
    static arma::vec fieldToVec(arma::mat &mat);
    static arma::mat vecToField(arma::vec &vec);
};


#endif //HMC_FORWARD_FD_MODEL_H
