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

    // Initialize static fields
    arma::mat la = arma::mat(nx,nz);
    arma::mat mu = arma::mat(nx,nz);
    arma::mat lm = arma::mat(nx,nz);
    arma::mat b_vx = arma::mat(nx,nz);
    arma::mat b_vz = arma::mat(nx,nz);

    // Stuff about the parametrization
    arma::field<arma::span> parametrizationB = arma::field<arma::span>(8, 2);
    arma::field<arma::span> parametrizationI = arma::field<arma::span>(8, 2);

    // Interior of domain (excluding boundary layer)
    arma::span interiorX = arma::span(np_boundary, np_boundary + nx_domain - 1);
    arma::span interiorZ = arma::span(0, nz_domain - 1);

    // Constructors
    model();

    void updateFields(arma::mat &_density, arma::mat &_lambda, arma::mat &_mu);
};


#endif //HMC_FORWARD_FD_MODEL_H
