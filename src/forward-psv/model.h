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
    arma::field<arma::span> paramSpans = arma::field<arma::span>(8, 2);
    arma::span topRowY = arma::span(0, nz_domain / 2 - 1);
    arma::span bottomRowY = arma::span(nz_domain / 2, nz - 1);
    arma::span col1X = arma::span(0, nx_domain / 4 - 1 + np_boundary);
    arma::span col2X = arma::span((nx_domain / 4) * 1 + np_boundary, (nx_domain / 4) * 2 + np_boundary - 1);
    arma::span col3X = arma::span((nx_domain / 4) * 2 + np_boundary, (nx_domain / 4) * 3 + np_boundary - 1);
    arma::span col4X = arma::span((nx_domain / 4) * 3 + np_boundary, (nx_domain / 4) * 4 + np_boundary * 2 - 1);

    // Interior of domain (excluding boundary layer)
    arma::span interiorX = arma::span(np_boundary, np_boundary + nx_domain - 1);
    arma::span interiorZ = arma::span(0, nz_domain - 1);

    // Constructors
    model();

    // Methods
    static arma::vec fieldToVec(arma::mat &mat);

    static arma::mat vecToField(arma::vec &vec);

    void updateFields(arma::vec lambda, arma::vec mu, arma::vec lightness);
};


#endif //HMC_FORWARD_FD_MODEL_H
