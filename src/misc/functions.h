//
// Created by lars on 03/04/18.
//

#ifndef HMC_FORWARD_FD_FUNCTIONS_H
#define HMC_FORWARD_FD_FUNCTIONS_H

#include <armadillo>

arma::mat generateGaussian(arma::uword nx, arma::uword nz, double std_dist, arma::uword ix, arma::uword iz);

arma::vec generateRicker(double dt, int nt, double freq);

#endif //HMC_FORWARD_FD_FUNCTIONS_H
