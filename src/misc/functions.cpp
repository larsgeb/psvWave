//
// Created by lars on 03/04/18.
//

#include "functions.h"

arma::mat generateGaussian(arma::uword nx, arma::uword nz, double std_dist, arma::uword ix, arma::uword iz) {
    arma::mat xcoor(nx, nz);
    xcoor.each_col() = arma::linspace<arma::dcolvec>(0, nx - 1, nx);
    arma::mat zcoor(nx, nz);
    zcoor.each_row() = arma::linspace<arma::rowvec>(0, nz - 1, nz);
    arma::mat gauss(nx, nz);
    gauss = arma::exp(-(arma::square(xcoor - ix) / (std_dist * 2) + arma::square(zcoor - iz) / (std_dist * 2)));
    return gauss;

}

arma::vec generateRicker(double dt, int nt, double freq) {
    double centralFrequency = freq;
    double tsource = 1.0 / centralFrequency;
    arma::vec time = arma::linspace(0, dt * (nt - 1), static_cast<const arma::uword>(nt));
    double t0 = tsource * 1.5;
    arma::vec tau = M_PI * (time - t0) / t0;
    return (1 - 4 * tau % tau) % arma::exp(-2.0 * tau % tau);
}