//
// Created by lars on 17.03.18.
//

#include "model.h"
#include <armadillo>

model::model() {
    arma::mat la = 4000000000 * arma::ones(nx, nz);
    arma::mat mu = 1000000000 * arma::ones(nx, nz);
    arma::mat lm = la + 2 * mu;
    arma::mat b_vx = (1.0 / 1500.0) * arma::ones(nx, nz);
    arma::mat b_vz = b_vx;
}

arma::mat model::vecToField(arma::vec &vec) {
    // TODO
    return arma::mat();
}

arma::vec model::fieldToVec(arma::mat &mat) {
    return arma::vectorise(mat);
}
