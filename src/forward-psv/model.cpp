//
// Created by lars on 17.03.18.
//

#include "model.h"
#include <armadillo>


void model::updateFields(arma::mat &_density, arma::mat &_lambda, arma::mat &_mu) {

    b_vx = 1.0 / _density;
    b_vz = b_vx;
    la = _lambda;
    mu = _mu;
    lm = _lambda + 2 * _mu;
}

model::model() {}
