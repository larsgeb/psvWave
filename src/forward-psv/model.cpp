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

    paramSpans(0, 0) = col1X;
    paramSpans(1, 0) = col1X;
    paramSpans(2, 0) = col2X;
    paramSpans(3, 0) = col2X;
    paramSpans(4, 0) = col3X;
    paramSpans(5, 0) = col3X;
    paramSpans(6, 0) = col4X;
    paramSpans(7, 0) = col4X;

    paramSpans(0, 1) = topRowY;
    paramSpans(1, 1) = bottomRowY;
    paramSpans(2, 1) = topRowY;
    paramSpans(3, 1) = bottomRowY;
    paramSpans(4, 1) = topRowY;
    paramSpans(5, 1) = bottomRowY;
    paramSpans(6, 1) = topRowY;
    paramSpans(7, 1) = bottomRowY;
}

arma::mat model::vecToField(arma::vec &vec) {
    // TODO
    return arma::mat();
}

arma::vec model::fieldToVec(arma::mat &mat) {
    return arma::vectorise(mat);
}

void model::updateFields(arma::vec _lambda, arma::vec _mu, arma::vec _lightness) {

    if (_lambda.n_elem != paramSpans.n_rows or _mu.n_elem != paramSpans.n_rows or _lightness.n_elem != paramSpans.n_rows){
        throw std::invalid_argument("Trying to update the model with a wrong number of parameters!");
    }

    for (arma::uword iP = 0; iP < paramSpans.n_rows; ++iP) {
        auto subnx = paramSpans(iP, 0).b - paramSpans(iP, 0).a + 1;
        auto subnz = paramSpans(iP, 1).b - paramSpans(iP, 1).a + 1;
        la(paramSpans(iP, 0), paramSpans(iP, 1)) = _lambda(iP) * arma::ones(subnx, subnz);
        mu(paramSpans(iP, 0), paramSpans(iP, 1)) = (_mu(iP)) * arma::ones(subnx, subnz);
        lm(paramSpans(iP, 0), paramSpans(iP, 1)) = (_lambda(iP) + 2 * _mu(iP)) * arma::ones(subnx, subnz);
        b_vx(paramSpans(iP, 0), paramSpans(iP, 1)) = (_lightness(iP)) * arma::ones(subnx, subnz);
        b_vz(paramSpans(iP, 0), paramSpans(iP, 1)) = (_lightness(iP)) * arma::ones(subnx, subnz);
    }
}
