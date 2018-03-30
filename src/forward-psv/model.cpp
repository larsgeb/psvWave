//
// Created by lars on 17.03.18.
//

#include "model.h"
#include <armadillo>

model::model() {
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

void model::updateFields(arma::vec _lambda, arma::vec _mu, arma::vec _density) {

    if (_lambda.n_elem != paramSpans.n_rows or _mu.n_elem != paramSpans.n_rows or _density.n_elem != paramSpans.n_rows){
        throw std::invalid_argument("Trying to update the model with a wrong number of parameters!");
    }

    for (arma::uword iP = 0; iP < paramSpans.n_rows; ++iP) {
        auto subnx = paramSpans(iP, 0).b - paramSpans(iP, 0).a + 1;
        auto subnz = paramSpans(iP, 1).b - paramSpans(iP, 1).a + 1;
        la(paramSpans(iP, 0), paramSpans(iP, 1)) = _lambda(iP) * arma::ones(subnx, subnz);
        mu(paramSpans(iP, 0), paramSpans(iP, 1)) = (_mu(iP)) * arma::ones(subnx, subnz);
        lm(paramSpans(iP, 0), paramSpans(iP, 1)) = (_lambda(iP) + 2 * _mu(iP)) * arma::ones(subnx, subnz);
        b_vx(paramSpans(iP, 0), paramSpans(iP, 1)) = (1.0/_density(iP)) * arma::ones(subnx, subnz);
        b_vz(paramSpans(iP, 0), paramSpans(iP, 1)) = (1.0/_density(iP)) * arma::ones(subnx, subnz);
    }
}
