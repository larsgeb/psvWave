//
// Created by lars on 17.03.18.
//

#include "model.h"
#include <armadillo>

model::model(arma::uword hor, arma::uword ver) {

    // Create parametrization, TODO rework this for genral mxn discretization
    arma::span topRowY = arma::span(0, nz_domain / 2 - 1);
    arma::span bottomRowY = arma::span(nz_domain / 2, nz - 1);
    arma::span col1X = arma::span(0, nx_domain / 4 - 1 + np_boundary);
    arma::span col2X = arma::span((nx_domain / 4) * 1 + np_boundary, (nx_domain / 4) * 2 + np_boundary - 1);
    arma::span col3X = arma::span((nx_domain / 4) * 2 + np_boundary, (nx_domain / 4) * 3 + np_boundary - 1);
    arma::span col4X = arma::span((nx_domain / 4) * 3 + np_boundary, (nx_domain / 4) * 4 + np_boundary * 2 - 1);

    parametrizationB(0, 0) = col1X;
    parametrizationB(1, 0) = col1X;
    parametrizationB(2, 0) = col2X;
    parametrizationB(3, 0) = col2X;
    parametrizationB(4, 0) = col3X;
    parametrizationB(5, 0) = col3X;
    parametrizationB(6, 0) = col4X;
    parametrizationB(7, 0) = col4X;

    parametrizationB(0, 1) = topRowY;
    parametrizationB(1, 1) = bottomRowY;
    parametrizationB(2, 1) = topRowY;
    parametrizationB(3, 1) = bottomRowY;
    parametrizationB(4, 1) = topRowY;
    parametrizationB(5, 1) = bottomRowY;
    parametrizationB(6, 1) = topRowY;
    parametrizationB(7, 1) = bottomRowY;

    // Another parametrization
    topRowY = arma::span(0, nz_domain / 2 - 1);
    bottomRowY = arma::span(nz_domain / 2, nz_domain - 1);
    col1X = arma::span(0, nx_domain / 4 - 1);
    col2X = arma::span((nx_domain / 4) * 1, (nx_domain / 4) * 2 - 1);
    col3X = arma::span((nx_domain / 4) * 2, (nx_domain / 4) * 3 - 1);
    col4X = arma::span((nx_domain / 4) * 3, (nx_domain / 4) * 4 - 1);

    parametrizationI(0, 0) = col1X;
    parametrizationI(1, 0) = col1X;
    parametrizationI(2, 0) = col2X;
    parametrizationI(3, 0) = col2X;
    parametrizationI(4, 0) = col3X;
    parametrizationI(5, 0) = col3X;
    parametrizationI(6, 0) = col4X;
    parametrizationI(7, 0) = col4X;

    parametrizationI(0, 1) = topRowY;
    parametrizationI(1, 1) = bottomRowY;
    parametrizationI(2, 1) = topRowY;
    parametrizationI(3, 1) = bottomRowY;
    parametrizationI(4, 1) = topRowY;
    parametrizationI(5, 1) = bottomRowY;
    parametrizationI(6, 1) = topRowY;
    parametrizationI(7, 1) = bottomRowY;

}

void model::updateFields(arma::vec &_density, arma::vec &_lambda, arma::vec &_mu) {

    if (_lambda.n_elem != parametrizationB.n_rows or _mu.n_elem != parametrizationB.n_rows or
        _density.n_elem != parametrizationB.n_rows) {
        throw std::invalid_argument("Trying to update the model with a wrong number of parameters!");
    }

    for (arma::uword iP = 0; iP < parametrizationB.n_rows; ++iP) {
        auto subnx = parametrizationB(iP, 0).b - parametrizationB(iP, 0).a + 1;
        auto subnz = parametrizationB(iP, 1).b - parametrizationB(iP, 1).a + 1;
        la(parametrizationB(iP, 0), parametrizationB(iP, 1)) = _lambda(iP) * arma::ones(subnx, subnz);
        mu(parametrizationB(iP, 0), parametrizationB(iP, 1)) = (_mu(iP)) * arma::ones(subnx, subnz);
        lm(parametrizationB(iP, 0), parametrizationB(iP, 1)) = (_lambda(iP) + 2 * _mu(iP)) * arma::ones(subnx, subnz);
        b_vx(parametrizationB(iP, 0), parametrizationB(iP, 1)) = (1.0 / _density(iP)) * arma::ones(subnx, subnz);
        b_vz(parametrizationB(iP, 0), parametrizationB(iP, 1)) = (1.0 / _density(iP)) * arma::ones(subnx, subnz);
    }
}

model::model() {
    arma::span topRowY = arma::span(0, nz_domain / 2 - 1);
    arma::span bottomRowY = arma::span(nz_domain / 2, nz - 1);
    arma::span col1X = arma::span(0, nx_domain / 4 - 1 + np_boundary);
    arma::span col2X = arma::span((nx_domain / 4) * 1 + np_boundary, (nx_domain / 4) * 2 + np_boundary - 1);
    arma::span col3X = arma::span((nx_domain / 4) * 2 + np_boundary, (nx_domain / 4) * 3 + np_boundary - 1);
    arma::span col4X = arma::span((nx_domain / 4) * 3 + np_boundary, (nx_domain / 4) * 4 + np_boundary * 2 - 1);

    parametrizationB(0, 0) = col1X;
    parametrizationB(1, 0) = col1X;
    parametrizationB(2, 0) = col2X;
    parametrizationB(3, 0) = col2X;
    parametrizationB(4, 0) = col3X;
    parametrizationB(5, 0) = col3X;
    parametrizationB(6, 0) = col4X;
    parametrizationB(7, 0) = col4X;

    parametrizationB(0, 1) = topRowY;
    parametrizationB(1, 1) = bottomRowY;
    parametrizationB(2, 1) = topRowY;
    parametrizationB(3, 1) = bottomRowY;
    parametrizationB(4, 1) = topRowY;
    parametrizationB(5, 1) = bottomRowY;
    parametrizationB(6, 1) = topRowY;
    parametrizationB(7, 1) = bottomRowY;

    // Another parametrization
    topRowY = arma::span(0, nz_domain / 2 - 1);
    bottomRowY = arma::span(nz_domain / 2, nz_domain - 1);
    col1X = arma::span(0, nx_domain / 4 - 1);
    col2X = arma::span((nx_domain / 4) * 1, (nx_domain / 4) * 2 - 1);
    col3X = arma::span((nx_domain / 4) * 2, (nx_domain / 4) * 3 - 1);
    col4X = arma::span((nx_domain / 4) * 3, (nx_domain / 4) * 4 - 1);

    parametrizationI(0, 0) = col1X;
    parametrizationI(1, 0) = col1X;
    parametrizationI(2, 0) = col2X;
    parametrizationI(3, 0) = col2X;
    parametrizationI(4, 0) = col3X;
    parametrizationI(5, 0) = col3X;
    parametrizationI(6, 0) = col4X;
    parametrizationI(7, 0) = col4X;

    parametrizationI(0, 1) = topRowY;
    parametrizationI(1, 1) = bottomRowY;
    parametrizationI(2, 1) = topRowY;
    parametrizationI(3, 1) = bottomRowY;
    parametrizationI(4, 1) = topRowY;
    parametrizationI(5, 1) = bottomRowY;
    parametrizationI(6, 1) = topRowY;
    parametrizationI(7, 1) = bottomRowY;

}
