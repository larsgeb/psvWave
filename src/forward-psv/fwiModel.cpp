//
// Created by lars on 17.03.18.
//

#include "fwiModel.h"
#include <armadillo>

using namespace arma;

void fwiModel::updateFields(mat &_density, mat &_lambda, mat &_mu) {

    if (_density.n_rows != nx or _density.n_cols != nz or _lambda.n_rows != nx or _lambda.n_cols != nz
        or _mu.n_rows != nx or _mu.n_cols != nz) {
        throw std::invalid_argument("Dimension of updated fields is not equal to domain + non-reflecting boundary.");
    }

    b_vx = 1.0 / _density;
    b_vz = b_vx;
    la = _lambda;
    mu = _mu;
    lm = _lambda + 2 * _mu;
}

void fwiModel::updateInnerFields(mat &_density, mat &_lambda, mat &_mu) {

    if (_density.n_rows != nx_domain or _density.n_cols != nz_domain or _lambda.n_rows != nx_domain or
        _lambda.n_cols != nz_domain
        or _mu.n_rows != nx_domain or _mu.n_cols != nz_domain) {
        throw std::invalid_argument("Dimension of updated fields is not equal to domain.");
    }

    b_vx(interiorX, interiorZ) = 1.0 / _density;
    extendFields(b_vx);
    b_vz = b_vx;

    la(interiorX, interiorZ) = _lambda;
    mu(interiorX, interiorZ) = _mu;
    extendFields(la);
    extendFields(mu);

    lm = la + 2 * mu;

    setTimestepAuto(targetCourant);
}

void fwiModel::setTimestepAuto(double _targetCourant) {
    dt = _targetCourant * dx / (static_cast<double>(sqrt(lm.max() * b_vx.max())) * static_cast<double>(sqrt(2.0)));
    nt = static_cast<int>(ceil(samplingTime / dt));
}

void fwiModel::setTimestep(double _dt) {
    dt = _dt;
    nt = static_cast<int>(ceil(samplingTime / dt));

    if((dt * (static_cast<double>(sqrt(lm.max() * b_vx.max())) * static_cast<double>(sqrt(2.0))) / dx) >= 1){
        std::cout << "Courant criterion is NOT met!" << std::endl;
    }
}

void fwiModel::extendFields(mat &_outer, mat &_inner) {
    // lower
    _outer(span(np_boundary, nx_domain + np_boundary - 1), span(nz_domain, nz - 1)) =
            repmat(_inner(span::all, nz_domain - 1), 1, np_boundary);
    // left
    _outer(span(0, np_boundary - 1), span(0, nz_domain - 1)) = repmat(_inner(0, span::all), np_boundary, 1);
    // right
    _outer(span(np_boundary + nx_domain, nx - 1), span(0, nz_domain - 1)) =
            repmat(_inner(nx_domain - 1, span::all), np_boundary, 1);
    // lower-left
    _outer(span(0, np_boundary - 1), span(nz_domain, nz - 1)) = _outer(np_boundary, nz_domain - 1);
    // lower-right
    _outer(span(np_boundary + nx_domain), span(nz_domain, nz - 1)) = _outer(np_boundary + nx_domain - 1, nz_domain - 1);
}

void fwiModel::extendFields(mat &_outer) {
    // lower
    _outer(span(np_boundary, nx_domain + np_boundary - 1), span(nz_domain, nz - 1)) =
            repmat(_outer(span(np_boundary, nx_domain + np_boundary - 1), nz_domain - 1), 1, np_boundary);
    // left (complete)
    _outer(span(0, np_boundary - 1), span::all) =
            repmat(_outer(np_boundary, span::all), np_boundary, 1);
    // right (complete)
    _outer(span(np_boundary + nx_domain, nx - 1), span::all) =
            repmat(_outer(np_boundary + nx_domain - 1, span::all), np_boundary, 1);
}

fwiModel::fwiModel() {
    b_vx = arma::ones(nx, nz);
    b_vz = arma::ones(nx, nz);
    la = arma::ones(nx, nz);
    mu = arma::ones(nx, nz);
    lm = arma::ones(nx, nz);
}
