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

    de = _density;
    b_vx = 1.0 / _density;
    b_vz = b_vx;
    la = _lambda;
    mu = _mu;
    lm = _lambda + 2 * _mu;
}

void fwiModel::updateInnerFieldsElastic(mat &_density, mat &_lambda, mat &_mu) {

    if (_density.n_rows != nx_interior or _density.n_cols != nz_interior or _lambda.n_rows != nx_interior or
        _lambda.n_cols != nz_interior
        or _mu.n_rows != nx_interior or _mu.n_cols != nz_interior) {
        throw std::invalid_argument("Dimension of updated fields is not equal to domain.");
    }

    b_vx(interiorX, interiorZ) = 1.0 / _density;
    extendFields(b_vx);
    b_vz = b_vx;
    de = _density;
    extendFields(de);

    la(interiorX, interiorZ) = _lambda;
    mu(interiorX, interiorZ) = _mu;
    extendFields(la);
    extendFields(mu);

    lm = la + 2 * mu;

    calculateVelocityFields();

}

void fwiModel::updateInnerFieldsVelocity(mat &_density, mat &_vp, mat &_vs) {

    if (_density.n_rows != nx_interior or _density.n_cols != nz_interior or _vp.n_rows != nx_interior or
        _vp.n_cols != nz_interior or _vs.n_rows != nx_interior or _vs.n_cols != nz_interior) {
        throw std::invalid_argument("Dimension of updated fields is not equal to domain.");
    }

    b_vx(interiorX, interiorZ) = 1.0 / _density;
    extendFields(b_vx);
    b_vz = b_vx;
    de(interiorX, interiorZ) = _density;
    extendFields(de);

    vp(interiorX, interiorZ) = _vp;
    vs(interiorX, interiorZ) = _vs;
    extendFields(vp);
    extendFields(vs);

    calculateElasticFields();
}

void fwiModel::setTimestepAuto(double _targetCourant) {
    dt = _targetCourant * dx / (static_cast<double>(sqrt(lm.max() * b_vx.max())) * static_cast<double>(sqrt(2.0)));
    nt = static_cast<int>(ceil(samplingTime / dt));
}

void fwiModel::setTimestep(double _dt) {
    dt = _dt;
    nt = static_cast<int>(ceil(samplingTime / dt));

    if ((dt * (static_cast<double>(sqrt(lm.max() * b_vx.max())) * static_cast<double>(sqrt(2.0))) / dx) >= 1) {
        std::cout << "Courant criterion is NOT met!" << std::endl;
    }
}

void fwiModel::setTime(double _dt, int _nt, double _t) {
    dt = _dt;
    nt = _nt;
    samplingTime = _t;

    if ((dt * (static_cast<double>(sqrt(lm.max() * b_vx.max())) * static_cast<double>(sqrt(2.0))) / dx) >= 1) {
        std::cout << "Courant criterion is NOT met!" << std::endl;
    }
}

void fwiModel::extendFields(mat &_outer, mat &_inner) {
    // lower
    _outer(span(np_boundary, nx_interior + np_boundary - 1), span(nz_interior, nz - 1)) =
            repmat(_inner(span::all, nz_interior - 1), 1, np_boundary);
    // left
    _outer(span(0, np_boundary - 1), span(0, nz_interior - 1)) = repmat(_inner(0, span::all), np_boundary, 1);
    // right
    _outer(span(np_boundary + nx_interior, nx - 1), span(0, nz_interior - 1)) =
            repmat(_inner(nx_interior - 1, span::all), np_boundary, 1);
    // lower-left
    _outer(span(0, np_boundary - 1), span(nz_interior, nz - 1)) = _outer(np_boundary, nz_interior - 1);
    // lower-right
    _outer(span(np_boundary + nx_interior), span(nz_interior, nz - 1)) = _outer(np_boundary + nx_interior - 1, nz_interior - 1);
}

void fwiModel::extendFields(mat &_outer) {
    // lower
    _outer(span(np_boundary, nx_interior + np_boundary - 1), span(nz_interior, nz - 1)) =
            repmat(_outer(span(np_boundary, nx_interior + np_boundary - 1), nz_interior - 1), 1, np_boundary);
    // left (complete)
    _outer(span(0, np_boundary - 1), span::all) =
            repmat(_outer(np_boundary, span::all), np_boundary, 1);
    // right (complete)
    _outer(span(np_boundary + nx_interior, nx - 1), span::all) =
            repmat(_outer(np_boundary + nx_interior - 1, span::all), np_boundary, 1);
}

fwiModel::fwiModel() {
    b_vx = arma::ones(nx, nz);
    de = arma::ones(nx, nz);
    b_vz = arma::ones(nx, nz);
    la = arma::ones(nx, nz);
    mu = arma::ones(nx, nz);
    lm = arma::ones(nx, nz);
}

void fwiModel::calculateVelocityFields() {
    // TODO evaluate validity in staggered grid
    vp = sqrt((la + 2 * mu) % b_vx);
    vs = sqrt(mu % b_vx);
}

void fwiModel::calculateElasticFields() {
    // TODO evaluate validity in staggered grid
    mu = square(vs) / b_vx;
    lm = square(vp) / b_vx;
    la = lm - 2 * mu;
}

double fwiModel::get_dt() {
    return this->dt;
}

int fwiModel::get_nt() {
    return this->nt;
}

double fwiModel::get_samplingTime() {
    return this->samplingTime;
}

double fwiModel::get_targetCourant() {
    return targetCourant;
}
