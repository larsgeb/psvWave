//
// Created by lars on 17.03.18.
//
#include <armadillo>
#include "model.h"
#include "propagator.h"
#include "setup.h"

void propagator::propagate(model &_currentModel, bool accTraces, bool accFields, setup &setup) {
    std::cout << "P-wave speed: " << sqrt(_currentModel.lm.max() * _currentModel.b.max()) << std::endl;
    std::cout << "S-wave speed: " << sqrt(_currentModel.mu.max() * _currentModel.b.max()) << std::endl;
    std::cout << "Stability number: " << sqrt(_currentModel.lm.max() * _currentModel.b.max()) * dt *
                                         sqrt(1.0 / (_currentModel.dx * _currentModel.dx) +
                                              1.0 / (_currentModel.dz * _currentModel.dz)) << std::endl;

    double dx = _currentModel.dx;
    const arma::uword double nx = _currentModel.nx;
    double dz = _currentModel.dz;
    const arma::uword nz = _currentModel.nz;


    // dynamic fields
    arma::mat vx = arma::zeros(_currentModel.nx, _currentModel.nz);
    arma::mat vz = arma::zeros(_currentModel.nx, _currentModel.nz);
    arma::mat txx = arma::zeros(_currentModel.nx, _currentModel.nz);
    arma::mat tzz = arma::zeros(_currentModel.nx, _currentModel.nz);
    arma::mat txz = arma::zeros(_currentModel.nx, _currentModel.nz);

    if (accFields) {
        // we might need to save the fields in every time step, for that we need 3d matrices, or cubes. Pre-allocation
        // seems wise.
        arma::cube acc_vx(nx, nz, static_cast<const arma::uword>(nt));
        arma::cube acc_vz(nx, nz, static_cast<const arma::uword>(nt));
        arma::cube acc_txx(nx, nz, static_cast<const arma::uword>(nt));
        arma::cube acc_tzz(nx, nz, static_cast<const arma::uword>(nt));
        arma::cube acc_txz(nx, nz, static_cast<const arma::uword>(nt));
    }

    if (accTraces) {
        // we might need to record traces, so we need accumulators for that
        arma::mat acc_trace_vx;
        arma::mat acc_trace_vz;
        arma::mat acc_trace_txx;
        arma::mat acc_trace_tzz;
        arma::mat acc_trace_txz;
    }
}