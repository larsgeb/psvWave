//
// Created by lars on 17.03.18.
//
#include <armadillo>
#include "model.h"
#include "propagator.h"
#include "experiment.h"

void propagator::propagate(model &_currentModel, bool accTraces, bool accFields, arma::imat &receivers,
                           arma::imat &sources, arma::mat &sourceFunctions, arma::mat &data_obj_vx,
                           arma::mat &data_obj_vz, int nt, double dt, bool sameSource) {

    // Some standard output
//    std::cout << "P-wave speed: " << sqrt(_currentModel.lm.max() * _currentModel.b_vx.max()) << std::endl;
//    std::cout << "S-wave speed: " << sqrt(_currentModel.mu.max() * _currentModel.b_vx.max()) << std::endl;
    std::cout << "    Stability number: " << sqrt(_currentModel.lm.max() * _currentModel.b_vx.max()) * dt *
                                         sqrt(1.0 / (_currentModel.dx * _currentModel.dx) +
                                              1.0 / (_currentModel.dz * _currentModel.dz)) << std::endl;

    // Loading simulation parameters
    double dx = _currentModel.dx;
    const arma::uword nx = _currentModel.nx;
    double dz = _currentModel.dz;
    const arma::uword nz = _currentModel.nz;

    // Create dynamic fields
    arma::mat vx = arma::zeros(_currentModel.nx, _currentModel.nz);
    arma::mat vz = arma::zeros(_currentModel.nx, _currentModel.nz);
    arma::mat txx = arma::zeros(_currentModel.nx, _currentModel.nz);
    arma::mat tzz = arma::zeros(_currentModel.nx, _currentModel.nz);
    arma::mat txz = arma::zeros(_currentModel.nx, _currentModel.nz);

    // Create accumulation fields.
    if (accFields) {
        // We might need to save the fields in every time step, for that we need 3d matrices, or cubes. Pre-allocation
        // seems wise.
        arma::cube acc_vx(nx, nz, static_cast<const arma::uword>(nt - 1));
        arma::cube acc_vz(nx, nz, static_cast<const arma::uword>(nt - 1));
        arma::cube acc_txx(nx, nz, static_cast<const arma::uword>(nt - 1));
        arma::cube acc_tzz(nx, nz, static_cast<const arma::uword>(nt - 1));
        arma::cube acc_txz(nx, nz, static_cast<const arma::uword>(nt - 1));
    }
    if (accTraces) {
        // We might need to record traces, so we need accumulators for that as 2d matrices.
        arma::mat acc_trace_vx;
        arma::mat acc_trace_vz;
        arma::mat acc_trace_txx;
        arma::mat acc_trace_tzz;
        arma::mat acc_trace_txz;
    }

    // Create taper matrix
    arma::mat taper = _currentModel.np_boundary * arma::ones(nx, nz);
    for (int iTaper = 0; iTaper < _currentModel.np_boundary; ++iTaper) {
        taper.submat(iTaper, 0, nx - iTaper - 1, nz - iTaper - 1) =
                1 + iTaper * arma::ones(nx - 2 * iTaper, nz - iTaper);
    }
    taper = arma::exp(-arma::square(_currentModel.np_factor * (_currentModel.np_boundary - taper)));


    // Time marching
    for (int it = 0; it < nt - 1; ++it) {
//        std::cout << it << std::endl;
        // Inject source
        for (int source = 0; source < sources.n_rows; ++source) {
            int ix = sources.row(source)[0];
            int iz = sources.row(source)[1];

            if (sameSource){
                txx(ix + _currentModel.np_boundary, iz) += 0.5 * dt * sourceFunctions[it];
                tzz(ix + _currentModel.np_boundary, iz) += 0.5 * dt * sourceFunctions[it];
            }else{
                txx(ix + _currentModel.np_boundary, iz) += 0.5 * dt * sourceFunctions.row(source)[it];
                tzz(ix + _currentModel.np_boundary, iz) += 0.5 * dt * sourceFunctions.row(source)[it];
            }
        }

        // Record wavefield
        for (int receiver = 0; receiver < receivers.n_rows; ++receiver) {
//            int ix = receivers.row(receiver)[0];
//            int iz = receivers.row(receiver)[1];
//
//            data_obj_vx(receiver, it) = vx(ix + _currentModel.np_boundary, iz);
//            data_obj_vz(receiver, it) = vz(ix + _currentModel.np_boundary, iz);
        }

#pragma omp parallel
#pragma omp for
        for (int ix = 0; ix < nx; ++ix) {
            for (int iz = 0; iz < nz; ++iz) {
                if (iz > 1 and ix > 1 and ix < nx - 1 and iz < nz - 1) {
                    txx(ix, iz) = taper(ix, iz) *
                                  (txx(ix, iz) +
                                   (dt * _currentModel.lm(ix, iz) * (
                                           -coeff2 * vx(ix + 1, iz - 1) + coeff1 * vx(ix, iz - 1)
                                           - coeff1 * vx(ix - 1, iz - 1) + coeff2 * vx(ix - 2, iz - 1)
                                   ) / dx +
                                    (dt * _currentModel.la(ix, iz)) * (
                                            -coeff2 * vz(ix - 1, iz + 1) + coeff1 * vz(ix - 1, iz)
                                            - coeff1 * vz(ix - 1, iz - 1) + coeff2 * vz(ix - 1, iz - 2)
                                    ) / dz));
                    tzz(ix, iz) = taper(ix, iz) *
                                  (tzz(ix, iz) +
                                   (dt * _currentModel.la(ix, iz) * (
                                           -coeff2 * vx(ix + 1, iz - 1) + coeff1 * vx(ix, iz - 1)
                                           - coeff1 * vx(ix - 1, iz - 1) + coeff2 * vx(ix - 2, iz - 1)
                                   ) / dx +
                                    (dt * _currentModel.lm(ix, iz)) * (
                                            -coeff2 * vz(ix - 1, iz + 1) + coeff1 * vz(ix - 1, iz)
                                            - coeff1 * vz(ix - 1, iz - 1) + coeff2 * vz(ix - 1, iz - 2)
                                    ) / dz));
                    txz(ix, iz) = taper(ix, iz) *
                                  (txz(ix, iz) + dt * _currentModel.mu(ix, iz) * (
                                          (
                                                  -coeff2 * vx(ix - 1, iz + 1) + coeff1 * vx(ix - 1, iz)
                                                  - coeff1 * vx(ix - 1, iz - 1) + coeff2 * vx(ix - 1, iz - 2)
                                          ) / dz +
                                          (
                                                  -coeff2 * vz(ix + 1, iz - 1) + coeff1 * vz(ix, iz - 1)
                                                  - coeff1 * vz(ix - 1, iz - 1) + coeff2 * vz(ix - 2, iz - 1)
                                          ) / dx));
                } else {
                    txx(ix, iz) = txx(ix, iz) * taper(ix, iz);
                    txz(ix, iz) = txz(ix, iz) * taper(ix, iz);
                    tzz(ix, iz) = tzz(ix, iz) * taper(ix, iz);
                }
            }
        }

#pragma omp parallel
#pragma omp for
        for (int ix = 0; ix < nx; ++ix) {
            for (int iz = 0; iz < nz; ++iz) {
                if (iz < nz - 2 and ix < nx - 2 and ix > 0 and iz > 0) {
                    vx(ix, iz) =
                            taper(ix, iz) *
                            (vx(ix, iz) + _currentModel.b_vx(ix, iz) *
                                          (dt * (
                                                  -coeff2 * txx(ix + 2, iz + 1) + coeff1 * txx(ix + 1, iz + 1)
                                                  - coeff1 * txx(ix, iz + 1) + coeff2 * txx(ix - 1, iz + 1)
                                          ) / dx +
                                           dt * (
                                                   -coeff2 * txz(ix + 1, iz + 2) + coeff1 * txz(ix + 1, iz + 1)
                                                   - coeff1 * txz(ix + 1, iz) + coeff2 * txz(ix + 1, iz - 1)
                                           ) / dz));
                    vz(ix, iz) =
                            taper(ix, iz) *
                            (vz(ix, iz) + _currentModel.b_vz(ix, iz) *
                                          (dt * (
                                                  -coeff2 * txz(ix + 2, iz + 1) + coeff1 * txz(ix + 1, iz + 1)
                                                  - coeff1 * txz(ix, iz + 1) + coeff2 * txz(ix - 1, iz + 1)
                                          ) / dx +
                                           dt * (
                                                   -coeff2 * tzz(ix + 1, iz + 2) + coeff1 * tzz(ix + 1, iz + 1)
                                                   - coeff1 * tzz(ix + 1, iz) + coeff2 * tzz(ix + 1, iz - 1)
                                           ) / dz));
                } else {
                    vx(ix, iz) = vx(ix, iz) * taper(ix, iz);
                    vz(ix, iz) = vz(ix, iz) * taper(ix, iz);
                }
            }
        }
    }
}