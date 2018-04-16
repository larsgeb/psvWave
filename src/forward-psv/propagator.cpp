//
// Created by lars on 17.03.18.
//
#include <armadillo>
#include "model.h"
#include "propagator.h"
#include "experiment.h"

void propagator::propagateForward(model &_currentModel, shot &_shot) {

    // Some standard output
    if (1 < sqrt(_currentModel.lm.max() * _currentModel.b_vx.max()) * _shot.dt *
            sqrt(1.0 / (_currentModel.dx * _currentModel.dx) + 1.0 / (_currentModel.dz * _currentModel.dz))) {
        std::cout << "Max speed: " << sqrt(_currentModel.lm.max() * _currentModel.b_vx.max()) << std::endl
                  << "Max La+2mu: " << _currentModel.lm.max();
        throw std::invalid_argument("Warning! Numerical solution does not adhere to CFL-criterion");
    }

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

    // Create taper matrix
    arma::mat taper = _currentModel.np_boundary * arma::ones(nx, nz);
    for (arma::uword iTaper = 0; iTaper < _currentModel.np_boundary; ++iTaper) {
        taper.submat(iTaper, 0, nx - iTaper - 1, nz - iTaper - 1) =
                1 + iTaper * arma::ones(nx - 2 * iTaper, nz - iTaper);
    }
    taper = arma::exp(-arma::square(_currentModel.np_factor * (_currentModel.np_boundary - taper)));

    // Time marching through all time levels
    for (int it = 0; it < _shot.nt; ++it) {
        // Take snapshot of fields
        if (it % _shot.snapshotInterval == 0) {
            auto a = vx(_currentModel.interiorX, _currentModel.interiorZ);
            _shot.vxSnapshots.slice(it / _shot.snapshotInterval) = vx(_currentModel.interiorX, _currentModel.interiorZ);
            _shot.vzSnapshots.slice(it / _shot.snapshotInterval) = vz(_currentModel.interiorX, _currentModel.interiorZ);
            _shot.txxSnapshots.slice(it / _shot.snapshotInterval) = txx(_currentModel.interiorX,
                                                                        _currentModel.interiorZ);
            _shot.tzzSnapshots.slice(it / _shot.snapshotInterval) = tzz(_currentModel.interiorX,
                                                                        _currentModel.interiorZ);
            _shot.txzSnapshots.slice(it / _shot.snapshotInterval) = txz(_currentModel.interiorX,
                                                                        _currentModel.interiorZ);
        }

        // Record wavefield at receivers
#pragma omp parallel for collapse(1)
        for (arma::uword receiver = 0; receiver < _shot.receivers.n_rows; ++receiver) {
            int ix = _shot.receivers.row(receiver)[0] + _currentModel.np_boundary;
            int iz = _shot.receivers.row(receiver)[1];

            if (it == 0) {
                _shot.seismogramSyn_ux(receiver, it) = _shot.dt * vx(ix, iz) / (dx * dz);
                _shot.seismogramSyn_uz(receiver, it) = _shot.dt * vz(ix, iz) / (dx * dz);
            } else {
                _shot.seismogramSyn_ux(receiver, it) =
                        _shot.seismogramSyn_ux(receiver, it - 1) + _shot.dt * vx(ix, iz) / (dx * dz);
                _shot.seismogramSyn_uz(receiver, it) =
                        _shot.seismogramSyn_uz(receiver, it - 1) + _shot.dt * vz(ix, iz) / (dx * dz);
            }

        }

        // After this point is only integration, which doesn't have to be done at the last time level
        // Time integrate stress
#pragma omp parallel for collapse(1)
        for (arma::uword ix = 0; ix < nx; ++ix) {
            for (arma::uword iz = 0; iz < nz; ++iz) {
                if (ix > 1 and ix < nx - 2 and iz < nz - 2) {
                    txx(ix, iz) = taper(ix, iz) *
                                  (txx(ix, iz) +
                                   _shot.dt *
                                   (_currentModel.lm(ix, iz) * (
                                           coeff1 * (vx(ix + 1, iz) - vx(ix, iz)) +
                                           coeff2 * (vx(ix - 1, iz) - vx(ix + 2, iz))) / dx +
                                    _currentModel.la(ix, iz) * (
                                            coeff1 * (vz(ix, iz) - (iz > 0 ? vz(ix, iz - 1) : 0)) +
                                            coeff2 * ((iz > 1 ? vz(ix, iz - 2) : 0) - vz(ix, iz + 1))) / dz));
                    tzz(ix, iz) = taper(ix, iz) *
                                  (tzz(ix, iz) +
                                   _shot.dt *
                                   (_currentModel.la(ix, iz) * (
                                           coeff1 * (vx(ix + 1, iz) - vx(ix, iz)) +
                                           coeff2 * (vx(ix - 1, iz) - vx(ix + 2, iz))) / dx +
                                    (_currentModel.lm(ix, iz)) * (
                                            coeff1 * (vz(ix, iz) - (iz > 0 ? vz(ix, iz - 1) : 0)) +
                                            coeff2 * ((iz > 1 ? vz(ix, iz - 2) : 0) - vz(ix, iz + 1))) / dz));
                    txz(ix, iz) = taper(ix, iz) *
                                  (txz(ix, iz) + _shot.dt * _currentModel.mu(ix, iz) * (
                                          (coeff1 * (vx(ix, iz + 1) - vx(ix, iz)) +
                                           coeff2 * ((iz > 0 ? vx(ix, iz - 1) : 0) - vx(ix, iz + 2))) / dz +
                                          (coeff1 * (vz(ix, iz) - vz(ix - 1, iz)) +
                                           coeff2 * (vz(ix - 2, iz) - vz(ix + 1, iz))) / dx));
                } else {
                    txx(ix, iz) = txx(ix, iz) * taper(ix, iz);
                    txz(ix, iz) = txz(ix, iz) * taper(ix, iz);
                    tzz(ix, iz) = tzz(ix, iz) * taper(ix, iz);
                }
            }
        }

        // Time integrate velocity
#pragma omp parallel for collapse(1)
        for (arma::uword ix = 0; ix < nx; ++ix) {
            for (arma::uword iz = 0; iz < nz; ++iz) {
                if (iz < nz - 2 and ix < nx - 2 and ix > 1) {
                    vx(ix, iz) =
                            taper(ix, iz) *
                            (vx(ix, iz)
                             + _currentModel.b_vx(ix, iz) * _shot.dt * (
                                    (coeff1 * (txx(ix, iz) - txx(ix - 1, iz)) +
                                     coeff2 * (txx(ix - 2, iz) - txx(ix + 1, iz))) / dx +
                                    (coeff1 * (txz(ix, iz) - (iz > 0 ? txz(ix, iz - 1) : 0)) +
                                     coeff2 * ((iz > 1 ? txz(ix, iz - 2) : 0) - txz(ix, iz + 1))) / dz)
                            );
                    vz(ix, iz) =
                            taper(ix, iz) *
                            (vz(ix, iz)
                             + _currentModel.b_vz(ix, iz) * _shot.dt * (
                                    (coeff1 * (txz(ix + 1, iz) - txz(ix, iz)) +
                                     coeff2 * (txz(ix - 1, iz) - txz(ix + 2, iz))) / dx +
                                    (coeff1 * (tzz(ix, iz + 1) - tzz(ix, iz)) +
                                     coeff2 * ((iz > 0 ? tzz(ix, iz - 1) : 0) - tzz(ix, iz + 2))) / dz));
                } else {
                    vx(ix, iz) = vx(ix, iz) * taper(ix, iz);
                    vz(ix, iz) = vz(ix, iz) * taper(ix, iz);
                }
            }
        }

        // Inject explosive source
        for (arma::uword source = 0; source < _shot.source.n_rows; ++source) {
            int ix = _shot.source.row(source)[0] + _currentModel.np_boundary;
            int iz = _shot.source.row(source)[1];
            vx(ix, iz) += 0.5 * _shot.dt * _shot.sourceFunction[it] * _currentModel.b_vx(ix, iz) / (dx * dz);
            vz(ix, iz) += 0.5 * _shot.dt * _shot.sourceFunction[it] * _currentModel.b_vz(ix, iz) / (dx * dz);
        }

        // Print status bar
        if (it % (_shot.nt / 50) == 0) {
            char message[1024];
            sprintf(message, "\r \r    %i%%",
                    static_cast<int>(static_cast<double>(it) * 100.0 / static_cast<double>(_shot.nt)));
            std::cout << message << std::flush;
        }
    }

    std::cout << std::endl;
}


void propagator::propagateAdjoint(model &_currentModel, shot &_shot, arma::mat &_denistyKernel,
                                  arma::mat &_muKernel, arma::mat &_lambdaKernel) {

    // Some standard output
    if (1 < sqrt(_currentModel.lm.max() * _currentModel.b_vx.max()) * _shot.dt *
            sqrt(1.0 / (_currentModel.dx * _currentModel.dx) + 1.0 / (_currentModel.dz * _currentModel.dz))) {
        throw std::invalid_argument("Warning! Numerical solution does not adhere to CFL-criterion");
    }

    // Loading simulation parameters
    double dx = _currentModel.dx;
    const arma::sword nx = _currentModel.nx;
    double dz = _currentModel.dz;
    const arma::sword nz = _currentModel.nz;

    // Create dynamic fields
    arma::mat vx = arma::zeros(_currentModel.nx, _currentModel.nz);
    arma::mat vz = arma::zeros(_currentModel.nx, _currentModel.nz);
    arma::mat txx = arma::zeros(_currentModel.nx, _currentModel.nz);
    arma::mat tzz = arma::zeros(_currentModel.nx, _currentModel.nz);
    arma::mat txz = arma::zeros(_currentModel.nx, _currentModel.nz);

    // Create taper matrix
    arma::mat taper = _currentModel.np_boundary * arma::ones(nx, nz);
    for (arma::uword iTaper = 0; iTaper < _currentModel.np_boundary; ++iTaper) {
        taper.submat(iTaper, 0, nx - iTaper - 1, nz - iTaper - 1) =
                1 + iTaper * arma::ones(nx - 2 * iTaper, nz - iTaper);
    }
    taper = arma::exp(-arma::square(_currentModel.np_factor * (_currentModel.np_boundary - taper)));

    // Time marching through all time levels
    for (int it = _shot.nt - 1; it >= 0; --it) {

        // Compute correlation integral for the kernel at snapshots
        if (it % _shot.snapshotInterval == 0) {

            arma::mat f1 = _shot.vxSnapshots.slice(it / _shot.snapshotInterval) %
                           vx(_currentModel.interiorX, _currentModel.interiorZ);

            arma::mat f2 = _shot.vzSnapshots.slice(it / _shot.snapshotInterval) %
                           vz(_currentModel.interiorX, _currentModel.interiorZ);

            _denistyKernel -=
                    _shot.snapshotInterval * _shot.dt * (f1 + f2);

            // Compute strain
            arma::mat exxAdj = (txx(_currentModel.interiorX, _currentModel.interiorZ) -
                                (tzz(_currentModel.interiorX, _currentModel.interiorZ) %
                                 _currentModel.la(_currentModel.interiorX, _currentModel.interiorZ)) /
                                _currentModel.lm(_currentModel.interiorX, _currentModel.interiorZ)) /
                               (_currentModel.lm(_currentModel.interiorX, _currentModel.interiorZ) -
                                (arma::square(_currentModel.la(_currentModel.interiorX, _currentModel.interiorZ)) /
                                 (_currentModel.lm(_currentModel.interiorX, _currentModel.interiorZ))));
            arma::mat ezzAdj = (tzz(_currentModel.interiorX, _currentModel.interiorZ) -
                                (txx(_currentModel.interiorX, _currentModel.interiorZ) %
                                 _currentModel.la(_currentModel.interiorX, _currentModel.interiorZ)) /
                                _currentModel.lm(_currentModel.interiorX, _currentModel.interiorZ)) /
                               (_currentModel.lm(_currentModel.interiorX, _currentModel.interiorZ) -
                                (arma::square(_currentModel.la(_currentModel.interiorX, _currentModel.interiorZ)) /
                                 (_currentModel.lm(_currentModel.interiorX, _currentModel.interiorZ))));
            arma::mat exzAdj = txz(_currentModel.interiorX, _currentModel.interiorZ) /
                               (2 * _currentModel.mu(_currentModel.interiorX, _currentModel.interiorZ));

            arma::mat exx = (_shot.txxSnapshots.slice(it / _shot.snapshotInterval) -
                             (_shot.tzzSnapshots.slice(it / _shot.snapshotInterval) %
                              _currentModel.la(_currentModel.interiorX, _currentModel.interiorZ)) /
                             _currentModel.lm(_currentModel.interiorX, _currentModel.interiorZ)) /
                            (_currentModel.lm(_currentModel.interiorX, _currentModel.interiorZ) -
                             (arma::square(_currentModel.la(_currentModel.interiorX, _currentModel.interiorZ)) /
                              (_currentModel.lm(_currentModel.interiorX, _currentModel.interiorZ))));
            arma::mat ezz = (_shot.tzzSnapshots.slice(it / _shot.snapshotInterval) -
                             (_shot.txxSnapshots.slice(it / _shot.snapshotInterval) %
                              _currentModel.la(_currentModel.interiorX, _currentModel.interiorZ)) /
                             _currentModel.lm(_currentModel.interiorX, _currentModel.interiorZ)) /
                            (_currentModel.lm(_currentModel.interiorX, _currentModel.interiorZ) -
                             (arma::square(_currentModel.la(_currentModel.interiorX, _currentModel.interiorZ)) /
                              (_currentModel.lm(_currentModel.interiorX, _currentModel.interiorZ))));
            arma::mat exz = _shot.txzSnapshots.slice(it / _shot.snapshotInterval) /
                            (2 * _currentModel.mu(_currentModel.interiorX, _currentModel.interiorZ));

            _lambdaKernel += _shot.snapshotInterval * _shot.dt * ((exx + ezz) % (exxAdj + ezzAdj));
            _muKernel += _shot.snapshotInterval * _shot.dt * 2 * ((exxAdj % exx) + (ezzAdj % ezz) + 2 * (exzAdj % exz));
        }

        // After this point is only integration, which doesn't have to be done at the last time level
        // Time integrate stress
#pragma omp parallel for collapse(1)
        for (int ix = 0; ix < nx; ++ix) {
            for (int iz = 0; iz < nz; ++iz) {
                if (ix > 1 and ix < nx - 2 and iz < nz - 2) {
                    txx(ix, iz) = taper(ix, iz) *
                                  (txx(ix, iz) -
                                   _shot.dt *
                                   (_currentModel.lm(ix, iz) * (
                                           coeff1 * (vx(ix + 1, iz) - vx(ix, iz)) +
                                           coeff2 * (vx(ix - 1, iz) - vx(ix + 2, iz))) / dx +
                                    _currentModel.la(ix, iz) * (
                                            coeff1 * (vz(ix, iz) - (iz > 0 ? vz(ix, iz - 1) : 0)) +
                                            coeff2 * ((iz > 1 ? vz(ix, iz - 2) : 0) - vz(ix, iz + 1))) / dz));
                    tzz(ix, iz) = taper(ix, iz) *
                                  (tzz(ix, iz) -
                                   _shot.dt *
                                   (_currentModel.la(ix, iz) * (
                                           coeff1 * (vx(ix + 1, iz) - vx(ix, iz)) +
                                           coeff2 * (vx(ix - 1, iz) - vx(ix + 2, iz))) / dx +
                                    (_currentModel.lm(ix, iz)) * (
                                            coeff1 * (vz(ix, iz) - (iz > 0 ? vz(ix, iz - 1) : 0)) +
                                            coeff2 * ((iz > 1 ? vz(ix, iz - 2) : 0) - vz(ix, iz + 1))) / dz));
                    txz(ix, iz) = taper(ix, iz) *
                                  (txz(ix, iz) - _shot.dt * _currentModel.mu(ix, iz) * (
                                          (coeff1 * (vx(ix, iz + 1) - vx(ix, iz)) +
                                           coeff2 * ((iz > 0 ? vx(ix, iz - 1) : 0) - vx(ix, iz + 2))) / dz +
                                          (coeff1 * (vz(ix, iz) - vz(ix - 1, iz)) +
                                           coeff2 * (vz(ix - 2, iz) - vz(ix + 1, iz))) / dx));
                } else {
                    txx(ix, iz) = txx(ix, iz) * taper(ix, iz);
                    txz(ix, iz) = txz(ix, iz) * taper(ix, iz);
                    tzz(ix, iz) = tzz(ix, iz) * taper(ix, iz);
                }
            }
        }

        // Time integrate velocity
#pragma omp parallel for collapse(1)
        for (int ix = 0; ix < nx; ++ix) {
            for (int iz = 0; iz < nz; ++iz) {
                if (iz < nz - 2 and ix < nx - 2 and ix > 1) {
                    vx(ix, iz) =
                            taper(ix, iz) *
                            (vx(ix, iz)
                             - _currentModel.b_vx(ix, iz) * _shot.dt * (
                                    (coeff1 * (txx(ix, iz) - txx(ix - 1, iz)) +
                                     coeff2 * (txx(ix - 2, iz) - txx(ix + 1, iz))) / dx +
                                    (coeff1 * (txz(ix, iz) - (iz > 0 ? txz(ix, iz - 1) : 0)) +
                                     coeff2 * ((iz > 1 ? txz(ix, iz - 2) : 0) - txz(ix, iz + 1))) / dz)
                            );
                    vz(ix, iz) =
                            taper(ix, iz) *
                            (vz(ix, iz)
                             - _currentModel.b_vz(ix, iz) * _shot.dt * (
                                    (coeff1 * (txz(ix + 1, iz) - txz(ix, iz)) +
                                     coeff2 * (txz(ix - 1, iz) - txz(ix + 2, iz))) / dx +
                                    (coeff1 * (tzz(ix, iz + 1) - tzz(ix, iz)) +
                                     coeff2 * ((iz > 0 ? tzz(ix, iz - 1) : 0) - tzz(ix, iz + 2))) / dz));
                } else {
                    vx(ix, iz) = vx(ix, iz) * taper(ix, iz);
                    vz(ix, iz) = vz(ix, iz) * taper(ix, iz);
                }
            }
        }
        // Inject adjoint sources
        for (arma::uword receiver = 0; receiver < _shot.receivers.n_rows; ++receiver) {
            int ix = _shot.receivers.row(receiver)[0] + _currentModel.np_boundary;
            int iz = _shot.receivers.row(receiver)[1];
            vx(ix, iz) += _shot.dt * _currentModel.b_vx(ix, iz) * _shot.vxAdjointSource(receiver, it) / (dx * dz);
            vz(ix, iz) += _shot.dt * _currentModel.b_vz(ix, iz) * _shot.vzAdjointSource(receiver, it) / (dx * dz);
        }
        if (it % (_shot.nt / 50) == 0) {
            char message[1024];
            sprintf(message, "\r \r    %i%%",
                    static_cast<int>(static_cast<double>(it) * 100.0 / static_cast<double>(_shot.nt)));
            std::cout << message << std::flush;
        }
    }
    std::cout << std::endl;
}

