//
// Created by lars on 17.03.18.
//
#include <armadillo>
#include "fwiModel.h"
#include "fwiPropagator.h"
#include "fwiExperiment.h"

using namespace arma;

void fwiPropagator::propagateForward(fwiModel &_currentModel, fwiShot &_shot) {
    // Rewrite synthetic parameters in shot for interpolation in misfit calculation
    _shot.samplingTimestepSyn = _currentModel.get_dt();
    _shot.samplingAmountSyn = _currentModel.get_nt();

    vec stf;
    if (_shot.samplingTimestepSyn != _shot.samplingTimestep) {
        if (_shot.errorOnInterpolate) {
            std::cout << _shot.samplingTimestepSyn << " " << _shot.samplingTimestep << std::endl;
            throw std::invalid_argument("You're trying to interpolate numerical results! This leads to very error prone kernels.");
        } else {
            std::cout << "WARNING: interpolation of stf! Leads to kernels which are very error prone!" << std::endl;
        }
        // Interpolate stf
        vec t = linspace(0, _shot.samplingAmount * _shot.samplingTimestep, static_cast<const uword>(_shot.samplingAmount));
        vec t_interp = linspace(0, _currentModel.get_nt() * _currentModel.get_dt(), static_cast<const uword>(_currentModel.get_nt()));
        interp1(t, _shot.sourceFunction, t_interp, stf, "*linear", 0);  // faster than "linear", monotonically increasing
    } else {
        stf = _shot.sourceFunction;
    }

    // Loading simulation parameters
    double dx = _currentModel.dx;
    const uword nx = _currentModel.nx;
    double dz = _currentModel.dz;
    const uword nz = _currentModel.nz;

    // Create dynamic fields
    mat vx = zeros(_currentModel.nx, _currentModel.nz);
    mat vz = zeros(_currentModel.nx, _currentModel.nz);
    mat txx = zeros(_currentModel.nx, _currentModel.nz);
    mat tzz = zeros(_currentModel.nx, _currentModel.nz);
    mat txz = zeros(_currentModel.nx, _currentModel.nz);

    // Create taper matrix
    mat taper = _currentModel.np_boundary * ones(nx, nz);
    for (uword iTaper = 0; iTaper < _currentModel.np_boundary; ++iTaper) {
        taper.submat(iTaper, 0, nx - iTaper - 1, nz - iTaper - 1) =
                1 + iTaper * ones(nx - 2 * iTaper, nz - iTaper);
    }
    taper = exp(-square(_currentModel.np_factor * (_currentModel.np_boundary - taper)));

    // Create cubes for snapshots (size changes with nt)
    _shot.txxSnapshots = cube(_currentModel.nx_interior, _currentModel.nz_interior,
                              1 + static_cast<const uword>(_currentModel.get_nt() / _shot.snapshotInterval));
    _shot.tzzSnapshots = cube(_currentModel.nx_interior, _currentModel.nz_interior,
                              1 + static_cast<const uword>(_currentModel.get_nt() / _shot.snapshotInterval));
    _shot.txzSnapshots = cube(_currentModel.nx_interior, _currentModel.nz_interior,
                              1 + static_cast<const uword>(_currentModel.get_nt() / _shot.snapshotInterval));
    _shot.vxSnapshots = cube(_currentModel.nx_interior, _currentModel.nz_interior,
                             1 + static_cast<const uword>(_currentModel.get_nt() / _shot.snapshotInterval));
    _shot.vzSnapshots = cube(_currentModel.nx_interior, _currentModel.nz_interior,
                             1 + static_cast<const uword>(_currentModel.get_nt() / _shot.snapshotInterval));

    // Create synthetic seismogram matrices (size changes with nt)
    _shot.seismogramSyn_ux = zeros(_shot.receivers.n_rows, static_cast<const uword>(_currentModel.get_nt()));
    _shot.seismogramSyn_uz = zeros(_shot.receivers.n_rows, static_cast<const uword>(_currentModel.get_nt()));

    // Time marching through all time levels
    for (int it = 0; it < _currentModel.get_nt(); ++it) {

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
        for (uword receiver = 0; receiver < _shot.receivers.n_rows; ++receiver) {
            int ix = _shot.receivers.row(receiver)[0] + _currentModel.np_boundary;
            int iz = _shot.receivers.row(receiver)[1];

            if (it == 0) {
                _shot.seismogramSyn_ux(receiver, it) = _currentModel.get_dt() * vx(ix, iz) / (dx * dz);
                _shot.seismogramSyn_uz(receiver, it) = _currentModel.get_dt() * vz(ix, iz) / (dx * dz);
            } else {
                _shot.seismogramSyn_ux(receiver, it) =
                        _shot.seismogramSyn_ux(receiver, it - 1) + _currentModel.get_dt() * vx(ix, iz) / (dx * dz);
                _shot.seismogramSyn_uz(receiver, it) =
                        _shot.seismogramSyn_uz(receiver, it - 1) + _currentModel.get_dt() * vz(ix, iz) / (dx * dz);
            }
        }

        // After this point is only integration, which doesn't have to be done at the last time level // TODO BREAK

        // Time integrate stress
#pragma omp parallel for collapse(1)
        for (uword ix = 0; ix < nx; ++ix) {
            for (uword iz = 0; iz < nz; ++iz) {
                if (ix > 1 and ix < nx - 2 and iz < nz - 2) {
                    txx(ix, iz) = taper(ix, iz) *
                                  (txx(ix, iz) +
                                   _currentModel.get_dt() *
                                   (_currentModel.lm(ix, iz) * (
                                           coeff1 * (vx(ix + 1, iz) - vx(ix, iz)) +
                                           coeff2 * (vx(ix - 1, iz) - vx(ix + 2, iz))) / dx +
                                    _currentModel.la(ix, iz) * (
                                            coeff1 * (vz(ix, iz) - (iz > 0 ? vz(ix, iz - 1) : 0)) +
                                            coeff2 * ((iz > 1 ? vz(ix, iz - 2) : 0) - vz(ix, iz + 1))) / dz));
                    tzz(ix, iz) = taper(ix, iz) *
                                  (tzz(ix, iz) +
                                   _currentModel.get_dt() *
                                   (_currentModel.la(ix, iz) * (
                                           coeff1 * (vx(ix + 1, iz) - vx(ix, iz)) +
                                           coeff2 * (vx(ix - 1, iz) - vx(ix + 2, iz))) / dx +
                                    (_currentModel.lm(ix, iz)) * (
                                            coeff1 * (vz(ix, iz) - (iz > 0 ? vz(ix, iz - 1) : 0)) +
                                            coeff2 * ((iz > 1 ? vz(ix, iz - 2) : 0) - vz(ix, iz + 1))) / dz));
                    txz(ix, iz) = taper(ix, iz) *
                                  (txz(ix, iz) + _currentModel.get_dt() * _currentModel.mu(ix, iz) * (
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
        for (uword ix = 0; ix < nx; ++ix) {
            for (uword iz = 0; iz < nz; ++iz) {
                if (iz < nz - 2 and ix < nx - 2 and ix > 1) {
                    vx(ix, iz) =
                            taper(ix, iz) *
                            (vx(ix, iz)
                             + _currentModel.b_vx(ix, iz) * _currentModel.get_dt() * (
                                    (coeff1 * (txx(ix, iz) - txx(ix - 1, iz)) +
                                     coeff2 * (txx(ix - 2, iz) - txx(ix + 1, iz))) / dx +
                                    (coeff1 * (txz(ix, iz) - (iz > 0 ? txz(ix, iz - 1) : 0)) +
                                     coeff2 * ((iz > 1 ? txz(ix, iz - 2) : 0) - txz(ix, iz + 1))) / dz)
                            );
                    vz(ix, iz) =
                            taper(ix, iz) *
                            (vz(ix, iz)
                             + _currentModel.b_vz(ix, iz) * _currentModel.get_dt() * (
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
        for (uword source = 0; source < _shot.source.n_rows; ++source) {
            int ix = _shot.source.row(source)[0] + _currentModel.np_boundary;
            int iz = _shot.source.row(source)[1];

            switch (_shot.sourceType) {
                case fwiShot::explosiveSource: // explosive
                    txx(ix, iz) +=
                            0.5 * _currentModel.get_dt() * stf[it] * _currentModel.b_vx(ix, iz) / (dx * dz);
                    tzz(ix, iz) +=
                            0.5 * _currentModel.get_dt() * stf[it] * _currentModel.b_vz(ix, iz) / (dx * dz);
                case fwiShot::rotationalSource: // rotational
                    txz += _currentModel.get_dt() * stf[it] * _currentModel.b_vz(ix, iz) / (dx * dz);
                case fwiShot::momentSource: // Moment tensor * stf
                    // (x,x)-couple
                    vx(ix - 1, iz) -=
                            _shot.moment[0, 0] * stf[it] * _currentModel.get_dt() * _currentModel.b_vz(ix - 1, iz) / (dx * dx * dx * dx);
                    vx(ix, iz) +=
                            _shot.moment[0, 0] * stf[it] * _currentModel.get_dt() * _currentModel.b_vz(ix, iz) / (dx * dx * dx * dx);

                    // (z,z)-couple
                    vz(ix, iz - 1) -=
                            _shot.moment[1, 1] * stf[it] * _currentModel.get_dt() * _currentModel.b_vz(ix, iz - 1) / (dz * dz * dz * dz);
                    vz(ix, iz) +=
                            _shot.moment[1, 1] * stf[it] * _currentModel.get_dt() * _currentModel.b_vz(ix, iz) / (dz * dz * dz * dz);

                    // (x,z)-couple
                    vx(ix - 1, iz + 1) +=
                            0.25 * _shot.moment[0, 1] * stf[it] * _currentModel.get_dt() * _currentModel.b_vz(ix - 1, iz + 1) / (dx * dx * dx * dx);
                    vx(ix, iz + 1) +=
                            0.25 * _shot.moment[0, 1] * stf[it] * _currentModel.get_dt() * _currentModel.b_vz(ix, iz + 1) / (dx * dx * dx * dx);
                    vx(ix - 1, iz - 1) -=
                            0.25 * _shot.moment[0, 1] * stf[it] * _currentModel.get_dt() * _currentModel.b_vz(ix - 1, iz - 1) / (dx * dx * dx * dx);
                    vx(ix, iz - 1) -=
                            0.25 * _shot.moment[0, 1] * stf[it] * _currentModel.get_dt() * _currentModel.b_vz(ix, iz - 1) / (dx * dx * dx * dx);

                    // (z,x)-couple
                    vz(ix + 1, iz - 1) +=
                            _shot.moment[1, 0] * stf[it] * _currentModel.get_dt() * _currentModel.b_vz(ix + 1, iz - 1) / (dz * dz * dz * dz);
                    vz(ix + 1, iz) +=
                            _shot.moment[1, 0] * stf[it] * _currentModel.get_dt() * _currentModel.b_vz(ix + 1, iz) / (dz * dz * dz * dz);
                    vz(ix - 1, iz - 1) -=
                            _shot.moment[1, 0] * stf[it] * _currentModel.get_dt() * _currentModel.b_vz(ix - 1, iz - 1) / (dz * dz * dz * dz);
                    vz(ix - 1, iz) -=
                            _shot.moment[1, 0] * stf[it] * _currentModel.get_dt() * _currentModel.b_vz(ix - 1, iz) / (dz * dz * dz * dz);
            }

        }

        // Print status bar
        if (it % (_currentModel.get_nt() / 50) == 0) {
            char message[1024];
            sprintf(message, "\r \r    %i%%", static_cast<int>(static_cast<double>(it) * 100.0 / static_cast<double>(_currentModel.get_nt())));
            std::cout << message << std::flush;
        }
    }
    char message[1024];
    sprintf(message, "\r \r    %i%%", 100);
    std::cout << message << std::flush;
    std::cout << std::endl;

    if (_shot.samplingTimestepSyn != _shot.samplingTimestep) {
        _shot.interpolateSynthetics();
    }
}


void fwiPropagator::propagateAdjoint(fwiModel &_currentModel, fwiShot &_shot, mat &_denistyKernel, mat &_muKernel,
                                     mat &_lambdaKernel) {

    // Loading simulation parameters
    double dx = _currentModel.dx;
    const sword nx = _currentModel.nx;
    double dz = _currentModel.dz;
    const sword nz = _currentModel.nz;

    // Create dynamic fields
    mat vx = zeros(_currentModel.nx, _currentModel.nz);
    mat vz = zeros(_currentModel.nx, _currentModel.nz);
    mat txx = zeros(_currentModel.nx, _currentModel.nz);
    mat tzz = zeros(_currentModel.nx, _currentModel.nz);
    mat txz = zeros(_currentModel.nx, _currentModel.nz);

    // Create taper matrix
    mat taper = _currentModel.np_boundary * ones(nx, nz);
    for (uword iTaper = 0; iTaper < _currentModel.np_boundary; ++iTaper) {
        taper.submat(iTaper, 0, nx - iTaper - 1, nz - iTaper - 1) =
                1 + iTaper * ones(nx - 2 * iTaper, nz - iTaper);
    }
    taper = exp(-square(_currentModel.np_factor * (_currentModel.np_boundary - taper)));

    // Time marching through all time levels
    for (int it = _currentModel.get_nt() - 1; it >= 0; --it) {

        // Compute correlation integral for the kernel at snapshots
        if (it % _shot.snapshotInterval == 0) {
//            _shot.vxSnapshots.slice(it / _shot.snapshotInterval).save("inversion1/fields/vx_f_" + std::to_string(it) + ".txt", raw_ascii);
//            _shot.vzSnapshots.slice(it / _shot.snapshotInterval).save("inversion1/fields/vz_f_" + std::to_string(it) + ".txt", raw_ascii);
//            mat curvx = vx(_currentModel.interiorX, _currentModel.interiorZ);
//            curvx.save("inversion1/fields/vx_a_" + std::to_string(it) + ".txt", raw_ascii);
//            mat curvz = vz(_currentModel.interiorX, _currentModel.interiorZ);
//            curvz.save("inversion1/fields/vz_a_" + std::to_string(it) + ".txt", raw_ascii);

            // todo check computation of kernels with integration?
            mat f1 = _shot.vxSnapshots.slice(it / _shot.snapshotInterval) %
                     vx(_currentModel.interiorX, _currentModel.interiorZ);

            mat f2 = _shot.vzSnapshots.slice(it / _shot.snapshotInterval) %
                     vz(_currentModel.interiorX, _currentModel.interiorZ);

            _denistyKernel -=
                    _shot.snapshotInterval * _currentModel.get_dt() * (f1 + f2);
            // Compute strain
            mat exxAdj = (txx(_currentModel.interiorX, _currentModel.interiorZ) -
                          (tzz(_currentModel.interiorX, _currentModel.interiorZ) %
                           _currentModel.la(_currentModel.interiorX, _currentModel.interiorZ)) /
                          _currentModel.lm(_currentModel.interiorX, _currentModel.interiorZ)) /
                         (_currentModel.lm(_currentModel.interiorX, _currentModel.interiorZ) -
                          (square(_currentModel.la(_currentModel.interiorX, _currentModel.interiorZ)) /
                           (_currentModel.lm(_currentModel.interiorX, _currentModel.interiorZ))));
            mat ezzAdj = (tzz(_currentModel.interiorX, _currentModel.interiorZ) -
                          (txx(_currentModel.interiorX, _currentModel.interiorZ) %
                           _currentModel.la(_currentModel.interiorX, _currentModel.interiorZ)) /
                          _currentModel.lm(_currentModel.interiorX, _currentModel.interiorZ)) /
                         (_currentModel.lm(_currentModel.interiorX, _currentModel.interiorZ) -
                          (square(_currentModel.la(_currentModel.interiorX, _currentModel.interiorZ)) /
                           (_currentModel.lm(_currentModel.interiorX, _currentModel.interiorZ))));
            mat exzAdj = txz(_currentModel.interiorX, _currentModel.interiorZ) /
                         (2 * _currentModel.mu(_currentModel.interiorX, _currentModel.interiorZ));

            mat exx = (_shot.txxSnapshots.slice(it / _shot.snapshotInterval) -
                       (_shot.tzzSnapshots.slice(it / _shot.snapshotInterval) %
                        _currentModel.la(_currentModel.interiorX, _currentModel.interiorZ)) /
                       _currentModel.lm(_currentModel.interiorX, _currentModel.interiorZ)) /
                      (_currentModel.lm(_currentModel.interiorX, _currentModel.interiorZ) -
                       (square(_currentModel.la(_currentModel.interiorX, _currentModel.interiorZ)) /
                        (_currentModel.lm(_currentModel.interiorX, _currentModel.interiorZ))));
            mat ezz = (_shot.tzzSnapshots.slice(it / _shot.snapshotInterval) -
                       (_shot.txxSnapshots.slice(it / _shot.snapshotInterval) %
                        _currentModel.la(_currentModel.interiorX, _currentModel.interiorZ)) /
                       _currentModel.lm(_currentModel.interiorX, _currentModel.interiorZ)) /
                      (_currentModel.lm(_currentModel.interiorX, _currentModel.interiorZ) -
                       (square(_currentModel.la(_currentModel.interiorX, _currentModel.interiorZ)) /
                        (_currentModel.lm(_currentModel.interiorX, _currentModel.interiorZ))));
            mat exz = _shot.txzSnapshots.slice(it / _shot.snapshotInterval) /
                      (2 * _currentModel.mu(_currentModel.interiorX, _currentModel.interiorZ));

            _lambdaKernel += _shot.snapshotInterval * _currentModel.get_dt() * ((exx + ezz) % (exxAdj + ezzAdj));
            _muKernel += _shot.snapshotInterval * _currentModel.get_dt() * 2 *
                         ((exxAdj % exx) + (ezzAdj % ezz) + 2 * (exzAdj % exz));
        }

        // After this point is only integration, which doesn't have to be done at the last time level
        // Time integrate stress
#pragma omp parallel for collapse(1)
        for (int ix = 0; ix < nx; ++ix) {
            for (int iz = 0; iz < nz; ++iz) {
                if (ix > 1 and ix < nx - 2 and iz < nz - 2) {
                    txx(ix, iz) = taper(ix, iz) *
                                  (txx(ix, iz) -
                                   _currentModel.get_dt() *
                                   (_currentModel.lm(ix, iz) * (
                                           coeff1 * (vx(ix + 1, iz) - vx(ix, iz)) +
                                           coeff2 * (vx(ix - 1, iz) - vx(ix + 2, iz))) / dx +
                                    _currentModel.la(ix, iz) * (
                                            coeff1 * (vz(ix, iz) - (iz > 0 ? vz(ix, iz - 1) : 0)) +
                                            coeff2 * ((iz > 1 ? vz(ix, iz - 2) : 0) - vz(ix, iz + 1))) / dz));
                    tzz(ix, iz) = taper(ix, iz) *
                                  (tzz(ix, iz) -
                                   _currentModel.get_dt() *
                                   (_currentModel.la(ix, iz) * (
                                           coeff1 * (vx(ix + 1, iz) - vx(ix, iz)) +
                                           coeff2 * (vx(ix - 1, iz) - vx(ix + 2, iz))) / dx +
                                    (_currentModel.lm(ix, iz)) * (
                                            coeff1 * (vz(ix, iz) - (iz > 0 ? vz(ix, iz - 1) : 0)) +
                                            coeff2 * ((iz > 1 ? vz(ix, iz - 2) : 0) - vz(ix, iz + 1))) / dz));
                    txz(ix, iz) = taper(ix, iz) *
                                  (txz(ix, iz) - _currentModel.get_dt() * _currentModel.mu(ix, iz) * (
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
                             - _currentModel.b_vx(ix, iz) * _currentModel.get_dt() * (
                                    (coeff1 * (txx(ix, iz) - txx(ix - 1, iz)) +
                                     coeff2 * (txx(ix - 2, iz) - txx(ix + 1, iz))) / dx +
                                    (coeff1 * (txz(ix, iz) - (iz > 0 ? txz(ix, iz - 1) : 0)) +
                                     coeff2 * ((iz > 1 ? txz(ix, iz - 2) : 0) - txz(ix, iz + 1))) / dz)
                            );
                    vz(ix, iz) =
                            taper(ix, iz) *
                            (vz(ix, iz)
                             - _currentModel.b_vz(ix, iz) * _currentModel.get_dt() * (
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
        for (uword receiver = 0; receiver < _shot.receivers.n_rows; ++receiver) {
            int ix = _shot.receivers.row(receiver)[0] + _currentModel.np_boundary;
            int iz = _shot.receivers.row(receiver)[1];
            vx(ix, iz) += _currentModel.get_dt() * _currentModel.b_vx(ix, iz) * _shot.vxAdjointSource(receiver, it) /
                          (dx * dz);
            vz(ix, iz) += _currentModel.get_dt() * _currentModel.b_vz(ix, iz) * _shot.vzAdjointSource(receiver, it) /
                          (dx * dz);
        }
        if (it % (_currentModel.get_nt() / 50) == 0) {
            char message[1024];
            sprintf(message, "\r \r    %i%% ",
                    static_cast<int>(static_cast<double>(it) * 100.0 / static_cast<double>(_currentModel.get_nt())));
            std::cout << message << std::flush;
        }
    }
    std::cout << std::endl;
}

