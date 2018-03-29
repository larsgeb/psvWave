//
// Created by lars on 17.03.18.
//
#include <armadillo>
#include "model.h"
#include "propagator.h"
#include "experiment.h"

void propagator::propagateForward(model &_currentModel, shot &_shot, bool storeWavefieldBoundary) {

    // Some standard output
    std::cout << "    Stability number: " <<
              sqrt(_currentModel.lm.max() * _currentModel.b_vx.max()) * _shot.dt *
              sqrt(1.0 / (_currentModel.dx * _currentModel.dx) + 1.0 / (_currentModel.dz * _currentModel.dz))
              << std::endl;

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
    for (int iTaper = 0; iTaper < _currentModel.np_boundary; ++iTaper) {
        taper.submat(iTaper, 0, nx - iTaper - 1, nz - iTaper - 1) =
                1 + iTaper * arma::ones(nx - 2 * iTaper, nz - iTaper);
    }
    taper = arma::exp(-arma::square(_currentModel.np_factor * (_currentModel.np_boundary - taper)));


    // Create accumulator for visualization
//    arma::cube acc(nx, nz, static_cast<const arma::uword>(_shot.nt));

    // Time marching through all time levels
    for (int it = 0; it < _shot.nt; ++it) {

        // Inject explosive source
        for (int source = 0; source < _shot.source.n_rows; ++source) {
            int ix = _shot.source.row(source)[0];
            int iz = _shot.source.row(source)[1];

            txx(ix + _currentModel.np_boundary, iz) += 0.5 * _shot.dt * _shot.sourceFunction[it];
            tzz(ix + _currentModel.np_boundary, iz) += 0.5 * _shot.dt * _shot.sourceFunction[it];
        }

        // Take snapshot of fields
        if(it % _shot.snapshotInterval == 0){
            auto a = vx(_currentModel.interiorX,_currentModel.interiorZ);
            _shot.vxSnapshots.slice(it/_shot.snapshotInterval) = vx(_currentModel.interiorX,_currentModel.interiorZ);
            _shot.vzSnapshots.slice(it/_shot.snapshotInterval) = vz(_currentModel.interiorX,_currentModel.interiorZ);
            _shot.txxSnapshots.slice(it/_shot.snapshotInterval) = txx(_currentModel.interiorX,_currentModel.interiorZ);
            _shot.tzzSnapshots.slice(it/_shot.snapshotInterval) = tzz(_currentModel.interiorX,_currentModel.interiorZ);
            _shot.txzSnapshots.slice(it/_shot.snapshotInterval) = txz(_currentModel.interiorX,_currentModel.interiorZ);
        }

        // Record wavefield at receivers
        for (int receiver = 0; receiver < _shot.receivers.n_rows; ++receiver) {
            int ix = _shot.receivers.row(receiver)[0];
            int iz = _shot.receivers.row(receiver)[1];

            if (it == 0) {
                _shot.seismogramSyn_ux(receiver, it) = _shot.dt * vx(ix + _currentModel.np_boundary, iz);
                _shot.seismogramSyn_uz(receiver, it) = _shot.dt * vz(ix + _currentModel.np_boundary, iz);
            } else {
                _shot.seismogramSyn_ux(receiver, it) =
                        _shot.seismogramSyn_ux(receiver, it - 1) + _shot.dt * vx(ix + _currentModel.np_boundary, iz);
                _shot.seismogramSyn_uz(receiver, it) =
                        _shot.seismogramSyn_uz(receiver, it - 1) + _shot.dt * vz(ix + _currentModel.np_boundary, iz);
            }

        }

//        acc.slice(it) = vx; // takes a lot of ram

        // After this point is only integration, which doesn't have to be done at the last time level

        // Time integrate stress
#pragma omp parallel
#pragma omp for
        for (int ix = 0; ix < nx; ++ix) {
            for (int iz = 0; iz < nz; ++iz) {
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
#pragma omp parallel
#pragma omp for
        for (int ix = 0; ix < nx; ++ix) {
            for (int iz = 0; iz < nz; ++iz) {
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
        if (it % (_shot.nt / 50) == 0) {
            char message[1024];
            sprintf(message, "\r \r    %i%%",
                    static_cast<int>(static_cast<double>(it) * 100.0 / static_cast<double>(_shot.nt)));
            std::cout << message << std::flush;
        }
    }
//#pragma omp parallel
//#pragma omp for
//    for (int it = 0; it < _shot.nt; ++it) { // takes a lot of time
//        char filename[1024];
//        sprintf(filename, "output/shot%i/vx%i.txt", _shot.ishot, it);
//        acc.slice(it).save(filename, arma::raw_ascii);
//    }
}
