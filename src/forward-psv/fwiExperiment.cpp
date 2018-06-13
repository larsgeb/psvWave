//
// Created by Lars Gebraad on 17.03.18.
//

#include <omp.h>
#include "fwiExperiment.h"
#include "fwiPropagator.h"

using namespace arma;

// Constructor

fwiExperiment::fwiExperiment(double _dx,
                             double _dz,
                             arma::uword _nx_interior,
                             arma::uword _nz_interior,
                             arma::uword _np_boundary,
                             double _np_factor,
                             arma::imat _receivers,
                             arma::imat _sources,
                             arma::vec _sourceFunction,
                             double _samplingTimestep,
                             int _samplingAmount,
                             fwiShot::SourceTypes _sourceType) :
        fwiExperiment(_receivers,
                      _sources,
                      _sourceFunction,
                      _samplingTimestep,
                      _samplingAmount,
                      _sourceType) {
    model = fwiModel(_dx, _dz, _nx_interior, _nz_interior, _np_boundary, _np_factor);
    model.setTime(samplingTimestep, samplingAmount, samplingTime);
}

fwiExperiment::fwiExperiment(imat _receivers,
                             imat _sources,
                             vec _sourceFunction,
                             double _samplingTimestep,
                             int _samplingAmount,
                             fwiShot::SourceTypes _sourceType) {
    // Create a fwiExperiment
    receivers = std::move(_receivers);
    sources = std::move(_sources);
    sourceFunction = std::move(_sourceFunction);
    samplingTime = _samplingTimestep * _samplingAmount;
    samplingTimestep = _samplingTimestep;
    samplingAmount = _samplingAmount;
    model.setTime(samplingTimestep, samplingAmount, samplingTime);

    muKernel_par1 = zeros(model.nx_interior, model.nz_interior);
    densityKernel_par1 = zeros(model.nx_interior, model.nz_interior);
    lambdaKernel_par1 = zeros(model.nx_interior, model.nz_interior);

    // Check for positions
    for (auto &&yPosReceiver : receivers.col(1)) {
        if (yPosReceiver >= static_cast<int>(model.nz_interior)) {
            throw std::invalid_argument("Invalid y position for receiver (in or beyond the Gaussian taper).");
        }
    }
    for (auto &&yPosSource : sources.col(1)) {
        if (yPosSource >= static_cast<int>(model.nz_interior)) {
            throw std::invalid_argument("Invalid y position for receiver (in or beyond the Gaussian taper).");
        }
    }

    // This way all shots have the same receivers and sources
    for (uword ishot = 0; ishot < sources.n_rows; ++ishot) {
        shots.emplace_back(
                fwiShot(sources.row(ishot), receivers, sourceFunction, samplingAmount, samplingTimestep, samplingTime, ishot, snapshotInterval,
                        _sourceType));
    }

}


void fwiExperiment::forwardData() {
    for (uword iShot = 0; iShot < sources.n_rows; ++iShot) {
        fwiPropagator::propagateForward(model, shots[iShot], exportSnapshots, performFWI, snapshots);
    }
}

void fwiExperiment::writeShots(file_type type, std::string _folder) {
    for (auto &&shot : shots) {
        shot.writeShot(type, _folder);
    }
}

void fwiExperiment::computeKernel() {
    calculateAdjointSourcesL2();

    muKernel_par1 = zeros(model.nx_interior, model.nz_interior);
    densityKernel_par1 = zeros(model.nx_interior, model.nz_interior);
    lambdaKernel_par1 = zeros(model.nx_interior, model.nz_interior);
    backwardAdjoint();
    mapKernels();
}

void fwiExperiment::calculateMisfit() {
    misfit = 0;
    for (auto &&shot : shots) {
        misfit += 0.5 * shot.samplingTimestep * accu(square(shot.seismogramObs_ux - shot.seismogramSyn_ux));
        misfit += 0.5 * shot.samplingTimestep * accu(square(shot.seismogramObs_uz - shot.seismogramSyn_uz));
    }
}

void fwiExperiment::calculateAdjointSourcesL2() {
    for (auto &&shot : shots) {
        shot.calculateAdjointSources();
    }
}

void fwiExperiment::backwardAdjoint() {
    for (uword iShot = 0; iShot < sources.n_rows; ++iShot) {
        fwiPropagator::propagateAdjoint(model, shots[iShot], densityKernel_par1, muKernel_par1, lambdaKernel_par1);
    }
}

void fwiExperiment::loadShots(std::string _folder) {
    for (auto &&shot : shots) {
        shot.loadShot(_folder);
    }
}

fwiExperiment::fwiExperiment() {
    receivers = imat();
    sources = imat();
    sourceFunction = vec();
    muKernel_par1 = zeros(model.nx_interior, model.nz_interior);
    densityKernel_par1 = zeros(model.nx_interior, model.nz_interior);
    lambdaKernel_par1 = zeros(model.nx_interior, model.nz_interior);
    shots = std::vector<fwiShot>();
}

void fwiExperiment::mapKernels() {
    densityKernel_par2 = densityKernel_par1
                         + (square(model.vp(model.interiorX, model.interiorZ)) - 2 * square(model.vs(model.interiorX, model.interiorZ))) %
                           lambdaKernel_par1
                         + square(model.vs(model.interiorX, model.interiorZ)) % muKernel_par1;

    vpKernel_par2 = 2 * model.vp(model.interiorX, model.interiorZ) % lambdaKernel_par1 /
                    model.b_vx(model.interiorX, model.interiorZ);

    vsKernel_par2 =
            (2 * model.vs(model.interiorX, model.interiorZ) % muKernel_par1 - 4 * model.vs(model.interiorX, model.interiorZ) % lambdaKernel_par1) /
            model.b_vx(model.interiorX, model.interiorZ);
}

void fwiExperiment::update(mat _de, mat _vp, mat _vs) {
    model.updateInnerFieldsVelocity(_de, _vp, _vs);
}


