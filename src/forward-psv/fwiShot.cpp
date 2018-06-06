//
// Created by lars on 26/03/18.
//

#include "fwiShot.h"

using namespace arma;

fwiShot::fwiShot(irowvec _source, imat &_receivers, vec &_sourceFunction, int _samplingAmount, double _samplingTimestep, double _samplingTime,
                 uword _ishot, int _snapshotInterval, SourceTypes _sourceType) {
    source = std::move(_source);
    receivers = _receivers;
    sourceFunction = _sourceFunction;
    samplingAmount = _samplingAmount;
    samplingTime = _samplingTime;
    samplingTimestep = _samplingTimestep;
    ishot = _ishot;
    snapshotInterval = _snapshotInterval;

    sourceType = _sourceType;

    moment = mat(2, 2);
    moment(0, 0) = 1;
    moment(0, 1) = 0;
    moment(1, 0) = 0;
    moment(1, 1) = -1;
}

void fwiShot::writeShot(file_type type, std::string folder) {
    std::string filename = folder + "/seismogram" + std::to_string(static_cast<int>(ishot));
    seismogramSyn_ux.save(filename + (type == arma_binary ? "_ux.bin" : "_ux.txt"), type);
    seismogramSyn_uz.save(filename + (type == arma_binary ? "_uz.bin" : "_uz.txt"), type);
}

void fwiShot::calculateAdjointSources() {
    // only interpolate when timesteps don't match
    if (this->samplingTimestepSyn != this->samplingTimestep) {
        throw std::invalid_argument("You're trying to interpolate numerical results! This leads to very error prone kernels.");
    } else {
        vxAdjointSource = seismogramSyn_ux - seismogramObs_ux;
        vzAdjointSource = seismogramSyn_uz - seismogramObs_uz;
    }
}

void fwiShot::interpolateSynthetics() {
    if (errorOnInterpolate) {
        throw std::invalid_argument("You're trying to interpolate numerical results! This leads to very error prone kernels.");
    } else {
        std::cout << "WARNING: interpolation of adjoint sources! Leads to kernels which are very error prone!" << std::endl;
    }
    rowvec t_obs = linspace<rowvec>(0, samplingAmount * samplingTimestep, static_cast<const uword>(samplingAmount));
    rowvec t_syn = linspace<rowvec>(0, samplingTimestepSyn * samplingAmountSyn, static_cast<const uword>(samplingAmountSyn));

    mat new_ux = mat(seismogramSyn_ux.n_rows, t_obs.n_cols);
    mat new_uz = mat(seismogramSyn_ux.n_rows, t_obs.n_cols);

    for (uword iReceiver = 0; iReceiver < seismogramSyn_ux.n_rows; ++iReceiver) {
        rowvec old_ux = seismogramSyn_ux.row(iReceiver);
        rowvec old_uz = seismogramSyn_uz.row(iReceiver);

        rowvec temp_ux;
        rowvec temp_uz;

        interp1(t_syn, old_ux, t_obs, temp_ux, "*linear", 0);  // faster than "linear", monotonically increasing
        interp1(t_syn, old_uz, t_obs, temp_uz, "*linear", 0);  // faster than "linear", monotonically increasing

        new_ux.row(iReceiver) = temp_ux;
        new_uz.row(iReceiver) = temp_uz;
    }
    seismogramSyn_ux = new_ux;
    seismogramSyn_uz = new_uz;
}

void fwiShot::loadShot(std::string _folder) {
    // Load observed data for shot i
    char filename[1024];
    sprintf(filename, "%s/seismogram%i%s", _folder.c_str(), static_cast<int>(ishot), "_ux.bin");
    seismogramObs_ux.load(filename);
    sprintf(filename, "%s/seismogram%i%s", _folder.c_str(), static_cast<int>(ishot), "_uz.bin");
    seismogramObs_uz.load(filename);
}


