//
// Created by lars on 26/03/18.
//

#include "shot.h"

shot::shot(arma::irowvec _source, arma::imat &_receivers, arma::vec &_sourceFunction, int _nt, double _dt,
           model &_model, arma::uword _ishot, int _snapshotInterval) {
    source = std::move(_source);
    receivers = _receivers;
    sourceFunction = _sourceFunction;
    nt = _nt;
    dt = _dt;
    ishot = _ishot;
    snapshotInterval = _snapshotInterval;

    txxSnapshots = arma::cube(_model.nx_domain, _model.nz_domain,
                              static_cast<const arma::uword>(nt / _snapshotInterval));
    tzzSnapshots = arma::cube(_model.nx_domain, _model.nz_domain,
                              static_cast<const arma::uword>(nt / _snapshotInterval));
    txzSnapshots = arma::cube(_model.nx_domain, _model.nz_domain,
                              static_cast<const arma::uword>(nt / _snapshotInterval));
    vxSnapshots = arma::cube(_model.nx_domain, _model.nz_domain,
                             static_cast<const arma::uword>(nt / _snapshotInterval));
    vzSnapshots = arma::cube(_model.nx_domain, _model.nz_domain,
                             static_cast<const arma::uword>(nt / _snapshotInterval));

    seismogramSyn_ux = arma::zeros(receivers.n_rows, static_cast<const arma::uword>(nt));
    seismogramSyn_uz = arma::zeros(receivers.n_rows, static_cast<const arma::uword>(nt));
}

void shot::writeShot(arma::file_type type, std::string folder) {
    char filename[1024];
    sprintf(filename, "%s/seismogram%i%s", folder.c_str(), static_cast<int>(ishot), (type == arma::arma_binary ? "_ux.bin": "_ux.txt"));
    seismogramSyn_ux.save(filename, type);
    sprintf(filename, "%s/seismogram%i%s", folder.c_str(), static_cast<int>(ishot), (type == arma::arma_binary ? "_uz.bin": "_uz.txt"));
    seismogramSyn_uz.save(filename, type);
}

void shot::calculateAdjointSources() {
    vxAdjointSource = seismogramSyn_ux - seismogramObs_ux;
    vzAdjointSource = seismogramSyn_uz - seismogramObs_uz;
}

void shot::loadShot(std::string &_folder) {
    // Load observed data for shot i
    char filename[1024];
    sprintf(filename, "%s/seismogram%i%s", _folder.c_str(), static_cast<int>(ishot), "_ux.bin");
    seismogramObs_ux.load(filename);
    sprintf(filename, "%s/seismogram%i%s", _folder.c_str(), static_cast<int>(ishot), "_uz.bin");
    seismogramObs_uz.load(filename);
}


