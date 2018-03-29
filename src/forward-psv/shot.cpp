//
// Created by lars on 26/03/18.
//

#include "shot.h"

shot::shot(arma::irowvec _source, arma::imat &_receivers, arma::vec &_sourceFunction, int _nt, double _dt,
           model &_model, int _ishot, int _snapshotInterval) {
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

    // Load observed data for shot i
    char filename[1024];
    sprintf(filename, "experimentResult/seismogram%i%s", ishot, "_ux.txt");
    seismogramObs_ux.load(filename);
    sprintf(filename, "experimentResult/seismogram%i%s", ishot, "_uz.txt");
    seismogramObs_uz.load(filename);
}

void shot::writeShot() {
    char filename[1024];
    sprintf(filename, "output/seismogram%i%s", ishot, "_ux.txt");
    seismogramSyn_ux.save(filename, arma::arma_binary);
    sprintf(filename, "output/seismogram%i%s", ishot, "_uz.txt");
    seismogramSyn_uz.save(filename, arma::arma_binary);
}

void shot::calculateAdjointSources() {
    vxAdjointSource = seismogramSyn_ux - seismogramObs_ux;
    vzAdjointSource = seismogramSyn_uz - seismogramObs_uz;

    // Use FD scheme to get velocities
    auto last = static_cast<const arma::uword>(nt - 1);
    vxAdjointSource(arma::span::all, arma::span(1, last - 1))
            = (vxAdjointSource(arma::span::all, arma::span(2, last)) -
               vxAdjointSource(arma::span::all, arma::span(0, last - 2))) / (2 * dt);
    vxAdjointSource(arma::span::all, 0) =
            (vxAdjointSource(arma::span::all, 1) - vxAdjointSource(arma::span::all, 0)) / dt;
    vxAdjointSource(arma::span::all, last) =
            (vxAdjointSource(arma::span::all, last) - vxAdjointSource(arma::span::all, last - 1)) / dt;
    vzAdjointSource(arma::span::all, arma::span(1, last - 1))
            = (vzAdjointSource(arma::span::all, arma::span(2, last)) -
               vzAdjointSource(arma::span::all, arma::span(0, last - 2))) / (2 * dt);
    vzAdjointSource(arma::span::all, 0) =
            (vzAdjointSource(arma::span::all, 1) - vzAdjointSource(arma::span::all, 0)) / dt;
    vzAdjointSource(arma::span::all, last) =
            (vzAdjointSource(arma::span::all, last) - vzAdjointSource(arma::span::all, last - 1)) / dt;
}


