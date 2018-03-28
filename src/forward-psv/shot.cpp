//
// Created by lars on 26/03/18.
//

#include "shot.h"

shot::shot(arma::irowvec _source, arma::imat &_receivers, arma::vec &_sourceFunction, int _nt, double _dt,
           model &_model, int _ishot) {
    source = std::move(_source);
    receivers = _receivers;
    sourceFunction = _sourceFunction;
    nt = _nt;
    dt = _dt;
    ishot = _ishot;

    seismogramSyn_ux = arma::zeros(receivers.n_rows, static_cast<const arma::uword>(nt));
    seismogramSyn_uz = arma::zeros(receivers.n_rows, static_cast<const arma::uword>(nt));
}

void shot::writeShot() {
    char filename[1024];
    sprintf(filename, "output/seismogram%i%s", ishot, "_ux.txt");
    seismogramSyn_ux.save(filename, arma::raw_ascii);
    sprintf(filename, "output/seismogram%i%s", ishot, "_uz.txt");
    seismogramSyn_uz.save(filename, arma::raw_ascii);
}


