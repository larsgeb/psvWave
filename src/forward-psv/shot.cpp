//
// Created by lars on 26/03/18.
//

#include "shot.h"

shot::shot(arma::irowvec _source, arma::imat &_receivers, arma::vec &_sourceFunction, int _nt, double _dt,
           model &_model) {
    source = std::move(_source);
    receivers = _receivers;
    sourceFunction = _sourceFunction;
    nt = _nt;
    dt = _dt;

    seismogramSyn_ux = arma::zeros(receivers.n_rows, static_cast<const arma::uword>(nt));
    seismogramSyn_uz = arma::zeros(receivers.n_rows, static_cast<const arma::uword>(nt));
}
