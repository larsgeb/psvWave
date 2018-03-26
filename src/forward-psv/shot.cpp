//
// Created by lars on 26/03/18.
//

#include "shot.h"

shot::shot(arma::irowvec _source, arma::imat &_receivers, arma::vec &_sourceFunction, int nt) {
    source = std::move(_source);
    receivers = _receivers;
    sourceFunction = _sourceFunction;

    forwardData_vx = arma::ones(receivers.n_rows, static_cast<const arma::uword>(nt));
    forwardData_vz = arma::ones(receivers.n_rows, static_cast<const arma::uword>(nt));
}
