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

    forwardData_vx = arma::ones(receivers.n_rows, static_cast<const arma::uword>(nt));
    forwardData_vz = arma::ones(receivers.n_rows, static_cast<const arma::uword>(nt));

    boundaryRecVxTop = arma::mat(_model.nx_domain, nt, arma::fill::zeros);
    boundaryRecVzTop = arma::mat(_model.nx_domain, nt, arma::fill::zeros);
    boundaryRecTxxTop = arma::mat(_model.nx_domain, nt, arma::fill::zeros);
    boundaryRecTzzTop = arma::mat(_model.nx_domain, nt, arma::fill::zeros);
    boundaryRecTxzTop = arma::mat(_model.nx_domain, nt, arma::fill::zeros);

    boundaryRecVxBottom = arma::mat(_model.nx_domain, nt, arma::fill::zeros);
    boundaryRecVzBottom = arma::mat(_model.nx_domain, nt, arma::fill::zeros);
    boundaryRecTxxBottom = arma::mat(_model.nx_domain, nt, arma::fill::zeros);
    boundaryRecTzzBottom = arma::mat(_model.nx_domain, nt, arma::fill::zeros);
    boundaryRecTxzBottom = arma::mat(_model.nx_domain, nt, arma::fill::zeros);

    boundaryRecVxLeft = arma::mat(nt, _model.nz_domain - 2, arma::fill::zeros);
    boundaryRecVzLeft = arma::mat(nt, _model.nz_domain - 2, arma::fill::zeros);
    boundaryRecTxxLeft = arma::mat(nt, _model.nz_domain - 2, arma::fill::zeros);
    boundaryRecTzzLeft = arma::mat(nt, _model.nz_domain - 2, arma::fill::zeros);
    boundaryRecTxzLeft = arma::mat(nt, _model.nz_domain - 2, arma::fill::zeros);

    boundaryRecVxRight = arma::mat(nt, _model.nz_domain - 2, arma::fill::zeros);
    boundaryRecVzRight = arma::mat(nt, _model.nz_domain - 2, arma::fill::zeros);
    boundaryRecTxxRight = arma::mat(nt, _model.nz_domain - 2, arma::fill::zeros);
    boundaryRecTzzRight = arma::mat(nt, _model.nz_domain - 2, arma::fill::zeros);
    boundaryRecTxzRight = arma::mat(nt, _model.nz_domain - 2, arma::fill::zeros);
}
