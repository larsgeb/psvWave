//
// Created by lars on 17.03.18.
//

#ifndef HMC_FORWARD_FD_MODEL_H
#define HMC_FORWARD_FD_MODEL_H

#include <armadillo>

class fwiModel {
public:
    // Fields
    const double dx = 1.249;
    const double dz = 1.249;
    const arma::uword nx_domain = 400;
    const arma::uword nz_domain = 200;
    const arma::uword np_boundary = 50;
    const double np_factor = 0.0075;
    const arma::uword nx = nx_domain + 2 * np_boundary;
    const arma::uword nz = nz_domain + np_boundary;

    // Static simulation fields
    arma::mat la = arma::mat(nx, nz);
    arma::mat mu = arma::mat(nx, nz);
    arma::mat lm = arma::mat(nx, nz);
    arma::mat b_vx = arma::mat(nx, nz);
    arma::mat b_vz = arma::mat(nx, nz); // TODO consolidate this into single gridpoint?

    arma::mat vp = arma::mat(nx, nz);
    arma::mat vs = arma::mat(nx, nz);

    // Interior of domain (excluding boundary layer)
    arma::span interiorX = arma::span(np_boundary, np_boundary + nx_domain - 1);
    arma::span interiorZ = arma::span(0, nz_domain - 1);

    // Constructors
    fwiModel();

    // Public methods
    void updateInnerFieldsElastic(arma::mat &_density, arma::mat &_lambda, arma::mat &_mu);

    void updateInnerFieldsVelocity(arma::mat &_density, arma::mat &_vp, arma::mat &_vs);

    void setTimestepAuto(double _targetCourant);

    void setTimestep(double _dt);

    void setTime(double _dt, int _nt, double _t);

    double get_dt();

    int get_nt();

    double get_samplingTime();

    double get_targetCourant();

private:

    double dt;
    int nt;
    double samplingTime;
    double targetCourant = 0.5;

    void updateFields(arma::mat &_density, arma::mat &_lambda, arma::mat &_mu);

    void extendFields(arma::mat &_outer, arma::mat &_inner);

    void extendFields(arma::mat &_outer);

    void calculateVelocityFields();


    void calculateElasticFields();


};


#endif //HMC_FORWARD_FD_MODEL_H
