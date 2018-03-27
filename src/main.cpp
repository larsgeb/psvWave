#include <iostream>
#include <armadillo>

#define _USE_MATH_DEFINES

#include <cmath>

// Own includes
#include "forward-psv/experiment.h"
#include "forward-psv/propagator.h"
#include "forward-psv/shot.h"

int main() {

    arma::imat receivers = "50 0; 100 0; 150 0;";
    arma::imat sources = "0 0;0 10;0 20;";

    double dt = 0.00025;
    int nt = 3000;

    double centralFrequency = 50.0;
    double tsource = 1.0 / centralFrequency;
    arma::vec time = arma::linspace(0,  dt * (nt - 1), nt);
    double t0 = tsource * 1.5;
    arma::vec tau = M_PI * (time - t0) / t0;
    arma::vec sourceFunction = (1 - 4 * tau % tau) % arma::exp(-2.0 * tau % tau);

    experiment experiment_1(receivers, sources, sourceFunction, nt, dt);

    experiment_1.forwardData();

    return 0;
}