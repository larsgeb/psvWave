#include <iostream>
#include <armadillo>

#define _USE_MATH_DEFINES

#include <cmath>

// Own includes
#include "forward-psv/experiment.h"
#include "forward-psv/propagator.h"
#include "forward-psv/shot.h"

int main() {

    arma::imat receivers = "50 0; 150 0; 250 0; 350 0; 450 0;";
    arma::imat sources = "100 0;200 0;300 0;400 0;";

    double dt = 0.00025;
    int nt = 3500;

    double centralFrequency = 50.0;
    double tsource = 1.0 / centralFrequency;
    arma::vec time = arma::linspace(0, dt * (nt - 1), nt);
    double t0 = tsource * 1.5;
    arma::vec tau = M_PI * (time - t0) / t0;
    arma::vec sourceFunction = (1 - 4 * tau % tau) % arma::exp(-2.0 * tau % tau);

    experiment experiment_1(receivers, sources, sourceFunction, nt, dt);

    auto xoff = experiment_1.currentModel.np_boundary;

    experiment_1.currentModel.mu(arma::span(xoff + 100, 199), arma::span(0, 99)) =
            1.5 * experiment_1.currentModel.mu(arma::span(xoff + 100, 199), arma::span(0, 99));
    experiment_1.currentModel.mu(arma::span(xoff + 200, 299), arma::span(100, 199)) =
            1.5 * experiment_1.currentModel.mu(arma::span(xoff + 200, 299), arma::span(100, 199));

    experiment_1.forwardData();

    return 0;
}