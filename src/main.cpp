#include <iostream>
#include <armadillo>

#define _USE_MATH_DEFINES

#include <cmath>

// Own includes
#include "forward-psv/experiment.h"
#include "forward-psv/propagator.h"
#include "forward-psv/shot.h"

int main() {

    arma::imat receivers = "50 1; 100 1; 150 1;";
    arma::imat sources = "150 150;";

    double dt = 0.00025;
    int nt = 3000;

    double centralFrequency = 50.0;
    double tsource = 1.0 / centralFrequency;
    arma::vec time = arma::linspace(0,  dt * (nt - 1), nt);
    double t0 = tsource * 1.5;
    arma::vec tau = M_PI * (time - t0) / t0;
    arma::vec sourceFunction = (1 - 4 * tau % tau) % arma::exp(-2.0 * tau % tau);

    experiment experiment_1(receivers, sources, sourceFunction, nt, dt);
    experiment_1.currentModel.mu(arma::span(200, 250), arma::span(125, 150)) = 1.5 * experiment_1.currentModel.mu(arma::span(200, 250), arma::span(125, 150));
    experiment_1.forwardData();
    experiment_1.backwardData();

    experiment_1.shots[0].boundaryRecVxTop.save("vxTop.txt",arma::raw_ascii);

    return 0;
}