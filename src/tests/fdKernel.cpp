//
// Created by lars on 30/03/18.
//

#include <iostream>
#include <armadillo>

#define _USE_MATH_DEFINES

#include <cmath>

// Own includes
#include "../forward-psv/experiment.h"
#include "../forward-psv/propagator.h"
#include "../forward-psv/shot.h"

int main() {
    arma::imat receivers;
    arma::imat sources;
    arma::vec sourceFunction;

    receivers.load("experiment1/receivers.txt");
    sources.load("experiment1/sources.txt");
    sourceFunction.load("experiment1/source.txt");

    experiment experiment_1(receivers, sources, sourceFunction);

    arma::vec lambda = 4e9 * arma::ones(8, 1);
    arma::vec mu = 1e9 * arma::ones(8, 1);
    arma::vec density = 1510 * arma::ones(8, 1); // Original was modeled on 1500
    experiment_1.currentModel.updateFields(lambda, mu, density);

    experiment_1.loadShots(const_cast<char *>("experiment1"));
    experiment_1.currentModel.updateFields(lambda, mu, density);
    experiment_1.forwardData();
    experiment_1.calculateMisfit();
    double misfit1 = experiment_1.misfit;
    experiment_1.calculateAdjointSources();
    experiment_1.computeKernel();
    double dirGradient = experiment_1.densityKernel(250, 60);

    double epsilon = 0.1;

    experiment_1.currentModel.b_vx(300, 60) = 1.0 / (density(4) + epsilon * 1);
    experiment_1.currentModel.b_vz(300, 60) = 1.0 / (density(4) + epsilon * 1);

    // Calculate misfit and gradient
    experiment_1.forwardData();
    experiment_1.calculateMisfit();
    double misfit2 = experiment_1.misfit;

    std::cout << std::endl << "Misfit 1: " << misfit1 << std::endl;
    std::cout << "Misfit 2: " << misfit2 << std::endl;
    std::cout << "Difference: " << misfit2-misfit1 << std::endl;
    std::cout << "Directional derivative times step (dx/dx * epsilon): " << dirGradient * epsilon<< std::endl;
    std::cout << "Predicted misfit 2: " << misfit1 + epsilon * 1 * dirGradient<< std::endl;
    std::cout << "Factor difference: " << (misfit2 - misfit1) / (epsilon * 1 * dirGradient) << std::endl;

    return 0;

}