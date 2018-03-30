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

    receivers.load("experimentResult/receivers.txt");
    sources.load("experimentResult/sources_1.txt");
    sourceFunction.load("experimentResult/source.txt");

    experiment experiment_1(receivers, sources, sourceFunction);

    arma::vec lambda = 4e9 * arma::ones(8, 1);
    arma::vec mu = 1e9 * arma::ones(8, 1);
    arma::vec density = 1500.0 * arma::ones(8, 1);

    // Exact observed data
    mu(2) *= 1.5;
    mu(5) *= 1.5;

    experiment_1.currentModel.updateFields(lambda, mu, density);
    double deltam = 1;
    experiment_1.currentModel.b_vx(70+50,70) = 1.0 / (1500.0 + deltam);
    experiment_1.currentModel.b_vz(70+50,70) = 1.0 / (1500.0 + deltam);


    // Calculate misfit and gradient
    experiment_1.forwardData();
    experiment_1.calculateMisfit();
    double misfit1 = experiment_1.misfit;
    std::cout << "Misfit 1: " << misfit1 << std::endl;
    experiment_1.calculateAdjointSources();
    experiment_1.computeKernel();

    // Now we choose one direction (a specific parameter to change);


    std::cout << deltam << std::endl;
    std::cout << "Directional derivative: " << experiment_1.densityKernel(70, 70) << std::endl;

    experiment_1.currentModel.b_vx(70+50,70) = 1.0 / 1500.0;
    experiment_1.currentModel.b_vz(70+50,70) = 1.0 / 1500.0;
    // Calculate misfit and gradient
    experiment_1.forwardData();
    experiment_1.calculateMisfit();
    double misfit2 = experiment_1.misfit;
    std::cout << "Misfit 2: " << misfit2 << std::endl;
    experiment_1.calculateAdjointSources();
    experiment_1.computeKernel();

    return 0;
}