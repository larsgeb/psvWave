#include <iostream>
#include <armadillo>

#define _USE_MATH_DEFINES

#include <cmath>

// Own includes
#include "forward-psv/experiment.h"
#include "forward-psv/propagator.h"
#include "forward-psv/shot.h"

int main() {
    arma::imat receivers;
    arma::imat sources;
    arma::vec sourceFunction;

    receivers.load("experimentResult/receivers.txt");
    sources.load("experimentResult/sources.txt");
    sourceFunction.load("experimentResult/source.txt");

    experiment experiment_1(receivers, sources, sourceFunction);

    arma::vec lambda = 4e9 * arma::ones(8, 1);
    arma::vec mu = 1e9 * arma::ones(8, 1);
    arma::vec lightness = 1500.0 * arma::ones(8, 1);

//    // Exact observed data
    mu(2) *= 1.5;
    mu(5) *= 1.5;

    // Set startind model
    experiment_1.currentModel.updateFields(lambda, mu, lightness);

    experiment_1.forwardData();

    experiment_1.calculateMisfit();

    std::cout << experiment_1.misfit << std::endl;

    experiment_1.calculateAdjointSources();

    experiment_1.computeKernel();

    experiment_1.densityKernel.save("densityKernel.txt", arma::raw_ascii);
    experiment_1.muKernel.save("muKernel.txt", arma::raw_ascii);
    experiment_1.lambdaKernel.save("lambdaKernel.txt", arma::raw_ascii);

    return 0;
}