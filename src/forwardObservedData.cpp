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
    arma::vec density = 1500.0 * arma::ones(8, 1);
    mu(2) *= 1.5;
    mu(5) *= 1.5;

    experiment_1.currentModel.updateFields(lambda, mu, density);

    experiment_1.forwardData();

    experiment_1.writeShots(arma::arma_binary);

    sources.save("sources.txt", arma::raw_ascii);
    receivers.save("receivers.txt", arma::raw_ascii);

    return 0;
}