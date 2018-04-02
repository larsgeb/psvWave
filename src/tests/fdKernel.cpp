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

typedef std::vector<double> stdvec;

int main() {
    // settings
    const double density = 1520; // original => 1500
    const double lame1 = 3.9e9; // original => 4e9
    const double lame2 = 0.975e9; // original => 1e9

    std::string receiversFile = "experiment1/receivers.txt";
    std::string sourcesFile = "experiment1/sources.txt";
    std::string sourceFunctionFile = "experiment1/source.txt";
    std::string syntheticsFolder = "experiment1";

    arma::uword horizontaldivisions = 4;
    arma::uword verticaldivisions = 2;

    // end of settings
    arma::imat receivers;
    arma::imat sources;
    arma::vec sourcefunction;

    receivers.load(receiversFile);
    sources.load(sourcesFile);
    sourcefunction.load(sourceFunctionFile);

    experiment experiment_1(receivers, sources, sourcefunction, 0, 0);

    arma::vec lambda = lame1 * arma::ones(verticaldivisions * horizontaldivisions, 1);
    arma::vec mu = lame2 * arma::ones(verticaldivisions * horizontaldivisions, 1);
    arma::vec rho = density * arma::ones(verticaldivisions * horizontaldivisions, 1);

    experiment_1.currentModel.updateFields(rho, lambda, mu);

    experiment_1.loadShots(syntheticsFolder);

    experiment_1.forwardData();
    experiment_1.calculateMisfit();

    double misfit1 = experiment_1.misfit;

    experiment_1.calculateAdjointSources();
    experiment_1.computeKernel();
    experiment_1.consolidateKernel();

    experiment_1.densityKernel.save("densityKernel.txt", arma::raw_ascii);
    experiment_1.muKernel.save("muKernel.txt", arma::raw_ascii);
    experiment_1.lambdaKernel.save("lambdaKernel.txt", arma::raw_ascii);
    experiment_1.dxdm.save("gradient.txt", arma::raw_ascii);

    stdvec misfit2v;
    stdvec factorv;
    stdvec epsilonv;

    arma::vec dm = arma::zeros(24, 1);

    dm(0 + 2) = -20;

    double dirGrad = arma::dot(experiment_1.dxdm, dm);

    for (double exp = -17; exp <= 1; exp += 1) {
        double epsilon = pow(10, exp);

        arma::vec rhoNew = rho;
        arma::vec lambdaNew = lambda;
        arma::vec muNew = mu;

        rhoNew += epsilon * dm(arma::span(0, 7));
        lambdaNew += epsilon * dm(arma::span(8, 15));
        muNew += epsilon * dm(arma::span(16, 23));

        experiment_1.currentModel.updateFields(rhoNew, lambdaNew, muNew);

        try {
            // Calculate misfit and gradient
            experiment_1.forwardData();
            experiment_1.calculateMisfit();
            double misfit2 = experiment_1.misfit;

            factorv.emplace_back((misfit2 - misfit1) / (epsilon * dirGrad));
            misfit2v.emplace_back(misfit2);
            epsilonv.emplace_back(epsilon);
        } catch (const std::invalid_argument &e) {
            std::cout << std::endl << "Terminating iterative increase." << std::endl;
            break;
        }
    }
    std::cout << std::endl;
    // Output using armadillo functions
    (arma::conv_to<arma::colvec>::from(misfit2v)).save("misfits.txt", arma::raw_ascii);
    (arma::conv_to<arma::colvec>::from(epsilonv)).save("epsilons.txt", arma::raw_ascii);
    (arma::conv_to<arma::colvec>::from(factorv)).save("factors.txt", arma::raw_ascii);

    return 0;
}