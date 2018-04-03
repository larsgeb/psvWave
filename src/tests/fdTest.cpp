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
#include "../misc/functions.h"

using namespace arma;
using namespace std;
typedef vector<double> stdvec;

int main() {
    // Define actual medium parameters
    double density = 1500; // original => 1500
    double lame1 = 4e9; // original => 4e9
    double lame2 = 1e9; // original => 1e9

    string experimentFolder = "kernelTest";
    string receiversFile = experimentFolder + string("/receivers.txt");
    string sourcesFile = experimentFolder + string("/sources.txt");
//    string sourceFunctionFile = experimentFolder + string("/source.txt"); // If you want to load an already defined source

    // Load array
    imat receivers;
    imat sources;
    receivers.load(receiversFile);
    sources.load(sourcesFile);

    // Create stf
    vec sourcefunction;
    double dt = 0.00025;
    int nt = 3500;
    double freq = 50;
    sourcefunction = generateRicker(dt, nt, freq);

    // Create experiment object
    experiment experiment_1(receivers, sources, sourcefunction);

    // Create material fields
    uword nx = experiment_1.currentModel.nx;
    uword nz = experiment_1.currentModel.nz;
    mat rho = density * ones(nx, nz);
    mat lambda = lame1 * ones(nx, nz);
    mat mu = lame2 * ones(nx, nz);
    experiment_1.currentModel.updateFields(rho, lambda, mu);

    // Generate 'observed' data
    experiment_1.forwardData();
    experiment_1.writeShots(arma_binary, experimentFolder);

    // Load the observed data into the appropriate fields
    experiment_1.loadShots(experimentFolder);

    // Add gaussian blob to material
    mat Gaussian = 10 * generateGaussian(nx, nz, 50, 150, 100);
    rho += Gaussian;
    experiment_1.currentModel.updateFields(rho, lambda, mu);

    // Generate synthetics
    experiment_1.forwardData();

    // Calculate misfit
    experiment_1.calculateMisfit();
    double misfit1 = experiment_1.misfit;

    // Calculate kernel
    experiment_1.calculateAdjointSources();
    experiment_1.computeKernel();

    // Save kernels
    experiment_1.densityKernel.save("kernelTest/densityKernel.txt", raw_ascii);
    experiment_1.muKernel.save("kernelTest/muKernel.txt", raw_ascii);
    experiment_1.lambdaKernel.save("kernelTest/lambdaKernel.txt", raw_ascii);

    // Create accumulators for finite difference test
    stdvec misfits;
    stdvec factors;
    stdvec epsilons;

    // Generate direction
    mat dm = -Gaussian;

    // Calculate predicted change
    double dirGrad = dot(experiment_1.densityKernel, dm(span(50, nx - 1 - 50), span(0, nz - 50 - 1)));

    // Check the kernel in multiple magnitudes
    for (int exp = -16; exp <= 1; exp++) {

        double epsilon = pow(10, exp);

        mat rhoNew = rho;
        mat lambdaNew = lambda;
        mat muNew = mu;

        rhoNew = rho + epsilon * dm;

        experiment_1.currentModel.updateFields(rhoNew, lambdaNew, muNew);

        try {
            // Calculate misfit and gradient
            experiment_1.forwardData();
            experiment_1.calculateMisfit();
            double misfit2 = experiment_1.misfit;

            factors.emplace_back((misfit2 - misfit1) / (epsilon * dirGrad));
            misfits.emplace_back(misfit2);
            epsilons.emplace_back(epsilon);
        } catch (const invalid_argument &e) {
            cout << endl << "Terminating iterative increase.";
            break;
        }
    }
    cout << endl;
    // Output using armadillo functions
    (conv_to<colvec>::from(misfits)).save("kernelTest/misfits.txt", raw_ascii);
    (conv_to<colvec>::from(epsilons)).save("kernelTest/epsilons.txt", raw_ascii);
    (conv_to<colvec>::from(factors)).save("kernelTest/factors.txt", raw_ascii);

    return 0;
}
