//
// Created by lars on 30/03/18.
//

#include <iostream>
#include <armadillo>

#define _USE_MATH_DEFINES

#include <cmath>

// Own includes
#include "../forward-psv/fwiExperiment.h"
#include "../forward-psv/fwiPropagator.h"
#include "../forward-psv/fwiShot.h"
#include "../misc/functions.h"

using namespace arma;
using namespace std;
typedef vector<double> stdvec;

int main() {
    // Define actual medium parameters
    double density = 1500; // original => 1500
    double p = 2000;
    double s = 800;

    string experimentFolder = "kernelTest";
    string receiversFile = experimentFolder + string("/receivers.txt");
    string sourcesFile = experimentFolder + string("/sources.txt");

    // Load array
    imat receivers;
    imat sources;
    receivers.load(receiversFile);
    sources.load(sourcesFile);

    // Create stf
    double dt = 0.00025;
    int nt = 4000;
    double t = nt * dt;

    double freq = 50;
    vec sourcefunction = generateRicker(dt, nt, freq);

    // Create experiment object
    fwiExperiment experiment(receivers, sources, sourcefunction, t, dt, nt);

    // Create material fields
    uword nx = experiment.currentModel.nx_domain;
    uword nz = experiment.currentModel.nz_domain;
    mat rho = density * ones(nx, nz);
    mat vp = p * ones(nx, nz);
    mat vs = s * ones(nx, nz);

    vp -= 100 * generateGaussian(nx, nz, 50, 150, 100);
    experiment.currentModel.updateInnerFieldsVelocity(rho, vp, vs);

    // Generate 'observed' data
//    experiment.forwardData();
//    experiment.writeShots(arma_binary, experimentFolder);
//    experiment.writeShots(raw_ascii, experimentFolder);

    vp -= 400 * generateGaussian(nx, nz, 50, 150, 100);
    experiment.currentModel.updateInnerFieldsVelocity(rho, vp, vs);

    // Load the observed data into the appropriate fields
    experiment.loadShots(experimentFolder);

    // Generate synthetics
    experiment.forwardData();
    experiment.writeShots(raw_ascii, experimentFolder);

    // Calculate misfit
    experiment.calculateMisfit();
    double misfit1 = experiment.misfit;

    // Calculate kernel
    experiment.computeKernel();

    // Save kernels
    experiment.densityKernel_par2.save("kernelTest/densityKernel_par2.txt", raw_ascii);
    experiment.vpKernel_par2.save("kernelTest/vpKernel_par2.txt", raw_ascii);
    experiment.vsKernel_par2.save("kernelTest/vsKernel_par2.txt", raw_ascii);

    // Create accumulators for finite difference test
    stdvec misfits;
    stdvec factors;
    stdvec epsilons;

    // Generate direction
    mat dm = 500 * generateGaussian(nx, nz, 50, 150, 100);

    // Calculate predicted change
    double dirGrad = dot(experiment.vpKernel_par2, dm);

    std::cout << dirGrad << std::endl;

    // Check the kernel in multiple magnitudes
    for (int exp = -16; exp <= 1; exp++) {

        double epsilon = pow(10, exp);

        mat rhoNew = rho;
        mat vpNew = vp;
        mat vsNew = vs;

        vpNew = vp + epsilon * dm;

        experiment.currentModel.updateInnerFieldsVelocity(rhoNew, vpNew, vsNew);

        try {
            // Calculate misfit and gradient
            experiment.forwardData();
            experiment.calculateMisfit();
            double misfit2 = experiment.misfit;

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
