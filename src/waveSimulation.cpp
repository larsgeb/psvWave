//
// Created by Lars Gebraad on 30/05/18.
//

#include <iostream>
#include <armadillo>

#define _USE_MATH_DEFINES

#include <cmath>

// Own includes
#include "forward-psv/fwiExperiment.h"
#include "forward-psv/fwiPropagator.h"
#include "forward-psv/fwiShot.h"
#include "misc/functions.h"

using namespace arma;
using namespace std;
typedef vector<double> stdvec;

int main() {

    // Loading sources and receivers
    string experimentFolder = "anisotropy";
    string receiversFile = experimentFolder + string("/receivers.txt");
    string sourcesFile = experimentFolder + string("/sources.txt");
    imat receivers;
    imat sources;
    receivers.load(receiversFile);
    sources.load(sourcesFile);

    // Create stf
    vec sourcefunction;
    double dt = 0.00025;
    unsigned int nt = 3500;
    double freq = 50;
    sourcefunction = generateRicker(dt, nt, freq);

    // Loading set up

    fwiExperiment experiment(receivers, sources, sourcefunction, dt * nt, dt, nt);

    // Loading material parameters
    mat rho = 1500 * ones(experiment.model.nx_interior, experiment.model.nz_interior);
    mat vp = 2000 * ones(experiment.model.nx_interior, experiment.model.nz_interior);
    mat vs = 800 * ones(experiment.model.nx_interior, experiment.model.nz_interior);
    experiment.update(rho, vp, vs);

    experiment.exportSnapshots = true;
    experiment.performFWI = false;

    for (int i = 0; i < 3500; i+=50) {
        experiment.snapshots.emplace_back(i);
    }

    experiment.forwardData();

}