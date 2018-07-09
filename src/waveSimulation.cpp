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
    string experimentFolder = "Original600x600model";
    string receiversFile = experimentFolder + string("/receivers.txt");
    string sourcesFile = experimentFolder + string("/sources.txt");
    string dimensionFile = experimentFolder + string("/dimensions.txt");
    imat receivers;
    imat sources;
    vec dimensions;
    receivers.load(receiversFile);
    sources.load(sourcesFile);
    dimensions.load(dimensionFile);

    // Load dimensions
    uword nx = static_cast<uword>(dimensions[0]);
    double dx = dimensions[1];
    uword nz = static_cast<uword>(dimensions[2]);
    double dz = dimensions[3];
    uword np = static_cast<uword>(dimensions[4]); 
    double np_f = dimensions[5];

    // Create stf
    vec sourcefunction;
    double dt = dimensions[6];
    unsigned int nt = static_cast<unsigned int>(dimensions[7]);
    double freq = dimensions[8];
    sourcefunction = generateRicker(dt, nt, freq);

    // Loading set up
    fwiExperiment experiment(dx, dz, nx, nz, np, np_f, receivers, sources, sourcefunction, dt, nt, fwiShot::momentSource);

    // Loading material parameters
    mat rho = 1500 * ones(experiment.model.nx_interior, experiment.model.nz_interior);
    mat vp = 2000 * ones(experiment.model.nx_interior, experiment.model.nz_interior);
    mat vs = 800 * ones(experiment.model.nx_interior, experiment.model.nz_interior);

    vp(span(120, 125), span(100, 105)) -= 100;
//    vs(span(120, 121), span(100, 101)) -= 100;

    vp(span(280, 285), span(300, 305)) += 100;
//    vs(span(280, 281), span(300, 301)) += 100;

    experiment.update(rho, vp, vs);

    experiment.exportSnapshots = true;
//    experiment.useRamSnapshots = true;
    experiment.performFWI = false;

    for (unsigned int i = 0; i < nt; i += 10) {
        experiment.snapshots.emplace_back(i);
    }

    experiment.forwardData();

}