#include <iostream>
#include <armadillo>

#define _USE_MATH_DEFINES

#include <cmath>

// Own includes
#include "forward-psv/experiment.h"
#include "forward-psv/propagator.h"
#include "forward-psv/shot.h"

int main() {

    // These numbers don't include the boundary layers
//    arma::imat receivers = "0 0; 50 0; 100 0; 150 0; 200 0; 250 0;300 0; 350 0;400 0;";
    arma::imat receivers(40, 2, arma::fill::zeros);
    for (int irow = 0; irow < receivers.n_rows; irow++) {
        receivers(irow, 0) = irow*10;
    }
    arma::imat sources = "50 0;150 0;250 0;350 0;";

    double dt = 0.00025;
    int nt = 3500;

    double centralFrequency = 50.0;
    double tsource = 1.0 / centralFrequency;
    arma::vec time = arma::linspace(0, dt * (nt - 1), nt);
    double t0 = tsource * 1.5;
    arma::vec tau = M_PI * (time - t0) / t0;
    arma::vec sourceFunction = (1 - 4 * tau % tau) % arma::exp(-2.0 * tau % tau);

//    sourceFunction.save("source.txt");

    experiment experiment_1(receivers, sources, sourceFunction);

    auto xoff = experiment_1.currentModel.np_boundary;

    experiment_1.currentModel.mu(arma::span(xoff + 100, xoff + 199), arma::span(0, 99)) =
            1.5 * experiment_1.currentModel.mu(arma::span(xoff + 100, xoff + 199), arma::span(0, 99));
    experiment_1.currentModel.mu(arma::span(xoff + 200, xoff + 299), arma::span(100, 249)) =
            1.5 * experiment_1.currentModel.mu(arma::span(xoff + 200, xoff + 299), arma::span(100, 249));

    experiment_1.currentModel.lm = experiment_1.currentModel.la + 2 * experiment_1.currentModel.mu;

//    experiment_1.forwardData();

//    experiment_1.writeShots();

//    sources.save("sources.txt", arma::raw_ascii);
//    receivers.save("receivers.txt", arma::raw_ascii);

//    experiment_1.currentModel.mu.save("observed_data_mu.txt", arma::raw_ascii);
//    experiment_1.currentModel.la.save("observed_data_la.txt", arma::raw_ascii);
//    experiment_1.currentModel.lm.save("observed_data_lm.txt", arma::raw_ascii);
//    experiment_1.currentModel.b_vx.save("observed_data_li_vx.txt", arma::raw_ascii);
//    experiment_1.currentModel.b_vz.save("observed_data_li_vz.txt", arma::raw_ascii);

    return 0;
}