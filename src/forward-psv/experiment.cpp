//
// Created by lars on 17.03.18.
//

#include <omp.h>
#include "experiment.h"
#include "propagator.h"

experiment::experiment(arma::imat _receivers, arma::imat _sources, arma::vec _sourceFunction) {
    // Create a seismic experiment
    receivers = std::move(_receivers);
    sources = std::move(_sources);
    sourceFunction = std::move(_sourceFunction);
    dt = 0.00025;
    nt = 3500;

    // Check for positions
    for (auto &&yPosReceiver : receivers.col(1)) {
        if (yPosReceiver >= currentModel.nz_domain) {
            throw std::invalid_argument("Invalid y position for receiver (in or beyond the Gaussian taper).");
        }
    }

    for (auto &&yPosSource : sources.col(1)) {
        if (yPosSource >= currentModel.nz_domain) {
            throw std::invalid_argument("Invalid y position for receiver (in or beyond the Gaussian taper).");
        }
    }

    // This way all shots have the same receivers and sources
    for (int ishot = 0; ishot < sources.n_rows; ++ishot) {
        shots.emplace_back(shot(sources.row(ishot), receivers, sourceFunction, nt, dt, currentModel, ishot, snapshotInterval));
    }
}

void experiment::forwardData() {
    // Calculate forward data from each shot
    std::cout << "Starting forward modeling of shots." << std::endl;
    std::cout << "Total of " << sources.n_rows << " shots." << std::endl;

    double startTime = omp_get_wtime(); // for timing multithread code
    // Run forward simulation for all shots
    for (int iShot = 0; iShot < sources.n_rows; ++iShot) {
        // This directly modifies the forwardData fields
        std::cout << " -- Running shot: " << iShot << std::endl;
        propagator::propagateForward(currentModel, shots[iShot], true);
        std::cout << "    Done! " << std::endl;
    }
    double stopTime = omp_get_wtime();
    double secsElapsed = stopTime - startTime;

    std::cout << "Finished forward modeling of shots, elapsed time: " << secsElapsed << " seconds (wall)." << std::endl
              << std::endl;
}


void experiment::writeShots() {
    // Write forward data from shots out to text files
    for (auto &&shot : shots) {
        std::cout << " -- Exporting shot: " << shot.ishot << std::endl;
        shot.writeShot();
        std::cout << "    Done! " << std::endl;
    }
}

void experiment::computeKernel() {

}

double experiment::calculateMisfit() {
    std::cout << "Calculating misfit... ";
    misfit = 0;
    for (auto &&shot : shots) {
        misfit += arma::accu(arma::square(shot.seismogramObs_ux - shot.seismogramSyn_ux));
        misfit += arma::accu(arma::square(shot.seismogramObs_uz - shot.seismogramSyn_uz));
    }
    misfit = sqrt(misfit);
    std::cout << "done!"<< std::endl;
}

void experiment::calculateAdjointSources() {
    std::cout << "Calculating adjoint sources... ";
    for (auto &&shot : shots) {
        shot.calculateAdjointSources();
    }
    std::cout << "done!"<< std::endl;
}
