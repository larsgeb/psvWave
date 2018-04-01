//
// Created by lars on 17.03.18.
//

#include <omp.h>
#include "experiment.h"
#include "propagator.h"

// Constructor
experiment::experiment(arma::imat _receivers, arma::imat _sources, arma::vec _sourceFunction) {
    // Create a seismic experiment
    receivers = std::move(_receivers);
    sources = std::move(_sources);
    sourceFunction = std::move(_sourceFunction);
    dt = 0.00025;
    nt = 3500;

    muKernel = arma::zeros(static_cast<const arma::uword>(currentModel.nx_domain),
                           static_cast<const arma::uword>(currentModel.nz_domain));
    densityKernel = arma::zeros(static_cast<const arma::uword>(currentModel.nx_domain),
                                static_cast<const arma::uword>(currentModel.nz_domain));
    lambdaKernel = arma::zeros(static_cast<const arma::uword>(currentModel.nx_domain),
                               static_cast<const arma::uword>(currentModel.nz_domain));

    // Check for positions
    for (auto &&yPosReceiver : receivers.col(1)) {
        if (yPosReceiver >= static_cast<int>(currentModel.nz_domain)) {
            throw std::invalid_argument("Invalid y position for receiver (in or beyond the Gaussian taper).");
        }
    }

    for (auto &&yPosSource : sources.col(1)) {
        if (yPosSource >= static_cast<int>(currentModel.nz_domain)) {
            throw std::invalid_argument("Invalid y position for receiver (in or beyond the Gaussian taper).");
        }
    }

    // This way all shots have the same receivers and sources
    for (arma::uword ishot = 0; ishot < sources.n_rows; ++ishot) {
        shots.emplace_back(
                shot(sources.row(ishot), receivers, sourceFunction, nt, dt, currentModel, ishot, snapshotInterval));
    }
}

void experiment::forwardData() {
    // Calculate forward data from each shot
    std::cout << "    Starting forward modeling of shots." << std::endl;
    std::cout << "    Total of " << sources.n_rows << " shots." << std::endl;

    double startTime = omp_get_wtime();
    // Run forward simulation for all shots
    for (arma::uword iShot = 0; iShot < sources.n_rows; ++iShot) {
        std::cout << " -- Running shot: " << iShot << std::endl;
        propagator::propagateForward(currentModel, shots[iShot]);
        std::cout << "    Done! " << std::endl;
    }
    double stopTime = omp_get_wtime();
    double secsElapsed = stopTime - startTime;

    std::cout << "    Finished forward modeling of shots, elapsed time: " << secsElapsed << " seconds (wall)."
              << std::endl;
}

void experiment::writeShots(arma::file_type type, char folder[]) {
    // Write forward data from shots out to text files
    for (auto &&shot : shots) {
        std::cout << " -- Exporting shot: " << shot.ishot << std::endl;
        shot.writeShot(type, folder);
        std::cout << "    Done! " << std::endl;
    }
}

void experiment::computeKernel() {
    muKernel = arma::zeros(currentModel.nx_domain, currentModel.nz_domain);
    densityKernel = arma::zeros(currentModel.nx_domain, currentModel.nz_domain);
    lambdaKernel = arma::zeros(currentModel.nx_domain, currentModel.nz_domain);

    backwardAdjoint();
}

void experiment::calculateMisfit() {
    std::cout << "Calculating misfit... ";
    misfit = 0;
    for (auto &&shot : shots) {
        misfit += 0.5 * shot.dt * arma::accu(arma::square(shot.seismogramObs_ux - shot.seismogramSyn_ux));
        misfit += 0.5 * shot.dt * arma::accu(arma::square(shot.seismogramObs_uz - shot.seismogramSyn_uz));
    }
    std::cout << "done!" << std::endl;
}

void experiment::calculateAdjointSources() {
    std::cout << "Calculating adjoint sources... ";
    for (auto &&shot : shots) {
        shot.calculateAdjointSources();
    }
    std::cout << "done!" << std::endl;
}

void experiment::backwardAdjoint() {
    // Calculate forward data from each shot
    std::cout << "    Starting backward adjoint modeling of shots." << std::endl;
    std::cout << "    Total of " << sources.n_rows << " shots." << std::endl;

    double startTime = omp_get_wtime(); // for timing multithread code
    // Run forward simulation for all shots
    for (arma::uword iShot = 0; iShot < sources.n_rows; ++iShot) {
        // This directly modifies the forwardData fields
        std::cout << " -- Running shot: " << iShot << std::endl;
        propagator::propagateAdjoint(currentModel, shots[iShot], densityKernel, muKernel, lambdaKernel);
        std::cout << "    Done! " << std::endl;
    }
    double stopTime = omp_get_wtime();
    double secsElapsed = stopTime - startTime;

    std::cout << "    Finished backward adjoint modeling of shots, elapsed time: " << secsElapsed << " seconds (wall)."
              << std::endl;
}

void experiment::loadShots(char *folder) {
    // Write forward data from shots out to text files
    for (auto &&shot : shots) {
        std::cout << " -- Loading shot: " << shot.ishot << std::endl;
        shot.loadShot(folder);
        std::cout << "    Done! " << std::endl;
    }
}
