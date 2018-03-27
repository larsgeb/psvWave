//
// Created by lars on 17.03.18.
//

#include <omp.h>
#include "experiment.h"
#include "propagator.h"

experiment::experiment(arma::imat _receivers, arma::imat _sources, arma::vec _sourceFunction, int _nt, double _dt) {
    // Create a seismic experiment
    receivers = std::move(_receivers);
    sources = std::move(_sources);
    sourceFunction = std::move(_sourceFunction);
    dt = _dt;
    nt = _nt;
    tTot = (nt - 1) * dt;

    // Check for positions
    for (auto &&yPosReceiver : receivers.col(1)) {
        if (yPosReceiver == 0) {
            throw std::invalid_argument("Invalid y position for receiver (in the free surface layer).");
        }
        if (yPosReceiver >= currentModel.nz_domain) {
            throw std::invalid_argument("Invalid y position for receiver (in or beyond the Gaussian taper).");
        }
    }

    for (auto &&yPosSource : sources.col(1)) {
        if (yPosSource == 0) {
            throw std::invalid_argument("Invalid y position for source (in the free surface layer).");
        }
        if (yPosSource >= currentModel.nz_domain) {
            throw std::invalid_argument("Invalid y position for receiver (in or beyond the Gaussian taper).");
        }
    }

    // This way all shots have the same receivers and sources
    for (int iShot = 0; iShot < sources.n_rows; ++iShot) {
        shots.emplace_back(shot(sources.row(iShot), receivers, sourceFunction, nt, dt, currentModel));
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
        std::cout << " -- Running shot: " << iShot + 1 << std::endl;
        propagator::propagateForward(currentModel, shots[iShot], true);
        std::cout << "    Done! " << std::endl;
    }
    double stopTime = omp_get_wtime();
    double secsElapsed = stopTime - startTime;

    std::cout << "Finished forward modeling of shots, elapsed time: " << secsElapsed << " seconds (wall)." << std::endl
              << std::endl;
}

void experiment::backwardData() {
    // Calculate backward data from each shot
    std::cout << "Starting backward modeling of shots." << std::endl;
    std::cout << "Total of " << sources.n_rows << " shots." << std::endl;

    double startTime = omp_get_wtime(); // for timing multithread code
    // Run forward simulation for all shots
    for (int iShot = 0; iShot < sources.n_rows; ++iShot) {
        // This directly modifies the forwardData fields
        std::cout << " -- Running shot: " << iShot + 1 << std::endl;
        propagator::propagateBackward(currentModel, shots[iShot]);
        std::cout << "    Done! " << std::endl;
    }
    double stopTime = omp_get_wtime();
    double secsElapsed = stopTime - startTime;

    std::cout << "Finished backward modeling of shots, elapsed time: " << secsElapsed << " seconds (wall)." << std::endl
              << std::endl;
}

//void experiment::writeShots() {
//    // Write forward data from shots out to text files
//    for (int iShot = 0; iShot < sources.n_rows; ++iShot) {
//        // This directly modifies the forwardData fields
//        std::cout << " -- Exporting shot (using 1 based index): " << iShot + 1 << std::endl;
//        std::stringstream filename;
//        filename << "shot-" << iShot << ".txt" << std::endl;
//        shots[iShot].writeShot(filename.str());
//        std::cout << "    Done! " << std::endl;
//    }
//}
