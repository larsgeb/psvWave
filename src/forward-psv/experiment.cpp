//
// Created by lars on 17.03.18.
//

#include <omp.h>
#include "experiment.h"
#include "propagator.h"

experiment::experiment(arma::imat _receivers, arma::imat _sources, arma::vec _sourceFunction, int nt, double dt) {
    // Create a seismic experiment
    receivers = std::move(_receivers);
    sources = std::move(_sources);
    sourceFunction = std::move(_sourceFunction);
    dt = dt;
    nt = nt;
    tTot = (nt - 1) * dt;

    // This way all shots have the same receivers and sources
    for (int iShot = 0; iShot < sources.n_rows; ++iShot) {
        shots.emplace_back(shot(sources.row(iShot), receivers, sourceFunction, nt, dt));
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
        propagator::propagateForward(currentModel, shots[iShot]);
        std::cout << "    Done! " << std::endl;
    }
    double stopTime = omp_get_wtime();
    double secsElapsed = stopTime - startTime;

    std::cout << "Finished forward modeling of shots, elapsed time: " << secsElapsed << " seconds (wall)." << std::endl
              << std::endl;
}
//
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
