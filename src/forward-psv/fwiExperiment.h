//
// Created by Lars Gebraad on 17.03.18.
//

#ifndef HMC_FORWARD_FD_SETUP_H
#define HMC_FORWARD_FD_SETUP_H

#include <armadillo>
#include "fwiShot.h"
#include "fwiModel.h"

class fwiExperiment {
    // Class which contains the basic elements of a full waveform experiment. Contains all the basic functions for FWI, which are delegated to shots
    // and receiver. Also contains the model object and other simulation parameters. Can be used for synthetics as well as inversion.
public:
    // Fields
    arma::imat receivers; // Indexes on the numerical grid for receivers
    arma::imat sources; // Indexes on the numerical grid for sources
    arma::vec sourceFunction; // Single source-time function for all shots
    fwiModel currentModel; // Model object at current sample
    std::vector<fwiShot> shots; // All shots accumulated in vector
    double samplingTime; // Length duration of the actual experiment (not the numerical simulation)
    double samplingTimestep; // Time step of the actual experiment (not the numerical simulation)
    int samplingAmount; // Amount of samples of the actual experiment (not the numerical simulation)

    // FWI parameters
    int snapshotInterval = 10; // Interval used to store wavefields and compute kernels
    double misfit; // Misfit at current model. Needs to be explicitly computed first
    arma::mat muKernel; // Mu kernel for free parameters mu, lambda, rho
    arma::mat densityKernel; // Density kernel for free parameters mu, lambda, rho
    arma::mat lambdaKernel; // Lambda kernel for free parameters mu, lambda, rho

    // Constructors
    fwiExperiment(arma::imat _receivers, arma::imat _sources, arma::vec _sourceFunction, double samplingTime, double samplingTimestep,
                  int samplingAmount); // Standard constructor supplying all necessary info

    fwiExperiment(); // Default constructor

    // Methods
    void writeShots(arma::file_type type, std::string &_folder); // Write shots from synthetic simulation to specified file type

    void forwardData(); // Forward propagate the stf over the supplied parameters

    void calculateMisfit(); // Calculate misfits of observed and synthetic data

    void computeKernel(); // compute all three medium parametr kernels

    void loadShots(std::string &_string); // Load shots (from binary format) into observed data

private:

    void calculateAdjointSourcesL2(); // Calculate adjoint sources using L2 norm

    void backwardAdjoint(); // Back propagate the adjoint sources to compute kernels

};


#endif //HMC_FORWARD_FD_SETUP_H
