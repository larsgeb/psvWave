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
    fwiModel model; // Model object at current sample
    std::vector<fwiShot> shots; // All shots accumulated in vector

    // Wave analysis parameters
    bool exportSnapshots = false;
    std::vector<int> snapshots;

    // FWI parameters
    bool performFWI = true;
    int snapshotInterval = 10; // Interval used to store wavefields and compute kernels
    double misfit; // Misfit at current model. Needs to be explicitly computed first
    arma::mat muKernel_par1; // Mu kernel for free parameters mu, lambda, rho
    arma::mat densityKernel_par1; // Density kernel for free parameters mu, lambda, rho
    arma::mat lambdaKernel_par1; // Lambda kernel for free parameters mu, lambda, rho

    arma::mat densityKernel_par2; // Density kernel for free parameters rho, vp, vs
    arma::mat vpKernel_par2; // P-wave velocity kernel for free parameters rho, vp, vs
    arma::mat vsKernel_par2; // S-wave velocity uhhhhkernel for free parameters rho, vp, vs

    // Constructors
    fwiExperiment(arma::imat _receivers, arma::imat _sources, arma::vec _sourceFunction, double samplingTimestep, int samplingAmount,
                  fwiShot::SourceTypes sourceType); // Standard constructor supplying all necessary info

    fwiExperiment(double dx,
                  double dz,
                  arma::uword nx_interior,
                  arma::uword nz_interior,
                  arma::uword np_boundary,
                  double np_factor,
                  arma::imat _receivers,
                  arma::imat _sources,
                  arma::vec _sourceFunction,
                  double samplingTimestep,
                  int samplingAmount,
                  fwiShot::SourceTypes sourceType); // Standard constructor supplying all necessary info, but with variable size

    fwiExperiment(); // Default constructor

    // Methods
    void writeShots(arma::file_type type, std::string _folder); // Write shots from synthetic simulation to specified file type

    void forwardData(); // Forward propagateLeapFrog the stf over the supplied parameters

    void calculateMisfit(); // Calculate misfits of observed and synthetic data

    void computeKernel(); // compute all three medium parameter kernels

    void update(arma::mat _de, arma::mat _vp, arma::mat _vs);

    void loadShots(std::string _string); // Load shots (from binary format) into observed data

private:

    double samplingTime; // Length duration of the actual experiment (not the numerical simulation)
    double samplingTimestep; // Time step of the actual experiment (not the numerical simulation)
    int samplingAmount; // Amount of samples of the actual experiment (not the numerical simulation)

    void calculateAdjointSourcesL2(); // Calculate adjoint sources using L2 norm

    void backwardAdjoint(); // Back propagateLeapFrog the adjoint sources to compute kernels

    void mapKernels();

    bool useRamSnapshots = false;
};


#endif //HMC_FORWARD_FD_SETUP_H
