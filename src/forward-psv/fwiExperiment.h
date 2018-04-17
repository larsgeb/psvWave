//
// Created by lars on 17.03.18.
//

#ifndef HMC_FORWARD_FD_SETUP_H
#define HMC_FORWARD_FD_SETUP_H

#include <armadillo>
#include "shot.h"
#include "model.h"

class fwiExperiment {
public:
    // Fields
    arma::imat receivers;
    arma::imat sources;
    arma::vec sourceFunction;
    model currentModel;
    std::vector<shot> shots;
    double samplingTime;
    double samplingTimestep;
    int samplingAmount;

    // FWI parameters
    int snapshotInterval = 10;
    double misfit;
    arma::mat muKernel;
    arma::mat densityKernel;
    arma::mat lambdaKernel;

    // Constructors
    fwiExperiment(arma::imat _receivers, arma::imat _sources, arma::vec _sourceFunction, double samplingTime,
                  double samplingTimestep, int samplingAmount);

    fwiExperiment();

    // Methods
    void writeShots(arma::file_type type, std::string &_folder);

    void forwardData();

    void calculateMisfit();

    void computeKernel();

    void loadShots(std::string &_string);

private:

    void calculateAdjointSources();

    void backwardAdjoint();

};


#endif //HMC_FORWARD_FD_SETUP_H
