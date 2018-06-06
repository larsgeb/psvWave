//
// Created by lars on 26/03/18.
//

#ifndef HMC_FORWARD_FD_SHOT_H
#define HMC_FORWARD_FD_SHOT_H


#include <armadillo>
#include "fwiModel.h"

class fwiShot {
public:
    // Fields
    arma::mat moment;
    arma::irowvec source;
    arma::imat receivers;
    arma::vec sourceFunction;
    arma::mat seismogramSyn_ux;
    arma::mat seismogramSyn_uz;
    double samplingTimestep;
    double samplingTime;
    int samplingAmount;
    double samplingTimestepSyn;
    int samplingAmountSyn;
    arma::uword ishot;
    int snapshotInterval;

    arma::cube txxSnapshots;
    arma::cube tzzSnapshots;
    arma::cube txzSnapshots;
    arma::cube vxSnapshots;
    arma::cube vzSnapshots;

    arma::mat seismogramObs_ux;
    arma::mat seismogramObs_uz;

    arma::mat vxAdjointSource;
    arma::mat vzAdjointSource;

    enum SourceTypes {
        explosiveSource = 0, rotationalSource, momentSource, momentSourceHeaviside
    };

    // Constructor
    fwiShot(arma::irowvec _source, arma::imat &_receivers, arma::vec &_sourceFunction, int _samplingAmount, double samplingTimestep,
            double _samplingTime, arma::uword ishot, int _snapshotInterval, SourceTypes sourceType);

    // Methods
    void writeShot(arma::file_type type, std::string folder);

    void loadShot(std::string _folder);

    void calculateAdjointSources();

    void interpolateSynthetics();

    bool errorOnInterpolate = true;
    int sourceType;
};


#endif //HMC_FORWARD_FD_SHOT_H
