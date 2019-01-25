//
// Created by lars on 25.01.19.
//

#ifndef FDWAVEMODEL_H
#define FDWAVEMODEL_H


#if OPENACCCOMPILE == 1
    #define OPENACC 1
#else
    #define OPENACC 0
#endif

#define real float

class fdWaveModel {
public:
    fdWaveModel();

    // ---- METHODS ----

    int forwardSimulate(bool storeFields, bool verbose, int isource);

    // ----  FIELDS ----

    // -- Definition of simulation --
    // | Gaussian taper specs
    int np_boundary = 50;
    real np_factor = 0.0075; // todo determine otpimal
    // | Finite difference coefficients
    real c1 = real(9.0 / 8.0);
    real c2 = real(1.0 / 24.0);
    // | Simulation size
    const static int nt = 4000;
    const static int nx = 200;
    const static int nz = 150;
    // | Discretization size
    real dx = 1.249;
    real dz = 1.249;
    real dt = 0.00025;
    // | Background material parameters
    real scalar_rho = 1500;
    real scalar_vp = 2000;
    real scalar_vs = 800;
    real rho[nx][nz];
    real vp[nx][nz];
    real vs[nx][nz];
    real taper[nx][nz];
    // | Source parameters (Gaussian wavelet)
    int ix_source[2] = {10 + np_boundary, 10 + np_boundary};
    int iz_source[2] = {10 + np_boundary, 90 + np_boundary};
    real alpha = 1.0 / 50.0;
    real t0 = 0.005;
    // | stf/rtf_ux arrays
    real t[nt];
    real stf[nt];
    const static int nr = 100;
    int ix_receivers[nr];
    int iz_receivers[nr];
//    int ix_receivers[nr] = {90 + np_boundary, 90 + np_boundary, 90 + np_boundary, 50 + np_boundary, 50 + np_boundary, 10 + np_boundary};
//    int iz_receivers[nr] = {50 + np_boundary, 10 + np_boundary, 90 + np_boundary, 10 + np_boundary, 90 + np_boundary, 50 + np_boundary};
    real rtf_ux[2][nr][nt];
    real rtf_uz[2][nr][nt];
    // | Source moment
    real moment[2][2];
    // | Dynamic fields
    real vx[nx][nz];
    real vz[nx][nz];
    real txx[nx][nz];
    real tzz[nx][nz];
    real txz[nx][nz];
    // |  fields
    real lm[nx][nz] = {{1}};
    real la[nx][nz] = {{1}};
    real mu[nx][nz] = {{1}};
    real b_vx[nx][nz] = {{1}};
    real b_vz[nx][nz] = {{1}};
    // | accumulators and snapshot interval
    int snapshotInterval = 10;
    const static int snapshots = 400;
    real accu_vx[snapshots][nx][nz];
    real accu_vz[snapshots][nx][nz];
    real accu_txx[snapshots][nx][nz];
    real accu_tzz[snapshots][nx][nz];
    real accu_txz[snapshots][nx][nz];

};


#endif //FDWAVEMODEL_H
