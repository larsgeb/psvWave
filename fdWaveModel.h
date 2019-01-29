//
// Created by lars on 25.01.19.
//

#ifndef FDWAVEMODEL_H
#define FDWAVEMODEL_H


#include <vector>

#if OPENACCCOMPILE == 1
    #define OPENACC 1
#else
    #define OPENACC 0
#endif

#define real float

class fdWaveModel { // TODO restructure public vs private methods and fields
public:
    fdWaveModel();

    // ---- METHODS ----

    void forward_simulate(int i_shot, bool store_fields, bool verbose);

    void adjoint_simulate(int i_shot, bool verbose);

    void write_receivers();

    void load_receivers();

    void map_kernels_to_velocity();

    void update_from_velocity();

    real calculate_misfit();

    void calculate_adjoint_sources();

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
    const static int n_sources = 4;
    const static int n_shots = 4;
    std::vector<std::vector<int>> which_source_to_fire_in_which_shot = {{0},
                                                                        {1},
                                                                        {3},
                                                                        {2}};
    int ix_sources[n_sources] = {10, 10, 90, 90};
    int iz_sources[n_sources] = {30, 70, 70, 30};
    bool add_np_to_source_location = true;
    real alpha = static_cast<float>(1.0 / 50.0);
    real t0 = 0.005;
    // | stf/rtf_ux arrays
    real t[nt];
    real stf[nt];
    const static int nr = 12;
    int ix_receivers[nr] = {10, 50, 50, 90, 90, 90, 10, 10, 30, 30, 70, 70};
    int iz_receivers[nr] = {50, 10, 90, 10, 50, 90, 10, 90, 10, 90, 10, 90};
    bool add_np_to_receiver_location = true;
    real rtf_ux[n_shots][nr][nt];
    real rtf_uz[n_shots][nr][nt];
    real rtf_ux_true[n_shots][nr][nt];
    real rtf_uz_true[n_shots][nr][nt];

    real a_stf_ux[n_shots][nr][nt];
    real a_stf_uz[n_shots][nr][nt];

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
    int snapshot_interval = 10;
    const static int snapshots = 400;
    real accu_vx[n_shots][snapshots][nx][nz]; // todo Debate whether or not to add one dimension per shot, or just overwrite each simulation.
    real accu_vz[n_shots][snapshots][nx][nz];
    real accu_txx[n_shots][snapshots][nx][nz];
    real accu_tzz[n_shots][snapshots][nx][nz];
    real accu_txz[n_shots][snapshots][nx][nz];


    // -- Helper stuff for inverse problems --
    real data_variance_ux[n_shots][nr][nt];
    real data_variance_uz[n_shots][nr][nt];

    real density_l_kernel[nx][nz];
    real lambda_kernel[nx][nz];
    real mu_kernel[nx][nz];

    real vp_kernel[nx][nz];
    real vs_kernel[nx][nz];
    real density_v_kernel[nx][nz];
};


#endif //FDWAVEMODEL_H
