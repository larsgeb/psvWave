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

#define real float // todo Debug nan's on double

class fdWaveModel { // TODO restructure public vs private methods and fields
public:
    fdWaveModel();

    // ---- METHODS ----

    void forward_simulate(int i_shot, bool store_fields, bool verbose);

    void adjoint_simulate(int i_shot, bool verbose);

    void write_receivers();

    void write_sources();

    void load_receivers(bool verbose);

    void map_kernels_to_velocity();

    void update_from_velocity();

    real calculate_misfit();

    void calculate_adjoint_sources();

    // ----  FIELDS ----

    // -- Definition of simulation --
    // | Gaussian taper specs
    const static int np_boundary = 10;
    real np_factor = 0.075; // todo determine optimal
    // | Finite difference coefficients
    real c1 = real(9.0 / 8.0);
    real c2 = real(1.0 / 24.0);
    // | Simulation size
    const static int nt = 8000;
    const static int nx_inner = 200;
    const static int nz_inner = 100;
    const static int nx = nx_inner + np_boundary * 2;
    const static int nz = nz_inner + np_boundary;
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
    real starting_rho[nx][nz];
    real starting_vp[nx][nz];
    real starting_vs[nx][nz];
    real taper[nx][nz];
    // | Source parameters (Gaussian wavelet)
    const static int n_sources = 7;
    const static int n_shots = 1;
    std::vector<std::vector<int>> which_source_to_fire_in_which_shot = {{0, 1, 2, 3, 4, 5, 6}};
    real delay_per_shot = 12; // over f
    int ix_sources[n_sources] = {25, 50, 75, 100, 125, 150, 175};
    int iz_sources[n_sources] = {10, 10, 10, 10, 10, 10, 10};
    real moment_angles[n_sources] = {90, 81, 41, 300, 147, 252, 327};
    bool add_np_to_source_location = true;
    real alpha = static_cast<real>(1.0 / 50.0);
    real t0 = 0.005;
    // | stf/rtf_ux arrays
    real t[nt];
    real stf[n_sources][nt];
    const static int nr = 19;
    int ix_receivers[nr] = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190,};
    int iz_receivers[nr] = {90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90,};
    bool add_np_to_receiver_location = true;
    real rtf_ux[n_shots][nr][nt];
    real rtf_uz[n_shots][nr][nt];
    real rtf_ux_true[n_shots][nr][nt];
    real rtf_uz_true[n_shots][nr][nt];

    real a_stf_ux[n_shots][nr][nt];
    real a_stf_uz[n_shots][nr][nt];

    // | Source moment
    real moment[n_sources][2][2];
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
    const static int snapshots = 800;
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

    void load_target(std::string de_target_relative_path, std::string vp_target_relative_path, std::string vs_target_relative_path, bool verbose);

    void
    load_starting(std::string de_starting_relative_path, std::string vp_starting_relative_path, std::string vs_starting_relative_path, bool verbose);

    void reset_velocity_fields();

    void reset_velocity_fields(bool reset_de, bool reset_vp, bool reset_vs);
};


#endif //FDWAVEMODEL_H
