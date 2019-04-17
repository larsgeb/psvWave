//
// Created by lars on 25.01.19.
//

#ifndef FDWAVEMODEL_H
#define FDWAVEMODEL_H


#include <vector>


/** \typedef
 * \brief Typedef that determines simulation precision.
 *
 */
using real_simulation = float;

/** \brief Finite difference wave modelling class.
 *
 * Finite difference wave modelling class. Contains all configuration within the header, and as such is defined at compile time. Contains all
 * necessary functions to perform FWI, but lacks optimization schemes.
 */
class fdWaveModel { // TODO restructure public vs private methods and fields
public:
    // ---- CONSTRUCTORS AND DESTRUCTORS ----
    fdWaveModel();

    ~fdWaveModel();

    // ---- METHODS ----

    void forward_simulate(int i_shot, bool store_fields, bool verbose);

    void adjoint_simulate(int i_shot, bool verbose);

    void write_receivers();

    void write_sources();

    void load_receivers(bool verbose);

    void map_kernels_to_velocity();

    void update_from_velocity();

    /**
     * Function to calculate misfit between observed seismograms and synthetic seismograms.
     */
    void calculate_misfit();

    void calculate_adjoint_sources();

    void load_target(const std::string &de_target_relative_path, const std::string &vp_target_relative_path,
                     const std::string &vs_target_relative_path, bool verbose);

    void load_starting(const std::string &de_starting_relative_path, const std::string &vp_starting_relative_path,
                       const std::string &vs_starting_relative_path, bool verbose);

    void reset_velocity_fields();

    void reset_velocity_fields(bool reset_de, bool reset_vp, bool reset_vs);

    void run_model(bool verbose);

    void run_model(bool verbose, bool simulate_adjoint);

    void reset_kernels();

    void allocate_1d_array(real_simulation *&pDouble, int dim1);

    void allocate_2d_array(real_simulation **&pDouble, int dim1, int dim2);

    void allocate_3d_array(real_simulation ***&pDouble, int dim1, int dim2, int dim3);

    void allocate_4d_array(real_simulation ****&pDouble, int dim1, int dim2, int dim3, int dim4);

    void deallocate_1d_array(real_simulation *&pDouble);

    void deallocate_2d_array(real_simulation **&pDouble, int dim1);

    void deallocate_3d_array(real_simulation ***&pDouble, int dim1, int dim2);

    void deallocate_4d_array(real_simulation ****&pDouble, int dim1, int dim2, int dim3);

    void set_2d_array_to_zero(real_simulation **pDouble);

    // ----  FIELDS ----

    // |--< Utility fields >--
    // | Finite difference coefficients
    real_simulation c1 = real_simulation(9.0 / 8.0);
    real_simulation c2 = real_simulation(1.0 / 24.0);
    bool add_np_to_source_location = true;
    bool add_np_to_receiver_location = true;

    // |--< Spatial fields >--
    // | Dynamic physical fields
    real_simulation **vx;
    real_simulation **vz;
    real_simulation **txx;
    real_simulation **tzz;
    real_simulation **txz;
    // | Static physical fields
    real_simulation **lm;
    real_simulation **la;
    real_simulation **mu;
    real_simulation **b_vx;
    real_simulation **b_vz;
    real_simulation **rho;
    real_simulation **vp;
    real_simulation **vs;
    // | Sensitivity kernels in Lamé's basis
    real_simulation **lambda_kernel;
    real_simulation **mu_kernel;
    real_simulation **density_l_kernel;
    // | Sensitivity kernels in velocity basis
    real_simulation **vp_kernel;
    real_simulation **vs_kernel;
    real_simulation **density_v_kernel;
    // | Static physical fields for the starting model
    real_simulation **starting_rho;
    real_simulation **starting_vp;
    real_simulation **starting_vs;
    // | Taper field
    real_simulation **taper;

    // |--< Time dependent signals >--
    real_simulation *t;
    real_simulation **stf;
    real_simulation ***moment;
    real_simulation ***rtf_ux;
    real_simulation ***rtf_uz;
    real_simulation ***rtf_ux_true;
    real_simulation ***rtf_uz_true;
    real_simulation ***a_stf_ux;
    real_simulation ***a_stf_uz;
    real_simulation ****accu_vx;
    real_simulation ****accu_vz;
    real_simulation ****accu_txx;
    real_simulation ****accu_tzz;
    real_simulation ****accu_txz;

    // -- Definition of simulation --
    // | Input parameters
    const static int np_boundary = 10;
    real_simulation np_factor = 0.075;
    const static int nt = 8000;
    const static int nx_inner = 200;
    const static int nz_inner = 100;
    const static int nx_inner_boundary = 10;
    const static int nz_inner_boundary = 20;
    real_simulation dx = 1.249;
    real_simulation dz = 1.249;
    real_simulation dt = 0.00025;
    real_simulation scalar_rho = 1500;
    real_simulation scalar_vp = 2000;
    real_simulation scalar_vs = 800;
    const static int n_sources = 7;
    const static int n_shots = 1;
    std::vector<std::vector<int>> which_source_to_fire_in_which_shot = {{0, 1, 2, 3, 4, 5, 6}};
    real_simulation delay_per_shot = 12; // over f
    int ix_sources[n_sources] = {25, 50, 75, 100, 125, 150, 175};
    int iz_sources[n_sources] = {10, 10, 10, 10, 10, 10, 10};
    real_simulation moment_angles[n_sources] = {90, 81, 41, 300, 147, 252, 327};
    real_simulation alpha = static_cast<real_simulation>(1.0 / 50.0);
    real_simulation t0 = 0.005;
    const static int nr = 19;
    int ix_receivers[nr] = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190,};
    int iz_receivers[nr] = {90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90,};
    int snapshot_interval = 10;

    const static int snapshots = 800;
    const static int nx = nx_inner + np_boundary * 2;
    const static int nz = nz_inner + np_boundary;
    const static int nx_free_parameters = nx_inner - nx_inner_boundary * 2;
    const static int nz_free_parameters = nz_inner - nz_inner_boundary * 2;

    // -- Helper stuff for inverse problems --
    //real_simulation data_variance_ux[n_shots][nr][nt];
    //real_simulation data_variance_uz[n_shots][nr][nt];

    real_simulation misfit;
    std::string observed_data_folder = "observed_data";
    std::string stf_folder = "sources";

};


#endif //FDWAVEMODEL_H
