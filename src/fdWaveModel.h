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
    fdWaveModel(const char *configuration_file);

    ~fdWaveModel();

    // ---- METHODS ----

    void parse_configuration(const char config_file[]);

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

    void load_model(const std::string &de_path, const std::string &vp_path,
                    const std::string &vs_path, bool verbose);

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
    real_simulation ****accu_vx; // Todo there is some ram usage to be optimized here as not the full wavefields have to be stored.
    real_simulation ****accu_vz; // Todo there is some ram usage to be optimized here as not the full wavefields have to be stored.
    real_simulation ****accu_txx; // Todo there is some ram usage to be optimized here as not the full wavefields have to be stored.
    real_simulation ****accu_tzz; // Todo there is some ram usage to be optimized here as not the full wavefields have to be stored.
    real_simulation ****accu_txz; // Todo there is some ram usage to be optimized here as not the full wavefields have to be stored.

    // -- Definition of simulation --
    // | Domain
    int nt;
    int nx_inner;
    int nz_inner;
    int nx_inner_boundary;
    int nz_inner_boundary;
    real_simulation dx;
    real_simulation dz;
    real_simulation dt;
    // | Boundary
    int np_boundary;
    real_simulation np_factor;
    // | Medium
    real_simulation scalar_rho;
    real_simulation scalar_vp;
    real_simulation scalar_vs;
    // | Sources
    int n_sources;
    int n_shots;
    std::vector<std::vector<int>> which_source_to_fire_in_which_shot;
    real_simulation delay_cycles_per_shot; // over f
    int *ix_sources;
    int *iz_sources;
    real_simulation *moment_angles;
    real_simulation peak_frequency;
    real_simulation alpha;
    real_simulation t0;
    int nr;
    int* ix_receivers;
    int* iz_receivers;
    int snapshot_interval;

    int snapshots;
    int nx;
    int nz;
    int nx_free_parameters;
    int nz_free_parameters;

    // -- Helper stuff for inverse problems --
    //real_simulation data_variance_ux[n_shots][nr][nt];
    //real_simulation data_variance_uz[n_shots][nr][nt];

    real_simulation misfit;
    std::string observed_data_folder = ".";
    std::string stf_folder = ".";
};

template<class T>
void parse_string_to_vector(std::basic_string<char> string_to_parse, std::vector<T> *destination_vector);

void parse_string_to_nested_vector(std::basic_string<char> string_to_parse, std::vector<std::vector<int>> *destination_vector);

#endif //FDWAVEMODEL_H
