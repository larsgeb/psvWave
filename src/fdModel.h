//
// Created by lars on 25.01.19.
//

#ifndef FDMODEL_H
#define FDMODEL_H

#include <string>

#include <vector>

#include "Eigen/Dense"
#include "Eigen/Sparse"

#include "contiguous_arrays.h"
//! Typedef that determines simulation precision.
//! On x86_64 systems it is typically fastest to use double,
//! as this is the native precision. Double has about a 10 %
//! performance gain over floats as  tested on Ubuntu 18.04 x86_64.
using real_simulation = double;

//! Typedef that is a shorthand for the correct precision column vector.
//! This vector has the right precision and shape to be used in matrix equations.
//! It is of dynamic size.
using dynamic_vector = Eigen::Matrix<real_simulation, Eigen::Dynamic, 1>;

//! \brief Finite difference wave modelling class.
//!
//! This class contains everything needed to do finite difference wave forward
//! and adjoint modelling.  It contains the entire experimental parameters as
//! fields, which are loaded at runtime from the supplied .ini file. The class
//! contains all necessary functions to perform FWI, but lacks optimization
//! schemes.
class fdModel {
public:
  // ---- CONSTRUCTORS AND DESTRUCTORS ----
  //!  \brief Constructor for modelling class.
  //!
  //!  This constructor creates the modelling class from a configuration file
  //!  supplied at runtime. As such, all fields are created dynamically.
  //!
  //!  @param configuration_file_relative_path Relative path to the configuration
  //!  .ini file. This file should contain all the fields needed for simulation.
  //!  Arbitrary defaults are hardcoded into the binary as backup within the
  //!  parse_configuration() method.
  explicit fdModel(const char *configuration_file_relative_path);

  fdModel(const int nt, const int nx_inner, const int nz_inner,
          const int nx_inner_boundary, const int nz_inner_boundary,
          const real_simulation dx, const real_simulation dz, const real_simulation dt,
          const int np_boundary, const real_simulation np_factor,
          const real_simulation scalar_rho, const real_simulation scalar_vp,
          const real_simulation scalar_vs, const int npx, const int npz,
          const real_simulation peak_frequency, const real_simulation source_timeshift,
          const real_simulation delay_cycles_per_shot, const int n_sources,
          const int n_shots, const std::vector<int> ix_sources_vector,
          const std::vector<int> iz_sources_vector,
          const std::vector<real_simulation> moment_angles_vector,
          const std::vector<std::vector<int>> which_source_to_fire_in_which_shot,
          const int nr, const std::vector<int> ix_receivers_vector,
          const std::vector<int> iz_receivers_vector, const int snapshot_interval,
          const std::string observed_data_folder, const std::string stf_folder);

  fdModel(const fdModel &model);

  //!  \brief Destructor for the class.
  //!
  //!  The destructor properly addresses every used new keyword in the
  //!  constructor, freeing all memory.
  ~fdModel();

  void allocate_memory();

  void initialize_arrays();
  void copy_arrays(const fdModel &model);

  void parse_parameters(const std::vector<int> ix_sources_vector,
                        const std::vector<int> iz_sources_vector,
                        const std::vector<real_simulation> moment_angles_vector,
                        const std::vector<int> ix_receivers_vector,
                        const std::vector<int> iz_receivers_vector);

  // ---- METHODS ----
  //!  \brief Method that parses .ini configuration file. Only used in
  //!  fdModel().
  //!
  //!  @param configuration_file_relative_path Relative path to the configuration
  //!  .ini file.
  void parse_configuration_file(const char *configuration_file_relative_path);

  //!  \brief Method to forward simulate wavefields for a specific shot.
  //!
  //!  Forward simulate wavefields of shot i_shot based on currently loaded
  //!  models. Storage of wavefields and verbosity of run can be toggled.
  //!
  //!  @param i_shot Integer controlling which shot to simulate.
  //!  @param store_fields Boolean to control storage of wavefields. If storage is
  //!  not required (i.e. no adjoint modeling), forward simulation should be
  //!  faster without storage.
  //!  @param verbose Boolean controlling if modelling should be verbose.
  void forward_simulate(int i_shot, bool store_fields, bool verbose,
                        bool output_wavefields = false);

  //!  \brief Method to adjoint simulate wavefields for a specific shot.
  //!
  //!  Adjoint simulate wavefields of shot i_shot based on currently loaded models
  //!  and calculate adjoint sources. Verbosity of run can be toggled.
  //!
  //!  @param i_shot Integer controlling which shot to simulate.
  //!  @param verbose Boolean controlling if modelling should be verbose.
  void adjoint_simulate(int i_shot, bool verbose);

  //!  \brief Method to write out synthetic seismograms to plaintext.
  //!
  //!  This method writes out the synthetic seismograms to a plaintext file.
  //!  Allows one to e.g. subsequently import these files as observed seismograms
  //!  later using load_receivers(). Every shot generates a separate ux and uz
  //!  receiver file (rtf_ux/rtf_uz), with every receiver being a single line in
  //!  these files.
  void write_receivers();

  void write_receivers(std::string prefix);

  //!  \brief Method to write out source signals to plaintext.
  //!
  //!  This method writes out the source time function (without moment tensor) to
  //!  plaintext file. Useful for e.g. visualizing the source staggering.
  void write_sources();

  //!  \brief Method to load receiver files.
  //!
  //!  This method loads receiver data from observed_data_folder folder into the
  //!  object. The data has to exactly match the set-up, and be named according to
  //!  component and shot (as generated by write_receivers() ).
  //!
  //!  @param verbose Controls the verbosity of the method during loading.
  void load_receivers(bool verbose);

  //!  \brief Method to map kernels to velocity parameter set.
  //!
  //!  This method takes the kernels (lambda, mu, rho) as originally calculated on
  //!  the Lamé's parameter set and maps them to the velocity parameter set (vp,
  //!  vs, rho).
  void map_kernels_to_velocity();

  //!  \brief Method to map velocities into Lamé's parameter set.
  //!
  //!  This method updates Lamé's parameters of the current model based on the
  //!  velocity parameters of the current model. Typically has to be done every
  //!  time after updating velocity.
  void update_from_velocity();

  //!  \brief Method to calculate L2 misfit.
  //!
  //!  This method calculates L2 misfit between observed seismograms and synthetic
  //!  seismograms and stores it in the misfit field.
  void calculate_l2_misfit();

  //!  \brief Method to calculate L2 adjoint sources
  //!  This method calculates L2 misfit between observed seismograms and synthetic
  //!  seismograms and stores it in the misfit field.
  void calculate_l2_adjoint_sources();

  //!  \brief Method to load models from plaintext into the model.
  //!
  //!  This methods loads any appropriate model (expressed in density, P-wave
  //!  velocity, and S-wave velocity) into the class and updates the Lamé fields
  //!  accordingly.
  //!
  //!  @param de_path Relative path to plaintext de file.
  //!  @param vp_path Relative path to plaintext vp file.
  //!  @param vs_path Relative path to plaintext vs file.
  //!  @param verbose Boolean controlling the verbosity of the method.
  void load_model(const std::string &de_path, const std::string &vp_path,
                  const std::string &vs_path, bool verbose);

  //!  \brief Method to perform all steps necessary for FWI, with additional
  //!  control over adjoint simulation.
  //!
  //!  This method performs all necessary steps in FWI; forward modelling, misfit
  //!  calculation and optionally adjoint source calculation, adjoint modelling
  //!  and kernel projection.
  //!
  //!  @param verbose Boolean controlling the verbosity of the method.
  //!  @param simulate_adjoint Boolean controlling the execution of the adjoint
  //!  simulation and kernel computation.
  void run_model(bool verbose, bool simulate_adjoint);

  //!  \brief Method to reset all Lamé sensitivity kernels to zero.
  //!
  //!  This method resets all sensitivity kernels to zero. Essential before
  //!  performing new adjoint simulations, as otherwise the kernels of subsequent
  //!  simulations would stack.
  void reset_kernels();

  // ----  FIELDS ----
  // |--< Utility fields >--
  // | Finite difference coefficients
  real_simulation c1 = real_simulation(9.0 / 8.0);
  real_simulation c2 = real_simulation(1.0 / 24.0);
  // Todo refactor into configuration
  bool add_np_to_source_location = true;
  bool add_np_to_receiver_location = true;

  // |--< Spatial fields >--
  // | Dynamic physical fields
  real_simulation **vx;  //!< Dynamic horizontal velocity field used in the simulations.
  real_simulation **vz;  //!< Dynamic vertical velocity field used in the simulations.
  real_simulation **txx; //!< Dynamic horizontal stress field used in the simulations.
  real_simulation **tzz; //!< Dynamic vertical stress field used in the simulations.
  real_simulation **txz; //!< Dynamic shear stress field used in the simulations.
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
  int *ix_receivers;
  int *iz_receivers;
  int snapshot_interval;

  int snapshots;
  int nx;
  int nz;
  int nx_free_parameters;
  int nz_free_parameters;

  int basis_gridpoints_x = 1; // How many gridpoints there are in a basis function
  int basis_gridpoints_z = 1;
  int free_parameters;

  // -- Helper stuff for inverse problems --
  // real_simulation data_variance_ux[n_shots][nr][nt];
  // real_simulation data_variance_uz[n_shots][nr][nt];

  real_simulation misfit;
  std::string observed_data_folder;
  std::string stf_folder;

  void write_kernels();

  dynamic_vector get_model_vector();
  void set_model_vector(dynamic_vector m);
  dynamic_vector get_gradient_vector();
  dynamic_vector load_vector(const std::string &vector_path, bool verbose);
};

// Miscellaneous functions

//! \brief Function to parse strings containing lists to std::vectors.
//!
//! Parses any string in the format {a, b, c, d, ...} to a vector of int, float,
//! double. Only types that can be cast to floats are supported for now. This is
//! due to the usage of strtof(); no template variant is used as of yet. The
//! input string is allowed to be trailed by a comment leading with a semicolon.
//! Items are appended to the passed vector.
//!
//! @param T Arbitrary numeric type that has a cast to and from float.
//! @param string_to_parse Input string of the form "{<item1>, <item2>, <item3>,
//! ... } ; comment ".
//! @param destination_vector Pointer to an (not necessarily empty) vector of
//! suitable type.
template <class T>
void parse_string_to_vector(std::basic_string<char> string_to_parse,
                            std::vector<T> *destination_vector);

//! \brief Function to parse strings containing 2d integer lists to
//! std::vector<std::vector<int>>. Sublists do not need to be of the same length.
//!
//! @param string_to_parse Input string of the form "{{<item1.1>, <item1.2>,
//! < item1.3>, ...}, {<item2.1>, <item2.2>,
//! ...}, ... } ; comment ".
//! @param destination_vector Pointer to an (not necessarily empty)
//! std::vector<std::vector<int>>.
void parse_string_to_nested_int_vector(
    std::basic_string<char> string_to_parse,
    std::vector<std::vector<int>> *destination_vector);

std::string zero_pad_number(int num, int pad);

#endif // FDMODEL_H
