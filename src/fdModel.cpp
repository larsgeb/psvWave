//
// Created by Lars Gebraad on 25.01.19.
//
#include "fdModel.h"
#include "INIReader.h"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <omp.h>
#include <stdexcept>

#define PI 3.14159265

fdModel::fdModel(const char *configuration_file_relative_path)
{
  // --- Initialization section ---

  // Parse configuration from supplied file
  parse_configuration_file(configuration_file_relative_path);

  // Allocate all fields dynamically (grouped by type)
  allocate_memory();

  initialize_arrays();
}

fdModel::fdModel(
    const int nt, const int nx_inner, const int nz_inner, const int nx_inner_boundary,
    const int nz_inner_boundary, const real_simulation dx, const real_simulation dz,
    const real_simulation dt, const int np_boundary, const real_simulation np_factor,
    const real_simulation scalar_rho, const real_simulation scalar_vp,
    const real_simulation scalar_vs, const int npx, const int npz,
    const real_simulation peak_frequency, const real_simulation source_timeshift,
    const real_simulation delay_cycles_per_shot, const int n_sources, const int n_shots,
    const std::vector<int> ix_sources_vector, const std::vector<int> iz_sources_vector,
    const std::vector<real_simulation> moment_angles_vector,
    const std::vector<std::vector<int>> which_source_to_fire_in_which_shot,
    const int nr, const std::vector<int> ix_receivers_vector,
    const std::vector<int> iz_receivers_vector, const int snapshot_interval,
    const std::string observed_data_folder, const std::string stf_folder)
    : nt(nt), nx_inner(nx_inner), nz_inner(nz_inner),
      nx_inner_boundary(nx_inner_boundary), nz_inner_boundary(nz_inner_boundary),
      dx(dx), dz(dz), dt(dt), np_boundary(np_boundary), np_factor(np_factor),
      scalar_rho(scalar_rho), scalar_vp(scalar_vp), scalar_vs(scalar_vs),
      basis_gridpoints_x(npx), basis_gridpoints_z(npz), peak_frequency(peak_frequency),
      t0(source_timeshift), delay_cycles_per_shot(delay_cycles_per_shot),
      n_sources(n_sources), n_shots(n_shots),
      which_source_to_fire_in_which_shot(which_source_to_fire_in_which_shot), nr(nr),
      snapshot_interval(snapshot_interval), observed_data_folder(observed_data_folder),
      stf_folder(stf_folder)
{

  parse_parameters(ix_sources_vector, iz_sources_vector, moment_angles_vector,
                   ix_receivers_vector, iz_receivers_vector);

  // Allocate all fields dynamically (grouped by type)
  allocate_memory();

  initialize_arrays();
}

fdModel::fdModel(const fdModel &model)
    : nt(model.nt), nx_inner(model.nx_inner), nz_inner(model.nz_inner),
      nx_inner_boundary(model.nx_inner_boundary),
      nz_inner_boundary(model.nz_inner_boundary), dx(model.dx), dz(model.dz),
      dt(model.dt), np_boundary(model.np_boundary), np_factor(model.np_factor),
      scalar_rho(model.scalar_rho), scalar_vp(model.scalar_vp),
      scalar_vs(model.scalar_vs), basis_gridpoints_x(model.basis_gridpoints_x),
      basis_gridpoints_z(model.basis_gridpoints_z),
      peak_frequency(model.peak_frequency), t0(model.t0),
      delay_cycles_per_shot(model.delay_cycles_per_shot), n_sources(model.n_sources),
      n_shots(model.n_shots),
      which_source_to_fire_in_which_shot(model.which_source_to_fire_in_which_shot),
      nr(model.nr), snapshot_interval(model.snapshot_interval),
      observed_data_folder(model.observed_data_folder), stf_folder(model.stf_folder)
{

  std::vector<int> ix_sources_vector(model.ix_sources,
                                     model.ix_sources + model.n_sources);
  std::vector<int> iz_sources_vector(model.iz_sources,
                                     model.iz_sources + model.n_sources);
  std::vector<real_simulation> moment_angles_vector(
      model.moment_angles, model.moment_angles + model.n_sources);
  std::vector<int> ix_receivers_vector(model.ix_receivers,
                                       model.ix_receivers + model.nr);
  std::vector<int> iz_receivers_vector(model.iz_receivers,
                                       model.iz_receivers + model.nr);

  parse_parameters(ix_sources_vector, iz_sources_vector, moment_angles_vector,
                   ix_receivers_vector, iz_receivers_vector);

  allocate_memory();

  copy_arrays(model);
}

fdModel::~fdModel()
{
  deallocate_array(vx);
  deallocate_array(vz);
  deallocate_array(txx);
  deallocate_array(tzz);
  deallocate_array(txz);
  deallocate_array(lm);
  deallocate_array(la);
  deallocate_array(mu);
  deallocate_array(b_vx);
  deallocate_array(b_vz);
  deallocate_array(rho);
  deallocate_array(vp);
  deallocate_array(vs);
  deallocate_array(density_l_kernel);
  deallocate_array(lambda_kernel);
  deallocate_array(mu_kernel);
  deallocate_array(vp_kernel);
  deallocate_array(vs_kernel);
  deallocate_array(density_v_kernel);
  deallocate_array(starting_rho);
  deallocate_array(starting_vp);
  deallocate_array(starting_vs);
  deallocate_array(taper);
  deallocate_array(t);
  deallocate_array(stf);
  deallocate_array(moment);
  deallocate_array(accu_vx);
  deallocate_array(accu_vz);
  deallocate_array(accu_txx);
  deallocate_array(accu_tzz);
  deallocate_array(accu_txz);
  deallocate_array(ix_receivers);
  deallocate_array(iz_receivers);
  deallocate_array(ix_sources);
  deallocate_array(iz_sources);
  deallocate_array(moment_angles);
}

void fdModel::allocate_memory()
{
  shape_grid = {nx, nz};

  allocate_array(vx, shape_grid);
  allocate_array(vz, shape_grid);
  allocate_array(txx, shape_grid);
  allocate_array(tzz, shape_grid);
  allocate_array(txz, shape_grid);
  allocate_array(lm, shape_grid);
  allocate_array(la, shape_grid);
  allocate_array(mu, shape_grid);
  allocate_array(b_vx, shape_grid);
  allocate_array(b_vz, shape_grid);
  allocate_array(rho, shape_grid);
  allocate_array(vp, shape_grid);
  allocate_array(vs, shape_grid);
  allocate_array(density_l_kernel, shape_grid);
  allocate_array(lambda_kernel, shape_grid);
  allocate_array(mu_kernel, shape_grid);
  allocate_array(vp_kernel, shape_grid);
  allocate_array(vs_kernel, shape_grid);
  allocate_array(density_v_kernel, shape_grid);
  allocate_array(starting_rho, shape_grid);
  allocate_array(starting_vp, shape_grid);
  allocate_array(starting_vs, shape_grid);
  allocate_array(taper, shape_grid);

  shape_t = {nt};

  allocate_array(t, shape_t);

  shape_stf = {n_sources, nt};
  allocate_array(stf, shape_stf);

  shape_moment = {n_sources, 2, 2};
  allocate_array(moment, shape_moment);

  shape_receivers = {n_shots, nr, nt};
  allocate_array(rtf_ux, shape_receivers);
  allocate_array(rtf_uz, shape_receivers);
  allocate_array(rtf_ux_true, shape_receivers);
  allocate_array(rtf_uz_true, shape_receivers);
  allocate_array(a_stf_ux, shape_receivers);
  allocate_array(a_stf_uz, shape_receivers);

  shape_accu = {n_shots, snapshots, nx, nz};
  allocate_array(accu_vx, shape_accu);
  allocate_array(accu_vz, shape_accu);
  allocate_array(accu_txx, shape_accu);
  allocate_array(accu_tzz, shape_accu);
  allocate_array(accu_txz, shape_accu);
}

void fdModel::initialize_arrays()
{

  // Place sources/receivers inside the domain if required
  if (add_np_to_receiver_location)
  {
    for (int ir = 0; ir < nr; ++ir)
    {
      ix_receivers[ir] += np_boundary;
      iz_receivers[ir] += np_boundary;
    }
  }
  if (add_np_to_source_location)
  {
    for (int is = 0; is < n_sources; ++is)
    {
      ix_sources[is] += np_boundary;
      iz_sources[is] += np_boundary;
    }
  }

  // Assign source time function and time axis based on set-up
  for (int i_shot = 0; i_shot < n_shots; ++i_shot)
  {
    for (int i_source = 0; i_source < which_source_to_fire_in_which_shot[i_shot].size();
         ++i_source)
    {
      for (unsigned int it = 0; it < nt; ++it)
      {
        t[it] = it * dt;
        auto f = static_cast<real_simulation>(1.0 / alpha);
        auto shiftedTime = static_cast<real_simulation>(
            t[it] - 1.4 / f - delay_cycles_per_shot * i_source / f);
        stf[linear_IDX(which_source_to_fire_in_which_shot[i_shot][i_source], it, n_sources, nt)] =
            real_simulation((1 - 2 * pow(M_PI * f * shiftedTime, 2)) *
                            exp(-pow(M_PI * f * shiftedTime, 2)));
      }
    }
  }

  // Assign moment tensors based on rotation.
  for (int i_source = 0; i_source < n_sources;
       i_source++)
  { // todo allow for more complex moment tensors

    moment[linear_IDX(i_source, 0, 0, n_sources, 2, 2)] =
        static_cast<real_simulation>(cos(moment_angles[i_source] * PI / 180.0) * 1e15);
    moment[linear_IDX(i_source, 0, 1, n_sources, 2, 2)] =
        static_cast<real_simulation>(-sin(moment_angles[i_source] * PI / 180.0) * 1e15);
    moment[linear_IDX(i_source, 1, 0, n_sources, 2, 2)] =
        static_cast<real_simulation>(-sin(moment_angles[i_source] * PI / 180.0) * 1e15);
    moment[linear_IDX(i_source, 1, 1, n_sources, 2, 2)] =
        static_cast<real_simulation>(-cos(moment_angles[i_source] * PI / 180.0) * 1e15);
  }

// Set all fields to background value so as to at least initialize.
#pragma omp parallel for collapse(2)
  for (int ix = 0; ix < nx; ++ix)
  {
    for (int iz = 0; iz < nz; ++iz)
    {
      vp[linear_IDX(ix, iz, nx, nz)] = scalar_vp;
      vs[linear_IDX(ix, iz, nx, nz)] = scalar_vs;
      rho[linear_IDX(ix, iz, nx, nz)] = scalar_rho;
    }
  }

  // Update Lamé's fields from velocity fields.
  update_from_velocity();

// Initialize Gaussian taper by ...
#pragma omp parallel for collapse(2)
  for (int ix = 0; ix < nx; ++ix)
  { // ... starting with zero taper in every point, ...
    for (int iz = 0; iz < nz; ++iz)
    {
      taper[linear_IDX(ix, iz, nx, nz)] = 0.0;
    }
  }

  for (int id = 0; id < np_boundary;
       ++id)
  { // ... subsequently, move from outside inwards over the np,
    // adding one to every point ...
#pragma omp parallel for collapse(2)
    for (int ix = id; ix < nx - id; ++ix)
    {
      for (int iz = id; iz < nz - id; ++iz)
      { // (hardcoded free surface boundaries by
        // not moving iz < nz)
        taper[linear_IDX(ix, iz, nx, nz)]++;
      }
    }
  }
#pragma omp parallel for collapse(2)
  for (int ix = 0; ix < nx;
       ++ix)
  { // ... and finally setting the maximum taper value to taper 1
    // using exponential function, decaying outwards.
    for (int iz = 0; iz < nz; ++iz)
    {
      auto lin_idx = linear_IDX(ix, iz, nx, nz);
      taper[lin_idx] = static_cast<real_simulation>(
          exp(-pow(np_factor * (np_boundary - taper[lin_idx]), 2)));
    }
  }
}

void fdModel::copy_arrays(const fdModel &model)
{

#pragma omp parallel for collapse(2)
  for (int ix = 0; ix < nx; ix++)
  {
    for (int iz = 0; iz < nz; iz++)
    {
      auto idx = linear_IDX(ix, iz, nx, nz);
      vx[idx] = model.vx[idx];
      vz[idx] = model.vz[idx];
      txx[idx] = model.txx[idx];
      tzz[idx] = model.tzz[idx];
      txz[idx] = model.txz[idx];
      lm[idx] = model.lm[idx];
      la[idx] = model.la[idx];
      mu[idx] = model.mu[idx];
      b_vx[idx] = model.b_vx[idx];
      b_vz[idx] = model.b_vz[idx];
      rho[idx] = model.rho[idx];
      vp[idx] = model.vp[idx];
      vs[idx] = model.vs[idx];
      density_l_kernel[idx] = model.density_l_kernel[idx];
      lambda_kernel[idx] = model.lambda_kernel[idx];
      mu_kernel[idx] = model.mu_kernel[idx];
      vp_kernel[idx] = model.vp_kernel[idx];
      vs_kernel[idx] = model.vs_kernel[idx];
      density_v_kernel[idx] = model.density_v_kernel[idx];
      starting_rho[idx] = model.starting_rho[idx];
      starting_vp[idx] = model.starting_vp[idx];
      starting_vs[idx] = model.starting_vs[idx];
      taper[idx] = model.taper[idx];

#pragma omp parallel for collapse(2)
      for (int i_shot = 0; i_shot < n_shots; i_shot++)
      {
        for (int i_snapshot = 0; i_snapshot < snapshots; i_snapshot++)
        {
          auto accu_idx = linear_IDX(i_shot, i_snapshot, ix, iz, n_shots, snapshots, nx, nz);
          accu_vx[accu_idx] = model.accu_vx[accu_idx];
          accu_vz[accu_idx] = model.accu_vz[accu_idx];
          accu_txx[accu_idx] = model.accu_txx[accu_idx];
          accu_tzz[accu_idx] = model.accu_tzz[accu_idx];
          accu_txz[accu_idx] = model.accu_txz[accu_idx];
        }
      }
    }
  }
#pragma omp parallel for collapse(1)
  for (int it = 0; it < nt; it++)
  {
    t[it] = model.t[it];
    for (int i_source = 0; i_source < n_sources; i_source++)
    {
      auto idx = linear_IDX(i_source, it, n_sources, nt);
      stf[idx] = model.stf[idx];
    }
  }
#pragma omp parallel for collapse(3)
  for (int i_source = 0; i_source < n_sources; i_source++)
  {
    for (int mi_ = 0; mi_ < 2; mi_++)
    {
      for (int m_j = 0; m_j < 2; m_j++)
      {
        auto idx = linear_IDX(i_source, mi_, m_j, n_sources, 2, 2);
        moment[idx] = model.moment[idx];
      }
    }
  }
#pragma omp parallel for collapse(3)
  for (int i_shot = 0; i_shot < n_shots; i_shot++)
  {
    for (int ir = 0; ir < nr; ir++)
    {
      for (int it = 0; it < nt; it++)
      {
        auto idx = linear_IDX(i_shot, ir, it, n_shots, nr, nt);
        rtf_ux[idx] = model.rtf_ux[idx];
        rtf_uz[idx] = model.rtf_uz[idx];
        rtf_ux_true[idx] = model.rtf_ux_true[idx];
        rtf_uz_true[idx] = model.rtf_uz_true[idx];
        a_stf_ux[idx] = model.a_stf_ux[idx];
        a_stf_uz[idx] = model.a_stf_uz[idx];
      }
    }
  }
}

void fdModel::parse_parameters(const std::vector<int> ix_sources_vector,
                               const std::vector<int> iz_sources_vector,
                               const std::vector<real_simulation> moment_angles_vector,
                               const std::vector<int> ix_receivers_vector,
                               const std::vector<int> iz_receivers_vector)
{
  std::cout << "Parsing passed configuration." << std::endl;

  nx = nx_inner + np_boundary * 2;
  nz = nz_inner + np_boundary * 2;
  nx_free_parameters = nx_inner - nx_inner_boundary * 2;
  nz_free_parameters = nz_inner - nz_inner_boundary * 2;

  // Basis functions
  assert(nx_free_parameters % basis_gridpoints_x == 0 and
         nz_free_parameters % basis_gridpoints_z == 0);
  free_parameters = 3 * nx_free_parameters * nz_free_parameters /
                    (basis_gridpoints_x * basis_gridpoints_z);

  // Parse source setup.
  ix_sources = new int[n_sources];
  iz_sources = new int[n_sources];
  moment_angles = new real_simulation[n_sources];

  if (ix_sources_vector.size() != n_sources or iz_sources_vector.size() != n_sources or
      moment_angles_vector.size() != n_sources)
  {
    throw std::invalid_argument(
        "Dimension mismatch between n_sources and sources.ix_sources, "
        "sources.iz_sources or sources.moment_angles");
  }
  for (int i_source = 0; i_source < n_sources; ++i_source)
  {
    ix_sources[i_source] = ix_sources_vector[i_source];
    iz_sources[i_source] = iz_sources_vector[i_source];
    moment_angles[i_source] = moment_angles_vector[i_source];
  }
  // Parse source stacking
  if (which_source_to_fire_in_which_shot.size() != n_shots)
  {
    throw std::invalid_argument(
        "Mismatch between n_shots and sources.which_source_to_fire_in_which_shot");
  }
  int total_sources = 0;
  for (const auto &shot_sources : which_source_to_fire_in_which_shot)
  {
    total_sources += shot_sources.size();
  }
  if (total_sources != n_sources)
  {
    throw std::invalid_argument(
        "Mismatch between n_sources and sources.which_source_to_fire_in_which_shot.");
  }

  // Receivers geometry
  ix_receivers = new int[nr];
  iz_receivers = new int[nr];

  if (ix_receivers_vector.size() != nr or iz_receivers_vector.size() != nr)
  {
    throw std::invalid_argument(
        "Mismatch between nr and receivers.ix_receivers or receivers.iz_receivers");
  }
  for (int i_receiver = 0; i_receiver < nr; ++i_receiver)
  {
    ix_receivers[i_receiver] = ix_receivers_vector[i_receiver];
    iz_receivers[i_receiver] = iz_receivers_vector[i_receiver];
  }

  // Final calculations
  alpha = static_cast<real_simulation>(1.0 / peak_frequency);
  snapshots = ceil(nt / snapshot_interval);

  // Check if the amount of snapshots satisfies the other parameters.
  if (floor(double(nt) / snapshot_interval) != snapshots)
  {
    throw std::length_error("Snapshot interval and size of accumulator don't match!");
  }
}

void fdModel::parse_configuration_file(const char *configuration_file_relative_path)
{
  std::cout << "Loading configuration file: '" << configuration_file_relative_path
            << "'." << std::endl;

  INIReader reader(configuration_file_relative_path);
  if (reader.ParseError() < 0)
  {
    throw std::invalid_argument("Can't load .ini file.");
  }

  // Domain
  nt = reader.GetInteger("domain", "nt");
  nx_inner = reader.GetInteger("domain", "nx_inner");
  nz_inner = reader.GetInteger("domain", "nz_inner");
  nx_inner_boundary = reader.GetInteger("domain", "nx_inner_boundary");
  nz_inner_boundary = reader.GetInteger("domain", "nz_inner_boundary");
  dx = reader.GetReal("domain", "dx");
  dz = reader.GetReal("domain", "dz");
  dt = reader.GetReal("domain", "dt");
  np_boundary = reader.GetInteger("boundary", "np_boundary");
  np_factor = reader.GetReal("boundary", "np_factor");
  scalar_rho = reader.GetReal("medium", "scalar_rho");
  scalar_vp = reader.GetReal("medium", "scalar_vp");
  scalar_vs = reader.GetReal("medium", "scalar_vs");
  basis_gridpoints_x = reader.GetInteger("basis", "npx");
  basis_gridpoints_z = reader.GetInteger("basis", "npz");
  peak_frequency = reader.GetReal("sources", "peak_frequency");
  t0 = reader.GetReal("sources", "source_timeshift");
  delay_cycles_per_shot = reader.GetReal("sources", "delay_cycles_per_shot");
  n_sources = reader.GetInteger("sources", "n_sources");
  n_shots = reader.GetInteger("sources", "n_shots");
  std::vector<int> ix_sources_vector;
  std::vector<int> iz_sources_vector;
  std::vector<real_simulation> moment_angles_vector;
  parse_string_to_vector(reader.Get("sources", "ix_sources"), &ix_sources_vector);
  parse_string_to_vector(reader.Get("sources", "iz_sources"), &iz_sources_vector);
  parse_string_to_vector(reader.Get("sources", "moment_angles"), &moment_angles_vector);
  parse_string_to_nested_int_vector(
      reader.Get("sources", "which_source_to_fire_in_which_shot"),
      &which_source_to_fire_in_which_shot);
  nr = reader.GetInteger("receivers", "nr");
  std::vector<int> ix_receivers_vector;
  std::vector<int> iz_receivers_vector;
  parse_string_to_vector(reader.Get("receivers", "ix_receivers"), &ix_receivers_vector);
  parse_string_to_vector(reader.Get("receivers", "iz_receivers"), &iz_receivers_vector);
  snapshot_interval = reader.GetInteger("inversion", "snapshot_interval");
  observed_data_folder = reader.Get("output", "observed_data_folder");
  stf_folder = reader.Get("output", "stf_folder");

  // Parse the read parameters
  parse_parameters(ix_sources_vector, iz_sources_vector, moment_angles_vector,
                   ix_receivers_vector, iz_receivers_vector);
}

// Forward modeller
void fdModel::forward_simulate(int i_shot, bool store_fields, bool verbose,
                               bool output_wavefields)
{
// Set dynamic physical fields to zero to reflect initial conditions.
#pragma omp parallel for collapse(2)
  for (int ix = 0; ix < nx; ++ix)
  {
    for (int iz = 0; iz < nz; ++iz)
    {
      auto idx = linear_IDX(ix, iz, nx, nz);
      vx[idx] = 0.0;
      vz[idx] = 0.0;
      txx[idx] = 0.0;
      tzz[idx] = 0.0;
      txz[idx] = 0.0;
    }
  }

  // If verbose, clock time of modelling.
  double startTime = 0, stopTime = 0, secsElapsed = 0;
  if (verbose)
  {
    startTime = real_simulation(omp_get_wtime());
  }

  // Time-loop starts here
  for (int it = 0; it < nt; ++it)
  {
    // Take wavefield snapshot at requited intervals.
    if (it % snapshot_interval == 0 and store_fields)
    {
#pragma omp parallel for collapse(2)
      for (int ix = 0; ix < nx; ++ix)
      {
        for (int iz = 0; iz < nz; ++iz)
        {
          auto idx_grid = linear_IDX(ix, iz, nx, nz);
          auto idx_accu = linear_IDX(i_shot, it / snapshot_interval, ix, iz, n_shots, snapshots, nx, nz);

          accu_vx[idx_accu] = vx[idx_grid];
          accu_vz[idx_accu] = vz[idx_grid];
          accu_txx[idx_accu] = txx[idx_grid];
          accu_txz[idx_accu] = txz[idx_grid];
          accu_tzz[idx_accu] = tzz[idx_grid];
        }
      }
    }

// Record seismograms by integrating velocity into displacement for every
// time-step.
#pragma omp parallel for collapse(1)
    for (int i_receiver = 0; i_receiver < nr; ++i_receiver)
    {
      auto idx_rtf = linear_IDX(i_shot, i_receiver, it, n_shots, nr, nt);
      auto idx_loc = linear_IDX(ix_receivers[i_receiver], iz_receivers[i_receiver], nx, nz);

      if (it == 0)
      {
        rtf_ux[idx_rtf] = dt * vx[idx_loc] / (dx * dz);
        rtf_uz[idx_rtf] = dt * vz[idx_loc] / (dx * dz);
      }
      else
      {
        auto idx_rtf_t_min_1 = linear_IDX(i_shot, i_receiver, it - 1, n_shots, nr, nt);

        rtf_ux[idx_rtf] =
            rtf_ux[idx_rtf_t_min_1] + dt * vx[idx_loc] / (dx * dz);
        rtf_uz[idx_rtf] =
            rtf_uz[idx_rtf_t_min_1] + dt * vz[idx_loc] / (dx * dz);
      }
    }

// Time integrate dynamic fields for stress ...
#pragma omp parallel for collapse(2)
    for (int ix = 2; ix < nx - 2; ++ix)
    {
      for (int iz = 2; iz < nz - 2; ++iz)
      {

        int idx = linear_IDX(ix, iz, nx, nz);
        int idx_xp1 = linear_IDX(ix + 1, iz, nx, nz);
        int idx_xp2 = linear_IDX(ix + 2, iz, nx, nz);
        int idx_xm1 = linear_IDX(ix - 1, iz, nx, nz);
        int idx_xm2 = linear_IDX(ix - 2, iz, nx, nz);
        int idx_zm1 = linear_IDX(ix, iz - 1, nx, nz);
        int idx_zm2 = linear_IDX(ix, iz - 2, nx, nz);
        int idx_zp1 = linear_IDX(ix, iz + 1, nx, nz);
        int idx_zp2 = linear_IDX(ix, iz + 2, nx, nz);

        txx[idx] =
            taper[idx] *
            (txx[idx] + dt * (lm[idx] *
                                  (c1 * (vx[idx_xp1] - vx[idx]) +
                                   c2 * (vx[idx_xm1] - vx[idx_xp2])) /
                                  dx +
                              la[idx] *
                                  (c1 * (vz[idx] - vz[idx_zm1]) +
                                   c2 * (vz[idx_zm2] - vz[idx_zp1])) /
                                  dz));
        tzz[idx] =
            taper[idx] *
            (tzz[idx] + dt * (la[idx] *
                                  (c1 * (vx[idx_xp1] - vx[idx]) +
                                   c2 * (vx[idx_xm1] - vx[idx_xp2])) /
                                  dx +
                              (lm[idx]) *
                                  (c1 * (vz[idx] - vz[idx_zm1]) +
                                   c2 * (vz[idx_zm2] - vz[idx_zp1])) /
                                  dz));
        txz[idx] = taper[idx] *
                   (txz[idx] + dt * mu[idx] *
                                   ((c1 * (vx[idx_zp1] - vx[idx]) +
                                     c2 * (vx[idx_zm1] - vx[idx_zp2])) /
                                        dz +
                                    (c1 * (vz[idx] - vz[idx_xm1]) +
                                     c2 * (vz[idx_xm2] - vz[idx_xp1])) /
                                        dx));
      }
    }
// ... and time integrate dynamic fields for velocity.
#pragma omp parallel for collapse(2)
    for (int ix = 2; ix < nx - 2; ++ix)
    {
      for (int iz = 2; iz < nz - 2; ++iz)
      {
        int idx = linear_IDX(ix, iz, nx, nz);
        int idx_xp1 = linear_IDX(ix + 1, iz, nx, nz);
        int idx_xp2 = linear_IDX(ix + 2, iz, nx, nz);
        int idx_xm1 = linear_IDX(ix - 1, iz, nx, nz);
        int idx_xm2 = linear_IDX(ix - 2, iz, nx, nz);
        int idx_zm1 = linear_IDX(ix, iz - 1, nx, nz);
        int idx_zm2 = linear_IDX(ix, iz - 2, nx, nz);
        int idx_zp1 = linear_IDX(ix, iz + 1, nx, nz);
        int idx_zp2 = linear_IDX(ix, iz + 2, nx, nz);

        vx[idx] = taper[idx] *
                  (vx[idx] + b_vx[idx] * dt *
                                 ((c1 * (txx[idx] - txx[idx_xm1]) +
                                   c2 * (txx[idx_xm2] - txx[idx_xp1])) /
                                      dx +
                                  (c1 * (txz[idx] - txz[idx_zm1]) +
                                   c2 * (txz[idx_zm2] - txz[idx_zp1])) /
                                      dz));
        vz[idx] = taper[idx] *
                  (vz[idx] + b_vz[idx] * dt *
                                 ((c1 * (txz[idx_xp1] - txz[idx]) +
                                   c2 * (txz[idx_xm1] - txz[idx_xp2])) /
                                      dx +
                                  (c1 * (tzz[idx_zp1] - tzz[idx]) +
                                   c2 * (tzz[idx_zm1] - tzz[idx_zp2])) /
                                      dz));
      }
    }

    // Inject sources at appropriate location and times.
    for (const auto &i_source : which_source_to_fire_in_which_shot[i_shot])
    {
      // Don't parallelize in assignment! Creates race condition
      // |-inject source
      // | (x,x)-couple
      auto idx_mt = linear_IDX(i_source, 0, 0, n_sources, 2, 2);

      auto idx_stf = linear_IDX(i_source, it, n_sources, nt);

      auto idx = linear_IDX(ix_sources[i_source], iz_sources[i_source], nx, nz);

      auto idx_xm1 = linear_IDX(ix_sources[i_source] - 1, iz_sources[i_source], nx, nz);
      auto idx_zm1 = linear_IDX(ix_sources[i_source], iz_sources[i_source] - 1, nx, nz);

      auto idx_xp1 = linear_IDX(ix_sources[i_source] + 1, iz_sources[i_source], nx, nz);
      auto idx_zp1 = linear_IDX(ix_sources[i_source], iz_sources[i_source] + 1, nx, nz);

      auto idx_xp1zm1 = linear_IDX(ix_sources[i_source] + 1, iz_sources[i_source] - 1, nx, nz);
      auto idx_xm1zp1 = linear_IDX(ix_sources[i_source] - 1, iz_sources[i_source] + 1, nx, nz);
      auto idx_xm1zm1 = linear_IDX(ix_sources[i_source] - 1, iz_sources[i_source] - 1, nx, nz);

      vx[idx_xm1] -=
          moment[idx_mt] * stf[idx_stf] * dt *
          b_vz[idx_xm1] / (dx * dx * dx * dx);
      vx[idx] +=
          moment[idx_mt] * stf[idx_stf] * dt *
          b_vz[idx] / (dx * dx * dx * dx);

      // | (z,z)-couple
      idx_mt = linear_IDX(i_source, 1, 1, n_sources, 2, 2);
      vz[idx_zm1] -=
          moment[idx_mt] * stf[idx_stf] * dt *
          b_vz[idx_zm1] / (dz * dz * dz * dz);
      vz[idx] +=
          moment[idx_mt] * stf[idx_stf] * dt *
          b_vz[idx] / (dz * dz * dz * dz);

      // | (x,z)-couple
      idx_mt = linear_IDX(i_source, 0, 1, n_sources, 2, 2);
      vx[idx_xm1zp1] +=
          0.25 * moment[idx_mt] * stf[idx_stf] * dt *
          b_vz[idx_xm1zp1] /
          (dx * dx * dx * dx);
      vx[idx_zp1] +=
          0.25 * moment[idx_mt] * stf[idx_stf] * dt *
          b_vz[idx_zp1] / (dx * dx * dx * dx);
      vx[idx_xm1zm1] -=
          0.25 * moment[idx_mt] * stf[idx_stf] * dt *
          b_vz[idx_xm1zm1] /
          (dx * dx * dx * dx);
      vx[idx_zm1] -=
          0.25 * moment[idx_mt] * stf[idx_stf] * dt *
          b_vz[idx_zm1] / (dx * dx * dx * dx);

      // | (z,x)-couple
      idx_mt = linear_IDX(i_source, 1, 0, n_sources, 2, 2);
      vz[idx_xp1zm1] +=
          0.25 * moment[idx_mt] * stf[idx_stf] * dt *
          b_vz[idx_xp1zm1] /
          (dz * dz * dz * dz);
      vz[idx_xp1] +=
          0.25 * moment[idx_mt] * stf[idx_stf] * dt *
          b_vz[idx_xp1] / (dz * dz * dz * dz);
      vz[idx_xm1zm1] -=
          0.25 * moment[idx_mt] * stf[idx_stf] * dt *
          b_vz[idx_xm1zm1] /
          (dz * dz * dz * dz);
      vz[idx_xm1] -=
          0.25 * moment[idx_mt] * stf[idx_stf] * dt *
          b_vz[idx_xm1] / (dz * dz * dz * dz);
    }

    if (it % 10 == 0 and output_wavefields)
    {
      // Write wavefields
      std::string filename_vp = "snapshots/vx" + zero_pad_number(it, 5) + ".txt";
      std::string filename_vs = "snapshots/vz" + zero_pad_number(it, 5) + ".txt";

      std::ofstream file_vx(filename_vp);
      std::ofstream file_vz(filename_vs);

      for (int ix = 0; ix < nx; ++ix)
      {
        for (int iz = 0; iz < nz; ++iz)
        {
          auto idx = linear_IDX(ix, iz, nx, nz);

          file_vx << vx[idx] << " ";
          file_vz << vz[idx] << " ";
        }
        file_vx << std::endl;
        file_vz << std::endl;
      }
      file_vx.close();
      file_vz.close();
    }
  }

  // Output timing if verbose.
  if (verbose)
  {
    stopTime = omp_get_wtime();
    secsElapsed = stopTime - startTime;
    std::cout << "Seconds elapsed for forward wave simulation: " << secsElapsed
              << std::endl;
  }
}

void fdModel::adjoint_simulate(int i_shot, bool verbose)
{
  // Reset dynamical fields
  for (int ix = 0; ix < nx; ++ix)
  {
    for (int iz = 0; iz < nz; ++iz)
    {

      auto idx = linear_IDX(ix, iz, nx, nz);

      vx[idx] = 0.0;
      vz[idx] = 0.0;
      txx[idx] = 0.0;
      tzz[idx] = 0.0;
      txz[idx] = 0.0;
    }
  }

  // If verbose, count time
  double startTime = 0, stopTime = 0, secsElapsed = 0;
  if (verbose)
  {
    startTime = real_simulation(omp_get_wtime());
  }

  for (int it = nt - 1; it >= 0; --it)
  {
    // Correlate wavefields
    if (it % snapshot_interval == 0)
    { // Todo, [X] rewrite for only relevant
      // parameters [ ] Check if done properly
#pragma omp parallel for collapse(2)
      for (int ix = np_boundary + nx_inner_boundary;
           ix < np_boundary + nx_inner - nx_inner_boundary; ++ix)
      {
        for (int iz = np_boundary + nz_inner_boundary;
             iz < np_boundary + nz_inner - nz_inner_boundary; ++iz)
        {

          auto idx = linear_IDX(ix, iz, nx, nz);

          auto idx_accu = linear_IDX(i_shot, it / snapshot_interval, ix, iz, n_shots, snapshots, nx, nz);

          density_l_kernel[idx] -=
              snapshot_interval * dt *
              (accu_vx[idx_accu] * vx[idx] + accu_vz[idx_accu] * vz[idx]);

          lambda_kernel[idx] +=
              snapshot_interval * dt *
              (((accu_txx[idx_accu] - (accu_tzz[idx_accu] * la[idx]) /
                                          lm[idx]) +
                (accu_tzz[idx_accu] - (accu_txx[idx_accu] * la[idx]) /
                                          lm[idx])) *
               ((txx[idx] - (tzz[idx] * la[idx]) / lm[idx]) +
                (tzz[idx] - (txx[idx] * la[idx]) / lm[idx]))) /
              ((lm[idx] - ((la[idx] * la[idx]) / (lm[idx]))) *
               (lm[idx] - ((la[idx] * la[idx]) / (lm[idx]))));

          mu_kernel[idx] +=
              snapshot_interval * dt * 2 *
              ((((txx[idx] - (tzz[idx] * la[idx]) / lm[idx]) *
                 (accu_txx[idx_accu] - (accu_tzz[idx_accu] * la[idx]) /
                                           lm[idx])) +
                ((tzz[idx] - (txx[idx] * la[idx]) / lm[idx]) *
                 (accu_tzz[idx_accu] - (accu_txx[idx_accu] * la[idx]) /
                                           lm[idx]))) /
                   ((lm[idx] - ((la[idx] * la[idx]) / (lm[idx]))) *
                    (lm[idx] - ((la[idx] * la[idx]) / (lm[idx])))) +
               2 * (txz[idx] * accu_txz[idx_accu] /
                    (4 * mu[idx] * mu[idx])));
        }
      }
    }

// Reverse time integrate dynamic fields for stress
#pragma omp parallel for collapse(2)
    for (int ix = 2; ix < nx - 2; ++ix)
    {
      for (int iz = 2; iz < nz - 2; ++iz)
      {

        auto idx = linear_IDX(ix, iz, nx, nz);

        auto idx_xp1 = linear_IDX(ix + 1, iz, nx, nz);
        auto idx_xp2 = linear_IDX(ix + 2, iz, nx, nz);
        auto idx_xm1 = linear_IDX(ix - 1, iz, nx, nz);
        auto idx_xm2 = linear_IDX(ix - 2, iz, nx, nz);

        auto idx_zm1 = linear_IDX(ix, iz - 1, nx, nz);
        auto idx_zm2 = linear_IDX(ix, iz - 2, nx, nz);
        auto idx_zp1 = linear_IDX(ix, iz + 1, nx, nz);
        auto idx_zp2 = linear_IDX(ix, iz + 2, nx, nz);

        txx[idx] =
            taper[idx] *
            (txx[idx] - dt * (lm[idx] *
                                  (c1 * (vx[idx_xp1] - vx[idx]) +
                                   c2 * (vx[idx_xm1] - vx[idx_xp2])) /
                                  dx +
                              la[idx] *
                                  (c1 * (vz[idx] - vz[idx_zm1]) +
                                   c2 * (vz[idx_zm2] - vz[idx_zp1])) /
                                  dz));
        tzz[idx] =
            taper[idx] *
            (tzz[idx] - dt * (la[idx] *
                                  (c1 * (vx[idx_xp1] - vx[idx]) +
                                   c2 * (vx[idx_xm1] - vx[idx_xp2])) /
                                  dx +
                              (lm[idx]) *
                                  (c1 * (vz[idx] - vz[idx_zm1]) +
                                   c2 * (vz[idx_zm2] - vz[idx_zp1])) /
                                  dz));
        txz[idx] = taper[idx] *
                   (txz[idx] - dt * mu[idx] *
                                   ((c1 * (vx[idx_zp1] - vx[idx]) +
                                     c2 * (vx[idx_zm1] - vx[idx_zp2])) /
                                        dz +
                                    (c1 * (vz[idx] - vz[idx_xm1]) +
                                     c2 * (vz[idx_xm2] - vz[idx_xp1])) /
                                        dx));
      }
    }
// Reverse time integrate dynamic fields for velocity
#pragma omp parallel for collapse(2)
    for (int ix = 2; ix < nx - 2; ++ix)
    {
      for (int iz = 2; iz < nz - 2; ++iz)
      {

        auto idx = linear_IDX(ix, iz, nx, nz);

        auto idx_xp1 = linear_IDX(ix + 1, iz, nx, nz);
        auto idx_xp2 = linear_IDX(ix + 2, iz, nx, nz);
        auto idx_xm1 = linear_IDX(ix - 1, iz, nx, nz);
        auto idx_xm2 = linear_IDX(ix - 2, iz, nx, nz);

        auto idx_zm1 = linear_IDX(ix, iz - 1, nx, nz);
        auto idx_zm2 = linear_IDX(ix, iz - 2, nx, nz);
        auto idx_zp1 = linear_IDX(ix, iz + 1, nx, nz);
        auto idx_zp2 = linear_IDX(ix, iz + 2, nx, nz);

        vx[idx] = taper[idx] *
                  (vx[idx] - b_vx[idx] * dt *
                                 ((c1 * (txx[idx] - txx[idx_xm1]) +
                                   c2 * (txx[idx_xm2] - txx[idx_xp1])) /
                                      dx +
                                  (c1 * (txz[idx] - txz[idx_zm1]) +
                                   c2 * (txz[idx_zm2] - txz[idx_zp1])) /
                                      dz));
        vz[idx] = taper[idx] *
                  (vz[idx] - b_vz[idx] * dt *
                                 ((c1 * (txz[idx_xp1] - txz[idx]) +
                                   c2 * (txz[idx_xm1] - txz[idx_xp2])) /
                                      dx +
                                  (c1 * (tzz[idx_zp1] - tzz[idx]) +
                                   c2 * (tzz[idx_zm1] - tzz[idx_zp2])) /
                                      dz));
      }
    }

    // Inject adjoint sources
    for (int ir = 0; ir < nr; ++ir)
    {
      auto idx_rec_loc = linear_IDX(ix_receivers[ir], iz_receivers[ir], nx, nz);
      auto idx_rec = linear_IDX(i_shot, ir, it, n_shots, nr, nt);

      vx[idx_rec_loc] += dt * b_vx[idx_rec_loc] * a_stf_ux[idx_rec] / (dx * dz);
      vz[idx_rec_loc] += dt * b_vz[idx_rec_loc] * a_stf_uz[idx_rec] / (dx * dz);
    }
  }

  // Output timing
  if (verbose)
  {
    stopTime = omp_get_wtime();
    secsElapsed = stopTime - startTime;
    std::cout << "Seconds elapsed for adjoint wave simulation: " << secsElapsed
              << std::endl;
  }
}

void fdModel::write_receivers() { write_receivers(std::string("")); }

void fdModel::write_receivers(const std::string prefix)
{
  std::string filename_ux;
  std::string filename_uz;

  std::ofstream receiver_file_ux;
  std::ofstream receiver_file_uz;

  for (int i_shot = 0; i_shot < n_shots; ++i_shot)
  {
    filename_ux = observed_data_folder + "/rtf_ux" + std::to_string(i_shot) + ".txt";
    filename_uz = observed_data_folder + "/rtf_uz" + std::to_string(i_shot) + ".txt";

    receiver_file_ux.open(filename_ux);
    receiver_file_uz.open(filename_uz);

    receiver_file_ux.precision(std::numeric_limits<real_simulation>::digits10 + 10);
    receiver_file_uz.precision(std::numeric_limits<real_simulation>::digits10 + 10);

    for (int i_receiver = 0; i_receiver < nr; ++i_receiver)
    {
      receiver_file_ux << std::endl;
      receiver_file_uz << std::endl;
      for (int it = 0; it < nt; ++it)
      {
        auto idx_rec = linear_IDX(i_shot, i_receiver, it, n_shots, nr, nt);
        receiver_file_ux << rtf_ux[idx_rec] << " ";
        receiver_file_uz << rtf_uz[idx_rec] << " ";
      }
    }
    receiver_file_ux.close();
    receiver_file_uz.close();
  }
}

void fdModel::write_sources()
{
  std::string filename_sources;
  std::ofstream shot_file;

  for (int i_shot = 0; i_shot < n_shots; ++i_shot)
  {
    filename_sources = stf_folder + "/sources_shot_" + std::to_string(i_shot) + ".txt";

    shot_file.open(filename_sources);

    shot_file.precision(std::numeric_limits<real_simulation>::digits10 + 10);

    for (int i_source : which_source_to_fire_in_which_shot[i_shot])
    {
      shot_file << std::endl;
      for (int it = 0; it < nt; ++it)
      {
        auto idx_source = linear_IDX(i_source, it, n_sources, nt);
        shot_file << stf[idx_source] << " ";
      }
    }
    shot_file.close();
  }
}

void fdModel::update_from_velocity()
{
#pragma omp parallel for collapse(2)
  for (int ix = 0; ix < nx; ++ix)
  {
    for (int iz = 0; iz < nz; ++iz)
    {

      auto idx = linear_IDX(ix, iz, nx, nz);

      mu[idx] = real_simulation(pow(vs[idx], 2) * rho[idx]);
      lm[idx] = real_simulation(pow(vp[idx], 2) * rho[idx]);
      la[idx] = lm[idx] - 2 * mu[idx];
      b_vx[idx] = real_simulation(1.0 / rho[idx]);
      b_vz[idx] = b_vx[idx];
    }
  }
}

void fdModel::load_receivers(bool verbose)
{
  std::string filename_ux;
  std::string filename_uz;

  std::ifstream receiver_file_ux;
  std::ifstream receiver_file_uz;

  for (int i_shot = 0; i_shot < n_shots; ++i_shot)
  {
    // Create filename from folder and shot.
    filename_ux = observed_data_folder + "/rtf_ux" + std::to_string(i_shot) + ".txt";
    filename_uz = observed_data_folder + "/rtf_uz" + std::to_string(i_shot) + ".txt";

    // Attempt to open the file by its filename.
    receiver_file_ux.open(filename_ux);
    receiver_file_uz.open(filename_uz);

    // Check if the file actually exists
    if (verbose)
    {
      std::cout << "File for ux data at shot " << i_shot << " is "
                << (receiver_file_ux.good() ? "good (exists at least)." : "ungood.")
                << std::endl;
      std::cout << "File for uz data at shot " << i_shot << " is "
                << (receiver_file_uz.good() ? "good (exists at least)." : "ungood.")
                << std::endl;
    }
    if (!receiver_file_ux.good() or !receiver_file_uz.good())
    {
      throw std::invalid_argument("Not all data is present!");
    }

    real_simulation placeholder_ux;
    real_simulation placeholder_uz;

    for (int i_receiver = 0; i_receiver < nr; ++i_receiver)
    {
      for (int it = 0; it < nt; ++it)
      {
        receiver_file_ux >> placeholder_ux;
        receiver_file_uz >> placeholder_uz;

        auto idx_receiver = linear_IDX(i_shot, i_receiver, it, n_shots, nr, nt);

        rtf_ux_true[idx_receiver] = placeholder_ux;
        rtf_uz_true[idx_receiver] = placeholder_uz;
      }
    }

    // Check data was large enough for set up
    if (!receiver_file_ux.good() or !receiver_file_uz.good())
    {
      std::cout << "Received bad state of file at end of reading! Does "
                   "the data match the set up?"
                << std::endl;
      throw std::invalid_argument("Not enough data is present!");
    }
    // Try to load more data ...
    receiver_file_ux >> placeholder_ux;
    receiver_file_uz >> placeholder_uz;
    // ... which shouldn't be possible
    if (receiver_file_ux.good() or receiver_file_uz.good())
    {
      std::cout << "Received good state of file past reading! Does the "
                   "data match the set up?"
                << std::endl;
      throw std::invalid_argument("Too much data is present!");
    }

    receiver_file_uz.close();
    receiver_file_ux.close();
  }
}

void fdModel::calculate_l2_misfit()
{
  misfit = 0;
  for (int i_shot = 0; i_shot < n_shots; ++i_shot)
  {
    for (int i_receiver = 0; i_receiver < nr; ++i_receiver)
    {
      for (int it = 0; it < nt; ++it)
      {

        auto idx_receiver = linear_IDX(i_shot, i_receiver, it, n_shots, nr, nt);

        misfit += 0.5 * dt * pow(rtf_ux_true[idx_receiver] - rtf_ux[idx_receiver], 2);
        misfit += 0.5 * dt * pow(rtf_uz_true[idx_receiver] - rtf_uz[idx_receiver], 2);
      }
    }
  }
}

void fdModel::calculate_l2_adjoint_sources()
{
#pragma omp parallel for collapse(3)
  for (int is = 0; is < n_shots; ++is)
  {
    for (int ir = 0; ir < nr; ++ir)
    {
      for (int it = 0; it < nt; ++it)
      {
        auto idx_receiver = linear_IDX(is, ir, it, n_shots, nr, nt);
        a_stf_ux[idx_receiver] = rtf_ux[idx_receiver] - rtf_ux_true[idx_receiver];
        a_stf_uz[idx_receiver] = rtf_uz[idx_receiver] - rtf_uz_true[idx_receiver];
      }
    }
  }
}

void fdModel::map_kernels_to_velocity()
{
#pragma omp parallel for collapse(2)
  for (int ix = 0; ix < nx; ++ix)
  {
    for (int iz = 0; iz < nz; ++iz)
    {
      auto idx = linear_IDX(ix, iz, nx, nz);
      vp_kernel[idx] = 2 * vp[idx] * lambda_kernel[idx] / b_vx[idx];
      vs_kernel[idx] = (2 * vs[idx] * mu_kernel[idx] -
                        4 * vs[idx] * lambda_kernel[idx]) /
                       b_vx[idx];
      density_v_kernel[idx] =
          density_l_kernel[idx] +
          (vp[idx] * vp[idx] - 2 * vs[idx] * vs[idx]) *
              lambda_kernel[idx] +
          vs[idx] * vs[idx] * mu_kernel[idx];
    }
  }
}

void fdModel::load_model(const std::string &de_path, const std::string &vp_path,
                         const std::string &vs_path, bool verbose)
{
  std::ifstream de_file;
  std::ifstream vp_file;
  std::ifstream vs_file;

  de_file.open(de_path);
  vp_file.open(vp_path);
  vs_file.open(vs_path);

  // Check if the file actually exists
  if (verbose)
  {
    std::cout << "Loading models." << std::endl;
    std::cout << "File: " << de_path << std::endl;
    std::cout << "File for density is "
              << (de_file.good() ? "good (exists at least)." : "ungood.") << std::endl;
    std::cout << "File: " << vp_path << std::endl;
    std::cout << "File for P-wave velocity is "
              << (vp_file.good() ? "good (exists at least)." : "ungood.") << std::endl;
    std::cout << "File: " << vs_path << std::endl;
    std::cout << "File for S-wave velocity is "
              << (vs_file.good() ? "good (exists at least)." : "ungood.") << std::endl;
  }
  if (!de_file.good() or !vp_file.good() or !vs_file.good())
  {
    throw std::invalid_argument("The files for target models don't seem to exist.\r\n"
                                "Paths:\r\n"
                                "\tDensity target: " +
                                de_path +
                                "\r\n"
                                "\tP-wave target : " +
                                vp_path +
                                "\r\n"
                                "\tS-wave target : " +
                                vs_path + "\r\n");
  }

  real_simulation placeholder_de;
  real_simulation placeholder_vp;
  real_simulation placeholder_vs;

  int iter = 0;

  for (int ix = 0; ix < nx; ++ix)
  {
    for (int iz = 0; iz < nz; ++iz)
    {

      auto idx = linear_IDX(ix, iz, nx, nz);

      de_file >> placeholder_de;
      vp_file >> placeholder_vp;
      vs_file >> placeholder_vs;

      std::cout << iter << " " << ix << " " << iz << " " << de_file.good() << " "
                << std::endl;

      rho[idx] = placeholder_de;
      vp[idx] = placeholder_vp;
      vs[idx] = placeholder_vs;
      iter++;

      if (!de_file.good() or !vp_file.good() or !vs_file.good())
      {
        throw std::invalid_argument("Received bad state of one of the files at end of "
                                    "reading. Does the data match the domain?");
      }
    }
  }

  // Check data was large enough for set up
  if (!de_file.good() or !vp_file.good() or !vs_file.good())
  {
    std::cout << "Received bad state of one of the files at end of "
                 "reading. Does the data match the domain?"
              << std::endl
              << "File states: " << std::endl
              << "Density:" << de_file.good() << std::endl
              << "P-wave: " << vp_file.good() << std::endl
              << "S-Wave: " << vs_file.good() << std::endl;
    throw std::invalid_argument("Not enough data is present!");
  }
  // Try to load more data ...
  de_file >> placeholder_de;
  vp_file >> placeholder_vp;
  vs_file >> placeholder_vs;
  // ... which shouldn't be possible
  if (de_file.good() or vp_file.good() or vs_file.good())
  {
    std::cout << "Received good state of file past reading. Does the data "
                 "match the domain?"
              << std::endl;
    throw std::invalid_argument("Too much data is present!");
  }

  de_file.close();
  vp_file.close();
  vs_file.close();

  update_from_velocity();
  if (verbose)
    std::cout << std::endl;
}

void fdModel::run_model(bool verbose, bool simulate_adjoint)
{
  for (int i_shot = 0; i_shot < n_shots; ++i_shot)
  {
    forward_simulate(i_shot, true, verbose);
  }
  calculate_l2_misfit();
  if (simulate_adjoint)
  {
    calculate_l2_adjoint_sources();
    reset_kernels();
    for (int is = 0; is < n_shots; ++is)
    {
      adjoint_simulate(is, verbose);
    }
    map_kernels_to_velocity();
  }
}

void fdModel::reset_kernels()
{
  for (int ix = 0; ix < nx; ++ix)
  {
    for (int iz = 0; iz < nz; ++iz)
    {

      auto idx = linear_IDX(ix, iz, nx, nz);

      lambda_kernel[idx] = 0.0;
      mu_kernel[idx] = 0.0;
      density_l_kernel[idx] = 0.0;
    }
  }
}

void fdModel::write_kernels()
{
  std::string filename_kernel_vp = "kernel_vp.txt";
  ;
  std::string filename_kernel_vs = "kernel_vs.txt";
  ;
  std::string filename_kernel_density = "kernel_density.txt";
  ;

  std::ofstream file_kernel_vp(filename_kernel_vp);
  std::ofstream file_kernel_vs(filename_kernel_vs);
  std::ofstream file_kernel_density(filename_kernel_density);

  for (int ix = 0; ix < nx; ++ix)
  {
    for (int iz = 0; iz < nz; ++iz)
    {
      auto idx = linear_IDX(ix, iz, nx, nz);

      file_kernel_vp << vp_kernel[idx] << " ";
      file_kernel_vs << vs_kernel[idx] << " ";
      file_kernel_density << density_v_kernel[idx] << " ";
    }
    file_kernel_vp << std::endl;
    file_kernel_vs << std::endl;
    file_kernel_density << std::endl;
  }
  file_kernel_vp.close();
  file_kernel_vs.close();
  file_kernel_density.close();
}

dynamic_vector fdModel::get_model_vector()
{
  // Assert that the chosen parametrization makes full blocks (i.e. that we
  // don't have 3 horizontal subdivisions for 20 points, which would result in a
  // non-rounded number for points per cell)
  assert(nx_free_parameters % basis_gridpoints_x == 0 and
         nz_free_parameters % basis_gridpoints_z == 0);

  int n_free_per_par = nx_free_parameters * nz_free_parameters /
                       (basis_gridpoints_x * basis_gridpoints_z);

  dynamic_vector m = dynamic_vector(n_free_per_par * 3, 1);

  // Loop over points within the free zone, so excluding boundary and non-free
  // parameters
  for (int ix = 0; ix < nx_free_parameters / basis_gridpoints_x; ++ix)
  {
    for (int iz = 0; iz < nz_free_parameters / basis_gridpoints_z; ++iz)
    {
      // Model vector index
      int i_parameter = ix + iz * nx_free_parameters / basis_gridpoints_x;

      // Global x index
      int gix = ix * basis_gridpoints_x + nx_inner_boundary + np_boundary;
      // Global z index
      int giz = iz * basis_gridpoints_z + nz_inner_boundary + np_boundary;

      m[i_parameter] = 0;
      m[i_parameter + n_free_per_par] = 0;
      m[i_parameter + 2 * n_free_per_par] = 0;

      // Average over the points in the cell
      // We could easily just retrieve one of the values in the block, but when
      // loading a model onto the grid, all the gridpoindts might not be the
      // same value.
      for (int sub_ix = 0; sub_ix < basis_gridpoints_x; sub_ix++)
      {
        for (int sub_iz = 0; sub_iz < basis_gridpoints_z; sub_iz++)
        {

          auto idx = linear_IDX(gix + sub_ix, giz + sub_iz, nx, nz);

          m[i_parameter] += vp[idx] /
                            (basis_gridpoints_x * basis_gridpoints_z);
          m[i_parameter + n_free_per_par] += vs[idx] /
                                             (basis_gridpoints_x * basis_gridpoints_z);
          m[i_parameter + 2 * n_free_per_par] +=
              rho[idx] /
              (basis_gridpoints_x * basis_gridpoints_z);
        }
      }
    }
  }
  return m;
}

void fdModel::set_model_vector(dynamic_vector m)
{
  assert(nx_free_parameters % basis_gridpoints_x == 0 and
         nz_free_parameters % basis_gridpoints_z == 0);

  int n_free_per_par = nx_free_parameters * nz_free_parameters /
                       (basis_gridpoints_x * basis_gridpoints_z);

  assert(m.size() == n_free_per_par * 3);

  // Loop over points within the free zone, so excluding boundary and non-free
  // parameters
  for (int ix = 0; ix < nx_free_parameters / basis_gridpoints_x; ++ix)
  {
    for (int iz = 0; iz < nz_free_parameters / basis_gridpoints_z; ++iz)
    {
      // Model vector index
      int i_parameter = ix + iz * nx_free_parameters / basis_gridpoints_x;

      // Global x index
      int gix = ix * basis_gridpoints_x + nx_inner_boundary + np_boundary;
      // Global z index
      int giz = iz * basis_gridpoints_z + nz_inner_boundary + np_boundary;

      // Average over the points in the cell
      // We could easily just retrieve one of the values in the block, but when
      // loading a model onto the grid, all the gridpoindts might not be the
      // same value.
      for (int sub_ix = 0; sub_ix < basis_gridpoints_x; sub_ix++)
      {
        for (int sub_iz = 0; sub_iz < basis_gridpoints_z; sub_iz++)
        {

          auto idx = linear_IDX(gix + sub_ix, giz + sub_iz, nx, nz);

          vp[idx] = m[i_parameter];
          vs[idx] = m[i_parameter + n_free_per_par];
          rho[idx] = m[i_parameter + 2 * n_free_per_par];
        }
      }
    }
  }
  update_from_velocity();
}

dynamic_vector fdModel::get_gradient_vector()
{
  // Assert that the chosen parametrization makes full blocks (i.e. that we
  // don't have 3 horizontal subdivisions for 20 points, which would result in a
  // non-rounded number for points per cell)
  assert(nx_free_parameters % basis_gridpoints_x == 0 and
         nz_free_parameters % basis_gridpoints_z == 0);

  int n_free_per_par = nx_free_parameters * nz_free_parameters /
                       (basis_gridpoints_x * basis_gridpoints_z);

  dynamic_vector g = dynamic_vector(n_free_per_par * 3, 1);

  // Loop over points within the free zone, so excluding boundary and non-free
  // parameters
  for (int ix = 0; ix < nx_free_parameters / basis_gridpoints_x; ++ix)
  {
    for (int iz = 0; iz < nz_free_parameters / basis_gridpoints_z; ++iz)
    {
      // Model vector index
      int i_parameter = ix + iz * nx_free_parameters / basis_gridpoints_x;

      // Global x index
      int gix = ix * basis_gridpoints_x + nx_inner_boundary + np_boundary;
      // Global z index
      int giz = iz * basis_gridpoints_z + nz_inner_boundary + np_boundary;

      g[i_parameter] = 0;
      g[i_parameter + n_free_per_par] = 0;
      g[i_parameter + 2 * n_free_per_par] = 0;

      // Average over the points in the cell
      // We could easily just retrieve one of the values in the block, but when
      // loading a model onto the grid, all the gridpoindts might not be the
      // same value.
      for (int sub_ix = 0; sub_ix < basis_gridpoints_x; sub_ix++)
      {
        for (int sub_iz = 0; sub_iz < basis_gridpoints_z; sub_iz++)
        {

          auto idx = linear_IDX(gix + sub_ix, giz + sub_iz, nx, nz);

          g[i_parameter] +=
              vp_kernel[idx]; // / (basis_gridpoints_x *
                              // basis_gridpoints_z);
          g[i_parameter + n_free_per_par] +=
              vs_kernel[idx]; // / (basis_gridpoints_x *
                              // basis_gridpoints_z);
          g[i_parameter + 2 * n_free_per_par] +=
              density_v_kernel[idx]; //  / (basis_gridpoints_x
                                     //  * basis_gridpoints_z);
        }
      }
    }
  }
  return g;
}

dynamic_vector fdModel::load_vector(const std::string &model_vector_path,
                                    bool verbose)
{
  int n_free_per_par = nx_free_parameters * nz_free_parameters;

  dynamic_vector m = dynamic_vector(n_free_per_par * 3, 1);

  std::ifstream model_vector_file;
  model_vector_file.open(model_vector_path);

  // Check if the file actually exists
  if (verbose)
  {
    std::cout << "Loading model vector." << std::endl;
    std::cout << "File: " << model_vector_path << std::endl;
    std::cout << "File is "
              << (model_vector_file.good() ? "good (exists at least)." : "ungood.")
              << std::endl;
  }
  if (!model_vector_file.good())
  {
    throw std::invalid_argument("The files for target models don't seem to exist.\r\n"
                                "Paths:\r\n"
                                "\tDensity target: " +
                                model_vector_path + "\r\n");
  }

  real_simulation placeholder_model_vector;
  for (int i = 0; i < n_free_per_par * 3; ++i)
  {
    model_vector_file >> placeholder_model_vector;
    m[i] = placeholder_model_vector;
  }

  // Check data was large enough for set up
  if (!model_vector_file.good())
  {
    std::cout << "Received bad state of one of the files at end of "
                 "reading. Does the data match the domain?"
              << std::endl;
    throw std::invalid_argument("Not enough data is present!");
  }
  // Try to load more data ...
  model_vector_file >> placeholder_model_vector;
  // ... which shouldn't be possible
  if (model_vector_file.good())
  {
    std::cout << "Received good state of file past reading. Does the data "
                 "match the domain?"
              << std::endl;
    throw std::invalid_argument("Too much data is present!");
  }

  model_vector_file.close();
  return m;
}

template <class T>
void parse_string_to_vector(std::basic_string<char> string_to_parse,
                            std::vector<T> *destination_vector)
{
  // Erase all spaces
  string_to_parse.erase(
      remove_if(string_to_parse.begin(), string_to_parse.end(), isspace),
      string_to_parse.end());
  // Find end of data and cut afterwards
  size_t pos = string_to_parse.find("}");
  string_to_parse.erase(pos, string_to_parse.length());
  // Cut leading curly brace
  string_to_parse.erase(0, 1);
  // Split up string
  std::string delimiter = ",";
  pos = 0;
  std::string token;
  while ((pos = string_to_parse.find(delimiter)) != std::string::npos)
  {
    token = string_to_parse.substr(0, pos);
    destination_vector->emplace_back(atof(token.c_str()));
    string_to_parse.erase(0, pos + delimiter.length());
  }
  token = string_to_parse.substr(0, pos);
  destination_vector->emplace_back(atof(token.c_str()));
}

void parse_string_to_nested_int_vector(
    std::basic_string<char> string_to_parse,
    std::vector<std::vector<int>> *destination_vector)
{ // todo clean up
  // Erase all spaces
  string_to_parse.erase(
      remove_if(string_to_parse.begin(), string_to_parse.end(), isspace),
      string_to_parse.end());

  std::string delimiter_outer = "},{";
  string_to_parse.erase(0, 2);

  size_t pos_outer = 0;
  std::string token_outer;

  while ((pos_outer = string_to_parse.find(delimiter_outer)) != std::string::npos)
  {
    std::vector<int> sub_vec;

    token_outer = string_to_parse.substr(0, pos_outer);

    std::string delimiter_inner = ",";
    size_t pos_inner = 0;
    std::string token_inner;
    while ((pos_inner = token_outer.find(delimiter_inner)) != std::string::npos)
    {
      token_inner = token_outer.substr(0, pos_inner);
      sub_vec.emplace_back(atof(token_inner.c_str()));
      token_outer.erase(0, pos_inner + delimiter_inner.length());
    }
    token_inner = token_outer.substr(0, pos_inner);
    sub_vec.emplace_back(atof(token_inner.c_str()));

    destination_vector->emplace_back(sub_vec);

    string_to_parse.erase(0, pos_outer + delimiter_outer.length());
  }

  // Process last vector
  std::vector<int> sub_vec;
  pos_outer = string_to_parse.find("}};");
  token_outer = string_to_parse.substr(0, pos_outer);
  //    std::cout << token_outer << std::endl;
  std::string delimiter_inner = ",";
  size_t pos_inner = 0;
  std::string token_inner;
  while ((pos_inner = token_outer.find(delimiter_inner)) != std::string::npos)
  {
    token_inner = token_outer.substr(0, pos_inner);
    sub_vec.emplace_back(atof(token_inner.c_str()));
    //        std::cout << token_inner << std::endl;
    token_outer.erase(0, pos_inner + delimiter_inner.length());
  }
  token_inner = token_outer.substr(0, pos_inner);
  sub_vec.emplace_back(atof(token_inner.c_str()));
  //    std::cout << token_inner << std::endl;
  destination_vector->emplace_back(sub_vec);
  //    destination_vector->emplace_back(atof(token_outer.c_str()));
}

std::string zero_pad_number(int num, int pad)
{
  std::ostringstream ss;
  ss << std::setw(pad) << std::setfill('0') << num;
  return ss.str();
}
