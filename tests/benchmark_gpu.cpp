//
// Created by Lars Gebraad on 16/04/19.
//

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include "Metal/Metal.hpp"
#include "Foundation/Foundation.hpp"

// Includes
#include "../src/fdModel.h"
#include <fstream>
#include <assert.h>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <tgmath.h>
#include "../src/contiguous_arrays.h"

int main()
{
  MTL::Device *device = MTL::CreateSystemDefaultDevice();

  std::cout << "Maximum amount of OpenMP threads: " << omp_get_max_threads()
            << std::endl;

  auto conf_file = "tests/test_configurations/default_testing_configuration.ini";
  int nt = 2000;
  int nx_inner = 2000;
  int nz_inner = 2000;
  int nx_inner_boundary = 10;
  int nz_inner_boundary = 20;
  float dx = 1.249;
  float dz = 1.249;
  float dt = 0.00025;
  int np_boundary = 25;
  float np_factor = 0.015;
  float scalar_rho = 1500.0;
  float scalar_vp = 2000.0;
  float scalar_vs = 800.0;
  int npx = 1;
  int npz = 1;
  float peak_frequency = 50;
  float source_timeshift = 0.005;
  float delay_cycles_per_shot = 24;
  int n_sources = 4;
  int n_shots = 1;
  std::vector<int> ix_sources_vector{25, 75, 125, 175};
  std::vector<int> iz_sources_vector{10, 10, 10, 10};
  std::vector<float> moment_angles_vector{90, 180, 90, 180};
  std::vector<std::vector<int>> which_source_to_fire_in_which_shot{{0, 1, 2, 3}};
  int nr = 19;
  std::vector<int> ix_receivers_vector{10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
                                       110, 120, 130, 140, 150, 160, 170, 180, 190};
  std::vector<int> iz_receivers_vector{90, 90, 90, 90, 90, 90, 90, 90, 90, 90,
                                       90, 90, 90, 90, 90, 90, 90, 90, 90};
  int snapshot_interval = 10;
  std::string observed_data_folder(".");
  std::string stf_folder(".");

  auto *model_1 = new fdModel(
      device, nt, nx_inner, nz_inner, nx_inner_boundary, nz_inner_boundary, dx, dz, dt,
      np_boundary, np_factor, scalar_rho, scalar_vp, scalar_vs, npx, npz,
      peak_frequency, source_timeshift, delay_cycles_per_shot, n_sources, n_shots,
      ix_sources_vector, iz_sources_vector, moment_angles_vector,
      which_source_to_fire_in_which_shot, nr, ix_receivers_vector, iz_receivers_vector,
      snapshot_interval, observed_data_folder, stf_folder);

  for (int is = 0; is < model_1->n_shots; ++is)
  {
    std::cout << "Forward simulating on CPU:" << std::endl;
    model_1->forward_simulate(is, true, true, false, false);
    std::cout << "Forward simulating on GPU:" << std::endl;
    model_1->forward_simulate(is, true, true, false, true);
  }

  for (int is = 0; is < model_1->n_shots; ++is)
  {
    std::cout << "Adjoint simulating on GPU:" << std::endl;
    model_1->adjoint_simulate(is, true, true);
    std::cout << "Adjoint simulating on CPU:" << std::endl;
    model_1->adjoint_simulate(is, true, false);
  }

  delete model_1;
}
