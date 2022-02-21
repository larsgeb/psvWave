//
// Created by Lars Gebraad on 16/04/19.
//

// Includes
#include "../src/fdModel.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <tgmath.h>

int main() {
  std::cout << "Maximum amount of OpenMP threads:" << omp_get_max_threads()
            << std::endl;

  // Define test, should be the same set up as the configuration file:
  // "../tests/test_configurations/default_testing_configuration.ini"
  auto conf_file = "tests/test_configurations/default_testing_configuration.ini";
  int nt = 8000;
  int nx_inner = 200;
  int nz_inner = 100;
  int nx_inner_boundary = 10;
  int nz_inner_boundary = 20;
  real_simulation dx = 1.249;
  real_simulation dz = 1.249;
  real_simulation dt = 0.00025;
  int np_boundary = 25;
  real_simulation np_factor = 0.015;
  real_simulation scalar_rho = 1500.0;
  real_simulation scalar_vp = 2000.0;
  real_simulation scalar_vs = 800.0;
  int npx = 1;
  int npz = 1;
  real_simulation peak_frequency = 50;
  real_simulation source_timeshift = 0.005;
  real_simulation delay_cycles_per_shot = 24;
  int n_sources = 4;
  int n_shots = 1;
  std::vector<int> ix_sources_vector{25, 75, 125, 175};
  std::vector<int> iz_sources_vector{10, 10, 10, 10};
  std::vector<real_simulation> moment_angles_vector{90, 180, 90, 180};
  std::vector<std::vector<int>> which_source_to_fire_in_which_shot{{0, 1, 2, 3}};
  int nr = 19;
  std::vector<int> ix_receivers_vector{10,  20,  30,  40,  50,  60,  70,  80,  90, 100,
                                       110, 120, 130, 140, 150, 160, 170, 180, 190};
  std::vector<int> iz_receivers_vector{90, 90, 90, 90, 90, 90, 90, 90, 90, 90,
                                       90, 90, 90, 90, 90, 90, 90, 90, 90};
  int snapshot_interval = 10;
  std::string observed_data_folder(".");
  std::string stf_folder(".");

  auto *model_1 = new fdModel(
      nt, nx_inner, nz_inner, nx_inner_boundary, nz_inner_boundary, dx, dz, dt,
      np_boundary, np_factor, scalar_rho, scalar_vp, scalar_vs, npx, npz,
      peak_frequency, source_timeshift, delay_cycles_per_shot, n_sources, n_shots,
      ix_sources_vector, iz_sources_vector, moment_angles_vector,
      which_source_to_fire_in_which_shot, nr, ix_receivers_vector, iz_receivers_vector,
      snapshot_interval, observed_data_folder, stf_folder);

  auto *model_2 = new fdModel(conf_file);

  real_simulation deterministic_sum_1 = 0.0;
  real_simulation deterministic_sum_2 = 0.0;

  for (int is = 0; is < model_1->n_shots; ++is) {
    model_1->forward_simulate(is, false, true);
    model_2->forward_simulate(is, false, true);

    for (int i_shot = 0; i_shot < model_1->n_shots; ++i_shot) {
      for (int i_receiver = 0; i_receiver < model_1->nr; ++i_receiver) {
        for (int it = 0; it < model_1->nt; ++it) {
          deterministic_sum_1 += model_1->rtf_ux[i_shot][i_receiver][it];
          deterministic_sum_2 += model_2->rtf_ux[i_shot][i_receiver][it];
        }
      }
    }
  }

  if (deterministic_sum_1 == deterministic_sum_2) {
    std::cout << "All simulations produced the same seismograms. The test succeeded."
              << std::endl
              << std::endl;
    exit(0);
  } else {
    std::cout << "Simulations were not consistent. The test failed." << std::endl
              << std::endl;
    exit(1);
  }
}
