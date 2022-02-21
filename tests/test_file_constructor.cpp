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

  auto conf_file = "tests/test_configurations/default_testing_configuration.ini";

  auto *model = new fdModel(conf_file);

  auto startTime = omp_get_wtime();
  int n_tests = 2;
  std::cout << "Running forward simulation " << n_tests << " times." << std::endl;
  auto deterministic_sum = new real_simulation[n_tests];

  for (int i_test = 0; i_test < n_tests; ++i_test) {
    for (int is = 0; is < model->n_shots; ++is) {
      model->forward_simulate(is, false, true);
    }

    deterministic_sum[i_test] = 0;

    for (int i_shot = 0; i_shot < model->n_shots; ++i_shot) {
      for (int i_receiver = 0; i_receiver < model->nr; ++i_receiver) {
        for (int it = 0; it < model->nt; ++it) {
          deterministic_sum[i_test] += model->rtf_ux[i_shot][i_receiver][it];
        }
      }
    }
  }
  auto endTime = omp_get_wtime();
  std::cout << "Elapsed time for all forward wave simulations: " << endTime - startTime
            << std::endl;

  model->write_receivers();
  model->write_sources();

  // Check deterministic output
  for (int i = 0; i < n_tests; i++) {
    if (deterministic_sum[i] != deterministic_sum[0]) {
      std::cout << "Simulations were not consistent. The test failed." << std::endl
                << std::endl;
      exit(1);
    }
  }
  std::cout << "All simulations produced the same seismograms. The test succeeded."
            << std::endl
            << std::endl;
  exit(0);
}
