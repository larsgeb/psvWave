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

  auto *model_1 = new fdModel(conf_file);

  auto *model_2 = new fdModel(*model_1);

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

  delete model_1;
  delete model_2;

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
