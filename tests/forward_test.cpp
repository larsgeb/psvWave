//
// Created by Lars Gebraad on 16/04/19.
//

// Includes
#include <iostream>
#include <iomanip>
#include <fstream>
#include <tgmath.h>
#include <omp.h>
#include "../src/fdWaveModel.h"

int main() {

    auto *model = new fdWaveModel();

    // Simulating a forawrd model using the target model.

    model->load_target("../tests/test_setup/de_target.txt", "../tests/test_setup/vp_target.txt", "../tests/test_setup/vs_target.txt", false);
    model->load_starting("../tests/test_setup/de_starting.txt", "../tests/test_setup/vp_starting.txt", "../tests/test_setup/vs_starting.txt", false);

    std::cout << "Maximum amount of OpenMP threads:" << omp_get_max_threads() << std::endl;

    std::cout << std::endl << "Creating true data" << std::endl << std::flush;
    auto startTime = real_simulation(omp_get_wtime());
    for (int is = 0; is < fdWaveModel::n_shots; ++is) {
        (*model).forward_simulate(is, true, true);
    }
    auto endTime = real_simulation(omp_get_wtime());
    std::cout << "elapsed time: " << endTime - startTime << std::endl;

    (*model).write_receivers();
    (*model).write_sources();

    exit(0);
}

