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

int main(int argc, char **argv) {
    std::cout << "Maximum amount of OpenMP threads:" << omp_get_max_threads() << std::endl << std::endl;
    auto *model = new fdWaveModel(argv[1]);

    auto startTime = omp_get_wtime();
    for (int is = 0; is < model->n_shots; ++is) {
        model->forward_simulate(is, false, true);
    }
    auto endTime = omp_get_wtime();
    std::cout << "Elapsed time for forward wave simulations: " << endTime - startTime << std::endl;

    model->write_receivers();
    model->write_sources();

    model->load_receivers(true);
    model->load_model(argv[2], argv[3], argv[4], true);

    for (int is = 0; is < model->n_shots; ++is) {
        model->forward_simulate(is, true, true);
    }
    model->observed_data_folder = "../";
    model->write_receivers();

    model->calculate_l2_misfit();
    std::cout << "Misfit: " << model->misfit << std::endl;
    model->calculate_l2_adjoint_sources();
    model->reset_kernels();

    for (int is = 0; is < model->n_shots; ++is) {
        model->adjoint_simulate(is, true);
    }

    model->map_kernels_to_velocity();
    model->write_kernels();

    exit(0);
}

