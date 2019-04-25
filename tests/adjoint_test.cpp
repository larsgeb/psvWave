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

    auto configuration_file = argv[1];

    auto *model = new fdWaveModel(configuration_file);
    model->load_model("../tests/test_setup/de_starting.txt", "../tests/test_setup/vp_starting.txt", "../tests/test_setup/vs_starting.txt", true);

    auto startTime = omp_get_wtime();
    model->load_receivers(true);
    std::cout << "Running forward simulation once." << std::endl;
    for (int is = 0; is < model->n_shots; ++is) {
        model->forward_simulate(is, true, true);
    }

    // Compute L2 misfits and adjoint sources.
    model->calculate_l2_misfit();
    model->calculate_l2_adjoint_sources();
    auto endTime = omp_get_wtime();
    std::cout << "Elapsed time for one forward simulation: " << endTime - startTime << std::endl
              << "Misfit: " << model->misfit << std::endl << std::endl;


    // Prepare for test
    startTime = omp_get_wtime();
    int n_tests = 3;
    auto deterministic_sum = new real_simulation[n_tests];

    for (int i_test = 0; i_test < n_tests; ++i_test) {
        model->reset_kernels();
        for (int is = 0; is < model->n_shots; ++is) {
            model->adjoint_simulate(is, true);
        }
        model->map_kernels_to_velocity();

        deterministic_sum[i_test] = 0;
        for (int ix = 0; ix < model->nx; ++ix) {
            for (int iz = 0; iz < model->nz; ++iz) {
                deterministic_sum[i_test] += model->density_v_kernel[ix][iz];
                deterministic_sum[i_test] += model->vp_kernel[ix][iz];
                deterministic_sum[i_test] += model->vs_kernel[ix][iz];
            }
        }
        std::cout << deterministic_sum[i_test] << std::endl;
    }

    model->write_kernels();

    endTime = omp_get_wtime();
    std::cout << "Elapsed time for all adjoint simulations: " << endTime - startTime << std::endl;

    // Check deterministic output
    for (unsigned i = 0; i < n_tests; i++) {
        if (deterministic_sum[i] != deterministic_sum[0]) {
            exit(1);
        }
    }
    std::cout << "All adjoint simulations produced the same kernels. The test succeeded." << std::endl;
    exit(0);
}

