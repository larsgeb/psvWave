//
// Created by Lars Gebraad on 23/04/19.
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

    model->load_receivers(true);
    std::cout << "Running forward simulation once." << std::endl;
    for (int is = 0; is < model->n_shots; ++is) {
        model->forward_simulate(is, true, true);
    }
    std::cout << std::endl
              << "Computing cross-correlations for all traces." << std::endl;

    // Compute cross correlation.
    real_simulation *r;
    allocate_1d_array(r, 200);

    auto startTime = omp_get_wtime();
    for (int i_shot = 0; i_shot < model->n_shots; ++i_shot) {
        for (int i_receiver = 0; i_receiver < model->nr; ++i_receiver) {
            cross_correlate(model->rtf_ux_true[i_shot][i_receiver], model->rtf_ux[i_shot][i_receiver], r, model->nt, 100);
            cross_correlate(model->rtf_uz_true[i_shot][i_receiver], model->rtf_uz[i_shot][i_receiver], r, model->nt, 100);
        }
    }
    auto endTime = omp_get_wtime();
    std::cout << "Cross-correlating " << model->n_shots * model->nr * 2 << " recordings took " << endTime - startTime << " seconds." << std::endl;


    deallocate_1d_array(r);

    std::cout << "All cross correlations are computed successfully. The test succeeded." << std::endl;

    exit(0);
}

