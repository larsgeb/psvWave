//
// Created by Lars Gebraad on 28/12/18.
//

// Includes
#include <iostream>
#include <iomanip>
#include <fstream>
#include <tgmath.h>
#include <omp.h>
#include "fdWaveModel.h"

int main() {

    auto *model = new fdWaveModel();

    std::cout << std::endl << "Creating true data" << std::endl << std::flush;
    real_simulation startTime = real_simulation(omp_get_wtime());
    (*model).load_target("de_target.txt", "vp_target.txt", "vs_target.txt", false);
    (*model).load_starting("de_starting.txt", "vp_starting.txt", "vs_starting.txt", false);
    for (int is = 0; is < fdWaveModel::n_shots; ++is) {
        (*model).forward_simulate(is, true, false);
    }
    (*model).write_receivers();
    (*model).write_sources();
    real_simulation endTime = real_simulation(omp_get_wtime());
    std::cout << "elapsed time: " << endTime - startTime << std::endl;

    std::cout << std::endl << "Computing kernel" << std::endl << std::flush;

    startTime = real_simulation(omp_get_wtime());
    bool reset_de = false, reset_vp = true, reset_vs = true;
    (*model).reset_velocity_fields(reset_de, reset_vp, reset_vs);
    (*model).load_receivers(false);
    (*model).update_from_velocity();
    for (int is = 0; is < fdWaveModel::n_shots; ++is) {
        (*model).forward_simulate(is, true, false);
    }
    std::cout << "Misfit: " << (*model).calculate_misfit() << std::endl;
    (*model).calculate_adjoint_sources();
    for (int is = 0; is < fdWaveModel::n_shots; ++is) {
        (*model).adjoint_simulate(is, true);
    }
    (*model).map_kernels_to_velocity();
    endTime = real_simulation(omp_get_wtime());
    std::cout << "elapsed time: " << endTime - startTime << std::endl;

    std::ofstream mu_file;
    std::ofstream lambda_file;
    std::ofstream de_l_file;

    std::ofstream vp_file;
    std::ofstream vs_file;
    std::ofstream de_v_file;

    mu_file.open("mu_file.txt");
    lambda_file.open("lambda_file.txt");
    de_l_file.open("de_l_file.txt");

    vp_file.open("vp_file.txt");
    vs_file.open("vs_file.txt");
    de_v_file.open("de_v_file.txt");

    for (int ix = 0; ix < fdWaveModel::nx; ++ix) {
        for (int iz = 0; iz < fdWaveModel::nz; ++iz) {
            mu_file << (*model).mu_kernel[ix][iz] << " ";
            lambda_file << (*model).lambda_kernel[ix][iz] << " ";
            de_l_file << (*model).density_l_kernel[ix][iz] << " ";

            vp_file << (*model).vp_kernel[ix][iz] << " ";
            vs_file << (*model).vs_kernel[ix][iz] << " ";
            de_v_file << (*model).density_v_kernel[ix][iz] << " ";
        }

        mu_file << std::endl;
        lambda_file << std::endl;
        de_l_file << std::endl;
        vp_file << std::endl;
        vs_file << std::endl;
        de_v_file << std::endl;
    }

    mu_file.close();
    lambda_file.close();
    de_l_file.close();

    vp_file.close();
    vs_file.close();
    de_v_file.close();

    exit(0);
}

