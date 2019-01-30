//
// Created by Lars Gebraad on 28/12/18.
//

// Includes
#include <mpi.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <tgmath.h>
#include "fdWaveModel.h"

int main() {


    // Initialize MPI
    MPI_Init(nullptr, nullptr);

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Get the name of the processor
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    printf("Hello world from processor %s, rank %d out of %d processors\n", processor_name, world_rank, world_size);

    auto *model = new fdWaveModel();

    std::cout << std::endl << "Creating true data" << std::endl << std::flush;
    (*model).load_target("de_target.txt", "vp_target.txt", "vs_target.txt");
    for (int is = 0; is < fdWaveModel::n_shots; ++is) {
        (*model).forward_simulate(is, true, true);
    }
    (*model).write_receivers();

    std::cout << std::endl << "Computing kernel" << std::endl << std::flush;
    bool reset_de = false, reset_vp = false, reset_vs = true;
    (*model).reset_velocity_fields(reset_de, reset_vp, reset_vs);
    (*model).load_receivers();
    (*model).update_from_velocity();
    for (int is = 0; is < fdWaveModel::n_shots; ++is) {
        (*model).forward_simulate(is, true, true);
    }
    std::cout << "Misfit: " << (*model).calculate_misfit() << std::endl;
    (*model).calculate_adjoint_sources();
    for (int is = 0; is < fdWaveModel::n_shots; ++is) {
        (*model).adjoint_simulate(is, true);
    }
    (*model).map_kernels_to_velocity();

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

    MPI_Finalize();

    exit(0);
}

