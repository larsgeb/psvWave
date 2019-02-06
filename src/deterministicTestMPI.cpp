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
#include <mpi.h>

int main() {

    MPI_Init(NULL, NULL);

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

    // Print off a hello world message
    printf("Hello world from processor %s, rank %d out of %d processors\n", processor_name, world_rank, world_size);
    MPI_Barrier(MPI_COMM_WORLD);
    auto *model = new fdWaveModel();
    real_simulation startTime, endTime;
    (*model).load_target("de_target.txt", "vp_target.txt", "vs_target.txt", false);
    (*model).load_starting("de_starting.txt", "vp_starting.txt", "vs_starting.txt", false);
    MPI_Barrier(MPI_COMM_WORLD);
    // Generate data on MPI process 1
    if (world_rank == 0) {
        startTime = real_simulation(omp_get_wtime());
        std::cout << std::endl << "Creating true data on rank " << world_rank << std::endl << std::flush;
        for (int is = 0; is < fdWaveModel::n_shots; ++is) {
            (*model).forward_simulate(is, true, false);
        }
        (*model).write_receivers();
        (*model).write_sources();
        endTime = real_simulation(omp_get_wtime());
        std::cout << "elapsed time: " << endTime - startTime << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    std::cout << "Computing kernel on rank " << world_rank << std::endl << std::flush;
    MPI_Barrier(MPI_COMM_WORLD);
    startTime = real_simulation(omp_get_wtime());
    bool reset_de = false, reset_vp = false, reset_vs = false;

    switch (world_rank) {
        case 0:
            reset_de = true;
            break;
        case 1:
            reset_vp = true;
            break;
        case 2:
        default:
            reset_vs = true;
            break;
    }
    (*model).reset_velocity_fields(reset_de, reset_vp, reset_vs);
    (*model).load_receivers(false);
    (*model).update_from_velocity();
    for (int is = 0; is < fdWaveModel::n_shots; ++is) {
        (*model).forward_simulate(is, true, false);
    }
    std::cout << "Misfit on rank " << world_rank << ": " << (*model).calculate_misfit() << std::endl;
    (*model).calculate_adjoint_sources();
    for (int is = 0; is < fdWaveModel::n_shots; ++is) {
        (*model).adjoint_simulate(is, false);
    }
    (*model).map_kernels_to_velocity();
    endTime = real_simulation(omp_get_wtime());
    std::cout << "elapsed time on rank " << world_rank << ": " << endTime - startTime << std::endl;

    std::string vp_filename = "vp_" + std::to_string(world_rank) + "file.txt";
    std::string vs_filename = "vs_" + std::to_string(world_rank) + "file.txt";
    std::string de_filename = "de_" + std::to_string(world_rank) + "file.txt";
    std::ofstream vp_file;
    std::ofstream vs_file;
    std::ofstream de_file;

    vp_file.open(vp_filename);
    vs_file.open(vs_filename);
    de_file.open(de_filename);

    for (int ix = 0; ix < fdWaveModel::nx; ++ix) {
        for (int iz = 0; iz < fdWaveModel::nz; ++iz) {

            vp_file << (*model).vp_kernel[ix][iz] << " ";
            vs_file << (*model).vs_kernel[ix][iz] << " ";
            de_file << (*model).density_v_kernel[ix][iz] << " ";
        }

        vp_file << std::endl;
        vs_file << std::endl;
        de_file << std::endl;
    }

    vp_file.close();
    vs_file.close();
    de_file.close();


    // Finalize the MPI environment.
    MPI_Finalize();

    exit(0);
}

