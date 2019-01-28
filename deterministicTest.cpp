//
// Created by Lars Gebraad on 28/12/18.
//

// Includes
#include <iostream>
#include <iomanip>
#include <fstream>
#include <tgmath.h>
#include "fdWaveModel.h"

int main() {

    auto *model = new fdWaveModel();

    (*model).load_receivers();

    (*model).vs[100][120] = 450;
    (*model).vs[100][121] = 450;
    (*model).vs[101][120] = 450;
    (*model).vs[101][121] = 450;
    (*model).update_from_velocity();

    (*model).forward_simulate(1, true, true);
    (*model).forward_simulate(0, true, true);
    std::cout << "Misfit: " << (*model).calculate_misfit() << std::endl;
    (*model).calculate_adjoint_sources();
    (*model).adjoint_simulate(1, true);
    (*model).adjoint_simulate(0, true);
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


    exit(0);
}

