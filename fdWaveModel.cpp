//
// Created by lars on 25.01.19.
//

#include <omp.h>
#include <iostream>
#include <cmath>
#include <typeinfo>
#include <fstream>
#include <limits>
#include "fdWaveModel.h"

fdWaveModel::fdWaveModel() {

    // --- Informative section ---

    // Output whether or not compiled with OpenACC
//    if (OPENACC == 1) {
//        std::cout << std::endl << "OpenACC acceleration enabled, code should run on GPU." << std::endl;
//    } else {
//        std::cout << std::endl << "OpenACC acceleration not enabled, code should run on CPU." << std::endl;
//    }
    // Show real type (single or double precision)
    std::cout << "Code compiled with " << typeid(real).name() << " (d for double, accurate, f for float, fast)" << std::endl << std::flush;


    // --- Initialization section ---

    // Initialize data variance to one (take care of it outside of the code)
    std::fill(&data_variance_ux[0][0][0], &data_variance_ux[0][0][0] + sizeof(data_variance_ux) / sizeof(real), 1);
    std::fill(&data_variance_uz[0][0][0], &data_variance_uz[0][0][0] + sizeof(data_variance_uz) / sizeof(real), 1);

    // Assign stf/rtf_ux
    for (unsigned int it = 0; it < nt; ++it) {
        t[it] = it * dt;
        real f = 1.0 / alpha;
        real shiftedTime = t[it] - 1.4 / f;
        stf[it] = real((1 - 2 * pow(M_PI * f * shiftedTime, 2)) * exp(-pow(M_PI * f * shiftedTime, 2)));
    }

    moment[0][0] = 1;
    moment[0][1] = 0;
    moment[1][0] = 0;
    moment[1][1] = -1;

    // Setting all fields.
    std::fill(&vp[0][0], &vp[0][0] + sizeof(vp) / sizeof(real), scalar_vp);
    std::fill(&vs[0][0], &vs[0][0] + sizeof(vs) / sizeof(real), scalar_vs);
    std::fill(&rho[0][0], &rho[0][0] + sizeof(rho) / sizeof(real), scalar_rho);
    update_from_velocity();

    {
        // Initialize
        std::fill(&taper[0][0], &taper[0][0] + sizeof(taper) / sizeof(real), 0);
        for (int id = 0; id < np_boundary; ++id) {
            for (int ix = id; ix < nx - id; ++ix) {
                for (int iz = id; iz < nz; ++iz) {
                    taper[ix][iz]++;
                }
            }
        }
        for (auto &ix : taper) {
            for (float &element : ix) {
                element = static_cast<float>(exp(-pow(np_factor * (50 - element), 2)));
            }
        }
    }

    // Todo include more sanity checks
    if (floor(double(nt) / snapshot_interval) != snapshots) {
        throw std::length_error("Snapshot interval and size of accumulator don't match!");
    }

}

// Forward modeller
void fdWaveModel::forward_simulate(int i_source, bool store_fields, bool verbose) {

    // Reset dynamical fields
    std::fill(&vx[0][0], &vx[0][0] + sizeof(vx) / sizeof(int), 0);
    std::fill(&vz[0][0], &vz[0][0] + sizeof(vz) / sizeof(int), 0);
    std::fill(&txx[0][0], &txx[0][0] + sizeof(txx) / sizeof(int), 0);
    std::fill(&tzz[0][0], &tzz[0][0] + sizeof(tzz) / sizeof(int), 0);
    std::fill(&txz[0][0], &txz[0][0] + sizeof(txz) / sizeof(int), 0);

    // If verbose, count time
    double startTime = 0, stopTime = 0, secsElapsed = 0;
    if (verbose) { startTime = real(omp_get_wtime()); }

    for (int it = 0; it < nt; ++it) {

        // Take wavefield snapshot
        if (it % snapshot_interval == 0 and store_fields) {
            #pragma omp parallel for collapse(2)
            for (int ix = 0; ix < nx; ++ix) {
                for (int iz = 0; iz < nz; ++iz) {
                    accu_vx[i_source][it / snapshot_interval][ix][iz] = vx[ix][iz];
                    accu_vz[i_source][it / snapshot_interval][ix][iz] = vz[ix][iz];
                    accu_txx[i_source][it / snapshot_interval][ix][iz] = txx[ix][iz];
                    accu_txz[i_source][it / snapshot_interval][ix][iz] = txz[ix][iz];
                    accu_tzz[i_source][it / snapshot_interval][ix][iz] = tzz[ix][iz];
                }
            }
        }

        // Record seismograms by integrating velocity into displacement
        #pragma omp parallel for collapse(1)
        for (int i_receiver = 0; i_receiver < nr; ++i_receiver) {
            if (it == 0) {
                rtf_ux[i_source][i_receiver][it] = dt * vx[ix_receivers[i_receiver]][iz_receivers[i_receiver]] / (dx * dz);
                rtf_uz[i_source][i_receiver][it] = dt * vz[ix_receivers[i_receiver]][iz_receivers[i_receiver]] / (dx * dz);
            } else {
                rtf_ux[i_source][i_receiver][it] =
                        rtf_ux[i_source][i_receiver][it - 1] + dt * vx[ix_receivers[i_receiver]][iz_receivers[i_receiver]] / (dx * dz);
                rtf_uz[i_source][i_receiver][it] =
                        rtf_uz[i_source][i_receiver][it - 1] + dt * vz[ix_receivers[i_receiver]][iz_receivers[i_receiver]] / (dx * dz);
            }
        }

        // Time integrate dynamic fields for stress
        #pragma omp parallel for collapse(2)
        for (int ix = 2; ix < nx - 2; ++ix) {
            for (int iz = 2; iz < nz - 2; ++iz) {
                txx[ix][iz] = taper[ix][iz] *
                              (txx[ix][iz] +
                               dt *
                               (lm[ix][iz] * (
                                       c1 * (vx[ix + 1][iz] - vx[ix][iz]) +
                                       c2 * (vx[ix - 1][iz] - vx[ix + 2][iz])) / dx +
                                la[ix][iz] * (
                                        c1 * (vz[ix][iz] - vz[ix][iz - 1]) +
                                        c2 * (vz[ix][iz - 2] - vz[ix][iz + 1])) / dz));
                tzz[ix][iz] = taper[ix][iz] *
                              (tzz[ix][iz] +
                               dt *
                               (la[ix][iz] * (
                                       c1 * (vx[ix + 1][iz] - vx[ix][iz]) +
                                       c2 * (vx[ix - 1][iz] - vx[ix + 2][iz])) / dx +
                                (lm[ix][iz]) * (
                                        c1 * (vz[ix][iz] - vz[ix][iz - 1]) +
                                        c2 * (vz[ix][iz - 2] - vz[ix][iz + 1])) / dz));
                txz[ix][iz] = taper[ix][iz] *
                              (txz[ix][iz] + dt * mu[ix][iz] * (
                                      (c1 * (vx[ix][iz + 1] - vx[ix][iz]) +
                                       c2 * (vx[ix][iz - 1] - vx[ix][iz + 2])) / dz +
                                      (c1 * (vz[ix][iz] - vz[ix - 1][iz]) +
                                       c2 * (vz[ix - 2][iz] - vz[ix + 1][iz])) / dx));

            }
        }
        // Time integrate dynamic fields for velocity
        #pragma omp parallel for collapse(2)
        for (int ix = 2; ix < nx - 2; ++ix) {
            for (int iz = 2; iz < nz - 2; ++iz) {
                vx[ix][iz] =
                        taper[ix][iz] *
                        (vx[ix][iz]
                         + b_vx[ix][iz] * dt * (
                                (c1 * (txx[ix][iz] - txx[ix - 1][iz]) +
                                 c2 * (txx[ix - 2][iz] - txx[ix + 1][iz])) / dx +
                                (c1 * (txz[ix][iz] - txz[ix][iz - 1]) +
                                 c2 * (txz[ix][iz - 2] - txz[ix][iz + 1])) / dz));
                vz[ix][iz] =
                        taper[ix][iz] *
                        (vz[ix][iz]
                         + b_vz[ix][iz] * dt * (
                                (c1 * (txz[ix + 1][iz] - txz[ix][iz]) +
                                 c2 * (txz[ix - 1][iz] - txz[ix + 2][iz])) / dx +
                                (c1 * (tzz[ix][iz + 1] - tzz[ix][iz]) +
                                 c2 * (tzz[ix][iz - 1] - tzz[ix][iz + 2])) / dz));

            }
        }

        // |-inject source
        // | (x,x)-couple
        vx[ix_source[i_source] - 1][iz_source[i_source]] -=
                moment[0][0] * stf[it] * dt * b_vz[ix_source[i_source] - 1][iz_source[i_source]] / (dx * dx * dx * dx);
        vx[ix_source[i_source]][iz_source[i_source]] +=
                moment[0][0] * stf[it] * dt * b_vz[ix_source[i_source]][iz_source[i_source]] / (dx * dx * dx * dx);
        // | (z,z)-couple
        vz[ix_source[i_source]][iz_source[i_source] - 1] -=
                moment[1][1] * stf[it] * dt * b_vz[ix_source[i_source]][iz_source[i_source] - 1] / (dz * dz * dz * dz);
        vz[ix_source[i_source]][iz_source[i_source]] +=
                moment[1][1] * stf[it] * dt * b_vz[ix_source[i_source]][iz_source[i_source]] / (dz * dz * dz * dz);
        // | (x,z)-couple
        vx[ix_source[i_source] - 1][iz_source[i_source] + 1] +=
                0.25 * moment[0][1] * stf[it] * dt * b_vz[ix_source[i_source] - 1][iz_source[i_source] + 1] / (dx * dx * dx * dx);
        vx[ix_source[i_source]][iz_source[i_source] + 1] +=
                0.25 * moment[0][1] * stf[it] * dt * b_vz[ix_source[i_source]][iz_source[i_source] + 1] / (dx * dx * dx * dx);
        vx[ix_source[i_source] - 1][iz_source[i_source] - 1] -=
                0.25 * moment[0][1] * stf[it] * dt * b_vz[ix_source[i_source] - 1][iz_source[i_source] - 1] / (dx * dx * dx * dx);
        vx[ix_source[i_source]][iz_source[i_source] - 1] -=
                0.25 * moment[0][1] * stf[it] * dt * b_vz[ix_source[i_source]][iz_source[i_source] - 1] / (dx * dx * dx * dx);
        // | (z,x)-couple
        vz[ix_source[i_source] + 1][iz_source[i_source] - 1] +=
                0.25 * moment[1][0] * stf[it] * dt * b_vz[ix_source[i_source] + 1][iz_source[i_source] - 1] / (dz * dz * dz * dz);
        vz[ix_source[i_source] + 1][iz_source[i_source]] +=
                0.25 * moment[1][0] * stf[it] * dt * b_vz[ix_source[i_source] + 1][iz_source[i_source]] / (dz * dz * dz * dz);
        vz[ix_source[i_source] - 1][iz_source[i_source] - 1] -=
                0.25 * moment[1][0] * stf[it] * dt * b_vz[ix_source[i_source] - 1][iz_source[i_source] - 1] / (dz * dz * dz * dz);
        vz[ix_source[i_source] - 1][iz_source[i_source]] -=
                0.25 * moment[1][0] * stf[it] * dt * b_vz[ix_source[i_source] - 1][iz_source[i_source]] / (dz * dz * dz * dz);

    }

    // Output timing
    if (verbose) {
        stopTime = omp_get_wtime();
        secsElapsed = stopTime - startTime;
        std::cout << "Seconds elapsed for wave simulation: " << secsElapsed <<
                  std::endl;
    }
}


void fdWaveModel::adjoint_simulate(int i_source, bool verbose) {
    // Reset dynamical fields
    std::fill(&vx[0][0], &vx[0][0] + sizeof(vx) / sizeof(int), 0);
    std::fill(&vz[0][0], &vz[0][0] + sizeof(vz) / sizeof(int), 0);
    std::fill(&txx[0][0], &txx[0][0] + sizeof(txx) / sizeof(int), 0);
    std::fill(&tzz[0][0], &tzz[0][0] + sizeof(tzz) / sizeof(int), 0);
    std::fill(&txz[0][0], &txz[0][0] + sizeof(txz) / sizeof(int), 0);

    // If verbose, count time
    double startTime = 0, stopTime = 0, secsElapsed = 0;
    if (verbose) { startTime = real(omp_get_wtime()); }

    for (int it = nt - 1; it >= 0; --it) {

        // Correlate wavefields
        if (it % snapshot_interval == 0) { // Todo, rewrite for only relevant parameters
            #pragma omp parallel for collapse(2)
            for (int ix = 0; ix < nx; ++ix) {
                for (int iz = 0; iz < nz; ++iz) {
                    density_l_kernel[ix][iz] -= snapshot_interval * dt * (accu_vx[i_source][it / snapshot_interval][ix][iz] * vx[ix][iz] +
                                                                          accu_vz[i_source][it / snapshot_interval][ix][iz] * vz[ix][iz]);

                    lambda_kernel[ix][iz] += snapshot_interval * dt * // Todo ??
                                             (((accu_txx[i_source][it / snapshot_interval][ix][iz] -
                                                (accu_tzz[i_source][it / snapshot_interval][ix][iz] * la[ix][iz]) / lm[ix][iz]) +
                                               (accu_tzz[i_source][it / snapshot_interval][ix][iz] -
                                                (accu_txx[i_source][it / snapshot_interval][ix][iz] * la[ix][iz]) / lm[ix][iz]))
                                              * ((txx[ix][iz] - (tzz[ix][iz] * la[ix][iz]) / lm[ix][iz]) +
                                                 (tzz[ix][iz] - (txx[ix][iz] * la[ix][iz]) / lm[ix][iz]))) *
                                             pow(lm[ix][iz] - (pow(la[ix][iz], 2) / (lm[ix][iz])), -2); // todo optimize pows

                    mu_kernel[ix][iz] += snapshot_interval * dt * 2 *
                                         ((((txx[ix][iz] - (tzz[ix][iz] * la[ix][iz]) / lm[ix][iz]) *
                                            (accu_txx[i_source][it / snapshot_interval][ix][iz] -
                                             (accu_tzz[i_source][it / snapshot_interval][ix][iz] * la[ix][iz]) /
                                             lm[ix][iz])) +
                                           ((tzz[ix][iz] - (txx[ix][iz] * la[ix][iz]) / lm[ix][iz]) *
                                            (accu_tzz[i_source][it / snapshot_interval][ix][iz] -
                                             (accu_txx[i_source][it / snapshot_interval][ix][iz] * la[ix][iz]) /
                                             lm[ix][iz]))
                                          ) * pow(lm[ix][iz] - (pow(la[ix][iz], 2) / (lm[ix][iz])), -2) + // todo optimize pows
                                          2 *
                                          (txz[ix][iz] * accu_txz[i_source][it / snapshot_interval][ix][iz] *
                                           pow(2 * mu[ix][iz], -2))); // todo optimize pows
                }
            }
        }

        // Reverse time integrate dynamic fields for stress
        #pragma omp parallel for collapse(2)
        for (int ix = 2; ix < nx - 2; ++ix) {
            for (int iz = 2; iz < nz - 2; ++iz) {
                txx[ix][iz] = taper[ix][iz] *
                              (txx[ix][iz] -
                               dt *
                               (lm[ix][iz] * (
                                       c1 * (vx[ix + 1][iz] - vx[ix][iz]) +
                                       c2 * (vx[ix - 1][iz] - vx[ix + 2][iz])) / dx +
                                la[ix][iz] * (
                                        c1 * (vz[ix][iz] - vz[ix][iz - 1]) +
                                        c2 * (vz[ix][iz - 2] - vz[ix][iz + 1])) / dz));
                tzz[ix][iz] = taper[ix][iz] *
                              (tzz[ix][iz] -
                               dt *
                               (la[ix][iz] * (
                                       c1 * (vx[ix + 1][iz] - vx[ix][iz]) +
                                       c2 * (vx[ix - 1][iz] - vx[ix + 2][iz])) / dx +
                                (lm[ix][iz]) * (
                                        c1 * (vz[ix][iz] - vz[ix][iz - 1]) +
                                        c2 * (vz[ix][iz - 2] - vz[ix][iz + 1])) / dz));
                txz[ix][iz] = taper[ix][iz] *
                              (txz[ix][iz] - dt * mu[ix][iz] * (
                                      (c1 * (vx[ix][iz + 1] - vx[ix][iz]) +
                                       c2 * (vx[ix][iz - 1] - vx[ix][iz + 2])) / dz +
                                      (c1 * (vz[ix][iz] - vz[ix - 1][iz]) +
                                       c2 * (vz[ix - 2][iz] - vz[ix + 1][iz])) / dx));

            }
        }
        // Reverse time integrate dynamic fields for velocity
        #pragma omp parallel for collapse(2)
        for (int ix = 2; ix < nx - 2; ++ix) {
            for (int iz = 2; iz < nz - 2; ++iz) {
                vx[ix][iz] =
                        taper[ix][iz] *
                        (vx[ix][iz]
                         - b_vx[ix][iz] * dt * (
                                (c1 * (txx[ix][iz] - txx[ix - 1][iz]) +
                                 c2 * (txx[ix - 2][iz] - txx[ix + 1][iz])) / dx +
                                (c1 * (txz[ix][iz] - txz[ix][iz - 1]) +
                                 c2 * (txz[ix][iz - 2] - txz[ix][iz + 1])) / dz));
                vz[ix][iz] =
                        taper[ix][iz] *
                        (vz[ix][iz]
                         - b_vz[ix][iz] * dt * (
                                (c1 * (txz[ix + 1][iz] - txz[ix][iz]) +
                                 c2 * (txz[ix - 1][iz] - txz[ix + 2][iz])) / dx +
                                (c1 * (tzz[ix][iz + 1] - tzz[ix][iz]) +
                                 c2 * (tzz[ix][iz - 1] - tzz[ix][iz + 2])) / dz));

            }
        }

        // Inject adjoint sources
        for (int ir = 0; ir < nr; ++ir) {
            vx[ix_receivers[ir]][iz_receivers[ir]] += dt * b_vx[ix_receivers[ir]][iz_receivers[ir]] * a_stf_ux[i_source][ir][it] / (dx * dz);
            vz[ix_receivers[ir]][iz_receivers[ir]] += dt * b_vz[ix_receivers[ir]][iz_receivers[ir]] * a_stf_uz[i_source][ir][it] / (dx * dz);
        }
    }

    // Output timing
    if (verbose) {
        stopTime = omp_get_wtime();
        secsElapsed = stopTime - startTime;
        std::cout << "Seconds elapsed for wave simulation: " << secsElapsed <<
                  std::endl;
    }

}

void fdWaveModel::write_receivers() {
    std::string filename_ux;
    std::string filename_uz;

    std::ofstream receiver_file_ux;
    std::ofstream receiver_file_uz;

    for (int i_source = 0; i_source < ns; ++i_source) {

        filename_ux = "rtf_ux" + std::to_string(i_source) + ".txt";
        filename_uz = "rtf_uz" + std::to_string(i_source) + ".txt";

        receiver_file_ux.open(filename_ux);
        receiver_file_uz.open(filename_uz);

        receiver_file_ux.precision(std::numeric_limits<float>::digits10 + 10);
        receiver_file_uz.precision(std::numeric_limits<float>::digits10 + 10);

        for (int i_receiver = 0; i_receiver < nr; ++i_receiver) {
            receiver_file_ux << std::endl;
            receiver_file_uz << std::endl;
            for (int it = 0; it < nt; ++it) {
                receiver_file_ux << rtf_ux[i_source][i_receiver][it] << " ";
                receiver_file_uz << rtf_uz[i_source][i_receiver][it] << " ";

            }
        }
        receiver_file_ux.close();
        receiver_file_uz.close();
    }
}

void fdWaveModel::update_from_velocity() {
    #pragma omp parallel for collapse(2)
    for (int ix = 0; ix < nx; ++ix) {
        for (int iz = 0; iz < nz; ++iz) {
            mu[ix][iz] = real(pow(vs[ix][iz], 2) * rho[ix][iz]);
            lm[ix][iz] = real(pow(vp[ix][iz], 2) * rho[ix][iz]);
            la[ix][iz] = lm[ix][iz] - 2 * mu[ix][iz];
            b_vx[ix][iz] = real(1.0 / rho[ix][iz]);
            b_vz[ix][iz] = b_vx[ix][iz];
        }
    }
}

void fdWaveModel::load_receivers() { // Todo better file checking??
    std::string filename_ux;
    std::string filename_uz;

    std::ifstream receiver_file_ux;
    std::ifstream receiver_file_uz;

    for (int i_source = 0; i_source < 2; ++i_source) {
        filename_ux = "rtf_ux" + std::to_string(i_source) + ".txt";
        filename_uz = "rtf_uz" + std::to_string(i_source) + ".txt";

        receiver_file_ux.open(filename_ux);
        receiver_file_uz.open(filename_uz);

        // Check if the file actually exists
        std::cout << "File for ux data at source " << i_source << " is "
                  << (receiver_file_ux.good() ? "good (exists at least)." : "ungood.") << std::endl;
        std::cout << "File for uz data at source " << i_source << " is "
                  << (receiver_file_uz.good() ? "good (exists at least)." : "ungood.") << std::endl;
        if (!receiver_file_ux.good() or !receiver_file_uz.good()) {
            throw std::invalid_argument("Not all data is present!");
        }

        real placeholder_ux;
        real placeholder_uz;

        for (int i_receiver = 0; i_receiver < nr; ++i_receiver) {
            for (int it = 0; it < nt; ++it) {

                receiver_file_ux >> placeholder_ux;
                receiver_file_uz >> placeholder_uz;

                rtf_ux_true[i_source][i_receiver][it] = placeholder_ux;
                rtf_uz_true[i_source][i_receiver][it] = placeholder_uz;
            }
        }

        // Check data was large enough for set up
        if (!receiver_file_ux.good() or !receiver_file_uz.good()) {
            std::cout << "Received bad state of file at end of reading! Does the data match the set up?" << std::endl;
            throw std::invalid_argument("Not enough data is present!");
        }
        // Try to load more data ...
        receiver_file_ux >> placeholder_ux;
        receiver_file_uz >> placeholder_uz;
        // ... which shouldn't be possible
        if (receiver_file_ux.good() or receiver_file_uz.good()) {
            std::cout << "Received good state of file past reading! Does the data match the set up?" << std::endl;
            throw std::invalid_argument("Too much data is present!");
        }

        receiver_file_uz.close();
        receiver_file_ux.close();
    }

}

real fdWaveModel::calculate_misfit() { // todo Evaluate need for data variance
    real misfit = 0;
    for (int i_source = 0; i_source < ns; ++i_source) {
        for (int i_receiver = 0; i_receiver < nr; ++i_receiver) {
            for (int it = 0; it < nt; ++it) {
                misfit += 0.5 * dt * pow(rtf_ux_true[i_source][i_receiver][it] - rtf_ux[i_source][i_receiver][it], 2) /
                          data_variance_ux[i_source][i_receiver][it];
                misfit += 0.5 * dt * pow(rtf_uz_true[i_source][i_receiver][it] - rtf_uz[i_source][i_receiver][it], 2) /
                          data_variance_uz[i_source][i_receiver][it];
            }
        }
    }
    return misfit;
}

void fdWaveModel::calculate_adjoint_sources() {
    #pragma omp parallel for collapse(3)
    for (int is = 0; is < ns; ++is) {
        for (int ir = 0; ir < nr; ++ir) {
            for (int it = 0; it < nt; ++it) {
                a_stf_ux[is][ir][it] = rtf_ux[is][ir][it] - rtf_ux_true[is][ir][it];
                a_stf_uz[is][ir][it] = rtf_uz[is][ir][it] - rtf_uz_true[is][ir][it];
            }
        }

    }
}

void fdWaveModel::map_kernels_to_velocity() {
    #pragma omp parallel for collapse(2)
    for (int ix = 0; ix < nx; ++ix) {
        for (int iz = 0; iz < nz; ++iz) {
            vp_kernel[ix][iz] = 2 * vp[ix][iz] * lambda_kernel[ix][iz] / b_vx[ix][iz];
            vs_kernel[ix][iz] = (2 * vs[ix][iz] * mu_kernel[ix][iz] - 4 * vs[ix][iz] * lambda_kernel[ix][iz]) /
                                b_vx[ix][iz];
            density_v_kernel[ix][iz] = density_l_kernel[ix][iz]
                                       + (vp[ix][iz] * vp[ix][iz] - 2 * vs[ix][iz] * vs[ix][iz]) * lambda_kernel[ix][iz]
                                       + vs[ix][iz] * vs[ix][iz] * mu_kernel[ix][iz];
        }
    }
}
