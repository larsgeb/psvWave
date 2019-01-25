//
// Created by lars on 25.01.19.
//

#include <omp.h>
#include <iostream>
#include <cmath>
#include <typeinfo>
#include "fdWaveModel.h"

fdWaveModel::fdWaveModel() {

    for (int ir = 0; ir < nr; ++ir) {
        ix_receivers[ir] = ir + np_boundary;
        iz_receivers[ir] = 10;
    }

    // Output whether or not compiled with OpenACC
    if (OPENACC == 1) {
        std::cout << std::endl << "OpenACC acceleration enabled, code should run on GPU." << std::endl;
    } else {
        std::cout << std::endl << "OpenACC acceleration not enabled, code should run on CPU." << std::endl;
    }

    // Show real type (single or double precision)
    std::cout << "Code compiled with " << typeid(real).name() << " (d for double, accurate, f for float, fast)" << std::endl << std::flush;


    // Initialize fields
    for (int ix = 0; ix < nx; ++ix) {
        for (int iz = 0; iz < nz; ++iz) {
            rho[ix][iz] = scalar_rho;
            vp[ix][iz] = scalar_vp;
            vs[ix][iz] = scalar_vs;
            taper[ix][iz] = 1; // todo rewrite Gaussian taper
        }
    }

    // Assign stf/rtf_ux
    for (unsigned int it = 0; it < nt; ++it) {
        t[it] = it * dt;
//        stf[it] = real(exp(-alpha * pow(it - t0 / dt, 2))); Gaussian pulse
        real f = 1.0 / alpha;
        real shiftedTime = t[it] - 1.4 / f;
        stf[it] = real((1 - 2 * pow(M_PI * f * shiftedTime, 2)) * exp(-pow(M_PI * f * shiftedTime, 2)));
    }

    moment[0][0] = 1;
    moment[0][1] = 0;
    moment[1][0] = 0;
    moment[1][1] = -1;

    // Setting all fields.
    #pragma omp parallel for collapse(2)
    for (int ix = 0; ix < nx; ++ix) {
        for (int iz = 0; iz < nz; ++iz) {
            rho[ix][iz] = rho[0][0];
            vp[ix][iz] = vp[0][0];
            vs[ix][iz] = vs[0][0];
            taper[ix][iz] = 0;

            mu[ix][iz] = real(pow(vs[ix][iz], 2) * rho[ix][iz]);
            lm[ix][iz] = real(pow(vp[ix][iz], 2) * rho[ix][iz]);
            la[ix][iz] = lm[ix][iz] - 2 * mu[ix][iz];
            b_vx[ix][iz] = real(1.0 / rho[ix][iz]);
            b_vz[ix][iz] = b_vx[ix][iz];
        }
    }


    for (int id = 0; id < np_boundary; ++id) {
        for (int ix = id; ix < nx - id; ++ix) {
            for (int iz = id; iz < nz; ++iz) {
                taper[ix][iz]++;
            }
        }
    }

    for (int ix = 0; ix < nx; ++ix) {
        for (int iz = 0; iz < nz; ++iz) {
            taper[ix][iz] = exp(-pow(np_factor * (50 - taper[ix][iz]), 2));
        }
    }

    // Todo include more sanity checks
    if (floor(double(nt) / snapshotInterval) != snapshots) {
        throw std::length_error("Snapshot interval and size of accumulator don't match!");
    }

}


// Forward modeller
int fdWaveModel::forwardSimulate(bool storeFields, bool verbose, int isource) {

    // If verbose, count time
    double startTime = 0, stopTime = 0, secsElapsed = 0;
    if (verbose) { startTime = real(omp_get_wtime()); }

    for (int it = 0; it < nt; ++it) {

        // Record!
        for (int ireceiver = 0; ireceiver < nr; ++ireceiver) {
            if (it == 0) {
                rtf_ux[isource][ireceiver][it] = dt * vx[ix_receivers[ireceiver]][iz_receivers[ireceiver]] / (dx * dz);
                rtf_uz[isource][ireceiver][it] = dt * vz[ix_receivers[ireceiver]][iz_receivers[ireceiver]] / (dx * dz);
            } else {
                rtf_ux[isource][ireceiver][it] =
                        rtf_ux[isource][ireceiver][it - 1] + dt * vx[ix_receivers[ireceiver]][iz_receivers[ireceiver]] / (dx * dz);
                rtf_uz[isource][ireceiver][it] =
                        rtf_uz[isource][ireceiver][it - 1] + dt * vz[ix_receivers[ireceiver]][iz_receivers[ireceiver]] / (dx * dz);
            }
        }


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

        // (x,x)-couple
        vx[ix_source[isource] - 1][iz_source[isource]] -=
                moment[0][0] * stf[it] * dt * b_vz[ix_source[isource] - 1][iz_source[isource]] / (dx * dx * dx * dx);
        vx[ix_source[isource]][iz_source[isource]] += moment[0][0] * stf[it] * dt * b_vz[ix_source[isource]][iz_source[isource]] / (dx * dx * dx * dx);

        // (z,z)-couple
        vz[ix_source[isource]][iz_source[isource] - 1] -=
                moment[1][1] * stf[it] * dt * b_vz[ix_source[isource]][iz_source[isource] - 1] / (dz * dz * dz * dz);
        vz[ix_source[isource]][iz_source[isource]] += moment[1][1] * stf[it] * dt * b_vz[ix_source[isource]][iz_source[isource]] / (dz * dz * dz * dz);

        // (x,z)-couple
        vx[ix_source[isource] - 1][iz_source[isource] + 1] +=
                0.25 * moment[0][1] * stf[it] * dt * b_vz[ix_source[isource] - 1][iz_source[isource] + 1] / (dx * dx * dx * dx);
        vx[ix_source[isource]][iz_source[isource] + 1] +=
                0.25 * moment[0][1] * stf[it] * dt * b_vz[ix_source[isource]][iz_source[isource] + 1] / (dx * dx * dx * dx);
        vx[ix_source[isource] - 1][iz_source[isource] - 1] -=
                0.25 * moment[0][1] * stf[it] * dt * b_vz[ix_source[isource] - 1][iz_source[isource] - 1] / (dx * dx * dx * dx);
        vx[ix_source[isource]][iz_source[isource] - 1] -=
                0.25 * moment[0][1] * stf[it] * dt * b_vz[ix_source[isource]][iz_source[isource] - 1] / (dx * dx * dx * dx);

        // (z,x)-couple
        vz[ix_source[isource] + 1][iz_source[isource] - 1] +=
                0.25 * moment[1][0] * stf[it] * dt * b_vz[ix_source[isource] + 1][iz_source[isource] - 1] / (dz * dz * dz * dz);
        vz[ix_source[isource] + 1][iz_source[isource]] +=
                0.25 * moment[1][0] * stf[it] * dt * b_vz[ix_source[isource] + 1][iz_source[isource]] / (dz * dz * dz * dz);
        vz[ix_source[isource] - 1][iz_source[isource] - 1] -=
                0.25 * moment[1][0] * stf[it] * dt * b_vz[ix_source[isource] - 1][iz_source[isource] - 1] / (dz * dz * dz * dz);
        vz[ix_source[isource] - 1][iz_source[isource]] -=
                0.25 * moment[1][0] * stf[it] * dt * b_vz[ix_source[isource] - 1][iz_source[isource]] / (dz * dz * dz * dz);

        if (it % snapshotInterval == 0) {
            #pragma omp parallel for collapse(2)
            for (int ix = 0; ix < nx; ++ix) {
                for (int iz = 0; iz < nz; ++iz) {
                    accu_vx[it / snapshotInterval][ix][iz] = vx[ix][iz];
                    accu_vz[it / snapshotInterval][ix][iz] = vz[ix][iz];
                    accu_txx[it / snapshotInterval][ix][iz] = txx[ix][iz];
                    accu_txz[it / snapshotInterval][ix][iz] = txz[ix][iz];
                    accu_tzz[it / snapshotInterval][ix][iz] = tzz[ix][iz];
                }
            }
        }
    }

//    std::cout << std::endl << sizeof(accu_vx) << std::endl;

    // Output timing
    if (verbose) {
        stopTime = omp_get_wtime();
        secsElapsed = stopTime - startTime;
        std::cout << "Seconds elapsed for wave simulation: " << secsElapsed <<
                  std::endl;
    }

    // Cumulative check
//    real a = 0;
//    for (const auto &item : vx) {
//        for (const auto &item2 : item) {
//            a += item2;
//        }
//    }
//    std::cout << a << std::endl;

    // Cumulative check
//    real b = 0;
//    for (const auto &accu_it : accu_vx) {
//        for (const auto &accu_ix : accu_it) {
//            for (const auto &iter_iz : accu_ix) {
//                b += iter_iz;
//            }
//        }
//    }
//    std::cout << b << std::endl;

    return 0;
}
