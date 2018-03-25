#include <iostream>
#include <armadillo>

#define _USE_MATH_DEFINES

#include <cmath>

int main(int argc, char* argv[]) {
    // Simulation settings
    const int nt = 3000;
    const double dt = 0.00001;
    const double dx = 0.0667;
    const double dz = 0.0667;
    const arma::uword nx_domain = 300;
    const arma::uword nz_domain = 300;
    const arma::uword np_boundary = 0;
    const double np_factor = 0.015;
    const arma::uword nx = nx_domain + 2 * np_boundary;
    const arma::uword nz = nz_domain + np_boundary;

    // Dynamic fields
    arma::Mat<double> vx = arma::zeros(nx, nz);
    arma::Mat<double> vz = arma::zeros(nx, nz);
    arma::Mat<double> px = arma::zeros(nx, nz);
    arma::Mat<double> pz = arma::zeros(nx, nz);

    // Static fields
    double rho = 1500.0;
    arma::Mat<double> vp = (2000.0 * 2000.0) * arma::ones(nx, nz);
//    vp(arma::span(25, 30), arma::span(50, 55)) = 1.5 * vp(arma::span(25, 30), arma::span(50, 55));

    std::cout << "P-wave speed: " << sqrt(vp(0,0)) << std::endl;
    std::cout << "Stability number: " << sqrt(vp(0,0)) * dt * sqrt(1.0 / (dx * dx) + 1.0 / (dz * dz))
              << std::endl;

    // Source function (Ricker wavelet)
    double centralFrequency = 1000.0;
    double tsource = 1.0 / centralFrequency;
    arma::vec time = arma::linspace(0, dt * (nt - 1), nt);
    double t0 = tsource * 1.5;
    arma::vec tau = M_PI * (time - t0) / t0;
    arma::vec source = (1 - 4 * tau % tau) % arma::exp(-2 * tau % tau);
    source.save("output/source.txt", arma::raw_ascii);

    // Taper matrix
    arma::Mat<double> taper = np_boundary * arma::ones(nx, nz);
    for (int iTaper = 0; iTaper < np_boundary; ++iTaper) {
        taper.submat(iTaper, 0, nx - iTaper - 1, nz - iTaper - 1) =
                1 + iTaper * arma::ones(nx - 2 * iTaper, nz - iTaper);
    }
    taper = arma::exp(-arma::square(np_factor * (np_boundary - taper)));

    // Time marching
    double progress = 0.0;
    int barWidth = 70;
    for (int it = 0; it < nt; ++it) {

        px(51 + np_boundary, 5) += 0.5 * dt * source[it];
        pz(51 + np_boundary, 5) += 0.5 * dt * source[it];

        px(arma::span(1, nx - 1), arma::span(1, nz - 1)) -=
                ((dt / dx) * rho * vp(arma::span(1, nx - 1), arma::span(1, nz - 1))) %
                (vx(arma::span(1, nx - 1), arma::span(0, nz - 2)) -
                 vx(arma::span(0, nx - 2), arma::span(0, nz - 2)));
        pz(arma::span(1, nx - 1), arma::span(1, nz - 1)) -=
                ((dt / dz) * rho * vp(arma::span(1, nx - 1), arma::span(1, nz - 1))) %
                (vz(arma::span(0, nx - 2), arma::span(1, nz - 1)) -
                 vz(arma::span(0, nx - 2), arma::span(0, nz - 2)));

        px = px % taper;
        pz = pz % taper;

        vx(arma::span(0, nx - 2), arma::span(0, nz - 2)) -= (((dt / dx) / rho)) *
                                                            (px(arma::span(1, nx - 1), arma::span(1, nz - 1)) +
                                                             pz(arma::span(1, nx - 1), arma::span(1, nz - 1)) -
                                                             px(arma::span(0, nx - 2), arma::span(1, nz - 1)) -
                                                             pz(arma::span(0, nx - 2), arma::span(1, nz - 1)));
        vz(arma::span(0, nx - 2), arma::span(0, nz - 2)) -= (((dt / dz) / rho)) *
                                                            (px(arma::span(1, nx - 1), arma::span(1, nz - 1)) +
                                                             pz(arma::span(1, nx - 1), arma::span(1, nz - 1)) -
                                                             px(arma::span(1, nx - 1), arma::span(0, nz - 2)) -
                                                             pz(arma::span(1, nx - 1), arma::span(0, nz - 2)));

        vx = vx % taper;
        vz = vz % taper;

        char filename[1024];
        sprintf(filename, "output/p%i.txt", it);
        arma::Mat<double> p = (px + pz);
        p.save(filename, arma::raw_ascii);

    }


    return 0;
}