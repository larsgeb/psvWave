#include <iostream>
#include <armadillo>

#define _USE_MATH_DEFINES

#include <cmath>

int main(int argc, char *argv[]) {
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
    arma::Mat<double> txx = arma::zeros(nx, nz);
    arma::Mat<double> tzz = arma::zeros(nx, nz);
    arma::Mat<double> txz = arma::zeros(nx, nz);

    // Set these quantities
    double rho = 1500.0;
    double vp = 2000;
    double poissons = 0.25;

    // Compute the material properties ...
    double b_val = 1.0 / rho;
    double vs = vp * sqrt(0.5 * ((1 - 2 * poissons) / (2 * (1 - poissons))));
    double mu_val = vs * vs * rho;
    double la2mu_val = vp * vp * rho;
    double lambda_val = la2mu_val - 2 * mu_val;
    // ... and static fields
    const arma::Mat<double> b = b_val * arma::ones(nx, nz);
    const arma::Mat<double> mu = mu_val * arma::ones(nx, nz);
    const arma::Mat<double> la2mu = la2mu_val * arma::ones(nx, nz);
    const arma::Mat<double> lambda = lambda_val * arma::ones(nx, nz);

    std::cout << "P-wave speed: " << sqrt(la2mu_val * b_val) << std::endl;
    std::cout << "S-wave speed: " << sqrt(mu_val * b_val) << std::endl;
    std::cout << "Stability number: " << sqrt(la2mu_val * b_val) * dt * sqrt(1.0 / (dx * dx) + 1.0 / (dz * dz))
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
    for (int it = 0; it < nt; ++it) {
        // Inject time source
        txx(51 + np_boundary, 5) += 0.5 * dt * source[it];
        tzz(51 + np_boundary, 5) += 0.5 * dt * source[it];

        // Update stresses
        txx(arma::span(1, nx - 1), arma::span(1, nz - 1)) +=
                ((dt / dx) * la2mu(arma::span(1, nx - 1), arma::span(1, nz - 1))) %
                (vx(arma::span(1, nx - 1), arma::span(0, nz - 2)) -
                 vx(arma::span(0, nx - 2), arma::span(0, nz - 2)))
                +
                ((dt / dz) * lambda(arma::span(1, nx - 1), arma::span(1, nz - 1))) %
                (vz(arma::span(0, nx - 2), arma::span(1, nz - 1)) -
                 vz(arma::span(0, nx - 2), arma::span(0, nz - 2)));

        tzz(arma::span(1, nx - 1), arma::span(1, nz - 1)) +=
                ((dt / dx) * lambda(arma::span(1, nx - 1), arma::span(1, nz - 1))) %
                (vx(arma::span(1, nx - 1), arma::span(0, nz - 2)) -
                 vx(arma::span(0, nx - 2), arma::span(0, nz - 2)))
                +
                ((dt / dz) * la2mu(arma::span(1, nx - 1), arma::span(1, nz - 1))) %
                (vz(arma::span(0, nx - 2), arma::span(1, nz - 1)) -
                 vz(arma::span(0, nx - 2), arma::span(0, nz - 2)));

        txz(arma::span(1, nx - 1), arma::span(1, nz - 1)) +=
                (dt * mu(arma::span(1, nx - 1), arma::span(1, nz - 1))) %
                (
                        (vx(arma::span(0, nx - 2), arma::span(1, nz - 1)) -
                         vx(arma::span(0, nx - 2), arma::span(0, nz - 2))) / dz
                        +
                        (vz(arma::span(1, nx - 1), arma::span(0, nz - 2)) -
                         vz(arma::span(0, nx - 2), arma::span(0, nz - 2))) / dx
                );

//        txx = txx % taper;
//        tzz = tzz % taper;
//        txz = txz % taper;

        // Update velocities
        vx(arma::span(0, nx - 2), arma::span(0, nz - 2)) +=
                b(arma::span(0, nx - 2), arma::span(0, nz - 2)) % (
                        (dt / dx) *
                        (txx(arma::span(1, nx - 1), arma::span(1, nz - 1)) -
                         txx(arma::span(0, nx - 2), arma::span(1, nz - 1))) +
                        (dt / dz) *
                        (txz(arma::span(1, nx - 1), arma::span(1, nz - 1)) -
                         txz(arma::span(1, nx - 1), arma::span(0, nz - 2)))
                );

        vz(arma::span(0, nx - 2), arma::span(0, nz - 2)) +=
                b(arma::span(0, nx - 2), arma::span(0, nz - 2)) % (
                        (dt / dz) *
                        (tzz(arma::span(1, nx - 1), arma::span(1, nz - 1)) -
                         tzz(arma::span(1, nx - 1), arma::span(0, nz - 2))) +
                        (dt / dx) *
                        (txz(arma::span(1, nx - 1), arma::span(1, nz - 1)) -
                         txz(arma::span(0, nx - 2), arma::span(1, nz - 1)))
                );

//        vx = vx % taper;
//        vz = vz % taper;

        char filename[1024];
        sprintf(filename, "output/vx%i.txt", it);
        vx.save(filename, arma::raw_ascii);
    }
    return 0;
}