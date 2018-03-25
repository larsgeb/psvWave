#include <iostream>
#include <armadillo>

#define _USE_MATH_DEFINES

#include <cmath>

int main(int argc, char *argv[]) {
    // Simulation settings
    const int nt = 3000;
    const double dt = 0.00025;
    const double dx = 1.249;
    const double dz = 1.249;
    const arma::uword nx_domain = 400;
    const arma::uword nz_domain = 200;
    const arma::uword np_boundary = 50;
    const double np_factor = 0.0075;
    const arma::uword nx = nx_domain + 2 * np_boundary;
    const arma::uword nz = nz_domain + np_boundary;

    // Dynamic fields
    arma::Mat<double> vx = arma::zeros(nx, nz);
    arma::Mat<double> vz = arma::zeros(nx, nz);
    arma::Mat<double> txx = arma::zeros(nx, nz);
    arma::Mat<double> tzz = arma::zeros(nx, nz);
    arma::Mat<double> txz = arma::zeros(nx, nz);

    // Load static fields
    arma::mat la;
    la.load("la.txt");
    arma::mat mu;
    mu.load("mu.txt");
    arma::mat lm;
    lm.load("lm.txt");
    arma::mat b;
    b.load("de.txt");
    b = 1.0 / b;

    std::cout << "lm rows: " << lm.n_rows << ", lm cols: " << lm.n_cols << std::endl;
    std::cout << "vx rows: " << vx.n_rows << ", vx cols: " << vx.n_cols << std::endl;

    std::cout << "P-wave speed: " << sqrt(lm.max() * b.max()) << std::endl;
    std::cout << "S-wave speed: " << sqrt(mu.max() * b.max()) << std::endl;
    std::cout << "Stability number: " << sqrt(lm.max() * b.max()) * dt * sqrt(1.0 / (dx * dx) + 1.0 / (dz * dz))
              << std::endl;


    // Source function (Ricker wavelet)
    double centralFrequency = 50.0;
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
        txx(arma::span(1, nx - 1), arma::span(1, nz - 1)) =
                txx(arma::span(1, nx - 1), arma::span(1, nz - 1)) +
                ((dt / dx) *
                 lm(arma::span(1, nx - 1), arma::span(1, nz - 1))) %
                (vx(arma::span(1, nx - 1), arma::span(0, nz - 2)) -
                 vx(arma::span(0, nx - 2), arma::span(0, nz - 2)))
                +
                ((dt / dz) *
                 la(arma::span(1, nx - 1), arma::span(1, nz - 1))) %
                (vz(arma::span(0, nx - 2), arma::span(1, nz - 1)) -
                 vz(arma::span(0, nx - 2), arma::span(0, nz - 2)));

        tzz(arma::span(1, nx - 1), arma::span(1, nz - 1)) +=
                ((dt / dx) * la(arma::span(1, nx - 1), arma::span(1, nz - 1))) %
                (vx(arma::span(1, nx - 1), arma::span(0, nz - 2)) -
                 vx(arma::span(0, nx - 2), arma::span(0, nz - 2)))
                +
                ((dt / dz) * lm(arma::span(1, nx - 1), arma::span(1, nz - 1))) %
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

        txx = txx % taper;
        tzz = tzz % taper;
        txz = txz % taper;

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

        vx = vx % taper;
        vz = vz % taper;

        char filename[1024];
        sprintf(filename, "output/vx%i.txt", it);
        vx.save(filename, arma::raw_ascii);
    }
    std::cout << vx * vz;

    return 0;
}
