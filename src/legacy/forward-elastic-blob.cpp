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
    const arma::uword np_boundary = 75;
    const double np_factor = 0.004;
    const arma::uword nx = nx_domain + 2 * np_boundary;
    const arma::uword nz = nz_domain + np_boundary;

    // Dynamic fields
    arma::Mat<double> vx = arma::zeros(nx, nz);
    arma::Mat<double> vz = arma::zeros(nx, nz);
    arma::Mat<double> txx = arma::zeros(nx, nz);
    arma::Mat<double> tzz = arma::zeros(nx, nz);
    arma::Mat<double> txz = arma::zeros(nx, nz);

    // Load static fields
    arma::Mat<double> la = 4000000000 * arma::ones(nx, nz);
    arma::Mat<double> mu = 1000000000 * arma::ones(nx, nz);
    mu(arma::span(200, 250), arma::span(125, 150)) = 1.5 * mu(arma::span(200, 250), arma::span(125, 150));
    arma::Mat<double> lm = la + 2 * mu;
    arma::Mat<double> b = (1.0 / 1500.0) * arma::ones(nx, nz);

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

    arma::cube acc(nx, nz, nt);

    double coeff1 = 9.0 / 8.0;
//    double coeff1 = 4.0 / 3.0;
    double coeff2 = 1.0 / 24.0;
//    double coeff2 = 2.0 / 12.0;

    // Time marching
    for (int it = 0; it < nt; ++it) {
        // Inject time source

        txx(51 + np_boundary, 5) += 0.5 * dt * source[it];
        tzz(51 + np_boundary, 5) += 0.5 * dt * source[it];

#pragma omp parallel
#pragma omp for
        for (int ix = 0; ix < nx; ++ix) {
            for (int iz = 0; iz < nz; ++iz) {
                if (iz > 1 and ix > 1 and ix < nx - 1 and iz < nz - 1) {
                    txx(ix, iz) = taper(ix, iz) *
                                  (txx(ix, iz) +
                                   (dt * lm(ix, iz) * (
                                           -coeff2 * vx(ix + 1, iz - 1) + coeff1 * vx(ix, iz - 1)
                                           - coeff1 * vx(ix - 1, iz - 1) + coeff2 * vx(ix - 2, iz - 1)
                                   ) / dx +
                                    (dt * la(ix, iz)) * (
                                            -coeff2 * vz(ix - 1, iz + 1) + coeff1 * vz(ix - 1, iz)
                                            - coeff1 * vz(ix - 1, iz - 1) + coeff2 * vz(ix - 1, iz - 2)
                                    ) / dz));
                    tzz(ix, iz) = taper(ix, iz) *
                                  (tzz(ix, iz) +
                                   (dt * la(ix, iz) * (
                                           -coeff2 * vx(ix + 1, iz - 1) + coeff1 * vx(ix, iz - 1)
                                           - coeff1 * vx(ix - 1, iz - 1) + coeff2 * vx(ix - 2, iz - 1)
                                   ) / dx +
                                    (dt * lm(ix, iz)) * (
                                            -coeff2 * vz(ix - 1, iz + 1) + coeff1 * vz(ix - 1, iz)
                                            - coeff1 * vz(ix - 1, iz - 1) + coeff2 * vz(ix - 1, iz - 2)
                                    ) / dz));
                    txz(ix, iz) = taper(ix, iz) *
                                  (txz(ix, iz) + dt * mu(ix, iz) * (
                                          (
                                                  -coeff2 * vx(ix - 1, iz + 1) + coeff1 * vx(ix - 1, iz)
                                                  - coeff1 * vx(ix - 1, iz - 1) + coeff2 * vx(ix - 1, iz - 2)
                                          ) / dz +
                                          (
                                                  -coeff2 * vz(ix + 1, iz - 1) + coeff1 * vz(ix, iz - 1)
                                                  - coeff1 * vz(ix - 1, iz - 1) + coeff2 * vz(ix - 2, iz - 1)
                                          ) / dx));
                } else {
                    txx(ix, iz) = txx(ix, iz) * taper(ix, iz);
                    txz(ix, iz) = txz(ix, iz) * taper(ix, iz);
                    tzz(ix, iz) = tzz(ix, iz) * taper(ix, iz);
                }
            }
        }

#pragma omp parallel
#pragma omp for
        for (int ix = 0; ix < nx; ++ix) {
            for (int iz = 0; iz < nz; ++iz) {
                if (iz < nz - 2 and ix < nx - 2 and ix > 0 and iz > 0) {
                    vx(ix, iz) =
                            taper(ix, iz) *
                            (vx(ix, iz) + b(ix, iz) *
                                          (dt * (
                                                  -coeff2 * txx(ix + 2, iz + 1) +
                                                  coeff1 * txx(ix + 1, iz + 1)
                                                  - coeff1 * txx(ix, iz + 1) +
                                                  coeff2 * txx(ix - 1, iz + 1)
                                          ) / dx +
                                           dt * (
                                                   -coeff2 * txz(ix + 1, iz + 2) +
                                                   coeff1 * txz(ix + 1, iz + 1)
                                                   - coeff1 * txz(ix + 1, iz) +
                                                   coeff2 * txz(ix + 1, iz - 1)
                                           ) / dz));
                    vz(ix, iz) =
                            taper(ix, iz) *
                            (vz(ix, iz) + b(ix, iz) *
                                          (dt * (
                                                  -coeff2 * txz(ix + 2, iz + 1) +
                                                  coeff1 * txz(ix + 1, iz + 1)
                                                  - coeff1 * txz(ix, iz + 1) + coeff2 * txz(ix - 1, iz + 1)
                                          ) / dx +
                                           dt * (
                                                   -coeff2 * tzz(ix + 1, iz + 2) +
                                                   coeff1 * tzz(ix + 1, iz + 1)
                                                   - coeff1 * tzz(ix + 1, iz) + coeff2 * tzz(ix + 1, iz - 1)
                                           ) / dz));
                } else {
                    vx(ix, iz) = vx(ix, iz) * taper(ix, iz);
                    vz(ix, iz) = vz(ix, iz) * taper(ix, iz);
                }
            }
        }

        acc.slice(it) = vx; // takes a lot of ram
    }
//
#pragma omp parallel
#pragma omp for
    for (int it = 0; it < nt; ++it) { // takes a lot of time
        char filename[1024];
        sprintf(filename, "output/vx%i.txt", it);
        acc.slice(it).save(filename, arma::raw_ascii);
    }
    return 0;
}
