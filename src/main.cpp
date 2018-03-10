#include <iostream>
#include <armadillo>

#define _USE_MATH_DEFINES

#include <cmath>

int main() {
    const int nt = 3000;
    const double dt = 0.0001;
    const double dx = 10;
    const double dz = 10;
    const arma::uword nx_domain = 500;
    const arma::uword nz_domain = 500;
    const arma::uword np_boundary = 30;
    const arma::uword nx = nx_domain + 2 * np_boundary;
    const arma::uword nz = nz_domain + 2 * np_boundary;

    // Dynamic fields
    arma::Mat<double> vx = arma::zeros(nx, nz);
    arma::Mat<double> vz = arma::zeros(nx, nz);
    arma::Mat<double> txx = arma::zeros(nx, nz);
    arma::Mat<double> tzz = arma::zeros(nx, nz);
    arma::Mat<double> txz = arma::zeros(nx, nz);

    // Static fields
    const arma::Mat<double> beta = (1.0 / 2700.0) * arma::ones(nx, nz);
    const arma::Mat<double> lambda = 35000000000 * arma::ones(nx, nz);
    const arma::Mat<double> mu = 29000000000 * arma::ones(nx, nz);
    const arma::Mat<double> lp2m = lambda + (mu + mu);

    arma::Mat<double> taper = arma::ones(nx, nz);
    for (int iTaper = 1; iTaper < np_boundary + 1; ++iTaper) {
        taper.submat(iTaper, iTaper, nx - iTaper - 1, nz - iTaper - 1) =
                1 - (1 + iTaper) * arma::ones(nx - 2 * iTaper, nz - 2 * iTaper) / (np_boundary + 1);
    }
    taper = arma::exp(- (taper % taper) / 0.1);
    taper.col(0) *= 0;
    taper.col(nz-1) *= 0;
    taper.row(0) *= 0;
    taper.row(nx-1) *= 0;

//    std::cout << "Maximum P-wave speed: " << 1;
//    std::cout << "Poisson's ratio: " << 1;

    // Source function
    double centralFrequency = 1000;
    double tsource = 1 / centralFrequency;
    arma::vec time = arma::linspace(0, dt * (nt - 1), nt);
    double t0 = tsource * 1.5;
    arma::vec tau = M_PI * (time - t0) / t0;
    arma::vec source = (1 - 4 * tau % tau) % arma::exp(-2 * tau % tau);

    source.save("source.txt",arma::raw_ascii);

    // Time marching
    for (int it = 0; it < nt; ++it) {
        // Update dynamic fields at k+1/2 (vx and vz)
        vx.submat(1, 1, nx - 1, nz - 1) =
                vx.submat(1, 1, nx - 1, nz - 1) +
                beta.submat(1, 1, nx - 1, nz - 1) %
                (
                        (dt / dx) *
                        (txx.submat(1, 1, nx - 1, nz - 1) - txx.submat(0, 1, nx - 2, nz - 1))
                        +
                        (dt / dz) *
                        (txz.submat(1, 1, nx - 1, nz - 1) - txz.submat(1, 0, nx - 1, nz - 2))
                );
        vz.submat(0, 0, nx - 2, nz - 2) =
                vz.submat(0, 0, nx - 2, nz - 2) +
                beta.submat(0, 0, nx - 2, nz - 2) %
                (
                        (dt / dx) *
                        (txz.submat(1, 0, nx - 1, nz - 2) - txz.submat(0, 0, nx - 2, nz - 2))
                        +
                        (dt / dz) *
                        (tzz.submat(0, 1, nx - 2, nz - 1) - tzz.submat(0, 0, nx - 2, nz - 2))
                );

//        vx *= taper;
//        vz *= taper;

//        txx(45, 45) += 0.5 * dt * source[it];
//        tzz(45, 45) += 0.5 * dt * source[it];

        // Update dynamic fields at k+1 (txx, tzz and txz)
        txx.submat(0, 1, nx - 2, nz - 1) =
                txx.submat(0, 1, nx - 2, nz - 1) +
                lp2m.submat(0, 1, nx - 2, nz - 1) %
                (
                        (dt / dx) *
                        (vx.submat(1, 1, nx - 1, nz - 1) - vx.submat(0, 1, nx - 2, nz - 1))
                ) +
                lambda.submat(0, 1, nx - 2, nz - 1) % (
                        (dt / dz) *
                        (vz.submat(0, 1, nx - 2, nz - 1) - vz.submat(0, 0, nx - 2, nz - 2))
                );

        tzz.submat(0, 1, nx - 2, nz - 1) =
                tzz.submat(0, 1, nx - 2, nz - 1) +
                lp2m.submat(0, 1, nx - 2, nz - 1) %
                (
                        (dt / dz) *
                        (vz.submat(1, 1, nx - 1, nz - 1) - vz.submat(0, 1, nx - 2, nz - 1))
                ) +
                lambda.submat(0, 1, nx - 2, nz - 1) % (
                        (dt / dx) *
                        (vx.submat(0, 1, nx - 2, nz - 1) - vx.submat(0, 0, nx - 2, nz - 2))
                );


        txz.submat(1, 0, nx - 1, nz - 2) =
                txz.submat(1, 0, nx - 1, nz - 2) + mu.submat(1, 0, nx - 1, nz - 2) % (
                        (dt / dz) *
                        (vx.submat(1, 1, nx - 1, nz - 1) - vx.submat(1, 0, nx - 1, nz - 2))
                        +
                        (dt / dx) *
                        (vz.submat(1, 0, nx - 1, nz - 2) - vz.submat(0, 0, nx - 2, nz - 2))
                );

//        txx *= taper;
//        tzz *= taper;
//        txz *= taper;

        char filename[1024];
        sprintf(filename, "txx%i.txt", it);
        txx.save(filename, arma::raw_ascii);
    }


    return 0;
}