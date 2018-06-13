//
// Created by lars on 03/04/18.
//

#ifndef HMC_FORWARD_FD_FUNCTIONS_H
#define HMC_FORWARD_FD_FUNCTIONS_H

#include <armadillo>
#include <iterator>

arma::mat generateGaussian(arma::uword nx, arma::uword nz, double std_dist, double ix, double iz);

arma::vec generateRicker(double dt, arma::uword nt, double freq);

template<typename T>
std::ostream &operator<<(std::ostream &out, const std::vector<T> &v) {
    if (!v.empty()) {
        out << '[';
        std::copy(v.begin(), v.end(), std::ostream_iterator<T>(out, ", "));
        out << "\b\b]";
    }
    return out;
}

#endif //HMC_FORWARD_FD_FUNCTIONS_H
