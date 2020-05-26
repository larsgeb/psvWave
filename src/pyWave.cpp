#include "fdWaveModel.h"
#include <iostream>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <vector>

namespace py = pybind11;

template <class T>
py::array_t<T> array_to_numpy(T ****pointer, std::vector<ssize_t> shape) {
  return py::array_t<T>(py::buffer_info(
      ***pointer, sizeof(T), py::format_descriptor<T>::format(), 4, shape,
      std::vector<size_t>{shape[3] * shape[2] * shape[1] * sizeof(T),
                          shape[3] * shape[2] * sizeof(T), shape[3] * sizeof(T),
                          sizeof(T)}));
}
template <class T>
py::array_t<T> array_to_numpy(T ***pointer, std::vector<ssize_t> shape) {
  return py::array_t<T>(py::buffer_info(
      **pointer, sizeof(T), py::format_descriptor<T>::format(), 3, shape,
      std::vector<size_t>{shape[2] * shape[1] * sizeof(T), shape[2] * sizeof(T),
                          sizeof(T)}));
}
template <class T>
py::array_t<T> array_to_numpy(T **pointer, std::vector<ssize_t> shape) {
  return py::array_t<T>(py::buffer_info(
      *pointer, sizeof(T), py::format_descriptor<T>::format(), 2, shape,
      std::vector<size_t>{shape[1] * sizeof(T), sizeof(T)}));
}
template <class T>
py::array_t<T> array_to_numpy(T *pointer, std::vector<ssize_t> shape) {
  return py::array_t<T>(py::buffer_info(pointer, sizeof(T),
                                        py::format_descriptor<T>::format(), 1,
                                        shape, std::vector<size_t>{sizeof(T)}));
}

class fdWaveModelExtended : public fdWaveModel {
public:
  using fdWaveModel::fdWaveModel;

  py::tuple get_snapshots() {
    std::vector<ssize_t> shape = {n_shots, snapshots, nx, nz};
    return py::make_tuple(
        array_to_numpy(accu_vx, shape), array_to_numpy(accu_vz, shape),
        array_to_numpy(accu_txx, shape), array_to_numpy(accu_tzz, shape),
        array_to_numpy(accu_txz, shape));
  };

  py::tuple get_extent(bool include_absorbing_boundary = true) {
    if (include_absorbing_boundary) {
      return py::make_tuple(-np_boundary, dx * (nx_inner + np_boundary),
                            -np_boundary, dz * (nz_inner + np_boundary));
    } else {
      return py::make_tuple(0, dx * nx_inner, 0, dz * nz_inner);
    }
  };

  void forward_simulate_explicit_threads(int i_shot, bool store_fields,
                                         bool verbose, bool output_wavefields,
                                         int omp_threads_override) {
    auto old_limit = omp_get_num_threads();
    if (omp_threads_override != 0) {
      omp_set_num_threads(omp_threads_override);
    }
    forward_simulate(i_shot, store_fields, verbose, output_wavefields);
    omp_set_num_threads(old_limit);
  }

  py::tuple get_coordinates(bool in_units) {

    real_simulation **IX, **IZ;

    allocate_array(IX, nx, nz);
    allocate_array(IZ, nx, nz);

    if (in_units) {
      for (int ix = 0; ix < nx; ++ix) {
        for (int iz = 0; iz < nz; ++iz) {
          IX[ix][iz] = (ix - np_boundary) * dx;
          IZ[ix][iz] = (iz - np_boundary) * dz;
        }
      }
    } else {
      for (int ix = 0; ix < nx; ++ix) {
        for (int iz = 0; iz < nz; ++iz) {
          IX[ix][iz] = ix;
          IZ[ix][iz] = iz;
        }
      }
    }

    auto array_IX = array_to_numpy(IX, std::vector<ssize_t>{nx, nz});
    auto array_IZ = array_to_numpy(IZ, std::vector<ssize_t>{nx, nz});

    deallocate_array(IX);
    deallocate_array(IZ);

    return py::make_tuple(array_IX, array_IZ);
  }

  py::tuple get_parameter_fields() {
    auto array_vp = array_to_numpy(vp, std::vector<ssize_t>{nx, nz});
    auto array_vs = array_to_numpy(vs, std::vector<ssize_t>{nx, nz});
    auto array_rho = array_to_numpy(rho, std::vector<ssize_t>{nx, nz});
    return py::make_tuple(array_vp, array_vs, array_rho);
  }

  void set_parameter_fields(Eigen::MatrixXd _vp, Eigen::MatrixXd _vs,
                            Eigen::MatrixXd _rho) {
#pragma omp parallel for collapse(2)
    for (int ix = 0; ix < nx; ++ix) {
      for (int iz = 0; iz < nz; ++iz) {
        vp[ix][iz] = _vp(ix, iz);
        vs[ix][iz] = _vs(ix, iz);
        rho[ix][iz] = _rho(ix, iz);
      }
    }
    update_from_velocity();
  }
};

PYBIND11_MODULE(pyWave, m) {

  py::class_<fdWaveModelExtended>(m, "fdModel")
      .def(py::init<const char *>(), R"mydelimiter(
    The default constructor for fdWaveModel, a multithreaded finite
    difference solver for the elastic wave equation, built for FWI.

    Parameters
    ----------
    pathToIni : str
        String that contains the path to the .ini file describing the 
        FWI problem.
)mydelimiter")
      .def("forward_shot",
           &fdWaveModelExtended::forward_simulate_explicit_threads,
           py::arg("i_shot"), py::arg("store_fields") = true,
           py::arg("verbose") = false, py::arg("output_wavefields") = false,
           py::arg("omp_threads_override") = 0)
      .def_readonly("n_shots", &fdWaveModelExtended::n_shots)
      .def_readonly("n_sources", &fdWaveModelExtended::n_sources)
      .def("get_model_vector", &fdWaveModelExtended::get_model_vector)
      .def("set_model_vector", &fdWaveModelExtended::set_model_vector)
      .def("get_gradient_vector", &fdWaveModelExtended::get_gradient_vector)
      .def("load_vector", &fdWaveModelExtended::load_vector)
      .def("get_snapshots", &fdWaveModelExtended::get_snapshots,
           py::return_value_policy::move)
      .def_readonly("dt", &fdWaveModelExtended::dt)
      .def_readonly("dz", &fdWaveModelExtended::dz)
      .def_readonly("dx", &fdWaveModelExtended::dx)
      .def_readonly("time_step", &fdWaveModelExtended::dt)
      .def_readonly("snapshot_interval",
                    &fdWaveModelExtended::snapshot_interval)
      .def_readonly("snapshots", &fdWaveModelExtended::snapshots)
      .def("get_extent", &fdWaveModelExtended::get_extent)
      .def("get_coordinates", &fdWaveModelExtended::get_coordinates)
      .def("get_parameter_fields", &fdWaveModelExtended::get_parameter_fields)
      .def("set_parameter_fields", &fdWaveModelExtended::set_parameter_fields);
  ;
}
