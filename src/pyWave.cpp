#include "fdWaveModel.h"
#include <iostream>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <vector>

namespace py = pybind11;

class fdWaveModelExtended : public fdWaveModel {
public:
  using fdWaveModel::fdWaveModel;

  // Function to get the wavefields after a simulation as a tuple of 5
  // 4-dimensional NumPy arrays.
  py::tuple get_snapshots() {
    // Shape of the RTF's
    std::vector<ssize_t> shape = {n_shots, snapshots, nx, nz};

    py::array_t<real_simulation> numpy_accu_vx(py::buffer_info(
        accu_vx->arr,
        sizeof(real_simulation), // itemsize
        py::format_descriptor<real_simulation>::format(),
        4, // ndim
        shape,
        std::vector<size_t>{
            shape[3] * shape[2] * shape[1] * sizeof(real_simulation),
            shape[3] * shape[2] * sizeof(real_simulation),
            shape[3] * sizeof(real_simulation), sizeof(real_simulation)}
        // strides
        ));

    return py::make_tuple(numpy_accu_vx);
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
      .def("get_extent", &fdWaveModelExtended::get_extent);
}
