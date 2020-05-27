#include "fdWaveModel.h"
#include <iostream>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <thread>
#include <vector>

namespace py = pybind11;

#include <algorithm>
#include <iterator>
#include <vector>

template <typename T>
std::ostream &operator<<(std::ostream &os, std::vector<T> vec) {
  os << "{ ";
  std::copy(vec.begin(), vec.end(), std::ostream_iterator<T>(os, " "));
  os << "}";
  return os;
}

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

template <class T> void copy_data(T *destination, T *source, int size) {
#pragma omp parallel for collapse(1)
  for (size_t i1 = 0; i1 < size; i1++) {
    destination[i1] = source[i1];
  }
}

template <class T1, class T2>
void copy_data_cast(T1 *destination, T2 *source, int size) {
#pragma omp parallel for collapse(1)
  for (size_t i1 = 0; i1 < size; i1++) {
    destination[i1] = (T1)source[i1];
  }
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

    const auto old_limit = omp_get_max_threads();
    const auto optimal = std::thread::hardware_concurrency();

    if (verbose) {
      std::cout << "OpenMP info:" << std::endl
                << "  Original thread limit: " << omp_get_max_threads()
                << std::endl
                << "  Hardware concurrency: " << optimal << std::endl;
    }

    if (omp_threads_override != 0) {
      if (verbose) {
        std::cout << "  Setting override number of threads: "
                  << omp_threads_override << std::endl;
      }
      omp_set_num_threads(omp_threads_override);
    } else {
      if (verbose) {
        std::cout << "  Setting original number of threads: "
                  << omp_get_max_threads() << std::endl;
      }
      omp_set_num_threads(omp_get_max_threads());
    }

    if (verbose) {
      std::cout << "  Actual threads: " << omp_get_max_threads() << std::endl;
    }

    forward_simulate(i_shot, store_fields, verbose, output_wavefields);

    if (verbose) {
      std::cout << "  Resetting threads to: " << old_limit << std::endl
                << std::endl;
    }
    omp_set_num_threads(old_limit);
  }

  void adjoint_simulate_explicit_threads(int i_shot, bool verbose,
                                         int omp_threads_override) {

    const auto old_limit = omp_get_max_threads();
    const auto optimal = std::thread::hardware_concurrency();

    if (verbose) {
      std::cout << "OpenMP info:" << std::endl
                << "  Original thread limit: " << omp_get_max_threads()
                << std::endl
                << "  Hardware concurrency: " << optimal << std::endl;
    }

    if (omp_threads_override != 0) {
      if (verbose) {
        std::cout << "  Setting override number of threads: "
                  << omp_threads_override << std::endl;
      }
      omp_set_num_threads(omp_threads_override);
    } else {
      if (verbose) {
        std::cout << "  Setting original number of threads: "
                  << omp_get_max_threads() << std::endl;
      }
      omp_set_num_threads(omp_get_max_threads());
    }

    if (verbose) {
      std::cout << "  Actual threads: " << omp_get_max_threads() << std::endl;
    }

    adjoint_simulate(i_shot, verbose);

    if (verbose) {
      std::cout << "  Resetting threads to: " << old_limit << std::endl
                << std::endl;
    }
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

  void set_parameter_fields(py::array_t<real_simulation> _vp,
                            py::array_t<real_simulation> _vs,
                            py::array_t<real_simulation> _rho) {

    // Get buffer information for the passed arrays
    py::buffer_info _vp_buffer = _vp.request();
    py::buffer_info _vs_buffer = _vs.request();
    py::buffer_info _rho_buffer = _rho.request();

    // Get the required size
    std::vector<ssize_t> shape{nx, nz};

    // Verify the buffer shape
    if (!(_vp_buffer.shape == shape)) {
      throw py::value_error(
          "The input ndarray _vp does not have the right shape.");
    };
    if (!(_vs_buffer.shape == shape)) {
      throw py::value_error(
          "The input ndarray _vs does not have the right shape.");
    };
    if (!(_rho_buffer.shape == shape)) {
      throw py::value_error(
          "The input ndarray _rho does not have the right shape.");
    };

    // Get the total size of the buffer
    int buffer_size = _vp_buffer.shape[0] * _vp_buffer.shape[1];

    // Get pointer to the start of the (contiguous) buffer
    real_simulation *_vp_ptr = (real_simulation *)_vp_buffer.ptr;
    real_simulation *_vs_ptr = (real_simulation *)_vs_buffer.ptr;
    real_simulation *_rho_ptr = (real_simulation *)_rho_buffer.ptr;

    // Copy the data
    copy_data(vp[0], _vp_ptr, buffer_size);
    copy_data(vs[0], _vs_ptr, buffer_size);
    copy_data(rho[0], _rho_ptr, buffer_size);

    // Recalculate Lamé's parameters
    update_from_velocity();
  }

  py::tuple get_kernels() {
    auto array_vp_kernel =
        array_to_numpy(vp_kernel, std::vector<ssize_t>{nx, nz});
    auto array_vs_kernel =
        array_to_numpy(vs_kernel, std::vector<ssize_t>{nx, nz});
    auto array_rho_kernel =
        array_to_numpy(density_v_kernel, std::vector<ssize_t>{nx, nz});
    return py::make_tuple(array_vp_kernel, array_vs_kernel, array_rho_kernel);
  }

  py::tuple get_synthetic_data() {
    auto array_rtf_ux =
        array_to_numpy(rtf_ux, std::vector<ssize_t>{n_shots, nr, nt});
    auto array_rtf_uz =
        array_to_numpy(rtf_uz, std::vector<ssize_t>{n_shots, nr, nt});
    return py::make_tuple(array_rtf_ux, array_rtf_uz);
  }

  py::tuple get_observed_data() {
    auto array_rtf_ux_true =
        array_to_numpy(rtf_ux_true, std::vector<ssize_t>{n_shots, nr, nt});
    auto array_rtf_uz_true =
        array_to_numpy(rtf_uz_true, std::vector<ssize_t>{n_shots, nr, nt});
    return py::make_tuple(array_rtf_ux_true, array_rtf_uz_true);
  }

  py::tuple get_receivers(bool in_units,
                          bool include_absorbing_boundary_as_index) {

    // Create new arrays
    py::array_t<real_simulation> x_receivers(
        py::buffer_info(nullptr, sizeof(real_simulation),
                        py::format_descriptor<real_simulation>::format(), 1,
                        std::vector<ssize_t>{nr},
                        std::vector<size_t>{sizeof(real_simulation)}));
    py::array_t<real_simulation> z_receivers(
        py::buffer_info(nullptr, sizeof(real_simulation),
                        py::format_descriptor<real_simulation>::format(), 1,
                        std::vector<ssize_t>{nr},
                        std::vector<size_t>{sizeof(real_simulation)}));

    copy_data_cast((real_simulation *)x_receivers.request().ptr, ix_receivers,
                   nr);

    copy_data_cast((real_simulation *)z_receivers.request().ptr, iz_receivers,
                   nr);

    if (!include_absorbing_boundary_as_index || in_units) {
      for (int ir = 0; ir < nr; ir++) {
        ((real_simulation *)x_receivers.request().ptr)[ir] -= np_boundary;
        ((real_simulation *)z_receivers.request().ptr)[ir] -= np_boundary;
        if (in_units) {
          ((real_simulation *)x_receivers.request().ptr)[ir] *= dx;
          ((real_simulation *)z_receivers.request().ptr)[ir] *= dz;
        }
      }
    }

    return py::make_tuple(x_receivers, z_receivers);
  }

  void set_synthetic_data(py::array_t<real_simulation> ux,
                          py::array_t<real_simulation> uz) {

    // Get buffer information for the passed arrays
    py::buffer_info ux_buffer = ux.request();
    py::buffer_info uz_buffer = uz.request();

    // Get the required size
    std::vector<ssize_t> shape{n_shots, nr, nt};

    // Verify the buffer shape
    if (!(ux_buffer.shape == shape)) {
      throw py::value_error(
          "The input ndarray ux does not have the right shape.");
    };
    if (!(uz_buffer.shape == shape)) {
      throw py::value_error(
          "The input ndarray uz does not have the right shape.");
    };

    // Get the total size of the buffer
    int buffer_size =
        ux_buffer.shape[0] * ux_buffer.shape[1] * ux_buffer.shape[2];

    // Get pointer to the start of the (contiguous) buffer
    real_simulation *ptr_ux = (real_simulation *)ux_buffer.ptr;
    real_simulation *ptr_uz = (real_simulation *)uz_buffer.ptr;

    // Copy the data
    copy_data(rtf_ux[0][0], ptr_ux, buffer_size);
    copy_data(rtf_uz[0][0], ptr_uz, buffer_size);
  }

  void set_observed_data(py::array_t<real_simulation> ux,
                         py::array_t<real_simulation> uz) {
    // Get buffer information for the passed arrays
    py::buffer_info ux_buffer = ux.request();
    py::buffer_info uz_buffer = uz.request();

    // Get the required size
    std::vector<ssize_t> shape{n_shots, nr, nt};

    // Verify the buffer shape
    if (!(ux_buffer.shape == shape)) {
      throw py::value_error(
          "The input ndarray ux does not have the right shape.");
    };
    if (!(uz_buffer.shape == shape)) {
      throw py::value_error(
          "The input ndarray ux does not have the right shape.");
    };

    // Get the total size of the buffer
    int buffer_size =
        ux_buffer.shape[0] * ux_buffer.shape[1] * ux_buffer.shape[2];

    // Get pointer to the start of the (contiguous) buffer
    real_simulation *ux_ptr = (real_simulation *)ux_buffer.ptr;
    real_simulation *uz_ptr = (real_simulation *)uz_buffer.ptr;

    // Copy the data
    copy_data(rtf_ux_true[0][0], ux_ptr, buffer_size);
    copy_data(rtf_uz_true[0][0], uz_ptr, buffer_size);
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
      .def("forward_simulate",
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
      .def("get_kernels", &fdWaveModelExtended::get_kernels)
      .def("set_parameter_fields", &fdWaveModelExtended::set_parameter_fields)
      .def("get_synthetic_data", &fdWaveModelExtended::get_synthetic_data)
      .def("get_observed_data", &fdWaveModelExtended::get_observed_data)
      .def("set_synthetic_data", &fdWaveModelExtended::set_synthetic_data)
      .def("set_observed_data", &fdWaveModelExtended::set_observed_data)
      .def("get_receivers", &fdWaveModelExtended::get_receivers,
           py::arg("in_units") = true,
           py::arg("include_absorbing_boundary_as_index") = true)
      .def("calculate_l2_misfit", &fdWaveModelExtended::calculate_l2_misfit)
      .def("calculate_l2_adjoint_sources",
           &fdWaveModelExtended::calculate_l2_adjoint_sources)
      .def_readonly("n_shots", &fdWaveModelExtended::n_shots)
      .def_readonly("misfit", &fdWaveModelExtended::misfit)
      .def("reset_kernels", &fdWaveModelExtended::reset_kernels)
      .def("adjoint_simulate",
           &fdWaveModelExtended::adjoint_simulate_explicit_threads,
           py::arg("i_shot"), py::arg("verbose") = false,
           py::arg("omp_threads_override") = 0)
      .def("map_kernels_to_velocity",
           &fdWaveModelExtended::map_kernels_to_velocity);

  ;
}
