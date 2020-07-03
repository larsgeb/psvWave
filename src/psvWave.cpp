#ifndef DOXYGEN_SHOULD_SKIP_THIS

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include <iostream>
#include <thread>
#include <vector>

#include "fdModel.h"

namespace py = pybind11;

#include <algorithm>
#include <iterator>
#include <vector>

template <typename T> std::ostream &operator<<(std::ostream &os, std::vector<T> vec) {
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
  return py::array_t<T>(
      py::buffer_info(*pointer, sizeof(T), py::format_descriptor<T>::format(), 2, shape,
                      std::vector<size_t>{shape[1] * sizeof(T), sizeof(T)}));
}
template <class T>
py::array_t<T> array_to_numpy(T *pointer, std::vector<ssize_t> shape) {
  return py::array_t<T>(py::buffer_info(pointer, sizeof(T),
                                        py::format_descriptor<T>::format(), 1, shape,
                                        std::vector<size_t>{sizeof(T)}));
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

class fdModelExtended : public fdModel {
public:
  using fdModel::fdModel;

  py::tuple get_snapshots() {
    std::vector<ssize_t> shape = {n_shots, snapshots, nx, nz};
    return py::make_tuple(
        array_to_numpy(accu_vx, shape), array_to_numpy(accu_vz, shape),
        array_to_numpy(accu_txx, shape), array_to_numpy(accu_tzz, shape),
        array_to_numpy(accu_txz, shape));
  };

  py::tuple get_extent(bool include_absorbing_boundary = true) {
    if (include_absorbing_boundary) {
      return py::make_tuple(-np_boundary, dx * (nx_inner + np_boundary), -np_boundary,
                            dz * (nz_inner + np_boundary));
    } else {
      return py::make_tuple(0, dx * nx_inner, 0, dz * nz_inner);
    }
  };

  void forward_simulate_explicit_threads(int i_shot, bool store_fields, bool verbose,
                                         bool output_wavefields,
                                         int omp_threads_override) {
    const auto old_limit = omp_get_max_threads();
    const auto optimal = std::thread::hardware_concurrency();

    if (verbose) {
      std::cout << "OpenMP info:" << std::endl
                << "  Original thread limit: " << omp_get_max_threads() << std::endl
                << "  Hardware concurrency: " << optimal << std::endl;
    }

    if (omp_threads_override != 0) {
      if (verbose) {
        std::cout << "  Setting override number of threads: " << omp_threads_override
                  << std::endl;
      }
      omp_set_num_threads(omp_threads_override);
    } else {
      if (verbose) {
        std::cout << "  Setting original number of threads: " << omp_get_max_threads()
                  << std::endl;
      }
      omp_set_num_threads(omp_get_max_threads());
    }

    if (verbose) {
      std::cout << "  Actual threads: " << omp_get_max_threads() << std::endl;
    }

    forward_simulate(i_shot, store_fields, verbose, output_wavefields);

    if (verbose) {
      std::cout << "  Resetting threads to: " << old_limit << std::endl << std::endl;
    }
    omp_set_num_threads(old_limit);
  }

  void adjoint_simulate_explicit_threads(int i_shot, bool verbose,
                                         int omp_threads_override) {
    const auto old_limit = omp_get_max_threads();
    const auto optimal = std::thread::hardware_concurrency();

    if (verbose) {
      std::cout << "OpenMP info:" << std::endl
                << "  Original thread limit: " << omp_get_max_threads() << std::endl
                << "  Hardware concurrency: " << optimal << std::endl;
    }

    if (omp_threads_override != 0) {
      if (verbose) {
        std::cout << "  Setting override number of threads: " << omp_threads_override
                  << std::endl;
      }
      omp_set_num_threads(omp_threads_override);
    } else {
      if (verbose) {
        std::cout << "  Setting original number of threads: " << omp_get_max_threads()
                  << std::endl;
      }
      omp_set_num_threads(omp_get_max_threads());
    }

    if (verbose) {
      std::cout << "  Actual threads: " << omp_get_max_threads() << std::endl;
    }

    adjoint_simulate(i_shot, verbose);

    if (verbose) {
      std::cout << "  Resetting threads to: " << old_limit << std::endl << std::endl;
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
      throw py::value_error("The input ndarray _vp does not have the right shape.");
    };
    if (!(_vs_buffer.shape == shape)) {
      throw py::value_error("The input ndarray _vs does not have the right shape.");
    };
    if (!(_rho_buffer.shape == shape)) {
      throw py::value_error("The input ndarray _rho does not have the right shape.");
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
    auto array_vp_kernel = array_to_numpy(vp_kernel, std::vector<ssize_t>{nx, nz});
    auto array_vs_kernel = array_to_numpy(vs_kernel, std::vector<ssize_t>{nx, nz});
    auto array_rho_kernel =
        array_to_numpy(density_v_kernel, std::vector<ssize_t>{nx, nz});
    return py::make_tuple(array_vp_kernel, array_vs_kernel, array_rho_kernel);
  }

  py::tuple get_synthetic_data() {
    auto array_rtf_ux = array_to_numpy(rtf_ux, std::vector<ssize_t>{n_shots, nr, nt});
    auto array_rtf_uz = array_to_numpy(rtf_uz, std::vector<ssize_t>{n_shots, nr, nt});
    return py::make_tuple(array_rtf_ux, array_rtf_uz);
  }

  py::tuple get_observed_data() {
    auto array_rtf_ux_true =
        array_to_numpy(rtf_ux_true, std::vector<ssize_t>{n_shots, nr, nt});
    auto array_rtf_uz_true =
        array_to_numpy(rtf_uz_true, std::vector<ssize_t>{n_shots, nr, nt});
    return py::make_tuple(array_rtf_ux_true, array_rtf_uz_true);
  }

  py::tuple get_receivers(bool in_units, bool include_absorbing_boundary_as_index) {
    // Create new arrays
    py::array_t<real_simulation> x_receivers(py::buffer_info(
        nullptr, sizeof(real_simulation),
        py::format_descriptor<real_simulation>::format(), 1, std::vector<ssize_t>{nr},
        std::vector<size_t>{sizeof(real_simulation)}));
    py::array_t<real_simulation> z_receivers(py::buffer_info(
        nullptr, sizeof(real_simulation),
        py::format_descriptor<real_simulation>::format(), 1, std::vector<ssize_t>{nr},
        std::vector<size_t>{sizeof(real_simulation)}));

    copy_data_cast((real_simulation *)x_receivers.request().ptr, ix_receivers, nr);

    copy_data_cast((real_simulation *)z_receivers.request().ptr, iz_receivers, nr);

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
  py::tuple get_sources(bool in_units, bool include_absorbing_boundary_as_index) {
    // Create new arrays
    py::array_t<real_simulation> x_sources(py::buffer_info(
        nullptr, sizeof(real_simulation),
        py::format_descriptor<real_simulation>::format(), 1,
        std::vector<ssize_t>{n_sources}, std::vector<size_t>{sizeof(real_simulation)}));
    py::array_t<real_simulation> z_sources(py::buffer_info(
        nullptr, sizeof(real_simulation),
        py::format_descriptor<real_simulation>::format(), 1,
        std::vector<ssize_t>{n_sources}, std::vector<size_t>{sizeof(real_simulation)}));

    copy_data_cast((real_simulation *)x_sources.request().ptr, ix_sources, n_sources);

    copy_data_cast((real_simulation *)z_sources.request().ptr, iz_sources, n_sources);

    if (!include_absorbing_boundary_as_index || in_units) {
      for (int ir = 0; ir < n_sources; ir++) {
        ((real_simulation *)x_sources.request().ptr)[ir] -= np_boundary;
        ((real_simulation *)z_sources.request().ptr)[ir] -= np_boundary;
        if (in_units) {
          ((real_simulation *)x_sources.request().ptr)[ir] *= dx;
          ((real_simulation *)z_sources.request().ptr)[ir] *= dz;
        }
      }
    }

    return py::make_tuple(x_sources, z_sources);
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
      throw py::value_error("The input ndarray ux does not have the right shape.");
    };
    if (!(uz_buffer.shape == shape)) {
      throw py::value_error("The input ndarray uz does not have the right shape.");
    };

    // Get the total size of the buffer
    int buffer_size = ux_buffer.shape[0] * ux_buffer.shape[1] * ux_buffer.shape[2];

    // Get pointer to the start of the (contiguous) buffer
    real_simulation *ptr_ux = (real_simulation *)ux_buffer.ptr;
    real_simulation *ptr_uz = (real_simulation *)uz_buffer.ptr;

    // Copy the data
    copy_data(rtf_ux[0][0], ptr_ux, buffer_size);
    copy_data(rtf_uz[0][0], ptr_uz, buffer_size);
  }

  fdModelExtended *copy() {
    // std::cout << "Returning" << std::endl;
    return std::move(new fdModelExtended(*this));
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
      throw py::value_error("The input ndarray ux does not have the right shape.");
    };
    if (!(uz_buffer.shape == shape)) {
      throw py::value_error("The input ndarray ux does not have the right shape.");
    };

    // Get the total size of the buffer
    int buffer_size = ux_buffer.shape[0] * ux_buffer.shape[1] * ux_buffer.shape[2];

    // Get pointer to the start of the (contiguous) buffer
    real_simulation *ux_ptr = (real_simulation *)ux_buffer.ptr;
    real_simulation *uz_ptr = (real_simulation *)uz_buffer.ptr;

    // Copy the data
    copy_data(rtf_ux_true[0][0], ux_ptr, buffer_size);
    copy_data(rtf_uz_true[0][0], uz_ptr, buffer_size);
  }
};

PYBIND11_MODULE(__psvWave_cpp, m) {
  py::options options;
  options.disable_function_signatures();

  py::class_<fdModelExtended>(m, "fdModel",
                              R"mydelimiter(fdModel(configuration_file_path: str)
    Class to simulate P-SV wave phyiscs and its adjoint state.

    :param configuration_file_path: Path to the desired configuration.
)mydelimiter")
      .def(py::init<const char *>())
      .def("copy", &fdModelExtended::copy,
           "copy() -> psvWave.fdModel\n"
           "\n"
           "Returns a copy of the object, duplicating all members.")
      .def("forward_simulate", &fdModelExtended::forward_simulate_explicit_threads,
           py::arg("i_shot"), py::arg("store_fields") = true,
           py::arg("verbose") = false, py::arg("output_wavefields") = false,
           py::arg("omp_threads_override") = 0,
           "forward_simulate(i_shot: int, store_fields: bool = True, verbose: bool = "
           "False, output_wavefields: bool = False, omp_threads_override: int = 0)\n"
           "\n"
           "Run forward simulations for a given 'shot'.\n"
           "\n"
           ":param i_shot: Integer representing which shot will be simulated.\n"
           ":type  i_shot: int\n"
           ":param store_fields: Boolean controlling whether or not wavefields are "
           "stored, defaults to `True`.\n"
           ":type  store_fields: int\n"
           ":param verbose: Boolean controlling the verbosity of the simulation.\n"
           ":type  verbose: bool\n"
           ":param output_wavefields: Boolean controlling whether or not wavefields "
           "are written to disk.\n"
           ":type  output_wavefields: boolean\n"
           ":param omp_threads_override: Integer determining the amounts of threads "
           "that will be used. Defaults to the environment variable if not passed / "
           "0.\n"
           ":type  omp_threads_override: int\n")
      .def_readonly(
          "n_sources", &fdModelExtended::n_sources,
          "Number of sources across shots. Does not indicate how many per shot.")
      .def("get_model_vector", &fdModelExtended::get_model_vector,
           "get_model_vector() -> numpy.ndarray\n"
           "\n"
           "Get the current model in the model as a numpy vector, flattened.\n"
           "\n"
           ":returns: Current model vector containing P-wave speed, S-wave speed, and "
           "density.\n"
           ":rtype: numpy.ndarray")
      .def("set_model_vector", &fdModelExtended::set_model_vector,
           "set_model_vector(m: numpy.ndarray)\n"
           "\n"
           "Update the model (vp, vs, rho) in the class.\n"
           "\n"
           ":param m: vector of shape (free_parameters, 1).\n"
           ":type m: numpy.ndarray\n"
           "\n")
      .def("get_gradient_vector", &fdModelExtended::get_gradient_vector,
           "get_gradient_vector() -> numpy.ndarray\n"
           "\n"
           "Returns the computed gradient vector. Should be retreived only after all "
           "the following functions are run:\n"
           "\n"
           "1. :meth:`~psvWave.fdModel.set_model_vector`, to set a new model for which "
           "   to calculate the gradient.\n"
           "2. For each shot, :meth:`~psvWave.fdModel.forward_simulate`, to simulate "
           "   the forward wavefields.\n"
           "3. :meth:`~psvWave.fdModel.calculate_l2_misfit` to calculate the L2 misfit "
           "   of the synthetic waveforms w.r.t. the data.\n"
           "4. :meth:`~psvWave.fdModel.calculate_l2_adjoint_sources` to calculate the "
           "   adjoint sources corresponding to the current model.\n"
           "5. :meth:`~psvWave.fdModel.calculate_l2_adjoint_sources` to calculate the "
           "   adjoint sources corresponding to the current model.\n"
           "6. :meth:`~psvWave.fdModel.adjoint_simulate` to simulate the adjoint "
           "   (backward) wavefield.\n"
           "7. :meth:`~psvWave.fdModel.reset_kernels` to empty the current kernels.\n"
           "8. For each shot :meth:`~psvWave.fdModel.adjoint_simulate` to simulate the "
           "   adjoint (backward) wavefield exactly once. Simulating one of the shots "
           "   twice will result in incorrect kernels.\n"
           "9. :meth:`~psvWave.fdModel.map_kernels_to_velocity` to map the "
           "   sensitivities in Lamé's parameters (lambda, mu, rho) to velocity (vp, "
           "   vs, rho).\n"
           "\n"
           ":returns: Current gradient vector.\n"
           ":rtype: numpy.ndarray")
      .def("load_vector", &fdModelExtended::load_vector,
           "load_vector(relative_path: str, verbose: bool) -> numpy.ndarray\n"
           "\n"
           "Loads a vector of shape (free_parameters, 1) from a text file. Read in "
           "precision `real_simulation`.\n")
      .def("get_snapshots", &fdModelExtended::get_snapshots,
           py::return_value_policy::move,
           "get_snapshots() -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, "
           "numpy.ndarray, numpy.ndarray]\n"
           "\n"
           "Get snapshots of all the dynamical fields generated across all the shots.")
      .def_readonly("dt", &fdModelExtended::dt, "Time discretization")
      .def_readonly("dz", &fdModelExtended::dz, "Vertical discretization")
      .def_readonly("dx", &fdModelExtended::dx, "Horizontal discretization")
      .def_readonly("nt", &fdModelExtended::nt, "Total time points")
      .def_readonly("nz", &fdModelExtended::nz,
                    "Total vertical points, including boundary layer")
      .def_readonly("nx", &fdModelExtended::nx,
                    "Total horizontal points, including boundary layer")
      .def_readonly("nx_free_parameters", &fdModelExtended::nx_free_parameters)
      .def_readonly("nz_free_parameters", &fdModelExtended::nz_free_parameters)
      .def_readonly("nx_inner", &fdModelExtended::nx_inner)
      .def_readonly("nz_inner", &fdModelExtended::nz_inner)
      .def_readonly("nx_inner_boundary", &fdModelExtended::nx_inner_boundary)
      .def_readonly("nz_inner_boundary", &fdModelExtended::nz_inner_boundary)
      .def_readonly("np_boundary", &fdModelExtended::np_boundary)
      .def_readonly("free_parameters", &fdModelExtended::free_parameters,
                    "Total free parameters in the model, vp, vs, rho combined.")
      .def_readonly("snapshot_interval", &fdModelExtended::snapshot_interval,
                    "The interval of timesteps between snapshots.")
      .def_readonly("snapshots", &fdModelExtended::snapshots,
                    "The total amount of snapshots per shot.")
      .def("get_extent", &fdModelExtended::get_extent)
      .def("get_coordinates", &fdModelExtended::get_coordinates)
      .def("get_parameter_fields", &fdModelExtended::get_parameter_fields)
      .def("get_kernels", &fdModelExtended::get_kernels)
      .def("set_parameter_fields", &fdModelExtended::set_parameter_fields)
      .def("get_synthetic_data", &fdModelExtended::get_synthetic_data)
      .def("get_observed_data", &fdModelExtended::get_observed_data)
      .def_readonly("n_shots", &fdModelExtended::n_shots, "Number of shots")
      .def_readonly("which_source_to_fire_in_which_shot",
                    &fdModelExtended::which_source_to_fire_in_which_shot,
                    "Which source fires in which shot.")
      .def("set_synthetic_data", &fdModelExtended::set_synthetic_data)
      .def("set_observed_data", &fdModelExtended::set_observed_data)
      .def("get_sources", &fdModelExtended::get_sources, py::arg("in_units") = true,
           py::arg("include_absorbing_boundary_as_index") = true)
      .def("get_receivers", &fdModelExtended::get_receivers, py::arg("in_units") = true,
           py::arg("include_absorbing_boundary_as_index") = true,
           "get_receivers(in_units: bool, include_absorbing_boundary_as_index: bool) "
           "-> Tuple[numpy.ndarray, numpy,ndarray]\n"
           "\n"
           "Get the receiver array coordinates either in meters or grid indices. \n"
           "\n"
           ":param in_units: Boolean controlling if the returned coordinates are "
           "physical distance from origin or grid indices.\n"
           ":type  in_units: bool\n"
           ":param include_absorbing_boundary_as_index: Boolean controlling whether or "
           "not to place the origin at the edge of the absording boundary or within "
           "it.\n"
           ":type  include_absorbing_boundary_as_index: bool\n"
           "\n"
           ":returns: A tuple of 2 numpy.ndarray's for horizontal and vertical "
           "coordinate, each of shape (n_receivers, 1). \n"
           ":rtype: Tuple[numpy.ndarray, numpy,ndarray]")
      .def("calculate_l2_misfit", &fdModelExtended::calculate_l2_misfit,
           "calculate_l2_misfit()\n"
           "\n"
           "Calculate L2 misfit for simulated waveforms w.r.t. observed data.")
      .def("calculate_l2_adjoint_sources",
           &fdModelExtended::calculate_l2_adjoint_sources,
           "calculate_l2_adjoint_sources()\n"
           "\n"
           "Calculate the adjoint sources corresponding to the L2 misfit w.r.t. the "
           "observed data.\n")
      .def_readonly(
          "misfit", &fdModelExtended::misfit,
          "Current misfit in the model.\n"
          "\n"
          "This value is updated after :meth:`~psvWave.fdModel.calculate_l2_misfit` "
          "is run.")
      .def("reset_kernels", &fdModelExtended::reset_kernels,
           "reset_kernels()\n"
           "\n"
           "Reset kernel accumulators to zero. Important for adjoint simulations, as "
           "they simply keep accumulating wavefield correlations across shots and "
           "iterations otherwise. Making this manual allows for flexibility (e.g. "
           "different misfit per shot).\n")
      .def("adjoint_simulate", &fdModelExtended::adjoint_simulate_explicit_threads,
           py::arg("i_shot"), py::arg("verbose") = false,
           py::arg("omp_threads_override") = 0,
           "adjoint_simulate(i_shot: int, verbose: bool, omp_threads_override: int)\n"
           "\n"
           "Adjoint simulate the wavefield for a given shot. This additionally "
           "correlates the adjoint wavefields for this shot with those stored in the "
           "snapshots to calculate the sensitivity kernel in Lamé's parameters.\n"
           "\n"
           ":param i_shot: Integer representing which shot will be simulated.\n"
           ":type  i_shot: int\n"
           ":param verbose: Boolean controlling the verbosity of the simulation.\n"
           ":type  verbose: bool\n"
           ":param omp_threads_override: Integer determining the amounts of threads "
           "that will be used. Defaults to the environment variable if not passed / "
           "0.\n"
           ":type  omp_threads_override: int\n")
      .def("map_kernels_to_velocity", &fdModelExtended::map_kernels_to_velocity,
           "map_kernels_to_velocity()\n"
           "\n"
           "Transform sensitivity kernels in Lamé parametrization (lambda, mu, rho) to "
           "velocity parametrization (vp, vs, rho).");

  ;
}

#endif /* DOXYGEN_SHOULD_SKIP_THIS */