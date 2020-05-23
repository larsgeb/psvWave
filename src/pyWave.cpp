#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "fdWaveModel.h"

using Eigen::MatrixXd;

namespace py = pybind11;

PYBIND11_MODULE(pyWave, m){
	py::class_<fdWaveModel> (m, "fdWaveModel")
		.def(py::init<const char *>())
		.def("forward_shot", &fdWaveModel::forward_simulate)
		.def_readonly("n_shots", &fdWaveModel::n_shots)
		.def_readonly("n_sources", &fdWaveModel::n_sources);
}

