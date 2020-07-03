import numpy as _numpy
import matplotlib.pyplot as _plt
from typing import Tuple as _Tuple

from ._version import get_versions
from __psvWave_cpp import fdModel as fdModel

__version__ = get_versions()["version"]
__full_revisionid__ = get_versions()["full-revisionid"]
__date__ = get_versions()["date"]

# No need to make get_versions publicly accessible
del get_versions


# Class attributes ---------------------------------------------------------------------

fdModel.units = [r"$m/s$", r"$m/s$", r"$kg/m^3$"]
fdModel.parameters = [
    r"$v_p$",
    r"$v_s$",
    r"$\rho$",
]

# Class methods


def _add_method(cls):
    def decorator(func):
        setattr(cls, func.__name__.strip("_"), func)
        return func

    return decorator


@_add_method(fdModel)
def _plot_data(
    self: fdModel, data: _Tuple[_numpy.ndarray, _numpy.ndarray], exagerration=5.0,
):
    ux, uz = data
    n_shots = self.n_shots
    t = _numpy.arange(self.nt) * self.dt
    figure, axes = _plt.subplots(n_shots, 2, figsize=(8, 3 * n_shots))
    if axes.shape.__len__() == 1:
        axes = axes[:, None]
    max_amp = _numpy.max([ux, uz]) / exagerration
    for i_shot in range(n_shots):
        _ = axes[i_shot, 0].plot(
            t,
            (ux[i_shot, :, :].T + max_amp * _numpy.arange(ux.shape[1])) / max_amp,
            "r",
        )
        _ = axes[i_shot, 1].plot(
            t,
            (uz[i_shot, :, :].T + max_amp * _numpy.arange(ux.shape[1])) / max_amp,
            "r",
        )
        axes[i_shot, 0].set_title(f"horizontal displacement shot {i_shot}")
        axes[i_shot, 1].set_title(f"vertical displacement shot {i_shot}")

        axes[i_shot, 0].set_xlabel("time [s]")
        axes[i_shot, 0].set_ylabel("channel id")
        axes[i_shot, 1].set_xlabel("time [s]")
        axes[i_shot, 1].set_ylabel("channel id")
    _plt.tight_layout()
    return figure, axes


@_add_method(fdModel)
def _plot_synthetic_data(self: fdModel, exagerration=5.0):
    self.plot_data(
        self.get_synthetic_data(), exagerration=exagerration,
    )


@_add_method(fdModel)
def _plot_observed_data(self: fdModel, exagerration=5.0):
    self.plot_data(
        self.get_observed_data(), exagerration=exagerration,
    )


@_add_method(fdModel)
def _plot_domain(self: fdModel, axis=None, shot_to_plot=None):
    figure = None
    if axis is None:
        figure, axis = _plt.subplots(1, 1, figsize=(4, 3))

    if shot_to_plot is not None and type(shot_to_plot) == int:
        shot_to_plot = [shot_to_plot]

    if shot_to_plot is not None:
        for shot in shot_to_plot:
            assert (
                shot < self.n_shots
            ), "There's fewer shots in the model than the requested shot."

    extent = self.get_extent(True)
    rx, rz = self.get_receivers()
    sx, sz = self.get_sources()
    shots = self.which_source_to_fire_in_which_shot

    axis.set_xlabel("x [m]")
    axis.set_ylabel("z [m]")
    axis.set_xlim(extent[0], extent[1])
    axis.set_ylim(extent[2], extent[3])
    axis.set_aspect("equal")

    axis.scatter(
        rx, rz, marker="v", s=5, c="k", label="receivers",
    )

    if shot_to_plot is not None:
        for i_shot in shot_to_plot:
            axis.scatter(
                sx[shots[i_shot]],
                sz[shots[i_shot]],
                marker="x",
                s=5,
                c="r",
                label="sources",
            )
    else:
        axis.scatter(
            sx, sz, marker="x", s=5, c="r", label="sources",
        )

    _plt.tight_layout()
    return figure, axis


@_add_method(fdModel)
def _plot_fields(
    self: fdModel,
    fields=None,
    axes=None,
    shot_to_plot=None,
    cmap=_plt.get_cmap("seismic"),
    vmin=[None] * 3,
    vmax=[None] * 3,
):
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    if axes is None:
        figure, axes = _plt.subplots(nrows=3, ncols=1, figsize=(4, 9))

    if fields is None:
        fields = self.get_parameter_fields()

    for field in fields:
        assert field.shape == (self.nx, self.nz,), "Field has the wrong shape"

    extent = self.get_extent(True)

    for i, axis in enumerate(axes):
        self.plot_domain(axis=axis, shot_to_plot=shot_to_plot)

        image = axis.imshow(
            fields[i].T,
            extent=[extent[0], extent[1], extent[3], extent[2],],
            cmap=cmap,
            vmin=vmin[i],
            vmax=vmax[i],
        )

        divider = make_axes_locatable(axis)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        _plt.colorbar(image, cax=cax)
        cax.set_ylabel(f"{self.parameters[i]} [{self.units[i]}]")


@_add_method(fdModel)
def _plot_model_vector(
    self: fdModel,
    m,
    axes=None,
    shot_to_plot=None,
    cmap=_plt.get_cmap("seismic"),
    vmin=[None] * 3,
    vmax=[None] * 3,
):
    if axes is None:
        figure, axes = _plt.subplots(nrows=3, ncols=1, figsize=(4, 9))

    vector_splits = _numpy.split(m, 3)

    fields = self.get_parameter_fields()
    for i, field in enumerate(fields):
        field[:] = _numpy.nan
        field[
            (self.nx_inner_boundary + self.np_boundary) : -(
                self.nx_inner_boundary + self.np_boundary
            ),
            (self.nz_inner_boundary + self.np_boundary) : -(
                self.nz_inner_boundary + self.np_boundary
            ),
        ] = (
            vector_splits[i]
            .reshape((self.nz_free_parameters, self.nx_free_parameters,))
            .T
        )

    return self.plot_fields(
        fields=fields,
        axes=axes,
        shot_to_plot=shot_to_plot,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
