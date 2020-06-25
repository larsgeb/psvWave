from matplotlib import animation
import psvWave
import matplotlib.pyplot as plt
import numpy

model = psvWave.fdModel(
    "../tests/test_configurations/forward_configuration_4_sources.ini"
)

# Create target model ---------------------------------------------------------

# Get the coordinates of every grid point
IX, IZ = model.get_coordinates(True)
extent = model.get_extent(True)
# Get the associated parameter fields
vp, vs, rho = model.get_parameter_fields()

vp_starting = vp
vs_starting = vs
rho_starting = rho

numpy.save("vp_starting", vp_starting)
numpy.save("vs_starting", vs_starting)
numpy.save("rho_starting", rho_starting)

x_middle = (IX.max() + IX.min()) / 2
z_middle = (IZ.max() + IZ.min()) / 2

circle = ((IX - x_middle) ** 2 + (IZ - z_middle) ** 2) ** 0.5 < 15
vs = vs * (1 - 0.1 * circle)
vp = vp * (1 - 0.1 * circle)

cmap = plt.get_cmap("seismic")
plt.subplot(311)
plt.imshow(vp.T, extent=extent, vmin=1600, vmax=2400, cmap=cmap)
plt.subplot(312)
plt.imshow(vs.T, extent=extent, vmin=600, vmax=1000, cmap=cmap)
plt.subplot(313)
plt.imshow(rho.T, extent=extent, vmin=1200, vmax=1800, cmap=cmap)
plt.show()

vp_target = vp
vs_target = vs
rho_target = rho

numpy.save("vp_target", vp_target)
numpy.save("vs_target", vs_target)
numpy.save("rho_target", rho_target)

model.set_parameter_fields(vp_target, vs_target, rho_target)

# Create true data ------------------------------------------------------------

for i_shot in range(model.n_shots):
    model.forward_simulate(i_shot, omp_threads_override=6)

# Cheating of course, as this is synthetically generated data.
ux_obs, uz_obs = model.get_synthetic_data()

# numpy.random.seed(0)
# std = 10.0
# ux_obs += std * numpy.random.randn(*ux_obs.shape)
# uz_obs += std * numpy.random.randn(*uz_obs.shape)

numpy.save("ux_obs", ux_obs)
numpy.save("uz_obs", uz_obs)
model.set_observed_data(ux_obs, uz_obs)


# Reverting the model to the starting model -----------------------------------

vp = vp_starting
vs = vs_starting
rho = rho_starting

model.set_parameter_fields(vp_starting, vs_starting, rho_starting)

for i_shot in range(model.n_shots):
    model.forward_simulate(i_shot, omp_threads_override=6)

ux, uz = model.get_synthetic_data()
ux_obs, uz_obs = model.get_observed_data()

max_waveform = max(ux.max(), uz.max(), ux_obs.max(), uz.max()) / 2

m_ux_obs = ux_obs.copy()
m_ux = ux.copy()

for i in range(ux_obs.shape[1]):
    m_ux_obs[0, i:, :] += max_waveform
    m_ux[0, i:, :] += max_waveform

plt.plot(m_ux[0, :, :].T, "r", label="synthetic", alpha=0.5)
plt.plot(m_ux_obs[0, :, :].T, "k", label="observed", alpha=0.5)
plt.show()

# Perform adjoint simulation --------------------------------------------------

model.calculate_l2_misfit()

print(f"Data misfit: {model.misfit:.2f}")

model.calculate_l2_adjoint_sources()
model.reset_kernels()
for i_shot in range(model.n_shots):
    model.adjoint_simulate(i_shot, omp_threads_override=6)
model.map_kernels_to_velocity()

g_vp, g_vs, g_rho = model.get_kernels()

extrema = numpy.abs(g_vp).max(), numpy.abs(g_vs).max(), numpy.abs(g_rho).max()

extent = (extent[0], extent[1], extent[3], extent[2])

gradients = [g_vp, g_vs, g_rho]

plt.figure(figsize=(10, 4))
for i in range(3):
    plt.subplot(1, 3, int(i + 1))
    plt.xlabel("x [m]")
    plt.ylabel("z [m]")
    plt.imshow(
        gradients[i].T,
        vmin=-extrema[i],
        vmax=extrema[i],
        cmap=plt.get_cmap("seismic"),
        extent=extent,
    )
    plt.gca().invert_yaxis()
    plt.colorbar()

plt.tight_layout()
plt.show()

# Start iterating -------------------------------------------------------------

m = model.get_model_vector()

print("Starting gradient descent")

fields_during_iteration = []

iterations = 15

try:
    for i in range(iterations):

        g = model.get_gradient_vector()

        # Amplify Vp gradient
        g[0:10800] *= 1000

        m -= 0.2 * g
        model.set_model_vector(m)

        fields_during_iteration.append(list(model.get_parameter_fields()))

        # Simulate forward
        for i_shot in range(model.n_shots):
            model.forward_simulate(i_shot, omp_threads_override=6)

        # Calculate misfit and adjoint sources
        model.calculate_l2_misfit()
        model.calculate_l2_adjoint_sources()
        print(f"Data misfit: {model.misfit:.2f}")

        # Simulate adjoint
        model.reset_kernels()
        for i_shot in range(model.n_shots):
            model.adjoint_simulate(i_shot, omp_threads_override=6)
        model.map_kernels_to_velocity()
except KeyboardInterrupt:
    m = model.get_model_vector()
    iterations = i

vp, vs, rho = model.get_parameter_fields()
fields = [vp, vs, rho]
maxf = [2400, 1000, 1800]
minf = [1600, 600, 1200]

fig = plt.figure(figsize=(10, 4))


def animate(j):
    images = []
    for i in range(3):
        plt.subplot(1, 3, int(i + 1))
        plt.cla()
        plt.xlabel("x [m]")
        plt.ylabel("z [m]")
        images.append(
            plt.imshow(
                fields_during_iteration[j][i].T,
                cmap=plt.get_cmap("seismic"),
                extent=extent,
                vmin=minf[i],
                vmax=maxf[i],
            )
        )
        plt.gca().invert_yaxis()
    plt.tight_layout()
    return tuple(images)


anim = animation.FuncAnimation(fig, animate, frames=iterations, interval=10)
plt.show()

# Bonus: Animating a wavefield ------------------------------------------------

fig = plt.figure(figsize=(4, 10))
ax = plt.subplot(211)
ax2 = plt.subplot(212)
plt.xlabel("x [m]")
plt.ylabel("z [m]")

vx, _, _, _, _ = model.get_snapshots()
vx = vx[0, :, :, :]

# Get the receivers
rx, rz = model.get_receivers()

dt = model.dt
nt = vx.shape[0]
snapshot_interval = model.snapshot_interval
abswave = numpy.max(numpy.abs(vx)) / 25

extent = (extent[0], extent[1], extent[3], extent[2])

t = numpy.linspace(0, dt * nt * snapshot_interval, nt * snapshot_interval)


def animate(i):
    z1 = vx[int(i), :, :].T
    ax.cla()
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.scatter(rx, rz, color="k", marker="v")

    ax.text(-5, -5, f"Time: {i * dt * snapshot_interval:.3f}")
    im1 = ax.imshow(
        z1, vmin=-abswave, vmax=abswave, cmap=plt.get_cmap("PRGn"), extent=extent,
    )
    ax.invert_yaxis()

    ax2.cla()
    ax2.set_ylim([0, t[-1]])

    ax2.set_xlim(ax.get_xlim())

    for ir in range(19):
        ln1 = ax2.plot(
            ux[0, ir, : i * snapshot_interval] / 100 + rx[ir],
            t[: i * snapshot_interval],
            "k",
            alpha=0.5,
        )
        ln1 = ax2.plot(
            uz[0, ir, : i * snapshot_interval] / 100 + rx[ir],
            t[: i * snapshot_interval],
            "k",
            alpha=0.5,
        )
    ax2.invert_yaxis()
    ax2.set_xlabel("x [m]")
    ax2.set_ylabel("t [s]")
    plt.tight_layout()
    return im1, ln1


anim = animation.FuncAnimation(fig, animate, frames=nt, interval=1)
anim.save("video.mp4")
