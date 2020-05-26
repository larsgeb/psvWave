import time
# from matplotlib import animation
import pyWave
# import matplotlib.pyplot as plt
# import numpy

model = pyWave.fdModel(
    "../tests/test_configurations/forward_configuration.ini")

samples = 1
t0 = time.time()
for i in range(samples):
    model.forward_shot(0, verbose=True, store_fields=True,
                       omp_threads_override=4)
t1 = time.time()
total = t1-t0
print(f"Average execution time: {total / samples:.2f}")

# IX, IZ = model.get_coordinates(True)
# extent = model.get_extent(True)

# vp, vs, rho = model.get_parameter_fields()


# def distance_from(IX, IZ, x, z):
#     return ((IX-x)**2 + (IZ-z)**2)**.5


# x_middle = (IX.max() + IX.min())/2
# z_middle = (IZ.max() + IZ.min())/2

# field = distance_from(IX, IZ, x_middle, z_middle) < 50

# vs = vs * (1-0.1 * field)
# vp = vp * (1-0.1 * field)

# model.set_parameter_fields(vp, vs, rho)

# model.forward_shot(0, verbose=True, store_fields=True,
#                    omp_threads_override=4)


# fig = plt.figure()
# ax = plt.axes()
# plt.xlabel("x [m]")
# plt.ylabel("z [m]")

# vx, _, _, _, _ = model.get_snapshots()

# vx = vx[0, :, :, :]

# dt = model.dt
# snapshot_interval = model.snapshot_interval
# abswave = numpy.max(numpy.abs(vx)) / 25


# def animate(i):
#     z1 = vx[int(i), :, :].T
#     plt.cla()
#     plt.xlabel("x [m]")
#     plt.ylabel("z [m]")

#     plt.text(-5, -5, f"Time: {i * dt * snapshot_interval:.3f}")
#     im1 = plt.imshow(z1, vmin=-abswave, vmax=abswave,
#                      cmap=plt.get_cmap("PRGn"), extent=extent)

#     return im1


# anim = animation.FuncAnimation(
#     fig, animate, frames=vx.shape[0], interval=1)
# plt.show()
