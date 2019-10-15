import matplotlib.pyplot as plt
import numpy as np
from obspy.imaging.beachball import beach

vp = np.loadtxt("vp_2file.txt")
vs = np.loadtxt("vs_2file.txt")
de = np.loadtxt("de_2file.txt")

kernels = [vp, vs, de]

fig = plt.figure(figsize=(8, 10))
axes = fig.subplots(nrows=3)

np_boundary = 10
nx_inner = 200
nz_inner = 100

receivers_x = np.array([100, 10, 30, 50, 70, 90, 110, 130, 150, 170, 190, 20, 40, 60, 80, 120, 140, 160, 180, ]) + np_boundary
receivers_z = np.array([90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, ]) + np_boundary

sources_x = np.array([25, 50, 75, 100, 125, 150, 175]) + np_boundary
sources_z = np.array([10, 10, 10, 10, 10, 10, 10]) + np_boundary
n_sources = sources_x.size
images = []

for i in range(3):
    kernel = kernels[i].T
    e = np.max(np.abs(kernel[np_boundary + 20:nx_inner - 20 + np_boundary, np_boundary + 20:nz_inner - 20 + np_boundary]))

    images.append(axes[i].imshow(kernel, vmin=-e, vmax=e, cmap=plt.get_cmap("seismic")))
    axes[i].invert_yaxis()
    axes[i].plot(np_boundary + np.array([20, nx_inner - 20, nx_inner - 20, 20, 20]),
                 np_boundary + np.array([20, 20, nz_inner - 20, nz_inner - 20, 20]))
    axes[i].scatter(receivers_x, receivers_z, color='k', marker='v')
    fig.colorbar(images[i], ax=axes[i])

angles = [90, 81, 41, 300, 147, 252, 327]
for i in range(n_sources):
    for j in range(3):
        beach_ball = beach([-45 + angles[i]/2, 90, 0], xy=(sources_x[i], sources_z[i]), width=10, linewidth=1, alpha=1)
        axes[j].add_collection(beach_ball)

axes[0].set_title('vp')
axes[1].set_title('vs')
axes[2].set_title('de')

plt.show()
