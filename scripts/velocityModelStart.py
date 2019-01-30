import matplotlib.pyplot as plt
import numpy as np
from obspy.imaging.beachball import beach
import optparse
import scipy as sp
import scipy.ndimage

parser = optparse.OptionParser()
parser.add_option('-p', '--plot',
                  action="store", dest="plot",
                  help="query string", default="true")
options, args = parser.parse_args()
print('Plotting?:', options.plot)

np_boundary = 10
plot_boundary = False
nx_inner = 200
nz_inner = 100

vp_back = 2000
vs_back = 800
de_back = 1500

de = np.loadtxt("de_target.txt")
vp = np.loadtxt("vp_target.txt")
vs = np.loadtxt("vs_target.txt")

# Flat model
# for ix in range(nx_inner + 2 * np_boundary):
#     de[ix, :] = de[3, :]
#     vp[ix, :] = vp[3, :]
#     vs[ix, :] = vs[3, :]

de_temp = np.copy(de)
vp_temp = np.copy(vp)
vs_temp = np.copy(vs)
# Smooth model
for i in range(50):
    for ix in range(np_boundary + 10, np_boundary + nx_inner - 10):
        for iz in range(np_boundary + 20, np_boundary + nz_inner - 20):
            de_temp[ix, iz] = (de[ix - 1, iz] + 0.5 * de[ix - 2, iz] +
                               de[ix + 1, iz] + 0.5 * de[ix + 2, iz] +
                               de[ix, iz - 1] + 0.5 * de[ix, iz - 2] +
                               de[ix, iz + 1] + 0.5 * de[ix, iz + 2] +
                               7 * de[ix, iz]) / 13
            vp_temp[ix, iz] = (vp[ix - 1, iz] + 0.5 * vp[ix - 2, iz] +
                               vp[ix + 1, iz] + 0.5 * vp[ix + 2, iz] +
                               vp[ix, iz - 1] + 0.5 * vp[ix, iz - 2] +
                               vp[ix, iz + 1] + 0.5 * vp[ix, iz + 2] +
                               7 * vp[ix, iz]) / 13
            vs_temp[ix, iz] = (vs[ix - 1, iz] + 0.5 * vs[ix - 2, iz] +
                               vs[ix + 1, iz] + 0.5 * vs[ix + 2, iz] +
                               vs[ix, iz - 1] + 0.5 * vs[ix, iz - 2] +
                               vs[ix, iz + 1] + 0.5 * vs[ix, iz + 2] +
                               7 * vs[ix, iz]) / 13
    de = np.copy(de_temp)
    vs = np.copy(vs_temp)
    vp = np.copy(vp_temp)

receivers_x = np.array([100, 10, 30, 50, 70, 90, 110, 130, 150, 170, 190, 20, 40, 60, 80, 120, 140, 160, 180, ]) + np_boundary
receivers_z = np.array([90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, ]) + np_boundary
print("number of receivers:", receivers_x.size)

sources_x = np.array([25, 50, 75, 100, 125, 150, 175]) + np_boundary
sources_z = np.array([10, 10, 10, 10, 10, 10, 10]) + np_boundary
n_sources = sources_x.size

np.savetxt("de_starting.txt", de)
np.savetxt("vp_starting.txt", vp)
np.savetxt("vs_starting.txt", vs)

print("number of sources:", n_sources)
print("wrote out starting velocity model")
if options.plot == 'false':
    exit(0)


fig = plt.figure(figsize=(15, 3))
# (ax1, cax1, ax2, cax2, ax3, cax3) = fig.subplots(ncols=6, gridspec_kw={"width_ratios": [13, 1, 13, 1, 13, 1]})
axes_a = fig.subplots(ncols=6, gridspec_kw={"width_ratios": [13, 1, 13, 1, 13, 1]})

perturbations = 5
every_x_percentage = 2
axes = axes_a[0::2]
caxes = axes_a[1::2]
titles = ['vp', 'vs', 'de']
fields = [vp, vs, de]
background_fields = [vp_back, vs_back, de_back]
images = []
cbars = []
caxbs = []
angles = [90, 81, 41, 300, 147, 252, 327]


for j in range(3):
    for i in range(n_sources):
        beach_ball = beach([-45 + angles[i]/2, 90, 0], xy=(sources_x[i], sources_z[i]), width=10, linewidth=1, alpha=1)
        axes[j].add_collection(beach_ball)
    axes[j].plot(np_boundary + np.array([10, nx_inner - 10, nx_inner - 10, 10, 10]),
                 np_boundary + np.array([20, 20, nz_inner - 20, nz_inner - 20, 20]), '--k')
    axes[j].set_title(titles[j])
    axes[j].scatter(receivers_x, receivers_z, color='k', marker='v')
    images.append(axes[j].imshow(fields[j].T, cmap=plt.get_cmap('seismic'), vmin=background_fields[j] * (1 - perturbations / 100),
                                 vmax=background_fields[j] * (1 + perturbations / 100)))
    cbars.append(plt.colorbar(images[j], cax=caxes[j]))

    axes[j].invert_yaxis()
    caxbs.append(caxes[j].twinx())
    if j == 2:
        caxes[j].set_ylabel("[kg/m³]")
    else:
        caxes[j].set_ylabel("[m/s]")
    cbars[j].ax.yaxis.set_label_position("left")
    ticks_labels = np.arange(-perturbations, perturbations + 1, every_x_percentage)
    ticks = (ticks_labels / 100 + 1) * fields[j][0, 0]
    cbars[j].set_ticks(ticks)
    caxbs[j].set_ylim(fields[j][0, 0] * (1 - perturbations / 100), fields[j][0, 0] * (1 + perturbations / 100))
    caxbs[j].set_yticks(ticks)
    caxbs[j].set_yticklabels(ticks_labels)
    caxbs[j].set_ylabel("+/- [%]")

    cbars[j].set_ticks(np.linspace(images[j].get_clim()[0], images[j].get_clim()[1], 6))
    caxes[j].set_yticklabels(background_fields[j] * np.array([0.95, 0.97, 0.99, 1.01, 1.03, 1.05]))

    if (plot_boundary):
        axes[j].fill(
            [0, np_boundary * 2 + nx_inner, np_boundary * 2 + nx_inner, np_boundary + nx_inner, np_boundary + nx_inner, np_boundary, np_boundary, 0],
            [0, 0, np_boundary + nz_inner, np_boundary + nz_inner, np_boundary, np_boundary, np_boundary + nz_inner, np_boundary + nz_inner],
            alpha=0.1, color='k')

    if (plot_boundary):
        axes[j].set_xlim([0, np_boundary * 2 + nx_inner])
        axes[j].set_ylim([0, np_boundary * 1 + nz_inner])
    else:
        axes[j].set_xlim([np_boundary, np_boundary * 1 + nx_inner])
        axes[j].set_ylim([np_boundary, np_boundary * 1 + nz_inner])

plt.tight_layout()
plt.show()
