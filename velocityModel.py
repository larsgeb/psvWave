import matplotlib.pyplot as plt
import numpy as np
from obspy.imaging.beachball import beach

vp = np.ones((200, 150)) * 2000
vs = np.ones((200, 150)) * 800
de = np.ones((200, 150)) * 1500

# Global anomaly
vp_perturb = np.ones((60, 60)) * 1.01
de_perturb = np.ones((60, 60)) * 1.01
vs_perturb = np.ones((60, 60)) * 1.01

# Layering
vp_perturb[:, ::8] = 1.02
vp_perturb[:, 1::8] = 1.02
vp_perturb[:, 2::8] = 1.02
vp_perturb[:, 3::8] = 1.02
de_perturb[:, ::8] = 1.02
de_perturb[:, 1::8] = 1.02
de_perturb[:, 2::8] = 1.02
de_perturb[:, 3::8] = 1.02
vs_perturb[:, ::8] = 0.99
vs_perturb[:, 1::8] = 0.99
vs_perturb[:, 2::8] = 0.99
vs_perturb[:, 3::8] = 0.99

# Blob
vp_perturb[8:30, 46:36:-1] = 1.03
vs_perturb[8:30, 46:36:-1] = 1.05
de_perturb[8:30, 46:36:-1] = 0.98

# Slanted layers
for i in range(60):
    vp_perturb[i, :18 + int(i / 5)] = 0.96
    vp_perturb[i, :14 + int(i / 5)] = 0.98
    vp_perturb[i, :10 + int(i / 5)] = 0.96
    vp_perturb[i, :6 + int(i / 5)] = 0.98
    vp_perturb[i, :2 + int(i / 5)] = 0.96
    vs_perturb[i, :18 + int(i / 5)] = 0.98
    vs_perturb[i, :14 + int(i / 5)] = 1.03
    vs_perturb[i, :10 + int(i / 5)] = 0.98
    vs_perturb[i, :6 + int(i / 5)] = 1.03
    vs_perturb[i, :2 + int(i / 5)] = 0.98
    de_perturb[i, :18 + int(i / 5)] = 0.96
    de_perturb[i, :14 + int(i / 5)] = 0.98
    de_perturb[i, :10 + int(i / 5)] = 0.95
    de_perturb[i, :6 + int(i / 5)] = 0.98
    de_perturb[i, :2 + int(i / 5)] = 0.96

# Save for reconstruction
vp_perturb_temp = vp_perturb
vs_perturb_temp = vs_perturb
de_perturb_temp = de_perturb

for i in range(3):
    vp_perturb = (vp_perturb - 1) * 0.5 + 1
    vp_perturb[i:-i, i:-i] = vp_perturb_temp[i:-i, i:-i]
    vs_perturb = (vs_perturb - 1) * 0.5 + 1
    vs_perturb[i:-i, i:-i] = vs_perturb_temp[i:-i, i:-i]
    de_perturb = (de_perturb - 1) * 0.5 + 1
    de_perturb[i:-i, i:-i] = de_perturb_temp[i:-i, i:-i]

vp[70:130, 70:130] *= vp_perturb
vs[70:130, 70:130] *= vs_perturb
de[70:130, 70:130] *= de_perturb

fig = plt.figure(figsize=(15, 3))
(ax1, cax1, ax2, cax2, ax3, cax3) = fig.subplots(ncols=6, gridspec_kw={"width_ratios": [13, 1, 13, 1, 13, 1]})

perturbations = 5
every_x_percentage = 2

np.savetxt("de_target.txt", de)
np.savetxt("vp_target.txt", vp)
np.savetxt("vs_target.txt", vs)

receivers_x = np.array([10, 50, 50, 90, 90, 90, 10, 10, 30, 30, 70, 70]) + 50
receivers_z = np.array([50, 10, 90, 10, 50, 90, 10, 90, 10, 90, 10, 90]) + 50

sources_x = np.array([10, 10, 90, 90]) + 50
sources_z = np.array([30, 70, 70, 30]) + 50
n_sources = sources_x.size

focalMechanismsAx1 = [[]]
axes = [ax1, ax2, ax3]

for i in range(n_sources):
    for j in range(3):
        beach_ball = beach([-45, 90, 0], xy=(sources_x[i], sources_z[i]), width=10, linewidth=1, alpha=1)
        axes[j].add_collection(beach_ball)

im1 = ax1.imshow(vp.T, cmap=plt.get_cmap('seismic'), vmin=vp[0, 0] * (1 - perturbations / 100), vmax=vp[0, 0] * (1 + perturbations / 100))
ax1.invert_yaxis()
ax1.plot(50 + np.array([20, 80, 80, 20, 20]), 50 + np.array([20, 20, 80, 80, 20]), '--k')
ax1.fill([0, 200, 200, 150, 150, 50, 50, 0], [0, 0, 150, 150, 50, 50, 150, 150], alpha=0.2)
ax1.scatter(receivers_x, receivers_z, color='k', marker='v')

ax1.set_xlim([0, 200])
ax1.set_ylim([0, 150])
ax1.set_title('vp')
cbar1 = plt.colorbar(im1, cax=cax1)
cax1b = cax1.twinx()
cax1.set_ylabel("[m/s]")
cbar1.ax.yaxis.set_label_position("left")
ticks_labels = np.arange(-perturbations, perturbations + 1, every_x_percentage)
ticks = (ticks_labels / 100 + 1) * vp[0, 0]
cbar1.set_ticks(ticks)
cax1b.set_ylim(vp[0, 0] * (1 - perturbations / 100), vp[0, 0] * (1 + perturbations / 100))
cax1b.set_yticks(ticks)
cax1b.set_yticklabels(ticks_labels)
cax1b.set_ylabel("+/- [%]")

im2 = ax2.imshow(vs.T, cmap=plt.get_cmap('seismic'), vmin=vs[0, 0] * (1 - perturbations / 100), vmax=vs[0, 0] * (1 + perturbations / 100))
ax2.invert_yaxis()
ax2.plot(50 + np.array([20, 80, 80, 20, 20]), 50 + np.array([20, 20, 80, 80, 20]), '--k')
ax2.fill([0, 200, 200, 150, 150, 50, 50, 0], [0, 0, 150, 150, 50, 50, 150, 150], alpha=0.2)
ax2.scatter(receivers_x, receivers_z, color='k', marker='v')

ax2.set_xlim([0, 200])
ax2.set_ylim([0, 150])
ax2.set_title('vs')
cbar2 = plt.colorbar(im2, cax=cax2)
cax2b = cax2.twinx()
cax2.set_ylabel("[m/s]")
cbar2.ax.yaxis.set_label_position("left")
ticks_labels = np.arange(-perturbations, perturbations + 1, every_x_percentage)
ticks = (ticks_labels / 100 + 1) * vs[0, 0]
cax2b.set_ylim(vs[0, 0] * (1 - perturbations / 100), vs[0, 0] * (1 + perturbations / 100))
cbar2.set_ticks(ticks)
cax2b.set_yticks(ticks)
cax2b.set_yticklabels(ticks_labels)
cax2b.set_ylabel("+/- [%]")

im3 = ax3.imshow(de.T, cmap=plt.get_cmap('seismic'), vmin=de[0, 0] * (1 - perturbations / 100), vmax=de[0, 0] * (1 + perturbations / 100))
ax3.invert_yaxis()
ax3.plot(50 + np.array([20, 80, 80, 20, 20]), 50 + np.array([20, 20, 80, 80, 20]), '--k')
ax3.fill([0, 200, 200, 150, 150, 50, 50, 0], [0, 0, 150, 150, 50, 50, 150, 150], alpha=0.2)
ax3.scatter(receivers_x, receivers_z, color='k', marker='v')

ax3.set_xlim([0, 200])
ax3.set_ylim([0, 150])
ax3.set_title('de')
cbar3 = plt.colorbar(im3, cax=cax3)
cax3b = cax3.twinx()
cax3.set_ylabel("[kg/m³]")
cbar3.ax.yaxis.set_label_position("left")
ticks_labels = np.arange(-perturbations, perturbations + 1, every_x_percentage)
ticks = (ticks_labels / 100 + 1) * de[0, 0]
cax3b.set_ylim(de[0, 0] * (1 - perturbations / 100), de[0, 0] * (1 + perturbations / 100))
cbar3.set_ticks(ticks)
cax3b.set_yticks(ticks)
cax3b.set_yticklabels(ticks_labels)
cax3b.set_ylabel("+/- [%]")

plt.tight_layout()
plt.show()
