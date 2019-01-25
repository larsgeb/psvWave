import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

vp = np.ones((100, 100)) * 2000
vs = np.ones((100, 100)) * 800
de = np.ones((100, 100)) * 1500

vp_perturb = np.ones((60, 60)) * 1.1
de_perturb = np.ones((60, 60)) * 1.1

vp_perturb[:20:4] = 1.05
vp_perturb[1:20:4] = 1.05

vp_perturb[20::8] = 1.05
vp_perturb[21::8] = 1.05
vp_perturb[22::8] = 1.05
vp_perturb[23::8] = 1.05

vp_perturb_temp = vp_perturb

vp_perturb[14:24, 8:30] = 0.9

for i in range(60):
    vp_perturb[44 - int(i / 5):60, i] = 0.8
    vp_perturb[48 - int(i / 5):60, i] = 0.9
    vp_perturb[52 - int(i / 5):60, i] = 0.8
    vp_perturb[56 - int(i / 5):60, i] = 0.9
    vp_perturb[60 - int(i / 5):60, i] = 0.8
    vp_perturb[64 - int(i / 5):60, i] = 0.9


# for ix in range(60):
#     for iz in range(60):
#         vp_perturb[ix,iz] += 0.001 * ix + 0.001 * iz - 0.01

for i in range(3):
    vp_perturb = (vp_perturb - 1) * 0.5 + 1
    vp_perturb[i:-i, i:-i] = vp_perturb_temp[i:-i, i:-i]


vs_perturb = vp_perturb
de_perturb= vp_perturb


vp[20:80, 20:80] *= vp_perturb
vs[20:80, 20:80] *=  vs_perturb
de[20:80, 20:80] *= de_perturb

fig = plt.figure(figsize=(15, 3))
(ax1, cax1, ax2, cax2, ax3, cax3) = fig.subplots(ncols=6, gridspec_kw={"width_ratios": [13, 1, 13, 1, 13, 1]})

perturbations = 40
every_x_percentage = 20

im1 = ax1.imshow(vp, cmap=plt.get_cmap('seismic'), vmin=vp[0, 0] * (1 - perturbations / 100), vmax=vp[0, 0] * (1 + perturbations / 100))
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

im2 = ax2.imshow(vs, cmap=plt.get_cmap('seismic'), vmin=vs[0, 0] * (1 - perturbations / 100), vmax=vs[0, 0] * (1 + perturbations / 100))
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

im3 = ax3.imshow(de, cmap=plt.get_cmap('seismic'), vmin=de[0, 0] * (1 - perturbations / 100), vmax=de[0, 0] * (1 + perturbations / 100))
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
