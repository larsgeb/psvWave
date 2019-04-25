import numpy as np
import matplotlib.pyplot as plt

kernel_vp = np.loadtxt("kernel_vp.txt").T
kernel_vs = np.loadtxt("kernel_vs.txt").T
kernel_density = np.loadtxt("kernel_density.txt").T

plt.subplot(131)
plt.imshow(kernel_vp / np.max(np.abs(kernel_vp)), vmax=1, vmin=-1, cmap=plt.get_cmap("seismic"))
plt.gca().invert_yaxis()
plt.subplot(132)
plt.imshow(kernel_vs / np.max(np.abs(kernel_vs)), vmax=1, vmin=-1, cmap=plt.get_cmap("seismic"))
plt.gca().invert_yaxis()
plt.subplot(133)
plt.imshow(kernel_density / np.max(np.abs(kernel_density)), vmax=1, vmin=-1, cmap=plt.get_cmap("seismic"))
plt.gca().invert_yaxis()
plt.show()
