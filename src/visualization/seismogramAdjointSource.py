import numpy as np
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt

# fig = plt.figure(figsize=(6, 3), dpi=400)

vxAdjoint= np.transpose(np.loadtxt("adjoint_vx.txt"))
vzAdjoint= np.transpose(np.loadtxt("adjoint_vz.txt"))

max1 = np.max(np.abs(vxAdjoint)) / 300
max2 = np.max(np.abs(vzAdjoint)) / 300


plt.subplot(1,2,1)
im = plt.imshow(vxAdjoint, cmap=plt.get_cmap('seismic'), aspect=1 / 70, vmax=max1, vmin=-max1)
plt.subplot(1,2,2)
im = plt.imshow(vzAdjoint, cmap=plt.get_cmap('seismic'), aspect=1 / 70, vmax=max2, vmin=-max2)

plt.show()
