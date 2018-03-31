import numpy as np
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt

# fig = plt.figure(figsize=(6, 3), dpi=400)

seismoX = np.transpose(np.loadtxt("experimentResult/seismogram0_ux.txt"))
seismoZ = np.transpose(np.loadtxt("experimentResult/seismogram0_uz.txt"))

max1 = np.max(np.abs(seismoX)) / 300
max2 = np.max(np.abs(seismoZ)) / 300


plt.subplot(2,1,1)
im = plt.imshow(seismoX, cmap=plt.get_cmap('seismic'), aspect=1 / 70, vmax=max1, vmin=-max1)
plt.subplot(2,1,2)
im = plt.imshow(seismoZ, cmap=plt.get_cmap('seismic'), aspect=1 / 70, vmax=max2, vmin=-max2)


plt.show()
