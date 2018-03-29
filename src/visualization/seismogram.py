import numpy as np
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt

# fig = plt.figure(figsize=(6, 3), dpi=400)

seismo1 = np.transpose(np.loadtxt("output-observed/seismogram0_ux.txt"))
seismo2 = np.transpose(np.loadtxt("output-observed/seismogram0_uz.txt"))
max1 = np.max(np.abs(seismo1))/1000
max2 = np.max(np.abs(seismo1))/1000


plt.subplot(1,2,1)
im = plt.imshow(seismo1, cmap=plt.get_cmap('seismic'), aspect=1/70, vmax=max1, vmin=-max1)
plt.subplot(1,2,2)
im = plt.imshow(seismo2, cmap=plt.get_cmap('seismic'), aspect=1/70, vmax=max2, vmin=-max2)
plt.show()
