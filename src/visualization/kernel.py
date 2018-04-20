import numpy as np
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt

kernel = np.transpose(np.loadtxt("kernelTest/vsKernel_par2.txt"))

max = np.max(np.abs(kernel)) / 2

plt.imshow(kernel, cmap=plt.get_cmap("seismic"), vmin=-max, vmax=max)

plt.scatter([100, 200], [100, 100])
plt.show()
