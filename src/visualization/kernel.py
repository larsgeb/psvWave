import numpy as np
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt

# fig = plt.figure(figsize=(6, 3), dpi=400)

# densityKernel = np.transpose(np.loadtxt("kernelTest/densityKernel_par2.txt"))
kernel = np.transpose(np.loadtxt("kernelTest/vpKernel_par2.txt"))

max = np.max(np.abs(kernel))/2

plt.imshow(kernel, cmap=plt.get_cmap("seismic"),vmin=-max,vmax=max)

plt.scatter([150],[100])
plt.show()