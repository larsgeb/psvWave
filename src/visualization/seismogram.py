import numpy as np
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 5), dpi=400)

seismo = np.transpose(np.loadtxt("vxTop.txt"))
max = np.max(np.abs(seismo[:,:]))

im = plt.imshow(seismo[:,:], cmap=plt.get_cmap('seismic'), aspect=1 / 10, vmax=max, vmin=-max)
plt.show()
