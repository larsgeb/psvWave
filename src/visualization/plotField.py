import numpy as np
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt

# fig = plt.figure(figsize=(6, 3), dpi=400)

field = np.transpose(np.loadtxt("snapshot500_vx.txt"))

im = plt.imshow(field, cmap=plt.get_cmap('seismic'))

plt.show()
