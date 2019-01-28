import numpy as np
import matplotlib.pyplot as plt

vp = np.loadtxt("vp_file.txt")
vs = np.loadtxt("vs_file.txt")
de = np.loadtxt("de_v_file.txt")

e =1e-28

plt.imshow(vs.T, vmin=-e, vmax=e, cmap=plt.get_cmap("seismic"))
plt.gca().invert_yaxis()
plt.plot(50 + np.array([20, 80, 80, 20, 20]), 50 + np.array([20, 20, 80, 80, 20]))
plt.show()
