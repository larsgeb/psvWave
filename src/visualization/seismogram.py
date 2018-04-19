import numpy as np
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt

# fig = plt.figure(figsize=(6, 3), dpi=400)

seismoX = np.transpose(np.loadtxt("kernelTest/seismogram0_ux.txt"))
seismoX_obs = np.transpose(np.loadtxt("kernelTest/seismogram0_ux_obs.txt"))
seismoZ = np.transpose(np.loadtxt("kernelTest/seismogram0_uz.txt"))
seismoZ_obs = np.transpose(np.loadtxt("kernelTest/seismogram0_uz_obs.txt"))

max1 = np.max(np.abs(seismoX)) / 30
max2 = np.max(np.abs(seismoZ)) / 30


plt.subplot(1,2,1)
im = plt.plot(seismoX)
im = plt.plot(seismoZ_obs)
plt.subplot(1,2,2)
im = plt.plot(seismoZ)
im = plt.plot(seismoZ_obs)


plt.show()
