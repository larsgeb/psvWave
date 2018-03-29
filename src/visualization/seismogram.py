import numpy as np
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt

# fig = plt.figure(figsize=(6, 3), dpi=400)

seismoXSyn = np.transpose(np.loadtxt("seismogramSyn_ux.txt"))
seismoXObs = np.transpose(np.loadtxt("seismogramObs_ux.txt"))
seismoZSyn = np.transpose(np.loadtxt("seismogramSyn_uz.txt"))
seismoZObs = np.transpose(np.loadtxt("seismogramObs_uz.txt"))

seismoXDiff = seismoXObs - seismoXSyn
seismoZDiff = seismoZObs - seismoZSyn

max1 = np.max(np.abs(seismoXObs)) / 300
max2 = np.max(np.abs(seismoZObs)) / 300


plt.subplot(2,2,1)
im = plt.imshow(seismoXObs, cmap=plt.get_cmap('seismic'), aspect=1 / 70, vmax=max1, vmin=-max1)
plt.subplot(2,2,2)
im = plt.imshow(seismoZObs, cmap=plt.get_cmap('seismic'), aspect=1 / 70, vmax=max2, vmin=-max2)
plt.subplot(2,2,3)
im = plt.imshow(seismoXSyn, cmap=plt.get_cmap('seismic'), aspect=1 / 70, vmax=max1, vmin=-max1)
plt.subplot(2,2,4)
im = plt.imshow(seismoZSyn, cmap=plt.get_cmap('seismic'), aspect=1 / 70, vmax=max2, vmin=-max2)

misfit = (np.sum(np.sum(np.square(seismoXDiff))) + np.sum(np.sum(np.square(seismoZDiff))))**0.5

print(misfit)

plt.show()
