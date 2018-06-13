import numpy as np
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt

# fig = plt.figure(figsize=(6, 3), dpi=400)
from scipy import signal

maxi = 1900
dt = 1

# vx = np.ndarray((maxi, 300, 550))
# ux = np.ndarray((maxi, 300, 550))
# vz = np.ndarray((maxi, 300, 550))
# uz = np.ndarray((maxi, 300, 550))
# for i in range(0, maxi):
#     vx[i, :, :] = np.transpose(np.loadtxt("snapshots/snapshot%i_vx.txt" % i))
#     vz[i, :, :] = np.transpose(np.loadtxt("snapshots/snapshot%i_vz.txt" % i))
#
#     ux[i, :, :] = ux[i - 1, :, :] + dt * vx[i, :, :]
#     uz[i, :, :] = uz[i - 1, :, :] + dt * vz[i, :, :]

vx = np.transpose(np.loadtxt("snapshots/snapshot%i_vx.txt" % maxi))
vz = np.transpose(np.loadtxt("snapshots/snapshot%i_vz.txt" % maxi))

maxAmp = 1e-10

dz = np.array([[0, -1, 0],
               [0, 0, 0],
               [0, 1, 0]])
dx = np.array([[0, 0, 0],
               [-1, 0, 1],
               [0, 0, 0]])

field1 = vx  # ux[-1, :, :]
field2 = vz  # uz[-1, :, :]

uxdz = signal.convolve2d(field1, dz, boundary='symm', mode='same')
uzdx = signal.convolve2d(field2, dx, boundary='symm', mode='same')

uxdx = signal.convolve2d(field1, dz, boundary='symm', mode='same')
uzdz = signal.convolve2d(field2, dx, boundary='symm', mode='same')

# plt.subplot(3, 1, 1)
# im = plt.imshow(field1[:-50, 50:-50], cmap=plt.get_cmap('seismic'), vmin=-maxAmp, vmax=maxAmp)
# # im = plt.imshow(uxdz[:-50, 50:-50] - uzdx[:-50, 50:-50], cmap=plt.get_cmap('seismic'), vmin=-maxAmp, vmax=maxAmp)
# plt.subplot(3, 1, 2)
# plt.imshow(field2[:-50, 50:-50], cmap=plt.get_cmap('seismic'), vmin=-maxAmp, vmax=maxAmp)
# # plt.imshow(uxdx[:-50, 50:-50] + uzdz[:-50, 50:-50], cmap=plt.get_cmap('seismic'), vmin=-maxAmp, vmax=maxAmp)
# plt.subplot(3, 1, 3)
field3 = (np.sqrt(np.square(field1[:-50, 50:-50]) + np.square(field2[:-50, 50:-50])))
plt.imshow(np.log10(field3), cmap=plt.get_cmap('Reds'), vmin=-12, vmax=-5)
plt.show()

