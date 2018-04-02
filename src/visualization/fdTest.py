import numpy as np
import matplotlib.pyplot as plt

factors = np.loadtxt("factors.txt")
epsilons = np.loadtxt("epsilons.txt")



plt.semilogx(epsilons,factors)
plt.xlabel("Epsilon")
plt.ylabel("dx/dm * epsilon  vs  x1 - x0")
plt.show()

customCMAP = plt.get_cmap('seismic')
customCMAP._segmentdata['alpha'] =  [(0.0, 0.8, 0.8), (0.5, 0.5, 0.5), (1.0, 0.8, 0.8)]

gradient = np.loadtxt("gradient.txt")
densityKernel = np.transpose(np.loadtxt("densityKernel.txt"))
lambdaKernel = np.transpose(np.loadtxt("lambdaKernel.txt"))
muKernel = np.transpose(np.loadtxt("muKernel.txt"))

densityKernelD = np.transpose(np.reshape(gradient[0:8], (4, 2)))
lambdaKernelD = np.transpose(np.reshape(gradient[8:16], (4, 2)))
muKernelD = np.transpose(np.reshape(gradient[16:], (4, 2)))

max1 = np.max(np.abs(densityKernel))/10
max1d = np.max(np.abs(densityKernelD))/10
max2 = np.max(np.abs(lambdaKernel))/10
max2d = np.max(np.abs(lambdaKernelD))/10
max3 = np.max(np.abs(muKernel))/10
max3d = np.max(np.abs(muKernelD))/10

plt.subplot(3,2,1)
plt.imshow(densityKernel, cmap=customCMAP,vmin = -max1, vmax = max1)
plt.subplot(3,2,2)
plt.imshow(densityKernelD,cmap=customCMAP,vmin = -max1d, vmax = max1d)
plt.subplot(3,2,3)
plt.imshow(lambdaKernel, cmap=customCMAP,vmin = -max2, vmax = max2)
plt.subplot(3,2,4)
plt.imshow(lambdaKernelD,cmap=customCMAP,vmin = -max2d, vmax = max2d)
plt.subplot(3,2,5)
plt.imshow(muKernel, cmap=customCMAP,vmin = -max3, vmax = max3)
plt.subplot(3,2,6)
plt.imshow(muKernelD,cmap=customCMAP,vmin = -max3d, vmax = max3d)

plt.show()

