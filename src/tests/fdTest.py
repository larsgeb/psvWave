import numpy as np
import matplotlib.pyplot as plt

factors = np.loadtxt("kernelTest/factors.txt")
epsilons = np.loadtxt("kernelTest/epsilons.txt")

plt.semilogx(epsilons, factors)
plt.xlabel("Epsilon")
plt.ylabel("dx/dm * dm * epsilon  vs  x1 - x0")
plt.show()

customCMAP = plt.get_cmap('seismic')
customCMAP._segmentdata['alpha'] = [(0.0, 0.8, 0.8), (0.5, 0.5, 0.5), (1.0, 0.8, 0.8)]

densityKernel = np.transpose(np.loadtxt("kernelTest/densityKernel.txt"))
lambdaKernel = np.transpose(np.loadtxt("kernelTest/lambdaKernel.txt"))
muKernel = np.transpose(np.loadtxt("kernelTest/muKernel.txt"))

max1 = np.max(np.abs(densityKernel))
max2 = np.max(np.abs(lambdaKernel))
max3 = np.max(np.abs(muKernel))

plt.subplot(3, 1, 1)
plt.imshow(densityKernel, cmap=customCMAP, vmin=-max1, vmax=max1)
plt.scatter([350,350,350],[75,100,125])
plt.subplot(3, 1, 2)
plt.imshow(lambdaKernel, cmap=customCMAP, vmin=-max2, vmax=max2)
plt.scatter([350,350,350],[75,100,125])
plt.subplot(3, 1, 3)
plt.imshow(muKernel, cmap=customCMAP, vmin=-max3, vmax=max3)
plt.scatter([350,350,350],[75,100,125])

plt.show()
