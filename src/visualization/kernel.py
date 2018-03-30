import numpy as np
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt

# fig = plt.figure(figsize=(6, 3), dpi=400)

densityKernel = np.transpose(np.loadtxt("densityKernel.txt"))
lambdaKernel = np.transpose(np.loadtxt("lambdaKernel.txt"))
muKernel = np.transpose(np.loadtxt("muKernel.txt"))

max1 = np.max(np.abs(densityKernel))
max2 = np.max(np.abs(lambdaKernel))
max3 = np.max(np.abs(muKernel))
link = False
maxC = max(max2,max3)

customCMAP = plt.get_cmap('seismic')
customCMAP._segmentdata['alpha'] =  [(0.0, 0.8, 0.8), (0.5, 0.5, 0.5), (1.0, 0.8, 0.8)]

mu = np.loadtxt("mu.txt")[50:-50,:-50]

plt.subplot(3,1,1)
# imMedium = plt.imshow(np.transpose(mu), animated=True, aspect=1, cmap=plt.get_cmap("copper"))
im = plt.imshow(densityKernel, cmap=customCMAP, vmax=max1, vmin=-max1)
plt.colorbar()
plt.subplot(3,1,2)
# imMedium = plt.imshow(np.transpose(mu), animated=True, aspect=1, cmap=plt.get_cmap("copper"))
im = plt.imshow(lambdaKernel, cmap=customCMAP, vmax=(maxC if link else max2), vmin=-(maxC if link else max2))
plt.colorbar()
plt.subplot(3,1,3)
# imMedium = plt.imshow(np.transpose(mu), animated=True, aspect=1, cmap=plt.get_cmap("copper"))
im = plt.imshow(muKernel, cmap=customCMAP, vmax=(maxC if link else max3), vmin=-(maxC if link else max3))
plt.colorbar()

plt.show()
