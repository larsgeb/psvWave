import numpy as np
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt

# fig = plt.figure(figsize=(6, 3), dpi=400)

densityKernel = np.transpose(np.loadtxt("densityKernel.txt"))
lambdaKernel = np.transpose(np.loadtxt("lambdaKernel.txt"))
muKernel = np.transpose(np.loadtxt("muKernel.txt"))

max1 = np.max(np.abs(densityKernel))/1
max2 = np.max(np.abs(lambdaKernel))/1
max3 = np.max(np.abs(muKernel))/1
link = False
maxC = max(max2,max3)

customCMAP = plt.get_cmap('seismic')
customCMAP._segmentdata['alpha'] =  [(0.0, 0.8, 0.8), (0.5, 0.5, 0.5), (1.0, 0.8, 0.8)]

mu = np.ones((400,200)) * 1e9
overlay = np.zeros((400,200)) * 1e9

mu[100:200,0:100] = mu[100:200,0:100] * 1.5

mu[200:300,100:200] = mu[200:300,100:200] * 1.5
plt.subplot(2,2,1)
imMedium = plt.imshow(np.transpose(mu), animated=True, aspect=1, cmap=plt.get_cmap("copper"))
im = plt.imshow(np.transpose(overlay), cmap=customCMAP, vmax=max1, vmin=-max1)
plt.colorbar()
mu[200:300,100:200] = mu[200:300,100:200] / 1.5
plt.subplot(2,2,1+1)
imMedium = plt.imshow(np.transpose(mu), animated=True, aspect=1, cmap=plt.get_cmap("copper"))
im = plt.imshow(densityKernel, cmap=customCMAP, vmax=max1, vmin=-max1)
plt.colorbar()
plt.subplot(2,2,2+1)
imMedium = plt.imshow(np.transpose(mu), animated=True, aspect=1, cmap=plt.get_cmap("copper"))
im = plt.imshow(lambdaKernel, cmap=customCMAP, vmax=(maxC if link else max2), vmin=-(maxC if link else max2))
plt.colorbar()
plt.subplot(2,2,3+1)
imMedium = plt.imshow(np.transpose(mu), animated=True, aspect=1, cmap=plt.get_cmap("copper"))
im = plt.imshow(muKernel, cmap=customCMAP, vmax=(maxC if link else max3), vmin=-(maxC if link else max3))
plt.colorbar()

plt.show()
