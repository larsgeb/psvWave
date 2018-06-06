import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import sys

# print(sys.argv)
ims = []
fig = plt.figure(figsize=(10, 5), dpi=400)

Writer = animation.writers['ffmpeg']
writer = Writer(fps=30, metadata=dict(artist='Lars Gebraad'), bitrate=5000)

customCMAP = plt.get_cmap('seismic')
customCMAP._segmentdata['alpha'] = [(0.0, 0.8, 0.8), (0.5, 0.5, 0.5), (1.0, 0.8, 0.8)]

mu = np.loadtxt("mu.txt")

imMedium = plt.imshow(np.transpose(mu), animated=True, aspect=1, cmap=plt.get_cmap("copper"))
plotBoundary = plt.plot([50, 50, 450, 450], [0, 200, 200, 0], animated=True)

for frame in np.arange(0, 3500, 10):
    imWave = plt.imshow(np.transpose(np.loadtxt("output/shot%i/vx%i.txt" % (int(sys.argv[1]), frame))), animated=True, vmin=-2e-13, vmax=2e-13,
                        aspect=1, cmap=customCMAP)
    ims.append([imMedium, imWave, plotBoundary[0]])

ani = animation.ArtistAnimation(fig, ims, interval=10, blit=True, repeat_delay=0)
ani.save('shotOverlay%i.mp4' % int(sys.argv[1]), writer=writer)
