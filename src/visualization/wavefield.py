import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

ims = []
fig = plt.figure(figsize=(10,5), dpi=400)

Writer = animation.writers['ffmpeg']
writer = Writer(fps=30, metadata=dict(artist='Lars Gebraad'), bitrate=5000)

for frame in np.arange(0,3500,10):
    im = plt.imshow(np.transpose(np.loadtxt("output/shot0/vx%i.txt" % frame)), animated=True,vmin=-5e-13, vmax=5e-13, aspect=1, cmap=plt.get_cmap('seismic'))
    ims.append([im])


ani = animation.ArtistAnimation(fig, ims, interval=10, blit=True, repeat_delay=0)
ani.save('shot0.mp4', writer=writer)

