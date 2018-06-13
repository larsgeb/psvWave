import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

ims = []
fig = plt.figure(figsize=(5, 2.5), dpi=400)

Writer = animation.writers['ffmpeg']
writer = Writer(fps=30, metadata=dict(artist='Lars Gebraad'), bitrate=2500)

max = 1e-9

for frame in range(0, 2000, 10):
    # vx = np.transpose(np.loadtxt("snapshots/snapshot%i_vx.txt" % frame))

    # im = plt.imshow(vx[:-50, 50:-50], animated=True, aspect=1, cmap=plt.get_cmap('seismic'), vmin=-max , vmax=max )

    vx = np.transpose(np.loadtxt("snapshots/snapshot%i_vx.txt" % frame))
    vz = np.transpose(np.loadtxt("snapshots/snapshot%i_vz.txt" % frame))
    field3 = (np.sqrt(np.square(vx[:-50, 50:-50]) + np.square(vz[:-50, 50:-50])))
    im = plt.imshow(np.log10(field3), cmap=plt.get_cmap('Reds'), vmin=-12, vmax=-5)

    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=1, blit=True, repeat_delay=0)
ani.save('shot0.mp4', writer=writer)
