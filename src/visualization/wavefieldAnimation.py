import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

ims = []
fig = plt.figure(figsize=(2, 1), dpi=400)

Writer = animation.writers['ffmpeg']
writer = Writer(fps=30, metadata=dict(artist='Lars Gebraad'), bitrate=5000)

max = 1e-9

for frame in range(0, 4500, 1):
    vx = np.transpose(np.loadtxt("snapshots/snapshot%i_vx.txt" % frame))

    im = plt.imshow(vx[:-50, 50:-50], animated=True, aspect=1, cmap=plt.get_cmap('seismic'), vmin=-max , vmax=max )
    # plt.savefig("%i.png"%frame)
    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=1, blit=True, repeat_delay=0)
ani.save('shot0.mp4', writer=writer)
