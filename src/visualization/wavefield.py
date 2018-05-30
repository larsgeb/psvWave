import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

ims = []
fig = plt.figure(figsize=(10, 5), dpi=400)

Writer = animation.writers['ffmpeg']
writer = Writer(fps=1, metadata=dict(artist='Lars Gebraad'), bitrate=5000)

for frame in range(0, 3500, 50):
    vx = np.transpose(np.loadtxt("snapshot%i_vx.txt" % frame))

    if frame == 0:
        max = np.max(np.abs(vx))

    im = plt.imshow(vx, animated=True, aspect=1, cmap=plt.get_cmap('seismic'), vmin=-max*1e6, vmax=max*1e6)
    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=1, blit=True, repeat_delay=0)
ani.save('shot0.mp4', writer=writer)
