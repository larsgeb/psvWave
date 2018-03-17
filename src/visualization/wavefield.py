import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# plt.rcParams['animation.ffmpeg_path'] = 'C:\\Program Files\\ffmpeg\\bin\\ffmpeg'

ims = []
fig = plt.figure(figsize=(16,8), dpi=300)

Writer = animation.writers['ffmpeg']
writer = Writer(fps=30, metadata=dict(artist='Lars Gebraad'), bitrate=8000)

for frame in np.arange(0,3000,10):
    im = plt.imshow(np.transpose(np.loadtxt("output/vx%i.txt" % frame)), animated=True,vmin=-5e-13, vmax=5e-13, aspect=1, cmap=plt.get_cmap('seismic'))
    ims.append([im])


ani = animation.ArtistAnimation(fig, ims, interval=10, blit=True, repeat_delay=0)
# plt.colorbar()
ani.save('animation.mp4', writer=writer)