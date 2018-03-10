import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# from matplotlib.animation import FuncAnimation

ims = []
fig = plt.figure()

for frame in np.arange(0,2000,5):
    im = plt.imshow(np.transpose(np.loadtxt("output/p%i.txt" % frame)), animated=True,vmin=-5e-7, vmax=5e-7, aspect=1)
    ims.append([im])


ani = animation.ArtistAnimation(fig, ims, interval=10, blit=True, repeat_delay=0)
plt.colorbar()
plt.show()