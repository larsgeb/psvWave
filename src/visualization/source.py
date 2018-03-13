import numpy as np
import matplotlib.pyplot as plt
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)
sourceWavelet = np.loadtxt("output/source.txt")

plt.plot(sourceWavelet)
plt.show()