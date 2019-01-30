import numpy as np
import matplotlib.pyplot as plt

rtf_ux0 = np.loadtxt("rtf_ux0.txt")
rtf_uz0 = np.loadtxt("rtf_uz0.txt")

plt.plot(rtf_ux0.T, 'k')
plt.plot(rtf_uz0.T, 'g')
plt.show()
