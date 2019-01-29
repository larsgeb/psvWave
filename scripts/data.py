import numpy as np
import matplotlib.pyplot as plt

rtf_ux0 = np.loadtxt("rtf_ux0.txt")
rtf_ux1 = np.loadtxt("rtf_ux1.txt")
rtf_uz0 = np.loadtxt("rtf_uz0.txt")
rtf_uz1 = np.loadtxt("rtf_uz1.txt")

plt.plot(rtf_ux0.T, 'k')
plt.plot(rtf_ux1.T, 'r')
plt.plot(rtf_uz0.T, 'g')
plt.plot(rtf_uz1.T, 'b')
plt.show()
