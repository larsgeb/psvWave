import matplotlib.pyplot as plt
import numpy as np

rtf_ux0 = np.loadtxt("/home/lars/Documents/PhD/forward-virieux/build/rtf_ux0.txt")
rtf_uz0 = np.loadtxt("/home/lars/Documents/PhD/forward-virieux/build/rtf_uz0.txt")

sources = np.loadtxt("/home/lars/Documents/PhD/forward-virieux/build/sources_shot_0.txt")

t = np.linspace(0, rtf_ux0.shape[1] / 4000., rtf_ux0.shape[1])

plt.figure(figsize=(10, 10))
plt.subplot(211)
for i in range(rtf_ux0.shape[0]):
    rtf_ux0[i, :] += 10000 * i
    rtf_uz0[i, :] += 10000 * i

plt.plot(t, rtf_ux0.T, 'k')
plt.plot(t, rtf_uz0.T, 'g')
plt.ylabel('from left to right (increasing x)')
plt.xlabel('time [s]')

plt.subplot(212)
for i in range(sources.shape[0]):
    sources[i, :] += i
plt.plot(t, sources.T, 'r')
plt.ylabel('from left to right (increasing x)')
plt.xlabel('time [s]')

plt.show()
