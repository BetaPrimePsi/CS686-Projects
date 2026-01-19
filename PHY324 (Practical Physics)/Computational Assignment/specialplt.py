# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 19:03:15 2023

@author: betap
"""

import matplotlib.pyplot as plt
import numpy as np
N=1000
time=np.arange(N)

f = np.sin(2*np.pi*(30*time/N+1)*time/N)
fft = np.fft.fft(f)
M = len(fft)
freq = np.arange(M)
plt.plot(time/N, f)
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.savefig('sint2.png',dpi=300)
plt.show()
plt.close()

plt.plot(freq, np.abs(fft) / M)
plt.xlabel('Frequency (1/s)')
plt.ylabel('Amplitude (m)')
plt.savefig('sint2fft.png',dpi=300)
plt.show()

plt.plot(freq, np.abs(fft) / M)
plt.xlabel('Frequency (1/s)')
plt.ylabel('Amplitude (m)')
plt.xlim(0,300)
plt.savefig('sint2lim.png',dpi=300)
plt.show()