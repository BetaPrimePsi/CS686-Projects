# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 18:48:26 2023

@author: betap
"""
import matplotlib.pyplot as plt
import numpy as np
N=200
time=np.arange(N)

A1=5.   # wave amplitude
T1=17.  # wave period
y1=A1*np.sin(2.*np.pi*time/T1)

A2=9.
T2=13.
y2=A2*np.sin(2.*np.pi*time/T2)

y=y1+y2

plt.plot(time/N, y)
plt.ylabel('Position (m)')
plt.xlabel('Time (s)')
plt.savefig('doublewave',dpi=300)
plt.show()
plt.close()

ffty=np.fft.fft(y)
M=len(ffty)
freq=np.arange(M)

plt.plot(freq, np.abs(ffty)/M)
plt.ylabel('Amplitude (m)')
plt.xlabel('Frequency (1/s)')
plt.savefig('doublewavefft',dpi=300)
plt.show()
plt.close()

plt.plot(freq, np.abs(ffty)/M)
plt.ylabel('Amplitude (m)')
plt.xlabel('Frequency (1/s)')
plt.xlim(0,30)
plt.savefig('doublewavefftlim',dpi=300)
plt.show()
plt.close()