    # -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 15:35:35 2017

@author: Brian
"""
from random import gauss
import matplotlib.pyplot as plt
import numpy as np
import pickle

with open('noisy_sine_wave','rb') as file:
     data_from_file=pickle.load(file)
"""
the above few lines makes an array called data_from_file which contains
a noisy sine wave as long as you downloaded the file "noisy_sine_wave" 
and put it in the same directory as this python file

pickle is a Python package which nicely saves data to files. it can be
a little tricky when you save lots of data, but this file only has one
object (an array) saved so it is pretty easy
"""
signal = np.array(data_from_file)
fft= np.fft.fft(signal)
disp = 2000

xmax = 300
time = np.arange(disp)
M= len(fft)
freq = np.arange(M)
peak1 = 118
peak2 = 154
peak3 = 286
width1 = 1
width2 = 1
width3 = 1
filter_function1 = np.exp(-(freq-peak1)**2/width1)+np.exp(-(freq+peak1-M)**2/width1)
filter_function2 = np.exp(-(freq-peak2)**2/width2)+np.exp(-(freq+peak2-M)**2/width2)
filter_function3 = np.exp(-(freq-peak3)**2/width3)+np.exp(-(freq+peak3-M)**2/width3)
filter_function = filter_function1 + filter_function2 + filter_function3
filtfft = fft*filter_function
filtsignal = np.fft.ifft(filtfft)
cleaned = 2.4 * np.sin(2*np.pi*peak1*time/disp)+5*np.sin(2*np.pi*peak2*time/disp)+9*np.sin(2*np.pi*peak3*time/disp)

plt.plot(time / disp, signal)
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.savefig('totalS.png',dpi=300)
plt.show()
plt.close()

plt.plot(time / disp, signal)
plt.xlim(0,xmax/2000)
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.savefig('partS.png',dpi=300)
plt.show()
plt.close()


plt.plot(freq, np.abs(fft) / disp)
plt.xlim(0,xmax)
plt.xlabel('Frequency (1/s)')
plt.ylabel('Amplitude (m)')
plt.savefig('sfft.png',dpi=300)
plt.show()
plt.close()

plt.plot(freq, np.abs(filtfft) / disp)
plt.xlim(0,xmax)
plt.xlabel('Frequency (1/s)')
plt.ylabel('Amplitude (m)')
plt.savefig('ffft.png',dpi=300)
plt.show()

plt.plot(filtsignal)
plt.xlim(0,xmax)
plt.show()
number=len(data_from_file)

fig, (ax1,ax2,ax3)=plt.subplots(3,1,sharex='col')
# this gives us an array of 3 graphs, vertically aligned
ax1.plot(freq, np.abs(fft) / M)  
ax2.plot(freq, np.abs(filter_function))
ax3.plot(freq, np.abs(filtfft) / M)
"""
note that in general, the fft is a complex function, hence we plot
the absolute value of it. in our case, the fft is real, but the
result is both positive and negative, and the absolute value is still
easier to understand

if we plotted (abs(fft))**2, that would be called the power spectra
"""

fig.subplots_adjust(hspace=0)
ax1.set_ylim(0,6)
ax2.set_ylim(0,1.2)
ax3.set_ylim(0,6)
ax1.set_xlim(0,xmax)
ax1.set_xlim(0,xmax)
ax1.set_xlim(0,xmax)
ax1.set_ylabel('Noisy FFT')
ax2.set_ylabel('Filter Function')
ax3.set_ylabel('Filtered FFT')
ax3.set_xlabel('Amplitude (m) - Frequency (1/s)')

plt.tight_layout() 
""" 
the \n in our xlabel does not save to file well without the
tight_layout() command
"""
plt.savefig('filtering2.png',dpi=300)
plt.show()
plt.close()

fig, (ax1,ax2,ax3)=plt.subplots(3,1,sharex='col',sharey='col')
ax1.plot(time / disp, signal)
ax2.plot(time / disp,np.fft.ifft(filtfft))
ax3.plot(time / disp,np.real(cleaned))
"""
we plot the real part of our cleaned data - but since the 
original data was real, the result of our tinkering should 
be real so we don't lose anything by doing this

if you don't explicitly plot the real part, python will 
do it anyway and give you a warning message about only
plotting the real part of a complex number. so really, 
it's just getting rid of a pesky warning message
"""

fig.subplots_adjust(hspace=0)
ax1.set_ylim(-25,25)
ax2.set_ylim(-20,20)
ax3.set_ylim(-20,20)
ax1.set_xlim(0,xmax/2000)
ax2.set_xlim(0,xmax/2000)
ax3.set_xlim(0,xmax/2000)
ax1.set_ylabel('Original Data')
ax2.set_ylabel('Filtered Data')
ax3.set_ylabel('Ideal Result')
ax3.set_xlabel('Position (m) - Time (s)')

plt.savefig('unknownfilter.png',dpi=300)
plt.show()
plt.close()

message="There are " + \
        str(number) + \
        " data points in total, only drawing the first " + \
        str(xmax)
print(message)
