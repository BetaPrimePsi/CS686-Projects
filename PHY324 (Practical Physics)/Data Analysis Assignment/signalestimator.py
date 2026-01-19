# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 19:04:23 2023

@author: betap
"""

from random import gauss
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
import scipy.optimize as optimize
import scipy.stats as stats
import pickle
def expo(x,A,b,c):
    return A*np.exp(-b*x)+c
def pulse_shape(t_rise, t_fall):
    xx=np.linspace(0, 4095, 4096)
    yy = -(np.exp(-(xx-1000)/t_rise)-np.exp(-(xx-1000)/t_fall))
    yy[:1000]=0
    yy /= np.max(yy)
    return yy
def fit_pulse(x, A):
    _pulse_template = pulse_shape(20,80)
    xx=np.linspace(0, 4095, 4096)
    return A*np.interp(x, xx, _pulse_template)
# fit_pulse can be used by curve_fit to fit a pulse to the pulse_shape
with open("signal_p3.pkl","rb") as file:
    signal_data=pickle.load(file)
pulse_template = pulse_shape(20,80)
for itrace in range(1000):
    plt.plot(signal_data['evt_%i'%itrace], alpha=0.3)
plt.xlabel('Trace Time ($\mu$s)')
plt.ylabel('Readout (V)')
plt.savefig('sigsamp', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

amp6 = np.zeros(1000)
for ievt in range(1000):
    current_data = signal_data['evt_%i'%ievt]
    fitA, uncA = optimize.curve_fit(fit_pulse, np.arange(4096), current_data)
    amp6[ievt] = fitA
amp6*=1000 # convert from V to mV 
amp6*=46.776148490343054 # convert to energy scale
num_bins6=20
bin_range6=(0,20)

n6, bin_edges6, _ = plt.hist(amp6, bins=num_bins6, range=bin_range6, color='k', 
histtype='step', label='Data')
# This plots the histogram AND saves the counts and bin_edges for later use
plt.xlabel('Pulse Fit Energy Spectrum (keV)')
plt.ylabel('Events / %2.2f keV'%((bin_range6[-1]-bin_range6[0])/num_bins6));
plt.xlim(bin_range6)  
# If the legend covers some data, increase the plt.xlim value, maybe (0,0.5)
bin_centers6 = 0.5*(bin_edges6[1:]+bin_edges6[:-1])
sig6 = np.sqrt(n6)
sig6=np.where(sig6==0, 1, sig6)
plt.errorbar(bin_centers6, n6, yerr=sig6, fmt='none', c='k')
# This adds errorbars to the histograms, where each uncertainty is sqrt(y)
popt6, pcov6 = optimize.curve_fit(expo, bin_centers6, n6, 
             sigma = sig6, p0=(210,4,0), absolute_sigma=True)
n6_fit = expo(bin_centers6, *popt6)
chisquared6 = np.sum( ((n6 - n6_fit)/sig6 )**2)
dof6 = num_bins6 - len(popt6)
# Number of degrees of freedom is the number of data points less the number of 
# fitted parameters
x_bestfit6 = np.linspace(bin_edges6[0], bin_edges6[-1], 1000)
y_bestfit6 = expo(x_bestfit6, *popt6)
plt.plot(x_bestfit6, y_bestfit6, label='Fit')
fontsize=12
plt.text(15, 135, r'$A$ = %3.2f'%(popt6[0]), fontsize=fontsize)
plt.text(15, 115, r'$\beta$ = %3.2f 1/keV'%(popt6[1]), fontsize=fontsize)
plt.text(15, 95, r'$c$ = %3.2f keV'%(popt6[2]), fontsize=fontsize)
plt.text(15, 75, r'$\chi^2$/DOF=', fontsize=fontsize)
plt.text(15, 55, r'%3.2f/%i'%(chisquared6,dof6), fontsize=fontsize)
plt.text(15, 35, r'$\chi^2$ prob.= %1.1f'%(1-stats.chi2.cdf(chisquared6,dof6)), 
fontsize=fontsize)
plt.legend(loc=1)
plt.savefig('pulsefitsig',dpi=300, bbox_inches='tight')
plt.show()
plt.close()