save=True # if True then we save images as files

from random import gauss
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
import scipy.optimize as optimize
import scipy.stats as stats
import pickle

font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size'   : 22}
rc('font', **font)
# This changes the fonts for all graphs to make them bigger.
def myGauss(x, A, mean, width, base):
    return A*np.exp(-(x-mean)**2/(2*width**2)) + base
# This is my fitting function, a Guassian with a uniform background.
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
with open("calibration_p3.pkl","rb") as file:
    calibration_data=pickle.load(file)
pulse_template = pulse_shape(20,80)
plt.plot(pulse_template/2000, label='Pulse Template', color='r')
for itrace in range(10):
    plt.plot(calibration_data['evt_%i'%itrace], alpha=0.3)
plt.xlabel('Trace Time ($\mu$s)')
plt.ylabel('Readout (V)')
plt.legend(loc=1)
plt.savefig('calsamp', dpi=300, bbox_inches='tight')
plt.show()
plt.close()
""" 
This shows the first 10 data sets on top of each other.
Always a good idea to look at some of your data before analysing it!
It also plots our pulse template which has been scaled to be slightly 
larger than any of the actual pulses to make it visible.
"""
amp1=np.zeros(1000)
amp2=np.zeros(1000)
amp3=np.zeros(1000)
amp4=np.zeros(1000)
amp5=np.zeros(1000)
amp6=np.zeros(1000)
# These are the 6 energy estimators as empty arrays of the correct size.
for ievt in range(1000):
    current_data = calibration_data['evt_%i'%ievt]
    amp1_calculation = np.max(np.abs(current_data))
    amp1[ievt] = amp1_calculation
    baselineavg = np.average(current_data[0:990])
    amp2[ievt] = np.max(np.abs(current_data - baselineavg))
    amp3[ievt] = np.average(current_data)
    amp4[ievt] = np.average(current_data-baselineavg)
    amp5[ievt] = np.abs(np.average(current_data[999:1100]))
    fitA, uncA = optimize.curve_fit(fit_pulse, np.arange(4096), current_data)
    amp6[ievt] = fitA
"""
This incorrectly calculates one of the amplitude estimators.
You will want to fix it, and then do the other 5 estimators 
inside this for loop. I.e. you will need to add:
    amp2[ievt] = ...
    area1[ievt] = ...
etc.
"""
amp1*=1000 # convert from V to mV   
num_bins1=40
bin_range1=(0.07,0.44) 

amp2*=1000 # convert from V to mV   
num_bins2=40
bin_range2=(0.07,0.43)

amp3*=1000 # convert from V to mV   
num_bins3=30
bin_range3=(0,0.025)

amp4*=1000
num_bins4=30
bin_range4=(0,0.015)

amp5*=1000
num_bins5=50
bin_range5=(0,0.3)

amp6*=1000 # convert from V to mV   
num_bins6=50
bin_range6=(0,0.4)
"""
These two values were picked by trial and error. You'll 
likely want different values for each estimator.
"""
# FIRST ESTIMATOR
n1, bin_edges1, _ = plt.hist(amp1, bins=num_bins1, range=bin_range1, color='k', 
histtype='step', label='Data')
# This plots the histogram AND saves the counts and bin_edges for later use
plt.xlabel('Energy Estimator: Maximum Value (mV)')
plt.ylabel('Events / %2.2f mV'%((bin_range1[-1]-bin_range1[0])/num_bins1));
plt.xlim(bin_range1)  
# If the legend covers some data, increase the plt.xlim value, maybe (0,0.5)
bin_centers1 = 0.5*(bin_edges1[1:]+bin_edges1[:-1])
sig1 = np.sqrt(n1)
sig1=np.where(sig1==0, 1, sig1)

plt.errorbar(bin_centers1, n1, yerr=sig1, fmt='none', c='k')
# This adds errorbars to the histograms, where each uncertainty is sqrt(y)
popt1, pcov1 = optimize.curve_fit(myGauss, bin_centers1, n1, 
             sigma = sig1, p0=(100,0.25,0.05,5), absolute_sigma=True)
n1_fit = myGauss(bin_centers1, *popt1)
chisquared1 = np.sum( ((n1 - n1_fit)/sig1 )**2)
dof1 = num_bins1 - len(popt1)
C1=10/popt1[1]
# Number of degrees of freedom is the number of data points less the number of 
# fitted parameters
x_bestfit1 = np.linspace(bin_edges1[0], bin_edges1[-1], 1000)
y_bestfit1 = myGauss(x_bestfit1, *popt1) 
# Best fit line smoothed with 1000 datapoints. Don't use best fit lines with 5 or 
# 10 data points!
fontsize=12
plt.plot(x_bestfit1, y_bestfit1, label='Fit')
plt.text(0.08, 40, r'$C$ = %3.2f keV/mV'%(C1), fontsize=fontsize)
plt.text(0.08, 140, r'$\mu$ = %3.2f mV'%(popt1[1]), fontsize=fontsize)
plt.text(0.08, 120, r'$\sigma$ = %3.2f mV'%(popt1[2]), fontsize=fontsize)
plt.text(0.08, 100, r'$\chi^2$/DOF=', fontsize=fontsize)
plt.text(0.08, 80, r'%3.2f/%i'%(chisquared1,dof1), fontsize=fontsize)
plt.text(0.08, 60, r'$\chi^2$ prob.= %1.1f'%(1-stats.chi2.cdf(chisquared1,dof1)), 
fontsize=fontsize)
plt.legend(loc=1)
plt.savefig('max',dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# SECOND ESTIMATOR
n2, bin_edges2, _ = plt.hist(amp2, bins=num_bins2, range=bin_range2, color='k', 
histtype='step', label='Data')
# This plots the histogram AND saves the counts and bin_edges for later use
plt.xlabel('Energy Estimator: Baseline Max (mV)')
plt.ylabel('Events / %2.2f mV'%((bin_range2[-1]-bin_range2[0])/num_bins2));
plt.xlim(bin_range2)  
# If the legend covers some data, increase the plt.xlim value, maybe (0,0.5)
bin_centers2 = 0.5*(bin_edges2[1:]+bin_edges2[:-1])
sig2 = np.sqrt(n2)
sig2=np.where(sig2==0, 1, sig2)

plt.errorbar(bin_centers2, n2, yerr=sig2, fmt='none', c='k')
# This adds errorbars to the histograms, where each uncertainty is sqrt(y)
popt2, pcov2 = optimize.curve_fit(myGauss, bin_centers2, n2, 
             sigma = sig2, p0=(100,0.25,0.05,5), absolute_sigma=True)
n2_fit = myGauss(bin_centers2, *popt2)
chisquared2 = np.sum( ((n2 - n2_fit)/sig2 )**2)
dof2 = num_bins2 - len(popt2)
C2=10/popt2[1]
# Number of degrees of freedom is the number of data points less the number of 
# fitted parameters
x_bestfit2 = np.linspace(bin_edges2[0], bin_edges2[-1], 1000)
y_bestfit2 = myGauss(x_bestfit2, *popt2) 
# Best fit line smoothed with 1000 datapoints. Don't use best fit lines with 5 or 
# 10 data points!
fontsize=12
plt.plot(x_bestfit2, y_bestfit2, label='Fit')
plt.text(0.08, 140, r'$\mu$ = %3.2f mV'%(popt2[1]), fontsize=fontsize)
plt.text(0.08, 120, r'$\sigma$ = %3.2f mV'%(popt2[2]), fontsize=fontsize)
plt.text(0.08, 100, r'$\chi^2$/DOF=', fontsize=fontsize)
plt.text(0.08, 80, r'%3.2f/%i'%(chisquared2,dof2), fontsize=fontsize)
plt.text(0.08, 60, r'$\chi^2$ prob.= %1.1f'%(1-stats.chi2.cdf(chisquared2,dof2)), 
fontsize=fontsize)
plt.text(0.08, 40, r'$C$ = %3.2f keV/mV'%(C2), fontsize=fontsize)
plt.legend(loc=1)
plt.savefig('maxbase',dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# THIRD ESTIMATOR
n3, bin_edges3, _ = plt.hist(amp3, bins=num_bins3, range=bin_range3, color='k', 
histtype='step', label='Data')
# This plots the histogram AND saves the counts and bin_edges for later use
plt.xlabel('Energy Estimator: Integral (mV $\mu s$)')
plt.ylabel('Events / %2.3f mV $\mu s$'%((bin_range3[-1]-bin_range3[0])/num_bins3));
plt.xlim(bin_range3)  
# If the legend covers some data, increase the plt.xlim value, maybe (0,0.5)
bin_centers3 = 0.5*(bin_edges3[1:]+bin_edges3[:-1])
sig3 = np.sqrt(n3)
sig3=np.where(sig3==0, 1, sig3)
plt.errorbar(bin_centers3, n3, yerr=sig3, fmt='none', c='k')
# This adds errorbars to the histograms, where each uncertainty is sqrt(y)
popt3, pcov3 = optimize.curve_fit(myGauss, bin_centers3, n3, 
             sigma = sig3, p0=(40,0.010,0.015,1), absolute_sigma=True)
n3_fit = myGauss(bin_centers3, *popt3)
chisquared3 = np.sum( ((n3 - n3_fit)/sig3 )**2)
dof3 = num_bins3 - len(popt3)
C3=10/popt3[1]
# Number of degrees of freedom is the number of data points less the number of 
# fitted parameters
x_bestfit3 = np.linspace(bin_edges3[0], bin_edges3[-1], 1000)
y_bestfit3 = myGauss(x_bestfit3, *popt3) 
# Best fit line smoothed with 1000 datapoints. Don't use best fit lines with 5 or 
# 10 data points!
fontsize=12
plt.plot(x_bestfit3, y_bestfit3, label='Fit')
plt.text(0.001, 22, r'$\mu$ = %3.3f mV $\mu s$'%(popt3[1]), fontsize=fontsize)
plt.text(0.001, 18, r'$\sigma$ = %3.3f mV $\mu s$'%(popt3[2]), fontsize=fontsize)
plt.text(0.001, 14, r'$\chi^2$/DOF=', fontsize=fontsize)
plt.text(0.001, 10, r'%3.2f/%i'%(chisquared3,dof3), fontsize=fontsize)
plt.text(0.001, 6, r'$\chi^2$ prob.= %1.1f'%(1-stats.chi2.cdf(chisquared3,dof3)), 
fontsize=fontsize)
plt.text(0.001, 2, r'$C$ = %3.2f keV/mV $\mu s$'%(C3), fontsize=fontsize)
plt.legend(loc=1)
plt.savefig('int',dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# FOURTH ESTIMATOR
n4, bin_edges4, _ = plt.hist(amp4, bins=num_bins4, range=bin_range4, color='k', 
histtype='step', label='Data')
# This plots the histogram AND saves the counts and bin_edges for later use
plt.xlabel('Energy Estimator: Baseline Int. (mV $\mu s$)')
plt.ylabel('Events / %2.3f mV $\mu s$'%((bin_range4[-1]-bin_range4[0])/num_bins4));
plt.xlim(bin_range4)  
# If the legend covers some data, increase the plt.xlim value, maybe (0,0.5)
bin_centers4 = 0.5*(bin_edges4[1:]+bin_edges4[:-1])
sig4 = np.sqrt(n4)
sig4=np.where(sig4==0, 1, sig4)
plt.errorbar(bin_centers4, n4, yerr=sig4, fmt='none', c='k')
# This adds errorbars to the histograms, where each uncertainty is sqrt(y)
popt4, pcov4 = optimize.curve_fit(myGauss, bin_centers4, n4, 
             sigma = sig4, p0=(100,0.007,0.005,5), absolute_sigma=True)
n4_fit = myGauss(bin_centers4, *popt4)
chisquared4 = np.sum( ((n4 - n4_fit)/sig4 )**2)
dof4 = num_bins4 - len(popt4)
C4=10/popt4[1]
# Number of degrees of freedom is the number of data points less the number of 
# fitted parameters
x_bestfit4 = np.linspace(bin_edges4[0], bin_edges4[-1], 1000)
y_bestfit4 = myGauss(x_bestfit4, *popt4) 
# Best fit line smoothed with 1000 datapoints. Don't use best fit lines with 5 or 
# 10 data points!
fontsize=12
plt.plot(x_bestfit4, y_bestfit4, label='Fit')
plt.text(0.005, 32, r'$\mu$ = %3.3f mV $\mu s$'%(popt4[1]), fontsize=fontsize)
plt.text(0.005, 26, r'$\sigma$ = %3.3f mV $\mu s$'%(popt4[2]), fontsize=fontsize)
plt.text(0.005, 20, r'$\chi^2$/DOF=', fontsize=fontsize)
plt.text(0.005, 14, r'%3.2f/%i'%(chisquared4,dof4), fontsize=fontsize)
plt.text(0.005, 8, r'$\chi^2$ prob.= %1.1f'%(1-stats.chi2.cdf(chisquared4,dof4)), 
fontsize=fontsize)
plt.text(0.005, 2, r'$C$ = %3.2f keV/mV $\mu s$'%(C4), fontsize=fontsize)
plt.legend(loc=1)
plt.savefig('intbase',dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# FIFTH ESTIMATOR
n5, bin_edges5, _ = plt.hist(amp5, bins=num_bins5, range=bin_range5, color='k', 
histtype='step', label='Data')
# This plots the histogram AND saves the counts and bin_edges for later use
plt.xlabel('Energy Estimator: Ranged Int. (mV $\mu s$)')
plt.ylabel('Events / %2.2f mV $\mu s$'%((bin_range5[-1]-bin_range5[0])/num_bins5));
plt.xlim(bin_range5)  
# If the legend covers some data, increase the plt.xlim value, maybe (0,0.5)
bin_centers5 = 0.5*(bin_edges5[1:]+bin_edges5[:-1])
sig5 = np.sqrt(n5)
sig5=np.where(sig5==0, 1, sig5)
plt.errorbar(bin_centers5, n5, yerr=sig5, fmt='none', c='k')
# This adds errorbars to the histograms, where each uncertainty is sqrt(y)
popt5, pcov5 = optimize.curve_fit(myGauss, bin_centers5, n5, 
             sigma = sig5, p0=(100,0.25,0.05,5), absolute_sigma=True)
n5_fit = myGauss(bin_centers5, *popt5)
chisquared5 = np.sum( ((n5 - n5_fit)/sig5 )**2)
dof5 = num_bins5 - len(popt5)
C5=10/popt5[1]
# Number of degrees of freedom is the number of data points less the number of 
# fitted parameters
x_bestfit5 = np.linspace(bin_edges5[0], bin_edges5[-1], 1000)
y_bestfit5 = myGauss(x_bestfit5, *popt5) 
# Best fit line smoothed with 1000 datapoints. Don't use best fit lines with 5 or 
# 10 data points!
fontsize=12
plt.plot(x_bestfit5, y_bestfit5, label='Fit')
plt.text(0.01, 120, r'$\mu$ = %3.3f mV $\mu s$'%(popt5[1]), fontsize=fontsize)
plt.text(0.01, 100, r'$\sigma$ = %3.3f mV $\mu s$'%(np.abs(popt5[2])), fontsize=fontsize)
plt.text(0.01, 80, r'$\chi^2$/DOF=', fontsize=fontsize)
plt.text(0.01, 60, r'%3.2f/%i'%(chisquared5,dof5), fontsize=fontsize)
plt.text(0.01, 40, r'$\chi^2$ prob.= %1.1f'%(1-stats.chi2.cdf(chisquared5,dof5)), 
fontsize=fontsize)
plt.text(0.01, 20, r'$C$ = %3.2f keV/mV $\mu s$'%(C5), fontsize=fontsize)
plt.legend(loc=1)
plt.savefig('rangint',dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# SIXTH ESTIMATOR
n6, bin_edges6, _ = plt.hist(amp6, bins=num_bins6, range=bin_range6, color='k', 
histtype='step', label='Data')
# This plots the histogram AND saves the counts and bin_edges for later use
plt.xlabel('Energy Estimator: Pulse Fit (mV)')
plt.ylabel('Events / %2.2f mV'%((bin_range6[-1]-bin_range6[0])/num_bins6));
plt.xlim(bin_range6)  
# If the legend covers some data, increase the plt.xlim value, maybe (0,0.5)
bin_centers6 = 0.5*(bin_edges6[1:]+bin_edges6[:-1])
sig6 = np.sqrt(n6)
sig6=np.where(sig6==0, 1, sig6)
plt.errorbar(bin_centers6, n6, yerr=sig6, fmt='none', c='k')
# This adds errorbars to the histograms, where each uncertainty is sqrt(y)
popt6, pcov6 = optimize.curve_fit(myGauss, bin_centers6, n6, 
             sigma = sig6, p0=(100,0.25,0.05,5), absolute_sigma=True)
n6_fit = myGauss(bin_centers6, *popt6)
chisquared6 = np.sum( ((n6 - n6_fit)/sig6 )**2)
dof6 = num_bins6 - len(popt6)
C6=10/popt6[1]
# Number of degrees of freedom is the number of data points less the number of 
# fitted parameters
x_bestfit6 = np.linspace(bin_edges6[0], bin_edges6[-1], 1000)
y_bestfit6 = myGauss(x_bestfit6, *popt6) 
# Best fit line smoothed with 1000 datapoints. Don't use best fit lines with 5 or 
# 10 data points!
fontsize=12
plt.plot(x_bestfit6, y_bestfit6, label='Fit')
plt.text(0.01, 95, r'$\mu$ = %3.2f mV'%(popt6[1]), fontsize=fontsize)
plt.text(0.01, 80, r'$\sigma$ = %3.2f mV'%(popt6[2]), fontsize=fontsize)
plt.text(0.01, 65, r'$\chi^2$/DOF=', fontsize=fontsize)
plt.text(0.01, 50, r'%3.2f/%i'%(chisquared6,dof6), fontsize=fontsize)
plt.text(0.01, 35, r'$\chi^2$ prob.= %1.1f'%(1-stats.chi2.cdf(chisquared6,dof6)), 
fontsize=fontsize)
plt.text(0.01, 20, r'$C$ = %3.2f keV/mV'%(C6), fontsize=fontsize)
plt.legend(loc=1)
plt.savefig('pulsefit',dpi=300, bbox_inches='tight')
plt.show()
plt.close()
"""
This gives us the x-data which are the centres of each bin.
This is visually better for plotting errorbars.
More important, it's the correct thing to do for fitting the
Gaussian to our histogram.
It also fixes the shape -- len(n1) < len(bin_edges1) so we
cannot use 
plt.plot(n1, bin_edges1)
as it will give us a shape error.
"""
print(C6)
# The uncertainty on 0 count is 1, not 0. Replace all 0s with 1s.

"""
n1_fit is our best fit line using our data points.
Note that if you have few enough bins, this best fit
line will have visible bends which look bad, so you
should not plot n1_fit directly. See below.
"""

"""
Look how bad that chi-squared value (and associated probability) is!
If you look closely, the first 5 data points (on the left) are
responsible for about half of the chi-squared value. It might be
worth excluding them from the fit and subsequent plot.
Now your task is to find the calibration factor which converts the
x-axis of this histogram from mV to keV such that the peak (mu) is 
by definition at 10 keV. You do this by scaling each estimator (i.e.
the values of amp1) by a multiplicative constant with units mV / keV.
Something like:
energy_amp1 = amp1 * conversion_factor1
where you have to find the conversion_factor1 value. Then replot and
refit the histogram using energy_amp1 instead of amp1. 
If you do it correctly, the new mu value will be 10 keV, and the new 
sigma value will be the energy resolution of this energy estimator.
Note: you should show this before/after conversion for your first
energy estimator. To save space, only show the after histograms for
the remaining 5 energy estimators.
"""