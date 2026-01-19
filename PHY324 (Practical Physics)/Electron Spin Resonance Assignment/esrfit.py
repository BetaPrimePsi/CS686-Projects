from random import gauss
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
import scipy.optimize as optimize
import scipy.stats as stats
import pickle

with open('data.csv', 'r') as file:
    data = np.loadtxt(file, delimiter=',', usecols=range(2))
data = np.transpose(data)

freq = data[0]*1000**2
curr = data[1]/2

uncfreq = freq*0.004/100
unccurr=0.0005+0.03*curr

n=320
mu = 4*np.pi*10**(-7)
uncmu=2*10**(-10)*mu
R = 7.25/100
uncR = 0.05/100
e = 1.602*10**(-19)
unce = 0.0001*10**(-19)
mass = 9.109*10**(-31)
uncmass = 2.8*10**(-40)

gamma0 = e/(2*mass)
gunc0 = gamma0*np.sqrt((unce/e)**2+(uncmass/mass)**2)

def Ivsf(I,m):
    return m*I

mopt, mcov = optimize.curve_fit(Ivsf, curr, freq, absolute_sigma=True, sigma=uncfreq)

mopt = mopt[0]
munc = np.sqrt(mcov[0][0])

freqfit = Ivsf(curr, mopt)
uncfit = freqfit*np.sqrt((unccurr/curr)**2+(munc/mopt)**2)

gamma = 5**(3/2)*np.pi*mopt*R/(4*mu*n)
gunc = gamma*np.sqrt((munc/mopt)**2+(uncmu/mu)**2+(uncR/R)**2)

plt.plot(curr, freq, linestyle='None', marker='o', markersize=4, label='Data')
plt.xlabel('Resonance Current ($A$)',fontsize=15)
plt.ylabel('Absorption Frequency ($Hz$)', fontsize=15)
plt.plot(curr, freqfit, label='Fitted')
plt.errorbar(curr, freq, xerr=unccurr, yerr=uncfreq,ecolor='black', linestyle='None', label='Errors')
plt.legend(loc=0)
plt.savefig('datafit',dpi=1000)
plt.show()
plt.close()

residuals = (freq-freqfit)
resunc = np.sqrt((uncfreq)**2+(uncfit)**2)
resid2 = residuals[:-3]
resunc2 = resunc[:-3]

plt.plot(curr, residuals, linestyle='None', color='red', marker='o', markersize=4, label='Residuals')
plt.xlabel('Resonance Current ($A$)',fontsize=15)
plt.ylabel('Frequency Residuals ($Hz$)',fontsize=15)
plt.errorbar(curr, residuals, yerr=resunc, linestyle='None', ecolor='black',label='Deviations')
plt.legend(loc=3)
plt.savefig('residuals1',dpi=1000)
plt.show()
plt.close()

plt.plot(curr[:-3], resid2, linestyle='None', color='red', marker='o', markersize=4, label='Residuals')
plt.xlabel('Resonance Current ($A$)',fontsize=15)
plt.ylabel('Frequency Residuals ($Hz$)',fontsize=15)
plt.errorbar(curr[:-3], resid2, yerr=resunc2, linestyle='None', ecolor='black',label='Deviations')
plt.legend(loc=3)
plt.savefig('residuals2',dpi=1000)
plt.show()
plt.close()

chisquared = np.sum((residuals/uncfreq)**2)/(len(freq)-1)

bins, edges, _ = plt.hist(residuals, bins=9,range=(-2.5*10**6,1.5*10**6),label='Histogram', facecolor='None', edgecolor='red')
uncbin = np.sqrt(bins)
centers = (edges[:-1]+edges[1:])/2
plt.errorbar(centers, bins, yerr=uncbin, linestyle='None',ecolor='black',label='Standard Error')
plt.xlabel('Residuals ($Hz$)',fontsize=15)
plt.ylabel('Counts / $4.4\\times10^{5}$ $Hz$',fontsize=15)
plt.legend(loc=1)
plt.savefig('hist',dpi=1000)
plt.show()
plt.close()
g = gamma/gamma0
ug = g*np.sqrt((gunc/gamma)**2+(gunc0/gamma0)**2)
print(mopt,munc)    
print(gamma, gunc/gamma)
print(gamma0,gunc0/gamma0)
print(g,ug)