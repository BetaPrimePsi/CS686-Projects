from random import gauss
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
import scipy.optimize as optimize
import scipy.stats as stats
import scipy.special as sp
import pickle

fontsize=12

with open('trial1.txt', 'r') as file:
    d1 = np.loadtxt(file, delimiter='\t', usecols=range(2), skiprows=2)
with open('trial2.txt', 'r') as file:
    d2 = np.loadtxt(file, delimiter='\t', usecols=range(2), skiprows=2)
with open('trial3.txt', 'r') as file:
    d3 = np.loadtxt(file, delimiter='\t', usecols=range(2), skiprows=2)
with open('trial4.txt', 'r') as file:
    d4 = np.loadtxt(file, delimiter='\t', usecols=range(2), skiprows=2)
with open('trial5.txt', 'r') as file:
    d5 = np.loadtxt(file, delimiter='\t', usecols=range(2), skiprows=2)
with open('trial6.txt', 'r') as file:
    d6 = np.loadtxt(file, delimiter='\t', usecols=range(2), skiprows=2)
with open('trial7.txt', 'r') as file:
    d7 = np.loadtxt(file, delimiter='\t', usecols=range(2), skiprows=2)
with open('trial8.txt', 'r') as file:
    d8 = np.loadtxt(file, delimiter='\t', usecols=range(2), skiprows=2)
with open('trial9.txt', 'r') as file:
    d9 = np.loadtxt(file, delimiter='\t', usecols=range(2), skiprows=2)
with open('trial10.txt', 'r') as file:
    d10 = np.loadtxt(file, delimiter='\t', usecols=range(2), skiprows=2)
with open('trial11.txt', 'r') as file:
    d11 = np.loadtxt(file, delimiter='\t', usecols=range(2), skiprows=2)
with open('trial12.txt', 'r') as file:
    d12 = np.loadtxt(file, delimiter='\t', usecols=range(2), skiprows=2)
with open('trial13.txt', 'r') as file:
    d13 = np.loadtxt(file, delimiter='\t', usecols=range(2), skiprows=2)
with open('trial14.txt', 'r') as file:
    d14 = np.loadtxt(file, delimiter='\t', usecols=range(2), skiprows=2)
with open('trial15.txt', 'r') as file:
    d15 = np.loadtxt(file, delimiter='\t', usecols=range(2), skiprows=2)
with open('trial16.txt', 'r') as file:
    d16 = np.loadtxt(file, delimiter='\t', usecols=range(2), skiprows=2)
with open('trial17.txt', 'r') as file:
    d17 = np.loadtxt(file, delimiter='\t', usecols=range(2), skiprows=2)
with open('trial18.txt', 'r') as file:
    d18 = np.loadtxt(file, delimiter='\t', usecols=range(2), skiprows=2)
with open('trial19.txt', 'r') as file:
    d19 = np.loadtxt(file, delimiter='\t', usecols=range(2), skiprows=2)
with open('trial20.txt', 'r') as file:
    d20 = np.loadtxt(file, delimiter='\t', usecols=range(2), skiprows=2)
with open('trial21.txt', 'r') as file:
    d21 = np.loadtxt(file, delimiter='\t', usecols=range(2), skiprows=2)

def rali(x,A):
    return ((x)/A)*np.exp(-(x)**2/(2*A))

def chi2(xexp,xval,sigma):
    return np.sum(((xexp-xval)/sigma)**2)

def mle(x):
    return np.sum(x**2)/(2*len(x))

def brownian(t, D):
    return 4*D*t

sample = d1-d1[0]
sample = np.transpose(sample)
plt.plot(sample[0],sample[1])
plt.xlabel('x ($\mu$m)')
plt.ylabel('y ($\mu$m)')
plt.savefig('samplemotion',dpi=300)
plt.show()
plt.close()

s1 = np.linalg.norm(d1-d1[0],axis=1)
s2 = np.linalg.norm(d2-d2[0],axis=1)
s3 = np.linalg.norm(d3-d3[0],axis=1)
s4 = np.linalg.norm(d4-d4[0],axis=1)
s5 = np.linalg.norm(d5-d5[0],axis=1)
s6 = np.linalg.norm(d6-d6[0],axis=1)
s7 = np.linalg.norm(d7-d7[0],axis=1)
s8 = np.linalg.norm(d8-d8[0],axis=1)
s9 = np.linalg.norm(d9-d9[0],axis=1)
s10 = np.linalg.norm(d10-d10[0],axis=1)
s11 = np.linalg.norm(d11-d11[0],axis=1)
s12 = np.linalg.norm(d12-d12[0],axis=1)
s13 = np.linalg.norm(d13-d13[0],axis=1)
s14 = np.linalg.norm(d14-d14[0],axis=1)
s15 = np.linalg.norm(d15-d15[0],axis=1)
s16 = np.linalg.norm(d16-d16[0],axis=1)
s17 = np.linalg.norm(d17-d17[0],axis=1)
s18 = np.linalg.norm(d18-d18[0],axis=1)
s19 = np.linalg.norm(d19-d19[0],axis=1)
s20 = np.linalg.norm(d20-d20[0],axis=1)
s21 = np.linalg.norm(d21-d21[0],axis=1)

s = [s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15,s16,s17,s18,s19,s20,s21]

sigmas = np.sqrt(2)*(0.01)

meanr2 = np.zeros(len(s1))
sigmar2 = np.zeros(len(s1))
for thing in s:
    meanr2 = meanr2 + (thing*0.12048)**2
    sigmar2 = sigmar2 + 2*thing*sigmas
meanr2 = meanr2/len(s)
sigmar2 = sigmar2/len(s) 
trials = 120
total = 60
t = total/trials
sigmat = 0.03

T = 298
sigmaT = 0.5

mu = 0.001*(1-0.02*4.85)
sigmamu = 0.05*0.001

diam = 1.9*10**(-6)
sigmadia = 0.1*10**(-6)

drag = 3*np.pi*diam*mu
sigmadrag=drag*np.sqrt((sigmadia/diam)**2+(sigmamu/mu)**2)


trange = np.linspace(0,total,num=trials)

r2opt, r2cov = optimize.curve_fit(brownian, trange, meanr2,
                                 absolute_sigma=False, p0=(0.299))

r2dif = r2opt[0]
sigmadif2 = np.sqrt(r2cov[0][0])
r2fitted = brownian(trange,r2opt[0])

chisquaredr2 = chi2(r2fitted,meanr2,np.max(sigmar2))
dofr2 = trials-1

plt.text(0.45, 17, r'$D_{r}$ = %3.3f $\mu$m$^2$/s'%(r2dif), fontsize=fontsize)
plt.text(0.45, 15, r'$\chi^2$/DOF=', fontsize=fontsize)
plt.text(0.45, 13, r'%3.2f/%i'%(chisquaredr2,dofr2), fontsize=fontsize)

kr2 = (r2dif*10**(-12))*drag/T
sigmakr2 = kr2*np.sqrt((sigmadif2/r2dif)**2+(sigmadrag/drag)**2+(sigmaT/T)**2)

plt.plot(trange,meanr2,label='data')
plt.xlabel('Time (s)')
plt.ylabel('Mean Squared Displace ($\mu$m$^2$)')
plt.plot(trange, r2fitted,label='curvefitted')
plt.legend(loc=1)
plt.savefig('meanr2fit',dpi=300)
plt.show()
plt.close()

d1 = np.diff(d1,axis=0)
d2 = np.diff(d2, axis=0)
d3 = np.diff(d3, axis=0)
d4 = np.diff(d4, axis=0)
d5 = np.diff(d5, axis=0)
d6 = np.diff(d6, axis=0)
d7 = np.diff(d7, axis=0)
d8 = np.diff(d8, axis=0)
d9 = np.diff(d9, axis=0)
d10 = np.diff(d10, axis=0)
d11 = np.diff(d11, axis=0)
d12 = np.diff(d12, axis=0)
d13 = np.diff(d13, axis=0)
d14 = np.diff(d14, axis=0)
d15 = np.diff(d15, axis=0)
d16 = np.diff(d16, axis=0)
d17 = np.diff(d17, axis=0)
d18 = np.diff(d18, axis=0)
d19 = np.diff(d19, axis=0)
d20 = np.diff(d20, axis=0)
d21 = np.diff(d21, axis=0)


d = np.concatenate((d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14,d15,d16,d17,
                    d18,d19,d20,d21))
d = d*0.12048
sigmad = 0.01
r = np.linalg.norm(d, axis=1)
sigmar = np.sqrt(2)*sigmad

sep = 30
rang = (-0.01,1.245)
w = (rang[1]-rang[0])*len(r)/sep


freq, edges, _ = plt.hist(r, bins=sep, range=rang, color='k', 
                       histtype='step', label='data',weights=np.ones_like(r)/w)
plt.xlabel('Step Size Distribution ($\mu$m)')
plt.ylabel('Density / %2.2f $\mu$m'%((rang[-1]-rang[0])/sep))
plt.xlim(rang)

centers = 0.5*(edges[1:]+edges[:-1])
sigmaf = np.sqrt(freq / w)
sigmaf=np.where(sigmaf==0, 1/w, sigmaf)
plt.errorbar(centers, freq, yerr=sigmaf, fmt='none', c='k')


opt, cov = optimize.curve_fit(rali, centers, freq,sigma=sigmaf,absolute_sigma=True,p0=(0.299))

optdif = opt[0]/(2*t)
sigmadif = optdif*np.sqrt((np.sqrt(cov[0][0])/opt[0])**2+(sigmat/t)**2)

DOF=sep-len(opt)

fitted = rali(centers, optdif)

chisquared = chi2(fitted,freq,sigmaf)

plt.plot(centers, fitted,label='curvefitted')
plt.text(0.45, 2, r'$D_c$ = %3.3f $\mu$m$^2$/s'%(opt[0]), fontsize=fontsize)
plt.text(0.45, 1.75, r'$\chi^2$/DOF=', fontsize=fontsize)
plt.text(0.45, 1.5, r'%3.2f/%i'%(chisquared,DOF), fontsize=fontsize)
plt.legend(loc=1)
plt.savefig('rcurvefit',dpi=300)

mledif = mle(r)/(2*t)
sigmamdif = mledif*np.sqrt(np.sum((sigmar/r)**2)/len(r)+(sigmat/t)**2)
likfitted = rali(centers,mledif)
plt.plot(centers, likfitted, label='MLE')

plt.text(0.45, 2.25, r'$D_m$ = %3.3f $\mu$m$^2$/s'%(mledif), fontsize=fontsize)
plt.legend(loc=1)
plt.savefig('rlikfit',dpi=300)
kopt = (optdif*10**(-12))*drag/T
sigmakopt = kopt*np.sqrt((sigmadif/optdif)**2+(sigmadrag/drag)**2+(sigmaT/T)**2)
kmle = (mledif*10**(-12))*drag/T
sigmakmle = kmle*np.sqrt((sigmamdif/mledif)**2+(sigmadrag/drag)**2+(sigmaT/T)**2)


plt.show()
plt.close()

print(r2dif, sigmadif2)
print(optdif,sigmadif)
print(mledif, sigmamdif)
print('\n\n')
print(kr2,sigmakr2,sigmakr2/kr2)
print(kopt, sigmakopt,sigmakopt/kopt)
print(kmle, sigmakmle,sigmakmle/kmle)