from random import gauss
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
import scipy.optimize as optimize
import scipy.stats as stats
import pickle

with open('sample.csv', 'r') as file:
    sample = np.loadtxt(file, delimiter=',', usecols=range(1))
with open('fine.csv', 'r') as file:
    data = np.loadtxt(file, delimiter=',', usecols=range(3))

def linfit(x,m,b):
    return m*x+b

sample = np.transpose(sample)
data = np.transpose(data)
size = 32

l = 99.9/100
sigmal = 0.1/100

pos = data[0]/100
dowt = data[1]
upt = data[2]

sigmat = np.std(sample)
sigmap = 0.05/100

dperiod = dowt/size
uperiod = upt/size
sigmaper = sigmat/32
print(sigmaper)
uopt, ucov = optimize.curve_fit(linfit, pos, uperiod, sigma=np.ones(len(pos))*sigmaper, absolute_sigma=True)
dopt, dcov = optimize.curve_fit(linfit, pos, dperiod, sigma=np.ones(len(pos))*sigmaper, absolute_sigma=True)

um = uopt[0]
ub = uopt[1]
sigmaum = np.sqrt(ucov[0][0])
sigmaub = np.sqrt(ucov[1][1])
print('um=',um,'$\pm$',sigmaum)
print('ub=',ub,'$\pm$',sigmaub)
ufit = linfit(pos,um,ub)
ures = uperiod-ufit
uchi = np.sum((ures/sigmaper)**2)/9
print('uchi',uchi)
print('')

dm = dopt[0]
db = dopt[1]
sigmadm = np.sqrt(dcov[0][0])
sigmadb = np.sqrt(dcov[1][1])
print('dm=',dm,'$\pm$',sigmadm)
print('db=',db,'$\pm$',sigmadb)
dfit = linfit(pos,dm,db)
dres = dperiod-dfit
dchi = np.sum((dres/sigmaper)**2)/9
print('dchi',dchi)
print('')

diffb = ub-db
sigmadiffb = np.sqrt(sigmaub**2+sigmadb**2)
diffm=um-dm
sigmadiffm=np.sqrt(sigmaum**2+sigmadm**2)
intercept = -diffb/diffm
sigmaint = intercept*np.sqrt((sigmadiffm/diffm)**2+(sigmadiffb/diffb)**2)

T = linfit(intercept, um, ub)
sigmaT=np.sqrt((um*sigmaint)**2+(intercept*sigmaum)**2+(sigmaub)**2)

g = 4*(np.pi**2)*(l/T**2)
sigmag = g*np.sqrt((sigmal/l)**2+2*(sigmaT/T)**2)
print('int=',intercept,'$\pm$',sigmaint)
print('T=',T,'$\pm$',sigmaT)
print('g=',g,'$\pm$',sigmag)

plt.errorbar(pos, dperiod, yerr=sigmaper, xerr=sigmap, linestyle='None', marker='d',ecolor='black', label='Data')
plt.plot(pos, dfit, color='gray', label='Fit')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.xlabel('Relative Fine Mass Position (m)',fontsize=14)
plt.ylabel('Down Period (s)',fontsize=14)
plt.legend(loc=1)
plt.savefig('dp',dpi=1000)
plt.show()
plt.close()

plt.errorbar(pos, dres, yerr=sigmaper, xerr=sigmap, linestyle='None', marker='d',ecolor='black', label='Residuals')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.plot(pos, dfit-dfit, color='red', label='Zero')
plt.xlabel('Relative Fine Mass Position (m)',fontsize=14)
plt.ylabel('Down Residuals (s)',fontsize=14)
plt.legend(loc=1)
plt.savefig('dres',dpi=1000)
plt.show()
plt.close()

plt.errorbar(pos, uperiod, yerr=sigmaper, xerr=sigmap, linestyle='None', marker='d',ecolor='black', label='Data')
plt.plot(pos, ufit, color='green', label='Fit')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.xlabel('Relative Fine Mass Position (m)',fontsize=14)
plt.ylabel('Up Period (s)',fontsize=14)
plt.text
plt.legend(loc=1)
plt.savefig('up', dpi=1000)
plt.show()
plt.close()

plt.errorbar(pos, ures, yerr=sigmaper, xerr=sigmap, linestyle='None', marker='d',ecolor='black', label='Residuals')
plt.plot(pos, ufit-ufit, color='red', label='Zero')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.xlabel('Relative Fine Mass Position (m)',fontsize=14)
plt.ylabel('Up Residuals (s)',fontsize=14)
plt.text
plt.legend(loc=1)
plt.savefig('ures', dpi=1000)
plt.show()
plt.close()

rang = np.linspace(1*10**(-2),6*10**(-2),100)
uline = linfit(rang, um, ub)
dline = linfit(rang, dm, db)

plt.plot(rang, uline, color='green', label='Up')
plt.plot(rang, dline, color='gray', label='Down')
plt.errorbar(intercept, T, color='red',yerr=[sigmaT],xerr=[sigmaint], label='Intersection')
plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
plt.xlabel('Relative Fine Mass Position (m)',fontsize=14)
plt.ylabel('Periods (s)', fontsize=14)
plt.legend(loc=0)
plt.savefig('intersect',dpi=1000)
plt.show()
plt.close()

plt.plot(pos, ufit, color='green', label='Up')
plt.plot(pos, dfit, color='gray', label='Down')
plt.errorbar(intercept, T, color='red',yerr=[sigmaT],xerr=[sigmaint], label='Intersection')
plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
plt.xlabel('Relative Fine Mass Position (m)',fontsize=14)
plt.ylabel('Periods (s)', fontsize=14)
plt.legend(loc=0)
plt.savefig('intersectclose',dpi=1000)
plt.show()
plt.close()

bins1, edges1, _ = plt.hist(ures, bins=6,range=(-6*10**(-5),7*10**(-5)),label='Histogram', facecolor='red', edgecolor='red')
uncbin1 = np.sqrt(bins1)
centers1 = (edges1[:-1]+edges1[1:])/2
plt.errorbar(centers1, bins1, yerr=uncbin1, linestyle='None',ecolor='black',label='Standard Error')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.xlabel('Up Residuals (s)',fontsize=15)
plt.ylabel('Counts / $2.6\\times10^{-5}$ s',fontsize=15)
plt.legend(loc=1)
plt.savefig('uhist',dpi=1000)
plt.show()
plt.close()

bins2, edges2, _ = plt.hist(dres, bins=6,range=(-1*10**(-4),1*10**(-4)),label='Histogram', facecolor='red', edgecolor='red')
uncbin2 = np.sqrt(bins2)
centers2 = (edges2[:-1]+edges2[1:])/2
plt.errorbar(centers2, bins2, yerr=uncbin2, linestyle='None',ecolor='black',label='Standard Error')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.xlabel('Down Residuals (s)',fontsize=15)
plt.ylabel('Counts / $0.4\\times10^{-4}$ s',fontsize=15)
plt.legend(loc=1)
plt.savefig('dhist',dpi=1000)
plt.show()
plt.close()