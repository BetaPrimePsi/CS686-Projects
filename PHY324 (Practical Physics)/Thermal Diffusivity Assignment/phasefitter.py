from random import gauss
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
import scipy.optimize as optimize
import scipy.stats as stats
from scipy import signal, special
import pickle
with open('2min.csv', 'r') as file:
    dat20 = np.loadtxt(file, delimiter=',', usecols=range(2))
with open('2halfmin.csv', 'r') as file:
    dat25 = np.loadtxt(file, delimiter=',', usecols=range(2))
with open('3min.csv', 'r') as file:
    dat30 = np.loadtxt(file, delimiter=',', usecols=range(2))
with open('3halfmin.csv', 'r') as file:
    dat35 = np.loadtxt(file, delimiter=',', usecols=range(2))
with open('4min.csv', 'r') as file:
    dat40 = np.loadtxt(file, delimiter=',', usecols=range(2))

dat20 = np.transpose(dat20)
dat25 = np.transpose(dat25)
dat30 = np.transpose(dat30)
dat35 = np.transpose(dat35)
dat40 = np.transpose(dat40)

r = 0.45/100
sigmar = 0.1/100
sigmaTstat = 1
sigmaTdyn = 3
sigmaT = sigmaTstat+sigmaTdyn

tempC20 = 0
tempH20 = 95
T20 = 120
w20 = 2*np.pi/T20
sigmaw20=2*np.pi*sigmaT/(T20)**2
tempC25 = 0 
tempH25 = 98
T25 = 150
w25 = 2*np.pi/T25
sigmaw25=2*np.pi*sigmaT/(T25)**2
tempC30 = 0 
tempH30 = 95
T30 = 180
w30 = 2*np.pi/T30
sigmaw30=2*np.pi*sigmaT/(T30)**2
tempC35 = -2 
tempH35 = 98
T35 = 210
w35 = 2*np.pi/T35
sigmaw35=2*np.pi*sigmaT/(T35)**2
tempC40 = -1
tempH40 = 96
T40 = 240
w40 = 2*np.pi/T40
sigmaw40=2*np.pi*sigmaT/(T40)**2

sigmaw=np.array([sigmaw20,sigmaw25,sigmaw30,sigmaw35,sigmaw40])

def squarewave(t, A, w, c):
    return A*signal.square(w*t)+c

def inter(f,g):
    return np.argwhere(np.diff(np.sign(f-g))).flatten()

def fitter(w,m):
    return np.sqrt(w/m)*r

# 2 minutes
pulse20 = squarewave(dat20[0],(tempH20-tempC20)/2,w20,(tempH20+tempC20)/2)
plt.plot(dat20[0], dat20[1],label='Response')
plt.plot(dat20[0], pulse20,label='Square Wave')
plt.title('2 minute period')
plt.xlabel('Time (s)')
plt.ylabel('Temperature (${}^\circ$C)')
plt.savefig('2min',dpi=300)
plt.show()
plt.close()
t20 = np.array([42.5,35,35,32.5,27.5,32.5])

# 2.5 minutes
pulse25 = squarewave(dat25[0],(tempH25-tempC25)/2,w25,(tempH25+tempC25)/2)
plt.plot(dat25[0], dat25[1],label='Response')
plt.plot(dat25[0], pulse25, label='Square Wave')
plt.title('2.5 minute period')
plt.xlabel('Time (s)')
plt.ylabel('Temperature (${}^\circ$C)')
plt.legend(loc=1)
plt.savefig('2halfmin',dpi=300)
plt.show()
plt.close()
t25 =np.array([32.5,30,37.5,37.5,30,35,27.5,])

# 3 minutes
pulse30 = squarewave(dat30[0],(tempH30-tempC30)/2,w30,(tempH30+tempC30)/2)
plt.plot(dat30[0], dat30[1],label='Response')
plt.plot(dat30[0], pulse30, label='Square Wave')
plt.title('3 minute period')
plt.xlabel('Time (s)')
plt.ylabel('Temperature (${}^\circ$C)')
plt.legend(loc=1)
plt.savefig('3min',dpi=300)
plt.show()
plt.close()
t30 = np.array([32.5,35,40,42.5,42.5,40,27.5])

# 3.5 minutes
pulse35 = squarewave(dat35[0],(tempH35-tempC35)/2,w35,(tempH35+tempC35)/2)
plt.plot(dat35[0], dat35[1],label='Response')
plt.plot(dat35[0], pulse35, label='Square Wave')
plt.title('3.5 minute period')
plt.xlabel('Time (s)')
plt.ylabel('Temperature (${}^\circ$C)')
plt.legend(loc=1)
plt.savefig('3halfmin',dpi=300)
plt.show()
plt.close()
t35 = np.array([30,30,30,32.5,27.5,32.5])

# 4 minutes
pulse40 = squarewave(dat40[0],(tempH40-tempC40)/2,w40,(tempH40+tempC40)/2)
plt.plot(dat40[0], dat40[1],label='Response')
plt.plot(dat40[0], pulse40, label='Square Wave')
plt.title('4 minute period')
plt.xlabel('Time (s)')
plt.ylabel('Temperature (${}^\circ$C)')
plt.legend(loc=1)
plt.savefig('4min',dpi=300)
plt.show()
plt.close()
t40=np.array([27.5,30,30,30,30,30])

w = np.array([w20,w25,w30,w35,w40])

dt = np.array([np.average(t20),np.average(t25),np.average(t30),np.average(t35),
              np.average(t40)])
dp = w*dt

sigmadp=dp*np.sqrt((sigmaw/w)**2+(sigmaT/dt)**2)

x = np.linspace(0,4,1000)
ber = special.ber(x)
bei = special.bei(x)
phase = np.angle(ber+1j*bei)
P20=np.ones(len(x))*dp[0]
P25=np.ones(len(x))*dp[1]
P30=np.ones(len(x))*dp[2]
P35=np.ones(len(x))*dp[3]
P40=np.ones(len(x))*dp[4]

int20 = inter(P20, phase)
int25 = inter(P25, phase)
int30 = inter(P30, phase)
int35 = inter(P35,phase)
int40 = inter(P40, phase)

xint=np.array([x[int20[0]],x[int25[0]],x[int30[0]],x[int35[0]],x[int40[0]]])
sigmax = sigmadp

m = w*(r/xint)**2
sigmam = m*np.sqrt((sigmaw/w)**2+2*(sigmax/xint)**2+2*(sigmar/r)**2)
mavg = np.average(m)
sigmamavg = np.average(sigmam)

print(m)
print(sigmam)

mfit, munc = optimize.curve_fit(fitter, w, xint, absolute_sigma=False, 
                                sigma=sigmax, bounds=(0,np.infty))

chi2 = np.sum(((xint-fitter(w,mfit))/sigmax)**2)
dof = 4

plt.plot(x, ber, label='$ber_0$')
plt.plot(x, bei, label='$bei_0$')
plt.xlabel('x')
plt.ylabel('$ber_0$ and $bei_0$')
plt.legend(loc=1)
plt.savefig('berbei',dpi=300)
plt.show()
plt.close()
 
plt.plot(x,phase, label='phase')
plt.plot(x, P20, label='$\delta\phi_{120}$')
plt.plot(x, P25, label='$\delta\phi_{150}$')
plt.plot(x, P30, label='$\delta\phi_{180}$')
plt.plot(x, P35, label='$\delta\phi_{210}$')
plt.plot(x, P40, label='$\delta\phi_{240}$')
plt.plot(x[int20],phase[int20],'bo')
plt.plot(x[int25],phase[int25],'bo')
plt.plot(x[int30],phase[int30],'bo')
plt.plot(x[int35],phase[int35],'bo')
plt.plot(x[int40],phase[int40],'bo')
plt.text(0.01,2.4,r'$x_{120}$=%3.2f'%(xint[0]),fontsize=8)
plt.text(0.01,2.25,r'$x_{150}$=%3.2f'%(xint[1]),fontsize=8)
plt.text(0.01,2.1,r'$x_{180}$=%3.2f'%(xint[2]),fontsize=8)
plt.text(0.01,1.95,r'$x_{210}$=%3.2f'%(xint[3]),fontsize=8)
plt.text(0.01,1.8,r'$x_{240}$=%3.2f'%(xint[4]),fontsize=8)
plt.xlabel('x')
plt.ylabel('Phase Difference (rad)')
plt.legend(loc=1)
plt.savefig('xint',dpi=300)
plt.show()
plt.close()

plt.plot(w, xint, label='measured')
plt.plot(w, fitter(w,mfit),label='fitted')
plt.xlabel('Angular Frequency (rad/s)')
plt.ylabel('x intersections')
plt.text(0.026,3.1,r'$m_f$ = %3.3f E-7m^2/s'%(mfit*10**7))
plt.text(0.026, 3.0, r'$\chi^2$/DOF=')
plt.text(0.026, 2.9, r'%3.2f/%i'%(chi2,dof))
plt.text(0.026, 2.8, r'$\chi^2$ prob.= %3.2f'%(1-stats.chi2.cdf(chi2,dof)))
plt.legend(loc=1)
plt.savefig('mfit',dpi=300)
plt.show()
plt.close()

# See https://www.engineersedge.com/heat_transfer/thermal_diffusivity_table_13953.htm
actualm = 1.3*10**(-7) # m^2/s

print('Average value of m:', mavg,'pm',sigmamavg,'m^2/s')
print('Fitted value of m', mfit[0],'pm',munc[0][0],'m^2/s')
print('Actual Value of m:', actualm,'m^2/s')