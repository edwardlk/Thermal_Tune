################################################################################
##
##                             Thermal Tune from PSD
##
## 22 header lines on labview data files

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def func(w, a, b, c, w_0):
    return a/((w-w_0)**2 + b) + c

def stiffness(T, A, B):
    k_B = 1.3806 * 10**-23
    C = 0.8417
    return C*k_B*T*np.sqrt(B)/(np.pi*A)

data = np.genfromtxt("C:\Users\Ed\Desktop\PSD_0010.txt")
data_t = data.transpose()
print data_t

w_0 = float(input("Enter the resonance frequency in kHz: "))

w = np.linspace(0,100,200)
y = func(w, 10, 2, 1, 53)
yn = y + 0.2*np.random.normal(size=len(w))

tst = lambda x, a, b, c: func(x, a, b, c, w_0)

popt, pcov = curve_fit(tst, w, yn)

plt.figure()
plt.plot(w, yn, 'r.')
plt.plot(w, tst(w, popt[0], popt[1], popt[2]), 'b')

plt.show()

print "stiffness is ", stiffness(300, popt[0], popt[1])
