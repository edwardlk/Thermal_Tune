################################################################################
##
##                             Thermal Tune from PSD
##
##

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def func(w, a, b, c):
    return a/((w-53)**2 + b) + c

def stiffness(T, A, B):
    k_B = 1.3806 * 10**-23
    C = 0.8417
    return C*k_B*T*np.sqrt(B)/(np.pi*A)

w = np.linspace(0,100,200)
y = func(w, 10, 2, 1)
yn = y + 0.2*np.random.normal(size=len(w))

popt, pcov = curve_fit(func, w, yn)

plt.figure()
plt.plot(w, yn, 'r.')
plt.plot(w, func(w, popt[0], popt[1], popt[2]), 'b')

plt.show()

print "stiffness is ", stiffness(300, popt[0], popt[1])
