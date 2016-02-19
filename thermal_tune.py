################################################################################
##
##                             Thermal Tune from PSD
##
## 02/19/2016 Restarting this work. It turns out that the agilent AFM has FFT-
## capturing capabilities. Currently the curve is assumed to be smoothed + fit 
## externally (i.e. in orgin) with a Lorentzian, so this will just calculate 
## the stiffness from the parameters.

import numpy as np

def stiffness(T, s, A, L, D):
	k_B = 1.3806 * 10**-23
	S=1/s**2
	return 0.8174*k_B*T/(S*A)*((2*L-3*D)/(2*L-4*D))**2

def stiff_error(k, A, D, L, s_A, s_D, s_L):
	X = 4/(6*D**2-7*D*L+2*L**2)
	Y = (D*s_D/2)**2+(L*s_L/2)**2
	Z = (s_A/A)**2
	return k*np.sqrt(Z+X**2*Y)

data = np.genfromtxt("C:\Users\ADMIN\Desktop\Waveform01.txt", skip_header=1)
rows = data.shape[0]
columns = data.shape[1]


# Old Code from using oscilloscope

# 22 header lines on labview data files

# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.optimize import curve_fit

# def func(w, a, b, c, w_0):
    # return a/((w-w_0)**2 + b) + c

# def stiffness(T, A, B):
	# k_B = 1.3806 * 10**-23
	# C = 0.8417
	# sens = 50
	# s = 10**-7 / sens
	# return C*k_B*T*np.sqrt(B)/(np.pi*A*s**2)

# data = np.genfromtxt("C:\Users\ADMIN\Desktop\Waveform01.txt", skip_header=1)
# rows = data.shape[0]
# columns = data.shape[1]

# freq = np.zeros(rows)
# FFT_dBV = np.zeros(rows)
# FFT_V = np.zeros(rows)
# PSD_V2 = np.zeros(rows)

# for x1 in range(0, rows):
	# freq[x1] = data[x1, 0]
	# FFT_dBV[x1] = data[x1, 1]

# F_s = 200000
# N = 200

# for x in range(len(FFT_dBV)):
	# FFT_V[x] = 10**(FFT_dBV[x]/20)
	# PSD_V2[x] = abs(FFT_V[x])**2 / (F_s * N)
	
# peak = np.argmax(PSD_V2[10:])+10

# x_vals = freq[(peak-30):(peak+30)]
# y_vals = PSD_V2[(peak-30):(peak+30)] * 10**10

# tst = lambda x, a, b, c: func(x, a, b, c, freq[peak])
# popt, pcov = curve_fit(tst, freq, PSD_V2)

# plt.figure()
# plt.plot(freq, PSD_V2, 'r.')
# plt.plot(freq, tst(freq, popt[0], popt[1], popt[2]), 'b')
# plt.show()

# # plt.figure()
# # plt.plot(freq[(peak-30):(peak+30)], PSD_V2[(peak-30):(peak+30)], 'r.')
# # plt.show()

# print "stiffness is ", stiffness(300, popt[0], popt[1])

# # floor = min(yn)
# # PSD = yn - floor

# # max_pos = yn.tolist().index(max(yn))

# # ##w_0 = float(input("Enter the resonance frequency in kHz: "))

# # ##w = np.linspace(0,100,200)
# # ##y = func(w, 10, 2, 1, 53)
# # ##yn = y + 0.2*np.random.normal(size=len(w))

# # tst = lambda x, a, b, c: func(x, a, b, c, w[max_pos])

# # popt, pcov = curve_fit(tst, w, PSD)

# # plt.figure()
# # plt.plot(w, PSD, 'r.')
# # plt.plot(w, tst(w, popt[0], popt[1], popt[2]), 'b')

# # plt.show()

# # print "stiffness is ", stiffness(300, popt[0], popt[1])
