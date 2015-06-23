import Tkinter, tkFileDialog
from os import *
from numpy import *
from scipy import stats
import matplotlib.pyplot as plt

root = Tkinter.Tk()
root.withdraw()

info = 'Please select the folder that contains \
the data files you wish to combine.'

srcDir = tkFileDialog.askdirectory(parent=root, initialdir="/", title=info)

dataFiles = listdir(srcDir)

dataFiles.sort()

data = genfromtxt(path.join(srcDir,dataFiles[0]),skip_header=1)
rows = data.shape[0]

freq = array(data[:,0])
fft1 = array(data[:,1])

slope, intercept, r_value, p_value, std_err = stats.linregress(freq,fft1)

fft1a = zeros(rows)

for x in range(0, rows):
    fft1a[x] = fft1[x] - freq[x]*slope - intercept

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(freq,fft1a)
plt.show()
plt.close()




##columns = data.shape[1]


##for x in range(len(dataFiles)):
##    temp = matrix(genfromtxt(path.join(srcDir,dataFiles[x]),skip_header=1))
##
##    if x == 0:
##        L = zeros(temp.shape[0])
##
##    if temp.shape[1] == 2:
##        L = column_stack((L, array(temp)))
##    else:
##        L = column_stack((L, array(temp.transpose())))
##
##savetxt(path.join(srcDir,'output.txt'), L)
