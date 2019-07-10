import numpy as np
import matplotlib.pyplot as plt
import datetime
import time

plt.figure(figsize=(10,5))
#x = np.loadtxt('acc_z.txt', dtype=float)
y = np.loadtxt('pose_x.txt', dtype=float)
x10 = np.loadtxt('pose_y10.txt', dtype=float)
x11 = np.loadtxt('pose_y11.txt', dtype=float)
x12 = np.loadtxt('pose_y12.txt', dtype=float)
x13 = np.loadtxt('pose_y13.txt', dtype=float)
x14 = np.loadtxt('pose_y14.txt', dtype=float)
x15 = np.loadtxt('pose_y15.txt', dtype=float)
x16 = np.loadtxt('pose_y16.txt', dtype=float)
x17 = np.loadtxt('pose_y17.txt', dtype=float)
x18 = np.loadtxt('pose_y18.txt', dtype=float)
x19 = np.loadtxt('pose_y19.txt', dtype=float)
x20 = np.loadtxt('pose_y20.txt', dtype=float)
x21 = np.loadtxt('pose_y21.txt', dtype=float)
x22 = np.loadtxt('pose_y22.txt', dtype=float)
x23 = np.loadtxt('pose_y23.txt', dtype=float)
x24 = np.loadtxt('pose_y24.txt', dtype=float)
#plt.scatter(y, x)

plt.axis([1.5, 4, 3.5, 4.5])

#plt.plot(y, x10, label='10')
#plt.plot(y, x11, label='11')
#plt.plot(y, x12, label='12')
#plt.plot(y, x13, label='13')
#plt.plot(y, x14, label='14')
#plt.plot(y, x15, label='15')
#plt.plot(y, x16, label='16')
#plt.plot(y, x17, label='17')
plt.plot(y, x18, label='1')
plt.plot(y, x19, label='2')
plt.plot(y, x20, label='3')
plt.plot(y, x21, label='4')
plt.plot(y, x22, label='5')
#plt.plot(y, x23, label='23')
#plt.plot(y, x24, label='24')

x1, y1 = [1.75, 3.6], [4.0, 3.85]
plt.plot(x1, y1, label = '6', linewidth = 2, color = "black")

plt.legend(loc='lower left')

plt.xlabel('x pose')
plt.ylabel('y pose')
ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

#plt.savefig("height"+ st)
plt.savefig("y_x")
