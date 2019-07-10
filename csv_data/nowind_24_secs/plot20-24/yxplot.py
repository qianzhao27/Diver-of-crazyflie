import numpy as np
import matplotlib.pyplot as plt
import datetime
import time

plt.figure(figsize=(10,5))
#x = np.loadtxt('acc_z.txt', dtype=float)
y = np.loadtxt('pose_x.txt', dtype=float)
x1 = np.loadtxt('pose_y1.txt', dtype=float)
x2 = np.loadtxt('pose_y2.txt', dtype=float)
x3 = np.loadtxt('pose_y3.txt', dtype=float)
x4 = np.loadtxt('pose_y4.txt', dtype=float)
x5 = np.loadtxt('pose_y5.txt', dtype=float)
#plt.scatter(y, x)
plt.plot(y, x1)
plt.plot(y, x2)
plt.plot(y, x3)
plt.plot(y, x4)
plt.plot(y, x5)

plt.spines['left'].set_position('zero')
ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

#plt.savefig("height"+ st)
plt.savefig("y_x")
