import numpy as np
import matplotlib.pyplot as plt
import datetime
import time

plt.figure(figsize=(20,10))
#x = np.loadtxt('acc_z.txt', dtype=float)
y = np.loadtxt('pose_x.txt', dtype=float)
accx = np.loadtxt('acc_x.txt', dtype=float)
accy = np.loadtxt('acc_y.txt', dtype=float)
accz = np.loadtxt('acc_z.txt', dtype=float)

#plt.scatter(y, gx)
#plt.scatter(y, gy)
#plt.scatter(y, gz)
plt.subplot(311)
plt.plot(y, accx)
plt.ylabel('acc_x')

plt.subplot(312)
plt.plot(y, accy)
plt.ylabel('acc_y')

plt.subplot(313)
plt.plot(y, accz)
plt.ylabel('acc_z')

ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

#plt.savefig("acc"+ st)
plt.savefig("acc")
