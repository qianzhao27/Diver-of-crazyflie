import numpy as np
import matplotlib.pyplot as plt
import datetime
import time

plt.figure(figsize=(20,10))
#x = np.loadtxt('acc_z.txt', dtype=float)
y = np.loadtxt('pose_x.txt', dtype=float)
gx = np.loadtxt('gyro_x.txt', dtype=float)
gy = np.loadtxt('gyro_y.txt', dtype=float)
gz = np.loadtxt('gyro_z.txt', dtype=float)

#plt.scatter(y, gx)
#plt.scatter(y, gy)
#plt.scatter(y, gz)
plt.subplot(311)
plt.plot(y, gx)
plt.ylabel('gyro_x')

plt.subplot(312)
plt.plot(y, gy)
plt.ylabel('gyro_y')

plt.subplot(313)
plt.plot(y, gz)
plt.ylabel('gyro_z')

ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

#plt.savefig("gyro"+ st)
plt.savefig("gyro")
