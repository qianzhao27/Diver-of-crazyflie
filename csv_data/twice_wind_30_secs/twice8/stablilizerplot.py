import numpy as np
import matplotlib.pyplot as plt
import datetime
import time

plt.figure(figsize=(20,10))
#x = np.loadtxt('acc_z.txt', dtype=float)
y = np.loadtxt('pose_x.txt', dtype=float)
roll = np.loadtxt('roll.txt', dtype=float)
pitch = np.loadtxt('pitch.txt', dtype=float)
yaw = np.loadtxt('yaw.txt', dtype=float)

#plt.scatter(y, gx)
#plt.scatter(y, gy)
#plt.scatter(y, gz)
plt.subplot(311)
plt.plot(y, roll)
plt.ylabel('roll')

plt.subplot(312)
plt.plot(y, pitch)
plt.ylabel('pitch')

plt.subplot(313)
plt.plot(y, yaw)
plt.ylabel('yaw')

ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

# plt.savefig("stabilizer"+ st)
plt.savefig("stabilizer")
