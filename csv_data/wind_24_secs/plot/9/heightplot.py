import numpy as np
import matplotlib.pyplot as plt
import datetime
import time

plt.figure(figsize=(10,5))
#x = np.loadtxt('acc_z.txt', dtype=float)
y = np.loadtxt('pose_x.txt', dtype=float)
x = np.loadtxt('pose_z.txt', dtype=float)
#plt.scatter(y, x)
plt.plot(y, x)

ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

#plt.savefig("height"+ st)
plt.savefig("height")
