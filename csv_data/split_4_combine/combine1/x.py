import numpy as np

x = []

for i in range(0,4800):
    x = np.append(x, i/100)

np.savetxt('pose_x.txt', x, delimiter = ',')
