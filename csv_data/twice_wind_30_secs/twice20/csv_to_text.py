import pandas as pd
import numpy as np
df = pd.read_csv('twice20.csv')

gx = df['gyro.x']
np.savetxt('gyro_x.txt', gx)

gy = df['gyro.y']
np.savetxt('gyro_y.txt', gy)

gz = df['gyro.z']
np.savetxt('gyro_z.txt', gz)

ax = df['acc.x']
np.savetxt('acc_x.txt', ax)

ay = df['acc.y']
np.savetxt('acc_y.txt', ay)

az = df['acc.z']
np.savetxt('acc_z.txt', az)

x = df['stateEstimate.x']
np.savetxt('pose_x.txt', x)

y = df['stateEstimate.y']
np.savetxt('pose_y.txt', y)

z = df['stateEstimate.z']
np.savetxt('pose_z.txt', z)

pitch = df['stabilizer.pitch']
np.savetxt('pitch.txt', pitch)

roll = df['stabilizer.roll']
np.savetxt('roll.txt', roll)

yaw = df['stabilizer.yaw']
np.savetxt('yaw.txt', yaw)
