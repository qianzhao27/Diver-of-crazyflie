import pandas as pd
import numpy as np
df = pd.read_csv('nowind18.csv')

x = df['stateEstimate.x']
np.savetxt('pose_x.txt', x)

y = df['stateEstimate.y']
np.savetxt('pose_y.txt', y)

z = df['stateEstimate.z']
np.savetxt('pose_z.txt', z)

