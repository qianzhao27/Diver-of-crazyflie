import pandas as pd
import numpy as np
from scipy import signal

PATH = "data/"
LABEL_0 = "label_0/"
LABEL_1 = "label_1/"
LABEL_2 = "label_2/"
LABEL_2_5 = "label_2.5/"
LABEL_3 = "label_3/"
LABEL_4 = "label_4/"
FILE_PREFIX = "data_set_label_"
FILE_MIDDLE_PACKET = "_packet_"
FILE_MIDDLE_FACING = "_facing_"
FILE_SUFFIX = ".csv"

sliding_window = 1

def load_data(label, total_files, project='project1', drone="drone1", direction=""):
    path = PATH + project + '/' + drone + '/'

    if label == 0:
        path += LABEL_0
    elif label == 1:
        path += LABEL_1
    elif label == 2:
        path += LABEL_2
    elif label == 2.5:
        path += LABEL_2_5
    elif label == 3:
        path += LABEL_3
    elif label == 4:
        path += LABEL_4

    if direction != '':
        path = path[0:-1]+FILE_MIDDLE_FACING+direction+"/"
        direction = "_"+direction

    columns = ["timestamp_start", "timestamp_end", "stabilizer.roll", "stabilizer.pitch", "stabilizer.yaw", "gyro.x",
               "gyro.y", "gyro.z", "acc.x", "acc.y", "acc.z",
               "stateEstimate.x", "stateEstimate.y", "stateEstimate.z", "label"]
    data = pd.DataFrame(data=[], columns=columns)

    for i in range(total_files):
        fileName = FILE_PREFIX+str(label)+direction+FILE_MIDDLE_PACKET+str(i)+FILE_SUFFIX
        temp_data = pd.read_csv(path+fileName, index_col=0)
        temp_data["label"] = label
        # cut the first and last 100 = 1s after taking off and 1s before landing
        #temp_data = temp_data.iloc[900:-900, :]
        # only need 6000 data points, which is equivalent to 1 min of data
        temp_data = temp_data.iloc[:2400, :]
        data = data.append(temp_data, ignore_index=True)

    return data

def separate_data_based_on_apparatus(data):
    acc = data.iloc[:, 0:3]
    gyro = data.iloc[:, 3:6]
    pose = data.iloc[:, 7:10]
    stabilizer = data.iloc[:, 10:13]

    data_collection = {
        "acc": acc,
        "gyro": gyro,
        "stateEstimate": pose,
        "stabilizer": stabilizer
    }

    return data_collection

# after performing feature transformation on the training data
# the amount of data points will decrease becasue of the usage of sliding windwo.
# as a result, we need to reduce the number of label for each class, accordingly.
def adjust_label_amount(y, num_classes):
    rows = y.shape[0]
    rows_needed = int(sliding_window * 1000 / 10)

    if rows_needed > rows:
        print('Not enough data.')
        return None

    # calculate how many rows for one class
    num_data_points = rows//num_classes
    remainder = num_data_points % rows_needed
    counter = 0
    for k in range(num_data_points):
        if k + rows_needed <= num_data_points:
            counter += 1
        else:
            break
            
    if remainder / rows_needed > 0.9:
        counter += 1

    # generate new labels
    y_new = []
    for c in range(num_classes):
        label = [c for x in range(counter)]
        y_new.append(label)

    return np.array(y_new).flatten()

def reduce_noise_with_FFT(data, k):
    columns = data.columns.values
    n = data.shape[0]
    final_data = []
    
    for col in columns:
        data_fft = np.fft.fft(data[col])
        psd = data_fft*np.conj(data_fft)/n
        kth_psd = np.sort(psd.real)[-k]
        
        indices = psd > kth_psd
        data_fft = data_fft*indices
        data_ifft = np.fft.ifft(data_fft)
        final_data.append(data_ifft.real)
    
    final_data = pd.DataFrame(np.array(final_data).T, columns=columns)
    
    return final_data