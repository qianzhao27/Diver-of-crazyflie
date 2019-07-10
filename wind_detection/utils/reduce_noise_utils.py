import numpy as np
import pandas as pd
from scipy import signal

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