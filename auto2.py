import os
import numpy as np
from scipy import (signal, io)
import h5py
import tensorflow as tf
from keras.models import Sequential
from keras.layers import (Dense, Conv1D, GRU, LSTM, Recurrent, Bidirectional,
                          TimeDistributed, Dropout, Flatten, RepeatVector, Reshape)

def load_data(data_path):
    f = io.matlab.loadmat(data_path)
    #print(loop_data)
    loop_data = f['Loopdata_mixed'][()]
    X = loop_data.reshape(loop_data.shape[0]*loop_data.shape[0],-1)
    X -= np.nanmean(X)
    X[np.where(np.isfinite(x)==0)] = 0
    X /= np.std(X)
    X = np.atleast_3d(X)
    return X
