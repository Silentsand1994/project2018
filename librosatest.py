# Beat tracking example
from __future__ import print_function
import librosa
import numpy as np
from sklearn.cluster import KMeans

# 1. Get the file path to the included audio example
filename = librosa.util.example_audio_file()

# 2. Load the audio as a waveform `y`
#    Store the sampling rate as `sr`
y, sr = librosa.load("input/001_060_su.wav")

zero = librosa.zero_crossings(y)
zero = zero.tolist()

tra = librosa.core.time_to_frames(y)
#E = librosa.feature.rmse(y=y, hop_length=600)
y = np.resize(y, (int(y.shape[0]/511), 511))
print(y.shape)

features = np.zeros((y.shape[0], 14), dtype=y.dtype)

for x in range(0, y.shape[0]-1):
    zero = librosa.zero_crossings(y[x])
    zero = zero.tolist()
    features[x][0] = zero.count(True)
    features[x][1] = librosa.feature.rmse(y=y[x])
    temp = librosa.feature.mfcc(y=y[x], n_mfcc=12)
    features[x][2] = temp[0]
    features[x][3] = temp[1]
    features[x][4]= temp[2]
    features[x][5] = temp[3]
    features[x][6] = temp[4]
    features[x][7] = temp[5]
    features[x][8] = temp[6]
    features[x][9] = temp[7]
    features[x][10] = temp[8]
    features[x][11] = temp[9]
    features[x][12] = temp[10]
    features[x][13] = temp[11]


kmeans = KMeans().fit(features)
print(features.shape)

