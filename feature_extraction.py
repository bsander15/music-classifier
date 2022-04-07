import librosa
import librosa.display
import matplotlib.pyplot as plt
from pathlib import Path
import sklearn
import numpy as np

"""
Features Extracted So Far:

Zero Crossing Rate
Spectral Centroid
"""
# music_signals =np.array([librosa.load(p)[0] for p in Path().glob('data/music_wav/*.wav')])
# other_signals = np.array([librosa.load(p,duration=30)[0] for p in Path().glob('data/other_wav/*.wav')])

# np.save('data/music_signals', music_signals)
# np.save('data/other_signals', other_signals)

music_signals = np.array(np.load('data/music_signals.npy'))
other_signals = np.array(np.load('data/other_signals.npy'))


music_zcr = np.array([librosa.feature.zero_crossing_rate(x)[0] for x in music_signals])
other_zcr = np.array([librosa.feature.zero_crossing_rate(x)[0] for x in other_signals])

music_sc = np.array([librosa.feature.spectral_centroid(y=x)[0] for x in music_signals])
other_sc = np.array([librosa.feature.spectral_centroid(y=x)[0] for x in other_signals])

music_zcr_mean = np.mean(music_zcr, axis=1)
other_zcr_mean = np.mean(other_zcr, axis=1)
zcr_mean = np.concatenate((music_zcr_mean,other_zcr_mean), axis=0)
zcr_mean = np.reshape(zcr_mean,(159,1))

music_sc_mean = np.mean(music_sc, axis=1)
other_sc_mean = np.mean(other_sc, axis=1)
sc_mean = np.concatenate((music_sc_mean,other_sc_mean), axis=0)
sc_mean = np.reshape(sc_mean,(159,1))

feature_table1 = np.hstack((zcr_mean,sc_mean))
scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1,1))
training_features1 = scaler.fit_transform(feature_table1)
print(training_features1.shape)


plt.scatter(training_features1[:64,0],training_features1[:64,1],c='r')
plt.scatter(training_features1[64:,0],training_features1[64:,1],c='b')
plt.title('Mean')
plt.xlabel('Zero Crossing Rate Mean')
plt.ylabel('Spectral Centroid Mean')
plt.show()