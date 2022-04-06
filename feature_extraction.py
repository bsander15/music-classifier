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
# music_signals = [librosa.load(p)[0] for p in Path().glob('data/music_wav/*.wav')]
# other_signals = [librosa.load(p)[0] for p in Path().glob('data/other_wav/*.wav')]

# np.save('data/music_signals', music_signals)
# np.save('data/other_signals', other_signals)

music_signals = np.array(np.load('data/music_signals.npy'))
other_signals = np.array(np.load('data/other_signals.npy',allow_pickle=True))

def extract_features(signal):
    return [
        librosa.feature.zero_crossing_rate(signal)[0,0],
        librosa.feature.spectral_centroid(y=signal)[0,0]
    ]

music_features = np.array([extract_features(x) for x in music_signals])
other_features = np.array([extract_features(x) for x in other_signals])

# print(music_features)
# plt.figure(figsize=(14,5))
# plt.hist(music_features[:,0], color='g', range=(0,0.2), alpha=0.5, bins=20)
# plt.hist(other_features[:,0], color='r', range=(0,0.2), alpha=0.5, bins=20)
# plt.legend(('music','other'))
# plt.xlabel('Zero Crossing Rate')
# plt.ylabel('Count')
# plt.show()

feature_table = np.vstack((music_features, other_features))


scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1,1))
training_features = scaler.fit_transform(feature_table)

plt.scatter(training_features[:64,0], training_features[:64,1], c='b')
plt.scatter(training_features[64:,0], training_features[64:,1], c='r')
plt.xlabel("Zero Crossing Rate")
plt.ylabel("Spectral Centroid")
plt.show()
