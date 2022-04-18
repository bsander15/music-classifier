import librosa
import librosa.display
import matplotlib.pyplot as plt
from pathlib import Path
import sklearn
import numpy as np
import pandas

"""
Features Extracted So Far:

Zero Crossing Rate
Spectral Centroid
"""
durationLabel = ''
music_signals =np.array([librosa.load(p)[0] for p in Path().glob('data/music_wav{}/*.wav'.format(durationLabel))])
other_signals = np.array([librosa.load(p,duration=30)[0] for p in Path().glob('data/other_wav*{}/*.wav'.format(durationLabel))])

# np.save('data/music_signals', music_signals)
# np.save('data/other_signals', other_signals)

# music_signals = np.array(np.load('data/music_signals.npy'))
# other_signals = np.array(np.load('data/other_signals.npy'))

music_mfcc = librosa.feature.mfcc(y=music_signals)
other_mfcc = librosa.feature.mfcc(y=other_signals)
# print(music_mfcc.shape)



music_zcr = np.array([librosa.feature.zero_crossing_rate(x)[0] for x in music_signals])
other_zcr = np.array([librosa.feature.zero_crossing_rate(x)[0] for x in other_signals])
music_mfcc_mean = np.mean(music_mfcc,axis=2)
other_mfcc_mean = np.mean(other_mfcc,axis=2)
music_mfcc_var = np.var(music_mfcc, axis=2)
other_mfcc_var = np.var(other_mfcc,axis=2)
# print(music_mfcc_mean.shape);
# print(music_mfcc_var.shape);
# print(other_mfcc_mean.shape);
# print(other_mfcc_var.shape);

mfcc_mean = np.vstack((music_mfcc_mean,other_mfcc_mean))
mfcc_var = np.vstack((music_mfcc_var,other_mfcc_var))
# print(mfcc_mean.shape)
music_sc = np.array([librosa.feature.spectral_centroid(y=x)[0] for x in music_signals])
other_sc = np.array([librosa.feature.spectral_centroid(y=x)[0] for x in other_signals])

music_zcr_mean = np.mean(music_zcr, axis=1)
other_zcr_mean = np.mean(other_zcr, axis=1)
zcr_mean = np.concatenate((music_zcr_mean,other_zcr_mean), axis=0)
zcr_mean = np.reshape(zcr_mean,(music_signals.shape[0]+other_signals.shape[0],1))

music_sc_mean = np.mean(music_sc, axis=1)
other_sc_mean = np.mean(other_sc, axis=1)
sc_mean = np.concatenate((music_sc_mean,other_sc_mean), axis=0)
sc_mean = np.reshape(sc_mean,(music_signals.shape[0]+other_signals.shape[0],1))

music_zcr_var = np.var(music_zcr, axis=1)
other_zcr_var = np.var(other_zcr, axis=1)
zcr_var = np.concatenate((music_zcr_var,other_zcr_var), axis=0)
zcr_var = np.reshape(zcr_var,(music_signals.shape[0]+other_signals.shape[0],1))

music_sc_var = np.var(music_sc, axis=1)
other_sc_var = np.var(other_sc, axis=1)
sc_var = np.concatenate((music_sc_var,other_sc_var), axis=0)
sc_var = np.reshape(sc_var,(music_signals.shape[0]+other_signals.shape[0],1))

feature_table = np.hstack((zcr_mean,sc_mean,mfcc_mean,zcr_var,sc_var,mfcc_var))
print("feature table shape:", str(feature_table.shape))
scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1,1))
training_features = scaler.fit_transform(feature_table)
# print(normalized_means.shape)
# training_features = np.hstack((normalized_means,zcr_var,sc_var,mfcc_var))
print("training features shape: ",training_features.shape)

labels = ["zcr_mean","sc_mean","mfcc_mean1","mfcc_mean2","mfcc_mean3","mfcc_mean4",
            "mfcc_mean5", "mfcc_mean6", "mfcc_mean7", "mfcc_mean8", "mfcc_mean9",
            "mfcc_mean10", "mfcc_mean11", "mfcc_mean12","mfcc_mean13", "mfcc_mean14",
            "mfcc_mean15", "mfcc_mean16", "mfcc_mean17", "mfcc_mean18", "mfcc_mean19",
            "mfcc_mean20", "zcr_vcr", "sc_var", "mfcc_var1","mfcc_var2","mfcc_var3","mfcc_var4",
            "mfcc_var5", "mfcc_var6", "mfcc_var7", "mfcc_var8", "mfcc_var9",
            "mfcc_var10", "mfcc_var11", "mfcc_var12","mfcc_var13", "mfcc_var14",
            "mfcc_var15", "mfcc_var16", "mfcc_var17", "mfcc_var18", "mfcc_var19",
            "mfcc_var20"
]

music = ['music' for x in range(music_signals.shape[0])]
musicDf = pandas.DataFrame(music)
other = pandas.DataFrame(['other' for x in range(other_signals.shape[0])])
audioLabels = musicDf.append(other, ignore_index=True)
audioLabelsDf = pandas.DataFrame(audioLabels, dtype=pandas.StringDtype())


data = pandas.DataFrame(training_features, columns=labels)
dataset = pandas.concat([data,audioLabelsDf],axis=1)
dataset.to_csv("music_other_dataset{}.csv".format(durationLabel))



# print(training_features.shape)
# print(np.min(training_features, axis=0))
# print(np.max(training_features, axis=0))


# plt.scatter(training_features[:64,0],training_features[:64,1],c='r')
# plt.scatter(training_features[64:,0],training_features[64:,1],c='b')
# plt.xlabel('Zero Crossing Rate Mean')
# plt.ylabel('Spectral Centroid Mean')
# plt.show()

# plt.scatter(training_features[:64,2],training_features[:64,3],c='r')
# plt.scatter(training_features[64:,2],training_features[64:,3],c='b')
# plt.xlabel('Zero Crossing Rate Variance')
# plt.ylabel('Spectral Centroid Variance')
# plt.show()
