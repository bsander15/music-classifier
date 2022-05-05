

"""
mean and std-var
MFCC 
Spectral Centroid 
Zero crossing rate
Spectral Roll-off: frequency under which a large percentage of the frequencies exist
    https://librosa.org/doc/main/generated/librosa.feature.spectral_rolloff.html
Flux:
Beat detection: 
"""
from tkinter import E
import librosa as l
import numpy as np


class ExtractFeatures:
    def __init__(self, signals, feature_names):

        self.signals = signals

        self.feature_table = {'zcr': lambda signals: np.array([l.feature.zero_crossing_rate(x)[0] for x in signals]), 
            'sc': lambda signals: np.array([l.feature.spectral_centroid(y=x)[0] for x in signals]), 
            'mfcc': lambda signals: np.array([l.feature.mfcc(y=x) for x in signals]),
            'rolloff': lambda signals: np.array([l.feature.spectral_rolloff(y=x)[0] for x in signals]),
            'tempo': lambda signals: np.array([l.beat.tempo(y=x) for x in signals]),

            # ADDITIONAL FEATURES:

                # librosa.feature.rms
                    #returns: rmsnp.ndarray [shape=(…, 1, t)]
                # librosa.feature.chroma_cens
                    #returns: censnp.ndarray [shape=(…, n_chroma, t)]
                # librosa.feature.chroma_cqt
                    #returns: chromagramnp.ndarray [shape=(…, n_chroma, t)]
                # librosa.feature.chroma_stft
                    #returns: chromagramnp.ndarray [shape=(…, n_chroma, t)]
        }

        self.possible_names = { f'{pref}_{suf}' for suf in {'mean', 'var'} for pref in self.feature_table.keys()} 
        self.possible_names.add('tempo')
        for i in range(1, 21):
            self.possible_names.add(f'mfcc_mean{i}')
            self.possible_names.add(f'mfcc_var{i}')

        for ftr_name in feature_names:
            if ftr_name not in self.possible_names: 
                raise ValueError(f'Feature name: {ftr_name} is invalid')
        self.feature_vector = self.build_feature_vector(feature_names)

    def build_feature_vector(self, feature_names):

        features = []

        for ftr in feature_names:

            if ftr == 'tempo':
                feature_arr = self.feature_table[ftr](self.signals)
                features.append(feature_arr)
                continue

            prefix, suffix = ftr.split('_')
            feature_arr = self.feature_table[prefix](self.signals)

            if 'mean' in suffix:
                if prefix == 'mfcc':
                    if suffix == 'mean':
                        feature = np.mean(feature_arr, axis=2)
                    else: 
                        i = int(suffix.split('mean')[1])
                        mfcc_i = feature_arr[:,i-1,:]
                        feature = np.mean(mfcc_i, axis=1, keepdims=True)
                else:
                    feature = np.mean(feature_arr, axis=1, keepdims=True)
            elif 'var' in suffix:
                if prefix == 'mfcc':
                    if suffix == 'var':
                        feature = np.var(feature_arr, axis=2)
                    else: 
                        i = int(suffix.split('var')[1])
                        mfcc_i = feature_arr[:,i-1,:]
                        feature = np.var(mfcc_i, axis=1, keepdims=True)
                else:
                    feature = np.var(feature_arr, axis=1, keepdims=True)

            features.append(feature)

        return np.hstack(tuple(features))

    def get_feature_vector(self):
        return self.feature_vector
