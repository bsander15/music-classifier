

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

        for ftr_name in feature_names:
            if ftr_name not in self.possible_names: 
                raise ValueError(f'Feature name: {ftr_name} is invalid')
        self.feature_vector = self.build_feature_vector(feature_names)

    def build_feature_vector(self, feature_names):

        features = []

        for ftr in feature_names:
            prefix, suffix = ftr.split('_')
            feature_arr = self.feature_table[prefix](self.signals)

            if suffix == 'mean':
                if prefix == 'mfcc':
                    feature = np.mean(feature_arr, axis=2)
                else:
                    feature = np.mean(feature_arr, axis=1, keepdims=True)
            elif suffix == 'var':
                if prefix == 'mfcc':
                    feature = np.var(feature_arr, axis=2)
                else:
                    feature = np.var(feature_arr, axis=1, keepdims=True)

            features.append(feature)

        return np.hstack(tuple(features))

    def get_feature_vector(self):
        return self.feature_vector
