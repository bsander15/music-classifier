import os
import sys
import shutil
import librosa
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.metrics import classification_report, confusion_matrix
import soundfile as sf
from pickle import dump, load
import sklearn.preprocessing
from pydub import AudioSegment
from extract_features import ExtractFeatures

class MusicAgent:

    def __init__(self, model):
        self.model = model

    def predict(self, audio):
        Preprocess.segment_audio(audio,3,'predict_segments')
        features = Preprocess.extract_features('predict_segments')
        scaler = load(open('scaler.pkl', 'rb'))
        features_normalized = Preprocess.normalize(features,scaler)
        preds = self.model.predict(features_normalized)
        print(preds)
        if (np.mean(preds) > 0.5):
            return 'music'
        return 'other'


class Preprocess:
    
    def __init__(self,audio_dir):
        self.audio_dir = audio_dir

    def procces_audio(self):
        for dir in os.listdir(self.audio_files):
            for file in os.listdir(f'{self.audio_files}/{dir}'):
                f = f'{self.music_files}/{dir}/{file}'
                self.segment_audio(file,3,dir, dir)

    def segment_audio(self,audio_in,seconds,directory, from_dir):
        #can change to from_file if no difference in wav

        full_audio = AudioSegment.from_wav(f'genres/{from_dir}/{audio_in}')

        duration = full_audio.duration_seconds
        num_segments = int(duration//seconds)
        if not os.path.isdir(directory):
            os.mkdir(directory)
        for segment in range(num_segments):
            t0 = segment * seconds * 1000
            t1 = t0 + (seconds * 1000)
            full_audio[t0:t1].export(f'{directory}/{audio_in}_{segment}.wav', format='wav')
            audio, samplerate = sf.read(f'{directory}/{audio_in}_{segment}.wav')
            sf.write(f'{directory}/{audio_in}_{segment}.wav', audio, samplerate, subtype='PCM_16')

    def extract_features(self, directory, feature_names=None):
        if not feature_names:
            feature_names = ['zcr_mean', 'sc_mean', 'mfcc_mean', 'rolloff_mean', 'tempo_mean', 'mfcc_var', 'zcr_var', 'sc_var', 'rolloff_var']

        signals = np.array([librosa.load('{}/{}'.format(directory,f))[0] for f in os.listdir(directory)])

        fe = ExtractFeatures(signals, feature_names) 

        return fe.get_feature_vector()
    
    def normalize(self, features, scaler=None):
        if not scaler:
            scaler = sklearn.preprocessing.StandardScaler().fit(features)
            dump(scaler, open('scaler_genres.pkl', 'wb'))
            features_normalized = scaler.transform(features)
        else:
            print('here')
            features_normalized = scaler.transform(features)
        
        return features_normalized

    def create_training(self):
        audio_features = np.array([])
        labels = pd.DataFrame([])

        for directory in os.listdir(self.audio_dir):
            audio_features = np.vstack((audio_features,self.extract_features(f'{self.audio_dir}/{directory}')))
            labels.append(pd.DataFrame(f'{self.audio_dir}/{directory}',range(1,audio_features.shape[0]+1), columns=['labels']), ignore_index=True)

        feature_table_normalized = self.normalize(audio_features)
        
        columns = ["zcr_mean","sc_mean","mfcc_mean1","mfcc_mean2","mfcc_mean3","mfcc_mean4",
            "mfcc_mean5", "mfcc_mean6", "mfcc_mean7", "mfcc_mean8", "mfcc_mean9",
            "mfcc_mean10", "mfcc_mean11", "mfcc_mean12","mfcc_mean13", "mfcc_mean14",
            "mfcc_mean15", "mfcc_mean16", "mfcc_mean17", "mfcc_mean18", "mfcc_mean19",
            "mfcc_mean20", "rolloff_mean", "tempo", "mfcc_var1","mfcc_var2","mfcc_var3","mfcc_var4",
            "mfcc_var5", "mfcc_var6", "mfcc_var7", "mfcc_var8", "mfcc_var9",
            "mfcc_var10", "mfcc_var11", "mfcc_var12","mfcc_var13", "mfcc_var14",
            "mfcc_var15", "mfcc_var16", "mfcc_var17", "mfcc_var18", "mfcc_var19",
            "mfcc_var20", "zcr_var", "sc_var", "rolloff_var"]

        features = pd.DataFrame(feature_table_normalized, columns=columns)
        data = pd.concat([features,labels],axis=1)
        data.to_csv('data/genre_data.csv')
    

        
def main(argv):
    pass
if __name__ == '__main__':

    # ma = MusicAgent('./music', './audio', None, None)
    # ma.procces_audio()
    #main(sys.argv)

    # tb = TrainingBuilder(['music-segmented'],['other-segmented'])
    # tb.create_training()


    # ma = MusicAgent('./genres', '', None, None)
    # ma.procces_audio()
    tb = Preprocess('genres-segmented', [''])
    tb.create_training()

