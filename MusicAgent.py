import os
import sys
import shutil
import librosa
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.metrics import classification_report, confusion_matrix
import soundfile as sf
from pickle import dump, load
import sklearn.preprocessing
from pydub import AudioSegment
from classifiers.MOClassifier import MusicKNN
from extract_features import ExtractFeatures


class MusicAgent:
    
    def __init__(self,audio, music_other_model, genre_model):
        self.audio = audio
        self.music_other_model = music_other_model
        self.genre_model = genre_model


    # def procces_audio(self):


    def segment_audio(self,seconds):
        #can change to from_file if no difference in wav
        full_audio = AudioSegment.from_wav(self.audio)
        duration = full_audio.duration_seconds
        num_segments = int(duration//seconds)
        if os.path.isdir('segments'):
            shutil.rmtree('segments')
        os.mkdir('segments')
        for segment in range(num_segments):
            t0 = segment * seconds * 1000
            t1 = t0 + (seconds * 1000)
            full_audio[t0:t1].export('segments/file_{}.wav'.format(segment), format='wav')
            audio, samplerate = sf.read('segments/file_{}.wav'.format(segment))
            sf.write('segments/file_{}.wav'.format(segment), audio, samplerate, subtype='PCM_16')

    def extract_features(self, directory, feature_names=None):
        if not feature_names:
            feature_names = ['zcr_mean', 'sc_mean', 'mfcc_mean', 'mfcc_var', 'zcr_var', 'sc_var']

        signals = np.array([librosa.load('{}/{}'.format(directory,f))[0] for f in os.listdir(directory)])

        fe = ExtractFeatures(signals, feature_names) 

        return fe.get_feature_vector()
    
    def normalize(self, features, scaler=None):
        if not scaler:
            scaler = sklearn.preprocessing.StandardScaler().fit(features)
            dump(scaler, open('scaler.pkl', 'wb'))
            features_normalized = scaler.transform(features)
        else:
            print('here')
            features_normalized = scaler.transform(features)
        
        return features_normalized
 
    
    def predict(self):
        self.segment_audio(3)
        features = self.extract_features('segments')
        scaler = load(open('scaler.pkl', 'rb'))
        features_normalized = self.normalize(features,scaler)
        preds = self.music_other_model.predict(features_normalized)
        print(preds)
        if (np.mean(preds) > 0.5):
            return 'music'
        return 'other'


class TrainingBuilder(MusicAgent):

    def __init__(self, music_dir, other_dir):
        self.music_dir = music_dir
        self.other_dir = other_dir

    def create_training(self):
        music_features = np.array([])
        for directory in self.music_dir:
            if music_features.size == 0:
                music_features = super().extract_features(directory)
            else:
                music_features = np.vstack((other_features,super().extract_features(directory)))
        music_labels = np.ones((music_features.shape[0],1))

        other_features = np.array([])
        for directory in self.other_dir:
            if other_features.size == 0:
                other_features = super().extract_features(directory)
            else:
                other_features = np.vstack((other_features,super().extract_features(directory)))
        other_labels = np.zeros((other_features.shape[0],1))

        labels = np.vstack((music_labels,other_labels))

        feature_table = np.vstack((music_features,other_features))
        feature_table_normalized = super().normalize(feature_table)

        training_data = np.hstack((feature_table_normalized,labels))
        print(training_data.shape)
        np.savetxt('data.csv', training_data, delimiter=',')
    

        
def main(argv):
    # ma = MusicAgent(argv[1])
    # print(ma.audio)
    # ma.segment_audio(3)
    # ma.extract_features()
    # training = TrainingBuilder(['music_wav_3sec'],['other_wav1_3sec','other_wav2_3sec','other_wav3_3sec'])
    # training.create_training()
    data = np.genfromtxt('data/data.csv', delimiter=',')
    X = data[:,:44]
    y = data[:,44]
    knn = MusicKNN(X,y,5)
    ma = MusicAgent(argv[1],knn,None)
    print(ma.predict())
    # preds = knn.predict()
    # confusion_matrix, classification_report = knn.metrics(preds)
    # print(confusion_matrix, classification_report)

if __name__ == '__main__':
    main(sys.argv)



