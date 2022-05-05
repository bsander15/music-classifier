import os
from pdb import post_mortem
import sys
import shutil
from telnetlib import GA
import librosa
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import soundfile as sf
from pickle import dump, load
import sklearn.preprocessing
from pydub import AudioSegment
from classifiers.MOClassifier import MOClassifier
from extract_features import ExtractFeatures
from select_features import GAKnn, GANets

class MusicAgent:

    def __init__(self, model, binary_dir, genres_dir):
        self.model = model
        self.binary_dir = binary_dir
        self.genres_dir = genres_dir
        self.binary_data_full = None
        self.genres_data_full = None
        self.binary_params = None
        self.genres_params = None
        self.binary_model_optimal = MOClassifier(model)
        self.genres_model_optimal = MOClassifier(model)

    def predict(self, audio):
        print('PREDICTING GIVEN AUDIO FILE')
        p = Preprocess('predict')
        p.segment_audio(audio,3,'audio')
        binary_features = p.extract_features('predict_segmented/audio', self.binary_params[0])
        p.normalize(self.binary_data_full[self.binary_params[0]])
        scaler = load(open(f'scaler_predict.pkl', 'rb'))
        binary_features_normalized = p.normalize(binary_features,scaler)
        preds = self.binary_model_optimal.predict(binary_features_normalized)

        if self.predict_helper(preds) == 'music':
            genres_features = p.extract_features('predict_segmented/audio', self.genres_params[0])
            p.normalize(self.genres_data_full[self.genres_params[0]])
            scaler = load(open(f'scaler_predict.pkl','rb'))
            genres_features_normalized = p.normalize(genres_features,scaler)
            preds = self.genres_model_optimal.predict(genres_features_normalized)

            return self.predict_helper(preds)
        
        return 'Not a Music File'

    def predict_helper(self, preds):
        labels, pos = np.unique(preds, return_inverse=True)
        frequency = np.bincount(pos)
        maxpos = frequency.argmax()

        return labels[maxpos]
        
    
    def optimize_model(self):
        print('OPTIMIZING MODEL')
        if isinstance(self.model,KNeighborsClassifier):
            binary_ga = GAKnn(f'data/{self.binary_dir}_segmented_data.csv')
            genres_ga = GAKnn(f'data/{self.genres_dir}_segmented_data.csv')
            self.binary_params = binary_ga.optimize()[0]
            self.genres_params = genres_ga.optimize()[0]
            self.binary_model_optimal.setClassifier(KNeighborsClassifier(self.binary_params[1]))
            self.genres_model_optimal.setClassifier(KNeighborsClassifier(self.genres_params[1]))

        else:
            binary_ga = GANets(f'data/{self.binary_dir}_segmented_data.csv')
            genres_ga = GANets(f'data/{self.genres_dir}_segmented_data.csv')
            self.binary_params = binary_ga.optimize()[0]
            self.genres_params = genres_ga.optimize()[0]
            binary_net = np.full((self.binary_params[1]),self.binary_params[2])
            genres_net = np.full((self.genres_params[1]),self.genres_params[2])

            self.binary_model_optimal.setClassifier(MLPClassifier(binary_net))
            self.genres_model_optimal.setClassifier(MLPClassifier(genres_net))

        self.binary_model_optimal.fit(self.binary_data_full[self.binary_params[0]],self.binary_data_full.iloc[:,-1])
        self.genres_model_optimal.fit(self.genres_data_full[self.genres_params[0]],self.genres_data_full.iloc[:,-1])

    def preproccess(self):
        print('PREPROCCESSING DATA')
        if self.binary_dir:
            Preprocess(self.binary_dir).process_audio()
            self.binary_data_full = Preprocess(f'{self.binary_dir}_segmented').create_training()
        if self.genres_dir:
            Preprocess(self.genres_dir).process_audio()
            self.genres_data_full = Preprocess(f'{self.genres_dir}_segmented').create_training()

class Preprocess:
    
    def __init__(self,audio_dir):
        self.audio_dir = audio_dir

    def process_audio(self):
        for dir in os.listdir(self.audio_dir):
            if (dir != '.DS_Store'):
                for file in os.listdir(f'{self.audio_dir}/{dir}'):
                    self.segment_audio(file,3,dir)

    def segment_audio(self,audio_in,seconds,directory):
        print("SEGMENTING AUDIO")
        #can change to from_file if no difference in wav

        full_audio = AudioSegment.from_wav(f'{self.audio_dir}/{directory}/{audio_in}')

        duration = full_audio.duration_seconds
        num_segments = int(duration//seconds)
        if not os.path.isdir(f'{self.audio_dir}_segmented/{directory}'):
            os.makedirs(f'{self.audio_dir}_segmented/{directory}')
        for segment in range(num_segments):
            t0 = segment * seconds * 1000
            t1 = t0 + (seconds * 1000)
            full_audio[t0:t1].export(f'{self.audio_dir}_segmented/{directory}/{segment}_{audio_in}', format='wav')
            audio, samplerate = sf.read(f'{self.audio_dir}_segmented/{directory}/{segment}_{audio_in}')
            sf.write(f'{self.audio_dir}_segmented/{directory}/{segment}_{audio_in}', audio, samplerate, subtype='PCM_16')

    def extract_features(self, directory, feature_names=None):
        print("EXTRACTING FEATURES")
        if not feature_names:
            feature_names = ['zcr_mean', 'sc_mean', 'mfcc_mean', 'rolloff_mean', 'tempo_mean', 'mfcc_var', 'zcr_var', 'sc_var', 'rolloff_var']

        signals = np.array([librosa.load('{}/{}'.format(directory,f))[0] for f in os.listdir(directory) if f != '.DS_Store'])
        fe = ExtractFeatures(signals, feature_names) 
        return fe.get_feature_vector()
    
    def normalize(self, features, scaler=None):
        print("NORMALIZING DATA")
        if not scaler:
            scaler = sklearn.preprocessing.StandardScaler().fit(features)
            dump(scaler, open(f'scaler_{self.audio_dir}.pkl', 'wb'))
            features_normalized = scaler.transform(features)
        else:
            features_normalized = scaler.transform(features)
        
        return features_normalized

    def create_training(self):
        print("CREATING TRAINING DATA")
        audio_features = pd.DataFrame()
        labels = pd.DataFrame()
        columns = ["zcr_mean","sc_mean","mfcc_mean1","mfcc_mean2","mfcc_mean3","mfcc_mean4",
            "mfcc_mean5", "mfcc_mean6", "mfcc_mean7", "mfcc_mean8", "mfcc_mean9",
            "mfcc_mean10", "mfcc_mean11", "mfcc_mean12","mfcc_mean13", "mfcc_mean14",
            "mfcc_mean15", "mfcc_mean16", "mfcc_mean17", "mfcc_mean18", "mfcc_mean19",
            "mfcc_mean20", "rolloff_mean", "tempo", "mfcc_var1","mfcc_var2","mfcc_var3","mfcc_var4",
            "mfcc_var5", "mfcc_var6", "mfcc_var7", "mfcc_var8", "mfcc_var9",
            "mfcc_var10", "mfcc_var11", "mfcc_var12","mfcc_var13", "mfcc_var14",
            "mfcc_var15", "mfcc_var16", "mfcc_var17", "mfcc_var18", "mfcc_var19",
            "mfcc_var20", "zcr_var", "sc_var", "rolloff_var"]

        for directory in os.listdir(self.audio_dir):
            if directory != '.DS_Store':
                dir_features = pd.DataFrame(self.extract_features(f'{self.audio_dir}/{directory}'), columns=columns)
                audio_features = pd.concat([audio_features, dir_features], ignore_index=True)
                labels = pd.concat([labels,pd.DataFrame(f'{directory}',range(1,dir_features.shape[0]+1), columns=['labels'])], ignore_index=True)

        feature_table_normalized = self.normalize(audio_features)
        


        features = pd.DataFrame(feature_table_normalized, columns=columns)
        data = pd.concat([features,labels],axis=1)
        data.to_csv(f'data/{self.audio_dir}_data.csv')

        return data
    

        
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
    # tb = Preprocess('genres')
    # # tb.procces_audio()
    # tb.procces_audio()
    # tb.extract_features('genres/metal')
    # for dir in os.listdir('genres-segmented'):
    #     print(len(os.listdir(f'genres-segmented/{dir}')))
    ma = MusicAgent(KNeighborsClassifier(), 'binary-clips', 'genres-clips')
    ma.preproccess()
    ma.optimize_model()
    print(ma.predict('rock.00023.wav'))
