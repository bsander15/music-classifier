import os
import sys
import shutil
import librosa
import numpy as np
import soundfile as sf
import sklearn.preprocessing
from pydub import AudioSegment
from classifiers.music_other_knn import MusicKNN



class MusicAgent:
    
    def __init__(self,audio):
        self.audio = audio
        self.segments = []

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

    def extract_features(self, directory):
        signals = np.array([librosa.load('{}/{}'.format(directory,f))[0] for f in os.listdir(directory)])

        zcr = np.array([librosa.feature.zero_crossing_rate(x)[0] for x in signals])
        sc = np.array([librosa.feature.spectral_centroid(y=x)[0] for x in signals])
        mfcc = librosa.feature.mfcc(y=signals)

        zcr_mean = np.mean(zcr, axis=1, keepdims=True)
        sc_mean = np.mean(sc, axis=1, keepdims=True)
        mfcc_mean = np.mean(mfcc, axis=2)

        zcr_var = np.var(zcr, axis=1, keepdims=True)
        sc_var = np.var(sc, axis=1, keepdims=True)
        mfcc_var = np.var(mfcc, axis=2)

        feature_table = np.hstack((zcr_mean,sc_mean,mfcc_mean,zcr_var,sc_var,mfcc_var))
        
        return feature_table



def main(argv):
    ma = MusicAgent(argv[1])
    print(ma.audio)
    ma.segment_audio(3)
    ma.extract_features()

if __name__ == '__main__':
    main(sys.argv)



