
import os
import math
import pdb
import json
import librosa


"""
Run this file to rewrite genre-mfccs.json with MFFCs (Mel Frequency Cepstrum Coefficients) for each segment of audio file

parameters are subject to change (this file was copied from an online tutorial)
"""

mfccs_json_file = 'mfccs.json'
genres_dir = os.path.join(os.getcwd())
SAMPLE_RATE = 22050
DURATION = 30 # measure in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

def make_genre_map():
    genre_map = {}

    for dir_name, sub_dirs, wav_files in os.walk(genres_dir):
        if dir_name is not genres_dir:
            genre_name = dir_name.split('/')[-1]
            genre_map[genre_name] = wav_files

    return genre_map


def save_mfccs(genre_map, json_path, n_mfcc=13, n_fft=2048, \
    hop_length=512, num_segments=5):
    
    data = {
        'mapping': [],
        'mfcc': [],
        'labels': [], 
    }

    num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length)


    for i, (genre, wav_files) in enumerate(genre_map.items()):

        data['mapping'].append(genre)

        for wav_file in wav_files:

            file_path = os.path.join(genres_dir, genre, wav_file)
            # pdb.set_trace()

            # load audio file
            signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

            # process segments extracting mfcc and storing data
            for s in range(num_segments):
                start_sample = num_samples_per_segment * s 
                finish_sample = start_sample + num_samples_per_segment

                mfcc = librosa.feature.mfcc(signal[start_sample:finish_sample],
                        sr=sr, # sample rate
                        n_fft=n_fft,
                        n_mfcc=n_mfcc,
                        hop_length=hop_length
                    )

                mfcc = mfcc.T #transpose of the mfcc

                # store mfcc for segment if it has the expected length
                if len(mfcc) == expected_num_mfcc_vectors_per_segment:
                    data['mfcc'].append(mfcc.tolist()) # cast from numpy array to a normal python list
                    data['labels'].append(i-1)
                    print(f'segment: {s}')

    with open(json_path, 'w') as fp:
        json.dump(data, fp, indent=4)

if __name__ == '__main__':
    genre_map = make_genre_map()
    save_mfccs(genre_map, json_path='mfccs.json')

