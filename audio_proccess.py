import os
from pydub import AudioSegment

musicDir = os.fsencode('data/music_wav')
otherDir1 = os.fsencode('data/other_wav1')
otherDir2 = os.fsencode('data/other_wav2')
otherDir3 = os.fsencode('data/other_wav3')
directories = [musicDir, otherDir1, otherDir2, otherDir3]

for directory in directories:
    direct = os.fsdecode(directory)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        newAudio = AudioSegment.from_wav('{}/{}'.format(direct,filename))
        slices = filename.split('.')
        for i in range(10):
            t0 = i*3*1000
            t1 = (t0 + 3000)
            name = '{}_{}.{}'.format(slices[0],i,slices[1])
            segment = newAudio[t0:t1]
            segment.export('{}_3sec/{}'.format(direct,name), format='wav')
