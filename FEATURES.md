## Feature Extraction for Audio/Music Classification

### List of relevant features
1. Spectral Centroid
     - Relevance and characteristics: Good for detecting timbre/brightness
     - Research papers:
        1. https://web.archive.org/web/20110810004531/http://icmpc8.umn.edu/proceedings/ICMPC8/PDF/AUTHOR/MP040215.PDF
2. Zero crossing rate (ZCR)
     - Relevance and characteristics: Measures frequency at which the signal crosses the x axis in the time domain; predicts/characterizes periodicity, e.g. is used for classification percusivenss of different musical tracks.
     - Research papers:
        1. https://web.archive.org/web/20181029191653/https://pdfs.semanticscholar.org/6509/14f8be2c96ab2f55faec54d3e3876c5b1b69.pdf
    
3. Mel Frequency Spectrogram Coefficients (MFCC)
     - Relevance and characteristics: Apparantely best feature for NNs; shows the logs of frequencies (Mel scale) over time, and the log/Mel amplitudes of those frequencies. Virtually all Genre classification deepl learning papers mention using this feeature.
     - Research papers:
        1. This analysis uses MFCCs for NNs, KNN, and SVM: https://cs229.stanford.edu/proj2011/HaggbladeHongKao-MusicGenreClassification.pdf
            - To make this work for KNN, they used: [KL Divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence)
     - Other materials:
        1. Blog post explaining the concept: http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/
4. Chromagram
     - Relevance and characteristics: Organizes signal into 12 different pitch classes (i.e. notes) over time; timbre/instrument agnostic but harmonic/melodic-sensitive. 
     - Research papers:
     - Other materials:
        1. Explanation: https://www.saibhaskardevatha.co.in/projects/dsp/index.html with code: https://github.com/SaiBhaskarDevatha/Music-Genre-Classification
            - They created a feature vector of chromogram feature means; also summarize feature vectors (of different feature types) in well-known papers.

### Music/Other Classifier

  - ### Data sources, testing/training data

  - #### Preprocessing steps
    1. Merge original voice data set with other miscellaneous sounds (including animals, traffic, computer noise)
    2. In initially exracting the ZCR and Spectral Centroid features from these files (both music and other data sets), we found outliers in the non-music (or other) data set. In considering that some files were in Stereo format, and others were in Mono format, we normalized the data set by converting the stereo samples to mono. In doing this, we found that: [result pending].
    3. Other considerations: adjusting for loudness, removing silences, segmenting the wav files into segments.

  - #### Features to extract


### Musical Genre Classifier

  - ### Data sources, testing/training data
    - Source: GTZAN data set (this source has been cited by virtually every research paper we've read, and is said to be effective training data)
    - Link to [GTZAN data set](http://marsyas.info/downloads/datasets.html#)
    - This data source was used in this well known paper: [" Musical genre classification of audio signals " by G. Tzanetakis and P. Cook in IEEE Transactions on Audio and Speech Processing 2002](https://ieeexplore.ieee.org/document/1021072/figures#figures)

  - #### Preprocessing steps

  - #### Features to extract (these will likely also apply to the other classifier, albeit with different features (de)emphasized or omitted)
    - MFCC for NN (matrix, most popular method, it seems)
    - Chromogram vector
    - Vector that is assortment of features (e.g. Spectral centroid, ZCR)

