

### Audio Feature Selection and Classification

For this project, we wrote a genetic algorithm to find the near-optimal feature vectors for 
our audio classifiers. We have two classifiers, which each use a neural net and a KNN model 
to classify audio files as 1) music or non-music and 2) into separate musical genres. We selected
a set of common audio features used to describe qualities of the music such as brightness, tempo,
percusiveness, noise, etc. We then run a genetic algorithm on this initial population and evaluate each 
of the individuals (feature vectors) by training the models with their data and getting their accuracy.
After n number of generations, we return the fittest individual, i.e. the feature vector with the greatest
accuracy.

Index of the repositiory:

1. /classifiers
2. /experimental-code -- you can ignore this directory
3. extract_features.py
4. MusicAgent.py 
5. select_features.py
6. scaler_genres.pkl
7. scaler.pkl
