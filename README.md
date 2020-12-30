# Speech Recognition

This is program uses Convolutional Neural Networks, spectrograms, and various image processing techniques to create a software package that can create and recognize words from an "audio dictionary"

[Data used for this model](https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/data)

The current accuracy (based on an unseen data set) tends to be around 95%

## Add:
* discretize the training and implementation of the neural net
* implement in tensorflowRT in order to process continuous speech
* tqdm for training progress (maybe)
* put in draw.io diagram once uploaded to github and there's an actual link

## Fix:
* P1 - occassionally (usually one per training) invalid file names are selected. This results in an error message about an attempted divide by zero operation
* P1 - The training data should be shuffled
