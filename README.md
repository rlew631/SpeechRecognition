# Speech Recognition

This is program uses Convolutional Neural Networks, spectrograms, and various image processing techniques to create a software package that can create and recognize words from an "audio dictionary"

[Data used for this model](https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/data)

The current accuracy (based on an unseen data set) tends to be around 95% for the .py implementation

## Add:
* discretize the training and implementation of the neural net
* implement in tensorflowRT in order to process continuous speech
* add conv2d layer(s) or copy architecture from original .py file to the notebook

## Fix:
* P1 - "divide by zero" error when normalizing. This means max and min vals are the same and the sound file is empty or there was an error reading it. Update to not append these
