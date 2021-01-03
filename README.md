# Speech Recognition

This is program uses Convolutional Neural Networks, spectrograms, and various image processing techniques to create a software package that can create and recognize words from an "audio dictionary"

[Data used for this model](https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/data)

The current accuracy (based on an unseen data set) tends to be around 95% for the .py implementation

## Add:
* discretize the training and implementation of the neural net
* implement in tensorflowRT/TFLite in order to process continuous speech

## Bugs:
* Model is using keras V1 model saving/loading protocols, review docs and update to be able to resume model training progress

## References:
* [Custom ReduceLROnPlateau Callback](https://stackoverflow.com/questions/52227286/reducelronplateau-fallback-to-the-previous-weights-with-the-minimum-acc-loss)
* [One Hot Encoding with Keras](https://www.educative.io/edpresso/how-to-perform-one-hot-encoding-using-keras)
