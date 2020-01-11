# speech_recognition

This is a personal python project which aims to use Convolutional Neural Networks, spectrograms, and various image processing techniques to create a software package that can create and recognize words from an "audio dictionary"

The current accuracy (based on an unseen data set) tends to be around 95%

Currently the program: </br>
asks the user how many files to use for training/testing
looks at audio samples which are organized in "program_directory/word_to_be_learned/audio_clip.wav" format
creates an excel sheet with the first column containing a word_to_be_learned entry for each word and a random selection of files from that folder based on the number of training/testing samples specified
runs depth-first through the excel file and converts each wav file to a greyscale spectrogram storing it in x_train or x_test
creates an appropriate y_train or y_test entry
runs the testing/training data through keras to create the model and train it

NOTES: </br>
* this code is very rough and was/is just a learning exercise that wasn't intended to see the light of day as it currently stands
* the data used to train this model was found at: https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/data

Main functionalities that need to be added: </br>
* discretize the training and implementation of the neural net
* implement in tensorflowRT in order to process continuous speech

DISCLAIMER!!! There are many glitchy bits in the code which need to be addressed, the core ones afecting funtionality are:</br>
* The reshape function does not work with different train/test input values and exits the program

Lower priority bugs/features to be worked out: </br>
* it's possible to grab repeats of the same file with the way that they're currently selected
* the very first training file is not properly processed
* occassionally (usually one per training) invalid file names are selected. This results in an error message about an attempted divide by zero operation
* The training/testing data should be fed into keras randomly
* The progress messages only represent the progress for a given word, would be better to process all training data while displaying progress and then all testing data while displaying progress
