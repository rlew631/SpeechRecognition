import sqlite3
from sqlite3 import Error
import numpy as np
from speech_dependencies import save_model_to_disk, graph_spectrogram, load_waves
import keras
import matplotlib.pyplot as plt
import seaborn as sns

"""
These functions are used to work with storing/loading the data into sql
"""

def create_connection(db_file):
    """ create a database connection to a SQLite database """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        print(sqlite3.version)
    except Error as e:
        print(e)
    return conn

def encode_array_for_sql(array):
    """ takes in a numpy array and outputs a string to store in sql """
    return array.tobytes().decode('ISO-8859-1')

def decode_array_from_sql(string,x,y):
    """ takes in the string to turn into an array and its dimensions """
    return np.frombuffer(string.encode('ISO-8859-1')).reshape(y,x)

def train_val_load(X, y, mapping, num_classes, train_or_val):
    """
    Loads the sound clips and returns the spectrograms/labels
    """
    if train_or_val == 'train':
        print('loading training set')
    else:
        print('loading validation set')
        
    sound_clips = []
    for ind, fpath in enumerate(X):
        # update code to `try` loading pickles
        # or preprocess sound clips and save as pickles
        redgram = load_waves(fpath)
        if redgram.shape[0] != 14:
            print('\tbad redgram shape')
            # if experiencing sporadic accuracy between batches
            # see if it's on the 'bad normgram shape' batches
            # might be an indexing error
            y = np.delete(y, ind)
            continue
        sound_clips.append(redgram)
    sound_clips = np.asarray(sound_clips, dtype=np.float32)
#     sound_clips = np.array(sound_clips, dtype=np.float32)
    print(f'\tx shape (before reshape):{sound_clips.shape}')
    sound_clips = sound_clips.reshape( # maybe this is the error? looks like one of the sound clips had a super choppy spectrogram
                                sound_clips.shape[0],
                                sound_clips.shape[1], 
                                sound_clips.shape[2],1)
    print(f'\tx shape (after reshape):{sound_clips.shape}')
    
    for y_ind in range(len(y)):
        y[y_ind] = mapping[y[y_ind]]
    y = keras.utils.to_categorical(y, num_classes) # this might be the part that's fubar, try keras ohe. should still work decent for the number of samples tho?
    print(f'\ty shape:{y.shape}')
    
    # to see if the spectrogram looks decent
    sns.heatmap(sound_clips[0][:,:,0])
    plt.show()
    
    return sound_clips, y