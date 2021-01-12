import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from skimage.measure import block_reduce

def save_model_to_disk(model):
    """
    Converts the model to a json and saves as an h5 file
    """
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")


def graph_spectrogram(wav_file, nfft=512, noverlap=511):
    """
    Converts the wav file to a B/W spectrogram for the NN to interpret
    """
    rate, data = wavfile.read(wav_file)
    fig,ax = plt.subplots(1)
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    ax.axis('off')
    pxx, freqs, bins, im = ax.specgram(x=data,
                                    Fs=rate,
                                    cmap="gray",
                                    noverlap=noverlap,
                                    NFFT=nfft)
#     pxx, freqs, bins, im = ax.specgram(x=data, Fs=rate, mode="magnitude", cmap="gray", noverlap=noverlap, NFFT=nfft)
    ax.axis('off')
    plt.rcParams['figure.figsize'] = [0.75,0.5]
#     plt.rcParams['figure.figsize'] = [3,2]

    fig.canvas.draw()
    size_inches  = fig.get_size_inches()
    dpi          = fig.get_dpi()
    width, height = fig.get_size_inches() * fig.get_dpi()

    mplimage = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    # image is greyscale so all channels are the same, just need one channel
    imarray = np.reshape(mplimage, (int(height), int(width), 3))[:,:,1]
    plt.close(fig)
    # normalize AKA min/max scale the array
    imarray = (imarray - imarray.min())/(imarray.max() - imarray.min())
    return imarray

def load_waves(fpath):
    fpath = 'train/audio/' + fpath
    normgram = graph_spectrogram(fpath)
    redgram = block_reduce(normgram, block_size = (3,3), func = np.mean)
    # looks like block_reduce here is gaussian blur w/out padding
    return redgram

def plot_results(history):
    """
    plot the results from training the keras model. Maybe move this to a new file along with the new keras def
    """
    # changes figsize back to default size
    plt.rcParams['figure.figsize'] = [6.4, 4.8]
    
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.ylim(0,1)
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()