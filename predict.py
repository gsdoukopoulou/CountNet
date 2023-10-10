import numpy as np
import soundfile as sf
import argparse
import os
import keras
import sklearn
import librosa
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, LSTM, Permute, Reshape, Dropout, MaxPooling1D, ZeroPadding2D
from keras.models import Sequential


eps = np.finfo(np.float64).eps # np.float64 instead of np.float


def class_mae(y_true, y_pred): # calculate mean absolute error
    return K.mean(
        K.abs(
            K.argmax(y_pred, axis=-1) - K.argmax(y_true, axis=-1)
        ),
        axis=-1
    )


def count(audio, model, scaler):
    # compute STFT
    # len(audio) (80000,) (D*fs, D = 5s, fs = 16KHz)

    X = np.abs(librosa.stft(audio, n_fft=400, hop_length=160)).T # hop length: 10ms(hop size) * fs

    # apply global (featurewise) standardization to mean1, var0
    X = scaler.transform(X)

    # cut to input shape length (500 frames x 201 STFT bins)
    X = X[:500, :]

    # apply l2 normalization
    Theta = np.linalg.norm(X, axis=1) + eps
    X /= np.mean(Theta)

    # add sample dimension
    X = X[np.newaxis, ...]

    if len(model.input_shape) == 4:
        X = X[:, np.newaxis, ...]
    print(X.shape)
    print(model.input_shape)
    ys = model.predict(X, verbose=0) # as it is X is (1, 1, 500, 201)
    return np.argmax(ys, axis=1)[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Load keras model and predict speaker count'
    )

    parser.add_argument(
        'audio',
        help='audio file (samplerate 16 kHz) of 5 seconds duration'
    )

    # parser.add_argument(
    #     '--model', default='CRNN',
    #     help='model name'
    # )

    args = parser.parse_args()

    # # load model
    # model = keras.models.load_model(
    #     os.path.join('models', args.model + '.h5'),
    #     custom_objects={
    #         'class_mae': class_mae,
    #         'exp': K.exp
    #     }
    # )

    model = Sequential()
    model.add(ZeroPadding2D(padding=(0 , 0) , dim_ordering='default' , input_shape=(1 , 500 , 201) , name='zer'))
    model.add(Conv2D(64 , 3 , 3 , border_mode='valid' , activation='relu' , name='conv1' , dim_ordering='th'))
    model.add(Conv2D(32 , 3 , 3 , border_mode='valid' , activation='relu' , name='conv2' , dim_ordering='th'))
    model.add(MaxPooling2D(pool_size=(3 , 3) , border_mode='valid' , name='pool1' , dim_ordering='th'))
    model.add(Dropout(0.5 , name='dropout_1'))
    model.add(Conv2D(128 , 3 , 3 , border_mode='valid' , activation='relu' , name='conv3' , dim_ordering='th'))
    model.add(Conv2D(64 , 3 , 3 , border_mode='valid' , activation='relu' , name='conv4' , dim_ordering='th'))
    model.add(MaxPooling2D(pool_size=(3 , 3) , border_mode='valid' , name='pool2' , dim_ordering='th'))
    model.add(Permute((2 , 1 , 3) , name='permute_1'))
    model.add(Reshape((53 , 1280) , name='reshape_1'))
    model.add(LSTM(40 , return_sequences=True , name='lstm_1'))
    model.add(MaxPooling1D(pool_length=2 , name='maxpooling1d_1'))
    model.add(Flatten(name='flatten_1'))
    model.add(Dense(11 , name='dense_1'))
    model.add(Activation('softmax' , name='activation_1'))

    model.compile(optimizer='adam' , loss='categorical_crossentropy' , metrics=[class_mae])

    model.load_weights('models/CRNN.h5')

    # print model configuration
    model.summary()
    # save as svg file
    # load standardisation parameters
    scaler = sklearn.preprocessing.StandardScaler()
    with np.load(os.path.join("models", 'scaler.npz')) as data:
        scaler.mean_ = data['arr_0']
        scaler.scale_ = data['arr_1']

    # compute audio
    audio, rate = sf.read(args.audio, always_2d=True)

    # downmix to mono
    audio = np.mean(audio, axis=1)
    estimate = count(audio, model, scaler)
    print("Speaker Count Estimate: ", estimate)
