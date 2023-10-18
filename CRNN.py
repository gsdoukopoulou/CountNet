import os
import keras
import h5py
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, LSTM, Permute, Reshape, Dropout, MaxPooling1D, ZeroPadding2D, ZeroPadding1D, Conv1D
from keras.models import Sequential, load_model

# def class_mae(y_true, y_pred): # calculate mean absolute error
#     return K.mean(
#         K.abs(
#             K.argmax(y_pred, axis=-1) - K.argmax(y_true, axis=-1)
#         ),
#         axis=-1
#     )

model = Sequential()
model.add(Conv1D(64 , 3 , border_mode='same' , input_shape=(500, 1) , name='encoder'))  # input_shape=(1 , 80000),
model.add(Reshape((1 , 500 , 64) , name='reshape_layer'))

model.add(ZeroPadding2D(padding=(0, 0), dim_ordering='default', name='zero1')) #, input_shape=(1, 500, 201)
model.add(Conv2D(64, 3, 3, border_mode='valid', activation='relu', name='conv1', dim_ordering='th'))
model.add(Conv2D(32, 3, 3, border_mode='valid', activation='relu', name='conv2', dim_ordering='th'))
model.add(MaxPooling2D(pool_size=(3, 3), border_mode='valid', name='pool1', dim_ordering='th'))
model.add(Dropout(0.5, name='dropout_1'))
model.add(Conv2D(128, 3, 3, border_mode='valid', activation='relu', name='conv3', dim_ordering='th'))
model.add(Conv2D(64, 3, 3, border_mode='valid', activation='relu', name='conv4', dim_ordering='th'))
model.add(MaxPooling2D(pool_size=(3, 3), border_mode='valid', name='pool2', dim_ordering='th'))
model.add(Permute((2, 1, 3), name='permute_1'))
model.add(Reshape((53, 320), name='reshape_1'))
# model.add(LSTM(40, return_sequences=True, name='lstm_1'))
# model.add(MaxPooling1D(pool_length=2, name='maxpooling1d_1'))
# model.add(Flatten(name='flatten_1'))
# model.add(Dense(11, name='dense_1'))
# model.add(Activation('softmax', name='activation_1'))
model.summary()

# model = Sequential()
# model.add(ZeroPadding2D(padding=(0, 0), dim_ordering='default', name='zero1')) #, input_shape=(1, 500, 201)
# model.add(Conv2D(64, 3, 3, border_mode='valid', activation='relu', name='conv1', dim_ordering='th'))
# model.add(Conv2D(32, 3, 3, border_mode='valid', activation='relu', name='conv2', dim_ordering='th'))
# model.add(MaxPooling2D(pool_size=(3, 3), border_mode='valid', name='pool1', dim_ordering='th'))
# model.add(Dropout(0.5, name='dropout_1'))
# model.add(Conv2D(128, 3, 3, border_mode='valid', activation='relu', name='conv3', dim_ordering='th'))
# model.add(Conv2D(64, 3, 3, border_mode='valid', activation='relu', name='conv4', dim_ordering='th'))
# model.add(MaxPooling2D(pool_size=(3, 3), border_mode='valid', name='pool2', dim_ordering='th'))
# model.add(Permute((2, 1, 3), name='permute_1'))
# model.add(Reshape((53, 1280), name='reshape_1'))
# model.add(LSTM(40, return_sequences=True, name='lstm_1'))
# model.add(MaxPooling1D(pool_length=2, name='maxpooling1d_1'))
# model.add(Flatten(name='flatten_1'))
# model.add(Dense(11, name='dense_1'))
# model.add(Activation('softmax', name='activation_1'))
# model.summary()
#
# #
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[class_mae])
#
# # model.load_weights('models/CRNN.h5', by_name = True)
#
# # # # step 1: load all the weights of the pretrained model
# pretrained_model = load_model('models/CRNN.h5',custom_objects={'class_mae': class_mae,'exp': K.exp})
# conv1_pretrained = pretrained_model.get_layer('conv1')
# conv1_pretrained.add(Permute((3, 2, 4, 1) , name='permute_1'))
# # pretrained_weights = pretrained_weights_file['conv1']
# # reshaped_weights = conv1_pretrained.transpose(3, 1, 2, 0)
# model.get_layer('conv1').set_weights([conv1_pretrained])

# import glob
# # from scipy.io.wavfile import read
# import soundfile as sf
# import numpy as np
#
# # wavs = []
# # for filename in glob.glob('*.wav'):
# #     print(filename)
# #     wavs.append(read(filename))
#
# from pathlib import Path
# # base_path = Path(r"/home/gsdoukopoul/data/test")
# base_path = Path(r"D:\Documents\CNS\internship\CountNet\CountNet\test")
#
# wavs = []
# for filename in base_path.glob("10_*.wav"):
#     wavs.append(sf.read(filename))
#
# audio = wavs[-1][0]
# print(len(wavs))
#     # do something, e.g. with open(wav_file_path) as wav_file:
#
# y_true = np.ones(80000) * 10
#
# print(y_true)