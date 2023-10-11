import os
import keras
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, LSTM, Permute, Reshape, Dropout, MaxPooling1D, ZeroPadding2D, ZeroPadding1D, Conv1D
from keras.models import Sequential

def class_mae(y_true, y_pred): # calculate mean absolute error
    return K.mean(
        K.abs(
            K.argmax(y_pred, axis=-1) - K.argmax(y_true, axis=-1)
        ),
        axis=-1
    )

model = Sequential()
model.add(ZeroPadding2D(padding=(0, 0), dim_ordering='default', input_shape=(1, 500, 201), name='zero1'))
model.add(Conv2D(64, 3, 3, border_mode='valid', activation='relu', name='conv1', dim_ordering='th'))
model.add(Conv2D(32, 3, 3, border_mode='valid', activation='relu', name='conv2', dim_ordering='th'))
model.add(MaxPooling2D(pool_size=(3, 3), border_mode='valid', name='pool1', dim_ordering='th'))
model.add(Dropout(0.5, name='dropout_1'))
model.add(Conv2D(128, 3, 3, border_mode='valid', activation='relu', name='conv3', dim_ordering='th'))
model.add(Conv2D(64, 3, 3, border_mode='valid', activation='relu', name='conv4', dim_ordering='th'))
model.add(MaxPooling2D(pool_size=(3, 3), border_mode='valid', name='pool2', dim_ordering='th'))
model.add(Permute((2, 1, 3), name='permute_1'))
model.add(Reshape((53, 1280), name='reshape_1'))
model.add(LSTM(40, return_sequences=True, name='lstm_1'))
model.add(MaxPooling1D(pool_length=2, name='maxpooling1d_1'))
model.add(Flatten(name='flatten_1'))
model.add(Dense(11, name='dense_1'))
model.add(Activation('softmax', name='activation_1'))
model.summary()
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[class_mae])
#
# model.load_weights('models/CRNN.h5')

# Print the model summary

model = Sequential()
# model.add(ZeroPadding1D(padding=0 , input_shape=(1 , 80000) , name='zero1'))
model.add(Conv1D(64 , 3 , border_mode='valid' , activation='relu' , name='conv1', input_shape=(80000, 1)))
# model.add(Conv1D(64 , kernel_size=3 , activation='relu' , name='conv2'))
# model.add(MaxPooling1D(pool_size=3 , name='pool1'))
# model.add(LSTM(40 , return_sequences=True , name='lstm'))
# model.add(MaxPooling1D(pool_size=2 , name='pool2'))
# num_classes = 11  # Change this based on your classification task
# model.add(Dense(num_classes , name='dense_1'))
# model.add(Activation('softmax' , name='activation_1'))
model.summary()

