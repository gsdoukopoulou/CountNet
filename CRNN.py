import os
import keras
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, LSTM, Permute, Reshape, Dropout, MaxPooling1D, ZeroPadding2D
from keras.models import Sequential

def class_mae(y_true, y_pred): # calculate mean absolute error
    return K.mean(
        K.abs(
            K.argmax(y_pred, axis=-1) - K.argmax(y_true, axis=-1)
        ),
        axis=-1
    )


# load model
model = keras.models.load_model(
    os.path.join('models' , 'CRNN.h5') ,
    custom_objects={
        'class_mae': class_mae ,
        'exp': K.exp
    }
)

model.summary()

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

# model = Sequential([
#     ZeroPadding2D(padding=(0, 0), input_shape=(1, 500, 201), name='zeropadding2d_input_1'),
#     Conv2D(64, kernel_size=(3, 3), input_shape=(1, 500, 201), activation='relu', name='conv1'),
#     Conv2D(32, kernel_size=(3, 3), activation='relu', name='conv2'),
#     MaxPooling2D(pool_size=(3, 3), name='pool1'),
#     Conv2D(128, kernel_size=(3, 3), activation='relu', name='conv3'),
#     Conv2D(64, kernel_size=(3, 3), activation='relu', name='conv4'),
#     MaxPooling2D(pool_size=(3, 3), name='pool2'),
#     Dropout(0.5, name='dropout_1'),
#     Permute((2, 1, 3), name='permute_1'),
#     Reshape((53, 1280), name='reshape_1'),
#     LSTM(40, return_sequences=True, name='lstm_1'),
#     MaxPooling1D(pool_size=2, name='maxpooling1d_1'),
#     Flatten(name='flatten_1'),
#     Dense(11, name='dense_1'),
#     Activation('softmax', name='activation_1')
# ])

# Print the model summary
model.summary()

